import os
import math
from datetime import datetime, timezone
from typing import Dict, Any, List, Set, Tuple
import time
import random
import re

from flask import Flask, render_template, request, jsonify
import logging
import requests
from bs4 import BeautifulSoup  # type: ignore
from urllib.parse import unquote

try:
    from solana.rpc.api import Client  # type: ignore
    try:
        # Older solana-py path
        from solana.publickey import PublicKey  # type: ignore
    except Exception:
        # Newer solana-py uses solders for Pubkey
        from solders.pubkey import Pubkey as PublicKey  # type: ignore
except Exception:  # pragma: no cover
    # Allow app to import even if deps aren't installed yet; runtime will fail gracefully.
    Client = None  # type: ignore
    PublicKey = None  # type: ignore


app = Flask(__name__)
logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
logger = app.logger
_LAST_RPC_URL: str | None = None
WSOL_MINTS: Set[str] = {
    "So11111111111111111111111111111111111111112",
}
STABLE_MINTS: Set[str] = {
    "EPjFWdd5AuJnBfZQ64rW7v6k7hNE9F3iuQF3XzQk3tYE",  # USDC
    "Es9vMFrzaCQi3QjY6C9p8wG9ZsV5hHok6wJG8YcJ2n6q",  # USDT
}

# Dexscreener API base
DEXSCREENER_SEARCH = "https://api.dexscreener.com/latest/dex/search?q="
DEXSCREENER_TOKEN = "https://api.dexscreener.com/latest/dex/tokens/"
BIRDEYE_HISTORY = "https://public-api.birdeye.so/defi/history_price"
YAHOO_CHART = "https://query1.finance.yahoo.com/v8/finance/chart/"

# Routing hints for well-known tickers to avoid mapping to random on-chain tokens
YAHOO_STOCKS: Set[str] = {
    "AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA", "NVDA", "AMD", "INTC", "NFLX",
}
YAHOO_CRYPTO: Set[str] = {
    "BTC", "ETH", "LINK", "SOL", "BNB", "XRP", "ADA", "DOGE", "LTC", "BCH", "AVAX",
    "MATIC", "DOT", "ATOM", "OP", "ARB", "NEAR", "APT", "SUI", "SEI", "TIA",
}

# Single-scan tuning knobs (set via env for your RPC tier)
# Defaults tuned to stay under the frontend's 30s timeout; override via env for deeper scans
PAGE_LIMIT = int(os.environ.get("SCAN_PAGE_LIMIT", "1000"))  # signatures per page
MAX_PAGES = int(os.environ.get("SCAN_MAX_PAGES", "8"))       # pages of signatures
TIME_BUDGET_SEC = float(os.environ.get("SCAN_TIME_BUDGET_SEC", "25"))
MAX_ANALYZE_TX = int(os.environ.get("SCAN_MAX_ANALYZE_TX", "250"))

# USD filter for small moves
SOL_PRICE_USD = float(os.environ.get("SOL_PRICE_USD", "150"))
MIN_TRADE_USD = float(os.environ.get("MIN_TRADE_USD", "10"))


def get_solana_client() -> Client:
    rpc_url = os.environ.get("SOLANA_RPC_URL", "https://api.mainnet-beta.solana.com")
    global _LAST_RPC_URL
    if rpc_url != _LAST_RPC_URL:
        logger.info("rpc:selected url=%s", rpc_url)
        _LAST_RPC_URL = rpc_url
    return Client(rpc_url)


def human_label_from_score(score: int) -> str:
    # Special easter-egg score
    if score == 69:
        return "Enjoyer"

    # Top tiers
    if score >= 90:
        return "Connoisseur"
    if score >= 80:
        return "Cabal"
    if score >= 70:
        return "Chad"
    if score >= 60:
        return "Based"
    if score >= 50:
        return "Normie"
    if score >= 40:
        return "Degen"
    if score >= 30:
        return "Skeptic"
    if score >= 25:
        return "Tiktokker"
    if score >= 20:
        return "Rug Enjoyer"

    # 1–19 broken into 4 neutral tiers of 5 points each
    if score >= 16:
        return "Brahmin"
    if score >= 11:
        return "Kshatriya"
    if score >= 6:
        return "Vaishya"
    return "Shudra"


def human_label_from_score_twitter(score: int) -> str:
    if score >= 95:
        return "Oracle"
    if score >= 90:
        return "Alpha Caller"
    if score >= 80:
        return "Trusted"
    if score >= 70:
        return "Generally Solid"
    if score >= 60:
        return "Mixed"
    if score >= 50:
        return "Unreliable"
    if score >= 40:
        return "Heavy Shiller"
    return "Mega Shiller"

def grade_wallet(address: str) -> Dict[str, Any]:
    logger.info("grade_wallet:start address=%s", address)
    if Client is None or PublicKey is None:
        return {
            "error": "Dependencies not installed. Please install requirements first.",
        }

    try:
        if hasattr(PublicKey, "from_string"):
            public_key = PublicKey.from_string(address)  # type: ignore[attr-defined]
        else:
            public_key = PublicKey(address)
    except Exception:
        logger.warning("grade_wallet:invalid_address address=%s", address)
        return {"error": "Invalid Solana address."}

    client = get_solana_client()

    # Balance (solana 0.35 returns solders response)
    balance_lamports = 0
    try:
        balance_resp = client.get_balance(public_key)
        if hasattr(balance_resp, "value"):
            balance_lamports = getattr(balance_resp, "value", 0) or 0
        else:
            # Fallback for dict-like
            if isinstance(balance_resp, dict) and balance_resp.get("result"):
                balance_lamports = balance_resp["result"].get("value", 0) or 0
    except Exception:
        logger.exception("grade_wallet:balance_fetch_failed address=%s", address)
        balance_lamports = 0
    sol_balance = balance_lamports / 1_000_000_000

    # Unified scan: paginate signatures under a time/size budget
    signatures_json: List[Dict[str, Any]] = []
    before_sig: str | None = None
    started = time.time()

    def call_with_retries(fn, *args, **kwargs):
        attempts = 0
        delay = 0.4
        while True:
            try:
                return fn(*args, **kwargs)
            except Exception:
                attempts += 1
                if attempts >= 5:
                    raise
                sleep_for = delay * (2 ** (attempts - 1)) + random.uniform(0.2, 0.6)
                time.sleep(sleep_for)

    try:
        for _ in range(MAX_PAGES):
            if (time.time() - started) > TIME_BUDGET_SEC:
                break
            resp = call_with_retries(
                client.get_signatures_for_address,
                public_key,
                before=before_sig,
                limit=PAGE_LIMIT,
            )
            batch: List[Any] = []
            if hasattr(resp, "value"):
                batch = getattr(resp, "value", [])
                batch = [
                    {
                        "signature": getattr(x, "signature", None),
                        "blockTime": getattr(x, "block_time", None) or getattr(x, "blockTime", None),
                    }
                    for x in batch
                ]
            elif isinstance(resp, dict):
                batch = resp.get("result", []) or []
            if not batch:
                break
            signatures_json.extend(batch)
            before_sig = batch[-1].get("signature")
            if len(signatures_json) >= MAX_ANALYZE_TX:
                break
            # small pacing to avoid bursts even on paid RPC
            time.sleep(0.05)
    except Exception:
        logger.exception("grade_wallet:signatures_fetch_failed address=%s", address)
        # keep whatever we collected so far
    tx_count = len(signatures_json)

    # Estimate account age from oldest signature timestamp (if present)
    first_tx_time: datetime | None = None
    if signatures_json:
        try:
            oldest = signatures_json[-1]
            ts = oldest.get("blockTime")
            if ts:
                first_tx_time = datetime.fromtimestamp(ts, tz=timezone.utc)
        except Exception:
            first_tx_time = None

    days_old = 0
    if first_tx_time is not None:
        days_old = max(0, (datetime.now(timezone.utc) - first_tx_time).days)

    # Analyze up to MAX_ANALYZE_TX transactions collected above
    sample_sigs: List[str] = [s.get("signature") for s in signatures_json if s.get("signature")]
    sampled_tx: List[Tuple[int, Set[str]]] = []  # list of (timestamp, set(program_ids))
    trade_deltas: List[Tuple[int, float]] = []  # (timestamp, net SOL delta excluding fee)
    # Token-aware position tracking (per mint) using FIFO lots and SOL-equivalent cost/proceeds
    from collections import deque
    class _Lot:
        def __init__(self, ts: int, amount_ui: float, cost_sol: float) -> None:
            self.ts = int(ts)
            self.amount_ui = float(amount_ui)
            self.cost_sol = float(cost_sol)
    positions: Dict[str, deque[_Lot]] = {}
    realized_token: List[Dict[str, float | int | str]] = []
    # Local threshold for ignoring tiny flows
    min_move_sol_local = max(0.01, MIN_TRADE_USD / max(1e-6, SOL_PRICE_USD))

    for sig in sample_sigs:
        try:
            tx_resp = call_with_retries(client.get_transaction, sig, max_supported_transaction_version=0)
            tx_json: Dict[str, Any] | None = None
            if hasattr(tx_resp, "to_json"):
                import json as _json
                try:
                    tx_json = _json.loads(tx_resp.to_json())
                except Exception:
                    tx_json = None
            elif isinstance(tx_resp, dict):
                tx_json = tx_resp
            if not tx_json:
                continue
            result_obj = tx_json.get("result", {})
            msg = result_obj.get("transaction", {}).get("message", {})
            block_time = result_obj.get("blockTime") or result_obj.get("block_time")
            if not block_time:
                # Fallback to signatures listing for blockTime
                try:
                    block_time = next((s.get("blockTime") for s in signatures_json if s.get("signature") == sig), None)
                except Exception:
                    block_time = None
            if not block_time:
                    continue

            # Collect program IDs from instructions (best signal per tx)
            program_ids: Set[str] = set()
            instrs = msg.get("instructions", [])
            for ix in instrs:
                program_id = ix.get("programId") or ix.get("programIdIndex")
                if isinstance(program_id, str):
                    program_ids.add(program_id)

            # Build a robust account keys list that includes versioned tx loaded addresses
            account_keys_raw = msg.get("accountKeys")
            account_keys: List[Any] = []
            if isinstance(account_keys_raw, list) and account_keys_raw:
                account_keys = account_keys_raw
            else:
                # Legacy-v0 style fields
                static_keys = msg.get("staticAccountKeys", []) or []
                for key in static_keys:
                    account_keys.append(key)
                meta_obj = result_obj.get("meta", {})
                loaded = meta_obj.get("loadedAddresses") or {}
                for arr_name in ("writable", "readonly"):
                    for k in loaded.get(arr_name, []) or []:
                        account_keys.append(k)

            for key in account_keys:
                if isinstance(key, str):
                    program_ids.add(key)
                elif isinstance(key, dict) and "pubkey" in key:
                    program_ids.add(key["pubkey"])

            sampled_tx.append((int(block_time), program_ids))

            # Compute net SOL + wSOL delta for this wallet excluding fee on native leg, if metadata is available
            try:
                meta = result_obj.get("meta", {})
                pre_balances = meta.get("preBalances", [])
                post_balances = meta.get("postBalances", [])
                fee_lamports = int(meta.get("fee", 0) or 0)
                # Find wallet index in account keys
                wallet_str = str(public_key)
                wallet_index = -1
                # Ensure account_keys includes loaded addresses for versioned tx
                account_keys_flat: List[str] = []
                for k in account_keys:
                    if isinstance(k, str):
                        account_keys_flat.append(k)
                    elif isinstance(k, dict) and k.get("pubkey"):
                        account_keys_flat.append(k["pubkey"])
                for i, k in enumerate(account_keys_flat):
                    if (isinstance(k, str) and k == wallet_str) or (isinstance(k, dict) and k.get("pubkey") == wallet_str):
                        wallet_index = i
                        break
                native_delta_sol = 0.0
                if wallet_index >= 0 and wallet_index < len(pre_balances) and wallet_index < len(post_balances):
                    delta_lamports = int(post_balances[wallet_index]) - int(pre_balances[wallet_index]) + fee_lamports
                    native_delta_sol = float(delta_lamports) / 1_000_000_000.0

                # Include wSOL token balance delta (sum across all wallet-owned wSOL accounts) and stables in SOL-equivalent
                wsol_delta_tokens = 0.0
                stable_delta_sol_equiv = 0.0
                try:
                    pre_toks = meta.get("preTokenBalances", []) or []
                    post_toks = meta.get("postTokenBalances", []) or []
                    # Build maps (mint, owner, accountIndex?) -> ui amount
                    def extract_map(arr):
                        m = {}
                        for b in arr:
                            mint = b.get("mint")
                            owner = b.get("owner") or (b.get("accountOwner") if isinstance(b.get("accountOwner"), str) else None)
                            ui = None
                            amt = b.get("uiTokenAmount") or {}
                            if isinstance(amt, dict):
                                ui = amt.get("uiAmount")
                                if ui is None:
                                    # Fallback from amount/decimals
                                    try:
                                        raw = int(amt.get("amount", "0"))
                                        dec = int(amt.get("decimals", 0))
                                        ui = float(raw) / (10 ** dec)
                                    except Exception:
                                        ui = 0.0
                            elif isinstance(amt, (int, float)):
                                ui = float(amt)
                            if mint and owner:
                                m[(mint, owner)] = float(ui or 0.0)
                        return m
                    pre_map = extract_map(pre_toks)
                    post_map = extract_map(post_toks)
                    for (mint, owner), pre_ui in pre_map.items():
                        if owner != wallet_str:
                            continue
                        post_ui = post_map.get((mint, owner), 0.0)
                        delta_ui = (post_ui - pre_ui)
                        if mint in WSOL_MINTS:
                            wsol_delta_tokens += delta_ui
                        elif mint in STABLE_MINTS:
                            stable_delta_sol_equiv += float(delta_ui) / max(1e-6, SOL_PRICE_USD)
                    # Also handle accounts that only appear post-
                    for (mint, owner), post_ui in post_map.items():
                        if (mint, owner) not in pre_map and owner == wallet_str:
                            if mint in WSOL_MINTS:
                                wsol_delta_tokens += post_ui
                            elif mint in STABLE_MINTS:
                                stable_delta_sol_equiv += float(post_ui) / max(1e-6, SOL_PRICE_USD)
                except Exception:
                    wsol_delta_tokens = 0.0
                    stable_delta_sol_equiv = 0.0

                delta_sol_total = native_delta_sol + float(wsol_delta_tokens) + float(stable_delta_sol_equiv)
                if abs(delta_sol_total) > 1e-9:
                    trade_deltas.append((int(block_time), delta_sol_total))

                # Derive per-mint token deltas owned by this wallet (excluding SOL/wSOL/stables)
                try:
                    # Build maps again if needed
                    pre_toks = meta.get("preTokenBalances", []) or []
                    post_toks = meta.get("postTokenBalances", []) or []
                    def extract_map(arr):
                        m = {}
                        for b in arr:
                            mint = b.get("mint")
                            owner = b.get("owner") or (b.get("accountOwner") if isinstance(b.get("accountOwner"), str) else None)
                            ui = None
                            amt = b.get("uiTokenAmount") or {}
                            if isinstance(amt, dict):
                                ui = amt.get("uiAmount")
                                if ui is None:
                                    try:
                                        raw = int(amt.get("amount", "0"))
                                        dec = int(amt.get("decimals", 0))
                                        ui = float(raw) / (10 ** dec)
                                    except Exception:
                                        ui = 0.0
                            elif isinstance(amt, (int, float)):
                                ui = float(amt)
                            if mint and owner == wallet_str:
                                m[mint] = m.get(mint, 0.0) + float(ui or 0.0)
                        return m
                    pre_map_any = extract_map(pre_toks)
                    post_map_any = extract_map(post_toks)
                    token_deltas_ui: Dict[str, float] = {}
                    mints = set(pre_map_any.keys()) | set(post_map_any.keys())
                    for mint in mints:
                        if mint in WSOL_MINTS or mint in STABLE_MINTS:
                            continue
                        token_deltas_ui[mint] = float(post_map_any.get(mint, 0.0) - pre_map_any.get(mint, 0.0))

                    # Split buys/sells per mint and allocate SOL cost/proceeds
                    pos_ui = {m: d for m, d in token_deltas_ui.items() if d > 0}
                    neg_ui = {m: -d for m, d in token_deltas_ui.items() if d < 0}
                    if delta_sol_total < -min_move_sol_local and pos_ui:
                        total_ui = sum(pos_ui.values()) or 1.0
                        for mint, amt_ui in pos_ui.items():
                            alloc_cost = abs(delta_sol_total) * (amt_ui / total_ui)
                            lot = _Lot(int(block_time), amt_ui, alloc_cost)
                            if mint not in positions:
                                positions[mint] = deque()
                            positions[mint].append(lot)
                    if delta_sol_total > min_move_sol_local and neg_ui:
                        total_ui = sum(neg_ui.values()) or 1.0
                        for mint, amt_ui_sold in neg_ui.items():
                            alloc_proceeds = float(delta_sol_total) * (amt_ui_sold / total_ui)
                            # Realize against FIFO lots
                            lots = positions.get(mint)
                            remaining = float(amt_ui_sold)
                            cost_used = 0.0
                            entry_ts = None
                            while remaining > 1e-12 and lots and len(lots) > 0:
                                lot0 = lots[0]
                                take = min(lot0.amount_ui, remaining)
                                cost_per_ui = (lot0.cost_sol / lot0.amount_ui) if lot0.amount_ui > 0 else 0.0
                                cost_used += cost_per_ui * take
                                remaining -= take
                                lot0.amount_ui -= take
                                lot0.cost_sol -= cost_per_ui * take
                                if entry_ts is None:
                                    entry_ts = lot0.ts
                                if lot0.amount_ui <= 1e-12:
                                    lots.popleft()
                            if entry_ts is None:
                                entry_ts = int(block_time)
                            proceeds = alloc_proceeds
                            pnl = proceeds - cost_used
                            roi = (proceeds / cost_used - 1.0) if cost_used > 0 else (1.0 if proceeds > 0 else 0.0)
                            hold_sec = max(0, int(int(block_time) - entry_ts))
                            realized_token.append({
                                "mint": mint,
                                "outflow_sol": round(float(cost_used), 9),
                                "inflow_sol": round(float(proceeds), 9),
                                "pnl_sol": round(float(pnl), 9),
                                "roi": round(float(roi), 6),
                                "hold_seconds": int(hold_sec),
                            })
                except Exception:
                    pass
            except Exception:
                pass
        except Exception:
            logger.exception("grade_wallet:tx_fetch_failed sig=%s address=%s", sig, address)
            continue
        if (time.time() - started) > TIME_BUDGET_SEC:
            break

    # Profit-first analysis and scoring

    # Helper: compute median of a list of numbers
    def _median(nums: List[float]) -> float:
        if not nums:
            return 0.0
        nums_sorted = sorted(nums)
        n = len(nums_sorted)
        mid = n // 2
        if n % 2 == 1:
            return float(nums_sorted[mid])
        return float((nums_sorted[mid - 1] + nums_sorted[mid]) / 2.0)

    # Build realized roundtrips from SOL deltas using a simple cycle model
    trade_deltas_sorted = sorted(trade_deltas, key=lambda t: t[0])
    # Dynamic minimum move threshold in SOL based on USD size
    min_move_sol = max(0.01, MIN_TRADE_USD / max(1e-6, SOL_PRICE_USD))

    # FIFO proportional realization: close earliest outflow legs with inflows and realize PnL = inflow_consumed - outflow_consumed
    from collections import deque
    class _Leg:
        def __init__(self, ts: int, amt: float) -> None:
            self.ts = int(ts)
            self.amt = float(amt)

    outflow_legs: deque[_Leg] = deque()
    realized: List[Dict[str, float | int]] = []

    for ts, dsol in trade_deltas_sorted:
        if dsol < -min_move_sol:
            # Record cost leg
            outflow_legs.append(_Leg(ts, abs(dsol)))
        elif dsol > min_move_sol:
            inflow_remaining = float(dsol)
            if not outflow_legs:
                continue
            consumed_outflow = 0.0
            consumed_inflow = 0.0
            first_leg_ts = outflow_legs[0].ts
            while inflow_remaining > 1e-9 and outflow_legs:
                leg = outflow_legs[0]
                take = min(leg.amt, inflow_remaining)
                leg.amt -= take
                inflow_remaining -= take
                consumed_outflow += take
                consumed_inflow += take
                if leg.amt <= 1e-9:
                    outflow_legs.popleft()
            if consumed_outflow > 1e-9:
                outflow = consumed_outflow
                inflow = consumed_inflow
                pnl = inflow - outflow
                roi = (inflow / outflow - 1.0) if outflow > 0 else 0.0
                hold_sec = max(0, int(ts - first_leg_ts))
                realized.append({
                    "outflow_sol": round(outflow, 9),
                    "inflow_sol": round(inflow, 9),
                    "pnl_sol": round(pnl, 9),
                    "roi": round(roi, 6),
                    "hold_seconds": hold_sec,
                })
        # else small noise ignored

    # Aggregate metrics (combine SOL-cycle and token-aware realizations)
    combined = [
        {**t, "source": "sol"} for t in realized
    ] + [
        {**t, "source": "token"} for t in realized_token
    ]
    wins = [t for t in combined if t["pnl_sol"] > 0]
    losses = [t for t in combined if t["pnl_sol"] < 0]
    realized_trades = len(combined)
    wins_count = len(wins)
    losses_count = len(losses)
    total_profit_sol = float(sum(t["pnl_sol"] for t in wins)) if wins else 0.0
    total_loss_sol = float(sum(t["pnl_sol"] for t in losses)) if losses else 0.0  # negative
    net_profit_sol = total_profit_sol + total_loss_sol
    win_rate = round(wins_count / realized_trades, 4) if realized_trades > 0 else 0.0
    profit_factor = (total_profit_sol / abs(total_loss_sol)) if total_loss_sol < 0 else (99.0 if total_profit_sol > 0 else 0.0)

    best_trade_profit_sol = max([t["pnl_sol"] for t in wins], default=0.0)
    worst_trade_loss_sol = min([t["pnl_sol"] for t in losses], default=0.0)
    best_trade_roi = max([t["roi"] for t in combined], default=0.0)
    worst_trade_roi = min([t["roi"] for t in combined], default=0.0)

    # Quick wins by hold time thresholds (applied to winning trades only)
    win_hold_minutes = [t["hold_seconds"] / 60.0 for t in wins]
    median_win_hold_minutes = _median(win_hold_minutes)
    quick_30m = sum(1 for t in wins if t["hold_seconds"] <= 30 * 60)
    quick_2h = sum(1 for t in wins if 30 * 60 < t["hold_seconds"] <= 2 * 3600)
    quick_6h = sum(1 for t in wins if 2 * 3600 < t["hold_seconds"] <= 6 * 3600)

    # Big % profit counts
    roi_50 = sum(1 for t in combined if t["roi"] >= 0.5)
    roi_100 = sum(1 for t in combined if t["roi"] >= 1.0)
    roi_200 = sum(1 for t in combined if t["roi"] >= 2.0)

    # Tail loss count (very bad % losses)
    tail_loss_count = sum(1 for t in losses if t["roi"] <= -0.6)

    # Window-level proxy signals (fallback if no realized trades)
    window_positive_sol = float(sum(d for _, d in trade_deltas_sorted if d > min_move_sol))
    window_negative_sol = float(sum(abs(d) for _, d in trade_deltas_sorted if d < -min_move_sol))
    window_net_sol = window_positive_sol - window_negative_sol
    positive_events = sum(1 for _, d in trade_deltas_sorted if d > min_move_sol)
    negative_events = sum(1 for _, d in trade_deltas_sorted if d < -min_move_sol)
    best_positive_event_sol = max([d for _, d in trade_deltas_sorted if d > min_move_sol], default=0.0)

    # Score assembly (0-100), profit-first
    score = 0

    # Absolute profit size (0-40) — slightly buff gains
    if total_profit_sol >= 10:
        score += 40
    elif total_profit_sol >= 5:
        score += 32
    elif total_profit_sol >= 2:
        score += 24
    elif total_profit_sol >= 1:
        score += 16
    elif total_profit_sol >= 0.25:
        score += 8

    # Profit quality (0-20): win rate + profit factor
    quality_pts = 0
    if realized_trades >= 3:
        if win_rate >= 0.75:
            quality_pts += 14
        elif win_rate >= 0.65:
            quality_pts += 10
        elif win_rate >= 0.55:
            quality_pts += 6
        if profit_factor >= 3.0:
            quality_pts += 6
        elif profit_factor >= 2.0:
            quality_pts += 4
        elif profit_factor >= 1.5:
            quality_pts += 2
    score += min(20, quality_pts)

    # Quick profitable holds bonus (0-10)
    quick_pts = 3 * quick_30m + 2 * quick_2h + 1 * quick_6h
    score += min(10, quick_pts)

    # Big percentage profit bonus (0-20)
    pct_pts = 3 * roi_200 + 4 * roi_100 + 1 * roi_50
    score += min(20, pct_pts)

    # Big absolute win bonus (0-10)
    if best_trade_profit_sol >= 5:
        score += 10
    elif best_trade_profit_sol >= 2:
        score += 7
    elif best_trade_profit_sol >= 1:
        score += 5
    elif best_trade_profit_sol >= 0.5:
        score += 3

    # Loss magnitude penalty (up to -25), scaled to avoid collapsing everyone into 1–10
    loss_mag = abs(total_loss_sol)
    if loss_mag >= 20:
        score -= 25
    elif loss_mag >= 10:
        score -= 20
    elif loss_mag >= 5:
        score -= 14
    elif loss_mag >= 2:
        score -= 9
    elif loss_mag >= 1:
        score -= 5

    # Tail loss penalty (up to -8)
    tail_penalty = 1 * tail_loss_count
    if abs(worst_trade_loss_sol) >= 2.0:
        tail_penalty += 2
    score -= min(8, tail_penalty)

    # Fallback scoring if no realized trades were detected
    if realized_trades == 0:
        # Net gain over window
        if window_net_sol >= 5:
            score += 30
        elif window_net_sol >= 2:
            score += 22
        elif window_net_sol >= 1:
            score += 15
        elif window_net_sol >= 0.25:
            score += 8
        elif window_net_sol > 0:
            score += 6

        # Single strong positive event bonus
        if best_positive_event_sol >= 2:
            score += 7
        elif best_positive_event_sol >= 1:
            score += 5
        elif best_positive_event_sol >= 0.5:
            score += 3

        # Loss magnitude penalty
        if window_negative_sol >= 10:
            score -= 15
        elif window_negative_sol >= 5:
            score -= 12
        elif window_negative_sol >= 2:
            score -= 8
        elif window_negative_sol >= 1:
            score -= 5

        # Event balance hint
        if positive_events > 0 or negative_events > 0:
            balance_ratio = (positive_events - negative_events) / float(max(1, positive_events + negative_events))
            score += int(5 * max(-1.0, min(1.0, balance_ratio)))

    score = max(1, min(100, int(round(score))))

    label = human_label_from_score(score)

    # Profiles for display
    if median_win_hold_minutes <= 30 and wins_count > 0:
        hold_profile = "sniper scalper"
    elif median_win_hold_minutes <= 120 and wins_count > 0:
        hold_profile = "scalper"
    elif median_win_hold_minutes <= 360 and wins_count > 0:
        hold_profile = "quick swing"
    elif wins_count > 0:
        hold_profile = "swing/position"
    else:
        hold_profile = "no realized wins"

    if realized_trades <= 3:
        activity_profile = "low"
    elif realized_trades <= 10:
        activity_profile = "moderate"
    elif realized_trades <= 30:
        activity_profile = "high"
    else:
        activity_profile = "overactive"

    result_obj = {
        "address": str(public_key),
        "score": score,
        "label": label,
        "metrics": {
            "sol_balance": sol_balance,
            "recent_tx_count": tx_count,
            "account_age_days": days_old,
            "hold_profile": hold_profile,
            "activity_profile": activity_profile,
            # Realized performance
            "realized_trades": realized_trades,
            "wins": wins_count,
            "losses": losses_count,
            "win_rate": win_rate,
            "profit_factor": round(profit_factor, 4) if isinstance(profit_factor, float) else profit_factor,
            "total_profit_sol": round(total_profit_sol, 6),
            "total_loss_sol": round(total_loss_sol, 6),
            "net_profit_sol": round(net_profit_sol, 6),
            "best_trade_profit_sol": round(float(best_trade_profit_sol), 6),
            "worst_trade_loss_sol": round(float(worst_trade_loss_sol), 6),
            "best_trade_roi": round(float(best_trade_roi), 4),
            "worst_trade_roi": round(float(worst_trade_roi), 4),
            # Quick win and % profit highlights
            "quick_wins_30m": int(quick_30m),
            "quick_wins_2h": int(quick_2h),
            "quick_wins_6h": int(quick_6h),
            "big_roi_50_count": int(roi_50),
            "big_roi_100_count": int(roi_100),
            "big_roi_200_count": int(roi_200),
            "median_win_hold_minutes": round(float(median_win_hold_minutes), 2),
            # Window proxies
            "window_positive_sol": round(window_positive_sol, 6),
            "window_negative_sol": round(window_negative_sol, 6),
            "window_net_sol": round(window_net_sol, 6),
            "positive_events": int(positive_events),
            "negative_events": int(negative_events),
            "best_positive_event_sol": round(best_positive_event_sol, 6),
        },
    }
    logger.info("grade_wallet:done address=%s score=%s", address, score)
    return result_obj


def _analyze_signatures_full(client: Client, public_key: Any, signatures_json: List[Dict[str, Any]]) -> Dict[str, Any]:
    # Profit-first analyzer across a provided signatures set (used by deep scans)
    balance_lamports = 0
    try:
        balance_resp = client.get_balance(public_key)
        if hasattr(balance_resp, "value"):
            balance_lamports = getattr(balance_resp, "value", 0) or 0
        else:
            if isinstance(balance_resp, dict) and balance_resp.get("result"):
                balance_lamports = balance_resp["result"].get("value", 0) or 0
    except Exception:
        balance_lamports = 0
    sol_balance = balance_lamports / 1_000_000_000

    times_desc: List[int] = [s.get("blockTime") for s in signatures_json if s.get("blockTime")]
    times_asc: List[int] = sorted(times_desc)
    trade_deltas: List[Tuple[int, float]] = []
    def call_with_retries(fn, *args, **kwargs):
        attempts = 0
        delay = 0.5
        while True:
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                attempts += 1
                if attempts >= 6:
                    raise
                sleep_for = delay * (2 ** (attempts - 1)) + random.uniform(0, 0.25)
                time.sleep(sleep_for)

    # Respect a time budget while analyzing
    started = time.time()
    # Subsample to at most MAX_ANALYZE_TX transactions, evenly across history
    if len(signatures_json) > MAX_ANALYZE_TX:
        stride = max(1, len(signatures_json) // MAX_ANALYZE_TX)
        iter_sigs = signatures_json[::stride]
    else:
        iter_sigs = signatures_json

    for s in iter_sigs:
        sig = s.get("signature")
        if not sig:
            continue
        try:
            tx_resp = call_with_retries(client.get_transaction, sig, max_supported_transaction_version=0)
            tx_json = None
            if hasattr(tx_resp, "to_json"):
                import json as _json
                try:
                    tx_json = _json.loads(tx_resp.to_json())
                except Exception:
                    tx_json = None
            elif isinstance(tx_resp, dict):
                tx_json = tx_resp
            if not tx_json:
                continue
            result_obj = tx_json.get("result", {})
            msg = result_obj.get("transaction", {}).get("message", {})
            block_time = result_obj.get("blockTime") or result_obj.get("block_time") or s.get("blockTime")
            if not block_time:
                continue
            account_keys = msg.get("accountKeys", [])

            # deltas
            try:
                meta = result_obj.get("meta", {})
                pre_balances = meta.get("preBalances", [])
                post_balances = meta.get("postBalances", [])
                fee_lamports = int(meta.get("fee", 0) or 0)
                wallet_str = str(public_key)
                wallet_index = -1
                for i, k in enumerate(account_keys):
                    if (isinstance(k, str) and k == wallet_str) or (isinstance(k, dict) and k.get("pubkey") == wallet_str):
                        wallet_index = i
                        break
                if wallet_index >= 0 and wallet_index < len(pre_balances) and wallet_index < len(post_balances):
                    delta_lamports = int(post_balances[wallet_index]) - int(pre_balances[wallet_index]) + fee_lamports
                    delta_sol = float(delta_lamports) / 1_000_000_000.0
                    trade_deltas.append((int(block_time), delta_sol))
            except Exception:
                pass
        except Exception:
            continue
        if (time.time() - started) > TIME_BUDGET_SEC:
            break

    # Age and counts
    days_old = 0
    if times_asc:
        first_tx_time = datetime.fromtimestamp(times_asc[0], tz=timezone.utc)
        days_old = max(0, (datetime.now(timezone.utc) - first_tx_time).days)
    tx_count = len(signatures_json)

    # Reuse the same profit-first logic as in grade_wallet
    trade_deltas_sorted = sorted(trade_deltas, key=lambda t: t[0])
    min_move_sol = 0.01

    class _Cycle2:
        def __init__(self, start_ts: int, first_outflow: float) -> None:
            self.start_ts = int(start_ts)
            self.outflow = float(first_outflow)
            self.inflow = 0.0

    realized: List[Dict[str, float | int]] = []
    open_cycle: _Cycle2 | None = None
    for ts, dsol in trade_deltas_sorted:
        if dsol < -min_move_sol:
            if open_cycle is None:
                open_cycle = _Cycle2(ts, abs(dsol))
            else:
                open_cycle.outflow += abs(dsol)
        elif dsol > min_move_sol:
            if open_cycle is None:
                continue
            remaining_needed = max(0.0, open_cycle.outflow - open_cycle.inflow)
            consume = min(float(dsol), remaining_needed)
            open_cycle.inflow += consume
            if open_cycle.inflow + 1e-9 >= open_cycle.outflow:
                outflow = open_cycle.outflow
                inflow = open_cycle.inflow
                pnl = inflow - outflow
                roi = (inflow / outflow - 1.0) if outflow > 0 else 0.0
                hold_sec = max(0, int(ts - open_cycle.start_ts))
                realized.append({
                    "outflow_sol": round(outflow, 9),
                    "inflow_sol": round(inflow, 9),
                    "pnl_sol": round(pnl, 9),
                    "roi": round(roi, 6),
                    "hold_seconds": hold_sec,
                })
                open_cycle = None

    wins = [t for t in realized if t["pnl_sol"] > 0]
    losses = [t for t in realized if t["pnl_sol"] < 0]
    realized_trades = len(realized)
    wins_count = len(wins)
    losses_count = len(losses)
    total_profit_sol = float(sum(t["pnl_sol"] for t in wins)) if wins else 0.0
    total_loss_sol = float(sum(t["pnl_sol"] for t in losses)) if losses else 0.0
    net_profit_sol = total_profit_sol + total_loss_sol
    win_rate = round(wins_count / realized_trades, 4) if realized_trades > 0 else 0.0
    profit_factor = (total_profit_sol / abs(total_loss_sol)) if total_loss_sol < 0 else (99.0 if total_profit_sol > 0 else 0.0)

    best_trade_profit_sol = max([t["pnl_sol"] for t in wins], default=0.0)
    worst_trade_loss_sol = min([t["pnl_sol"] for t in losses], default=0.0)
    best_trade_roi = max([t["roi"] for t in realized], default=0.0)
    worst_trade_roi = min([t["roi"] for t in realized], default=0.0)

    win_hold_minutes = [t["hold_seconds"] / 60.0 for t in wins]
    def _median(vals: List[float]) -> float:
        if not vals:
            return 0.0
        vs = sorted(vals)
        n = len(vs)
        m = n // 2
        return float(vs[m] if n % 2 == 1 else (vs[m - 1] + vs[m]) / 2.0)
    median_win_hold_minutes = _median(win_hold_minutes)
    quick_30m = sum(1 for t in wins if t["hold_seconds"] <= 30 * 60)
    quick_2h = sum(1 for t in wins if 30 * 60 < t["hold_seconds"] <= 2 * 3600)
    quick_6h = sum(1 for t in wins if 2 * 3600 < t["hold_seconds"] <= 6 * 3600)
    roi_50 = sum(1 for t in realized if t["roi"] >= 0.5)
    roi_100 = sum(1 for t in realized if t["roi"] >= 1.0)
    roi_200 = sum(1 for t in realized if t["roi"] >= 2.0)
    tail_loss_count = sum(1 for t in losses if t["roi"] <= -0.6)

    score = 0
    if total_profit_sol >= 10:
        score += 35
    elif total_profit_sol >= 5:
        score += 28
    elif total_profit_sol >= 2:
        score += 20
    elif total_profit_sol >= 1:
        score += 12
    elif total_profit_sol >= 0.25:
        score += 6

    quality_pts = 0
    if realized_trades >= 3:
        if win_rate >= 0.75:
            quality_pts += 14
        elif win_rate >= 0.65:
            quality_pts += 10
        elif win_rate >= 0.55:
            quality_pts += 6
        if profit_factor >= 3.0:
            quality_pts += 6
        elif profit_factor >= 2.0:
            quality_pts += 4
        elif profit_factor >= 1.5:
            quality_pts += 2
    score += min(20, quality_pts)

    quick_pts = 3 * quick_30m + 2 * quick_2h + 1 * quick_6h
    score += min(10, quick_pts)

    pct_pts = 2 * roi_200 + 3 * roi_100 + 1 * roi_50
    score += min(15, pct_pts)

    if best_trade_profit_sol >= 5:
        score += 10
    elif best_trade_profit_sol >= 2:
        score += 7
    elif best_trade_profit_sol >= 1:
        score += 5
    elif best_trade_profit_sol >= 0.5:
        score += 3

    loss_mag = abs(total_loss_sol)
    if loss_mag >= 10:
        score -= 25
    elif loss_mag >= 5:
        score -= 18
    elif loss_mag >= 2:
        score -= 12
    elif loss_mag >= 1:
        score -= 7

    tail_penalty = 2 * tail_loss_count
    if abs(worst_trade_loss_sol) >= 2.0:
        tail_penalty += 2
    score -= min(10, tail_penalty)

    score = max(1, min(100, int(round(score))))
    label = human_label_from_score(score)

    if median_win_hold_minutes <= 30 and wins_count > 0:
        hold_profile = "sniper scalper"
    elif median_win_hold_minutes <= 120 and wins_count > 0:
        hold_profile = "scalper"
    elif median_win_hold_minutes <= 360 and wins_count > 0:
        hold_profile = "quick swing"
    elif wins_count > 0:
        hold_profile = "swing/position"
    else:
        hold_profile = "no realized wins"

    if realized_trades <= 3:
        activity_profile = "low"
    elif realized_trades <= 10:
        activity_profile = "moderate"
    elif realized_trades <= 30:
        activity_profile = "high"
    else:
        activity_profile = "overactive"

    return {
        "address": str(public_key),
        "score": score,
        "label": label,
        "metrics": {
            "sol_balance": sol_balance,
            "recent_tx_count": tx_count,
            "account_age_days": days_old,
            "hold_profile": hold_profile,
            "activity_profile": activity_profile,
            "realized_trades": realized_trades,
            "wins": wins_count,
            "losses": losses_count,
            "win_rate": win_rate,
            "profit_factor": round(profit_factor, 4) if isinstance(profit_factor, float) else profit_factor,
            "total_profit_sol": round(total_profit_sol, 6),
            "total_loss_sol": round(total_loss_sol, 6),
            "net_profit_sol": round(net_profit_sol, 6),
            "best_trade_profit_sol": round(float(best_trade_profit_sol), 6),
            "worst_trade_loss_sol": round(float(worst_trade_loss_sol), 6),
            "best_trade_roi": round(float(best_trade_roi), 4),
            "worst_trade_roi": round(float(worst_trade_roi), 4),
            "quick_wins_30m": int(quick_30m),
            "quick_wins_2h": int(quick_2h),
            "quick_wins_6h": int(quick_6h),
            "big_roi_50_count": int(roi_50),
            "big_roi_100_count": int(roi_100),
            "big_roi_200_count": int(roi_200),
            "median_win_hold_minutes": round(float(median_win_hold_minutes), 2),
        },
    }


def _run_deep_scan(scan_id: str, address: str) -> None:
    logger.info("deep_scan:start scan_id=%s address=%s", scan_id, address)
    SCANS[scan_id] = {
        "status": "running",
        "progress": 0,
        "processed": 0,
        "total": None,
        "error": None,
        "result": None,
        "address": address,
        "started_at": time.time(),
    }
    try:
        if hasattr(PublicKey, "from_string"):
            public_key = PublicKey.from_string(address)  # type: ignore[attr-defined]
        else:
            public_key = PublicKey(address)
    except Exception:
        logger.warning("deep_scan:invalid_address scan_id=%s address=%s", scan_id, address)
        SCANS[scan_id].update({"status": "error", "error": "Invalid Solana address."})
        return

    client = get_solana_client()
    all_sigs: List[Dict[str, Any]] = []
    before_sig: str | None = None
    max_loops = min(200, MAX_PAGES)  # safety cap
    try:
        def call_with_retries(fn, *args, **kwargs):
            attempts = 0
            delay = 0.5
            while True:
                try:
                    return fn(*args, **kwargs)
                except Exception as e:  # backoff on RPC limits / transient errors
                    attempts += 1
                    if attempts >= 6:
                        raise
                    sleep_for = delay * (2 ** (attempts - 1)) + random.uniform(0, 0.25)
                    logger.warning("deep_scan:rpc_retry attempt=%s delay=%.2fs err=%s", attempts, sleep_for, str(e)[:140])
                    time.sleep(sleep_for)

        started = time.time()
        for loop_idx in range(max_loops):
            if (time.time() - started) > TIME_BUDGET_SEC:
                logger.info("deep_scan:time_budget_reached scan_id=%s collected=%s", scan_id, len(all_sigs))
                break
            resp = call_with_retries(
                client.get_signatures_for_address, public_key, before=before_sig, limit=PAGE_LIMIT
            )
            batch: List[Any] = []
            if hasattr(resp, "value"):
                batch = getattr(resp, "value", [])
                batch = [
                    {
                        "signature": getattr(x, "signature", None),
                        "blockTime": getattr(x, "block_time", None) or getattr(x, "blockTime", None),
                    }
                    for x in batch
                ]
            elif isinstance(resp, dict):
                batch = resp.get("result", []) or []

            if not batch:
                logger.info("deep_scan:pagination_complete scan_id=%s total=%s", scan_id, len(all_sigs))
                break
            all_sigs.extend(batch)
            before_sig = batch[-1].get("signature")
            SCANS[scan_id].update({"processed": len(all_sigs), "progress": min(95, SCANS[scan_id]["processed"] % 95)})
            logger.info("deep_scan:page scan_id=%s loop=%s batch=%s cumulative=%s before=%s", scan_id, loop_idx, len(batch), len(all_sigs), before_sig)

            # Avoid monopolizing RPC
            time.sleep(0.12)

        # Analyze
        logger.info("deep_scan:analyze_begin scan_id=%s total=%s", scan_id, len(all_sigs))
        # Wrap per-tx fetches with retry & rate limit inside analyzer
        result = _analyze_signatures_full(client, public_key, all_sigs)
        SCANS[scan_id].update({
            "status": "completed",
            "progress": 100,
            "total": len(all_sigs),
            "result": result,
            "partial": (time.time() - started) > TIME_BUDGET_SEC,
            "finished_at": time.time(),
        })
        logger.info("deep_scan:completed scan_id=%s score=%s", scan_id, result.get("score"))
    except Exception as e:
        logger.exception("deep_scan:error scan_id=%s", scan_id)
        SCANS[scan_id].update({"status": "error", "error": str(e)})


# Deep-scan endpoints removed; unified /grade scan handles both pagination and analysis under a time budget


@app.route("/", methods=["GET", "POST"])
def index():
    result: Dict[str, Any] | None = None
    error: str | None = None

    if request.method == "POST":
        # Legacy fallback: still render page if form posts without JS
        raw = (request.form.get("handle", "") or request.form.get("address", "")).strip()
        if not raw:
            error = "Please enter a Twitter handle (e.g., @name)."
        else:
            # If the input looks like a twitter handle, route to twitter grading for SSR fallback
            if raw.startswith("@") or ("twitter.com/" in raw) or ("x.com/" in raw):
                result = grade_twitter(raw)
            else:
                result = grade_wallet(raw)
            error = result.get("error") if isinstance(result, dict) and result.get("error") else None

    return render_template("index.html", result=result, error=error)


@app.post("/grade")
def grade_api():
    data = request.get_json(silent=True) or {}
    address = (data.get("address") or request.form.get("address") or "").strip()
    if not address:
        return jsonify({"error": "Please enter a Solana wallet address."}), 400
    logger.info("grade_api:request address=%s", address)
    result = grade_wallet(address)
    status = 200 if not result.get("error") else 400
    logger.info("grade_api:response address=%s status=%s", address, status)
    return jsonify(result), status


# ---------------- Twitter reliability grading -----------------

def _extract_handle(raw: str) -> str | None:
    s = (raw or "").strip()
    if not s:
        return None
    if s.startswith("@"):
        s = s[1:]
    # URLs
    m = re.search(r"(?:https?://)?(?:www\.)?(?:x|twitter)\.com/([A-Za-z0-9_]{1,15})", s)
    if m:
        return m.group(1)
    # Raw handle
    m2 = re.match(r"^[A-Za-z0-9_]{1,15}$", s)
    if m2:
        return s
    return None


def _extract_status_id(raw: str) -> str | None:
    s = (raw or "").strip()
    m = re.search(r"(?:https?://)?(?:www\.)?(?:x|twitter)\.com/[^/]+/status/(\d+)", s)
    return m.group(1) if m else None


def _get_bearer_token() -> str | None:
    """Return a normalized Twitter Bearer token.
    Some dashboards show it URL-encoded; decode if needed.
    """
    token = os.environ.get("TWITTER_BEARER_TOKEN")
    if not token:
        return None
    token = token.strip().strip('"').strip("'")
    # Heuristic: if looks URL-encoded, decode once
    if "%" in token or "+" in token:
        try:
            decoded = unquote(token)
            # sanity: decoded should be longer or contain '/' characters
            if decoded and ("/" in decoded or decoded.count('%') < token.count('%')):
                token = decoded
        except Exception:
            pass
    return token

def _twitter_fetch_from_nitter(handle: str, max_results: int = 50) -> List[Dict[str, Any]]:
    instances = [
        "https://nitter.net",
        "https://nitter.poast.org",
        "https://n.opnxng.com",
        "https://nitter.privacydev.net",
    ]
    headers = {"User-Agent": "Mozilla/5.0"}
    handle_path = (handle or "").lstrip("@")
    for base in instances:
        try:
            url = f"{base}/{handle_path}"
            resp = requests.get(url, headers=headers, timeout=10)
            if not resp.ok or not resp.text:
                continue
            soup = BeautifulSoup(resp.text, "html.parser")
            tweets: List[Dict[str, Any]] = []
            # Each tweet link looks like /{handle}/status/{id} (case on handle varies). Match any status link.
            for a in soup.select('a[href*="/status/"]'):
                href = a.get('href') or ''
                m = re.search(r"/status/(\d+)", href)
                if not m:
                    continue
                tid = m.group(1)
                # Fetch the tweet page to reliably get text + timestamp
                item = _fetch_single_tweet(tid, handle_path)
                if item:
                    tweets.append(item)
                if len(tweets) >= max_results:
                    break
            if tweets:
                return tweets
        except Exception:
            continue
    return []


TWEET_SCAN_LIMIT = int(os.environ.get("TWEET_SCAN_LIMIT", "800"))
TWEET_TIME_BUDGET_SEC = float(os.environ.get("TWEET_TIME_BUDGET_SEC", "12"))


def _twitter_fetch_recent(handle: str, max_results: int = 100) -> List[Dict[str, Any]]:
    # Try Twitter API v2 if bearer provided
    bearer = _get_bearer_token()
    force_nitter = os.environ.get("FORCE_NITTER", "").lower() in ("1", "true", "yes")
    if bearer and not force_nitter:
        try:
            # Lookup user id
            r1 = requests.get(
                f"https://api.twitter.com/2/users/by/username/{handle}",
                headers={"Authorization": f"Bearer {bearer}"}, timeout=12,
            )
            if r1.ok:
                uid = r1.json().get("data", {}).get("id")
                if uid:
                    r2 = requests.get(
                        f"https://api.twitter.com/2/users/{uid}/tweets",
                        params={
                            "max_results": str(max_results),
                            "exclude": "retweets",
                            "tweet.fields": "created_at,entities",
                        },
                        headers={"Authorization": f"Bearer {bearer}"}, timeout=12,
                    )
                    if r2.ok:
                        out: List[Dict[str, Any]] = []
                        data = r2.json()
                        out.extend({
                            "id": t.get("id"),
                            "text": t.get("text", ""),
                            "created_at": t.get("created_at"),
                            "entities": t.get("entities") or {},
                        } for t in data.get("data", []) or [])
                        # Paginate
                        next_token = (data.get("meta") or {}).get("next_token")
                        started = time.time()
                        while next_token and len(out) < TWEET_SCAN_LIMIT and (time.time() - started) < TWEET_TIME_BUDGET_SEC:
                            r3 = requests.get(
                                f"https://api.twitter.com/2/users/{uid}/tweets",
                                params={
                                    "max_results": "100",
                                    "exclude": "retweets",
                                    "tweet.fields": "created_at,entities",
                                    "pagination_token": next_token,
                                },
                                headers={"Authorization": f"Bearer {bearer}"}, timeout=12,
                            )
                            if not r3.ok:
                                break
                            jd = r3.json()
                            out.extend({
                                "id": t.get("id"),
                                "text": t.get("text", ""),
                                "created_at": t.get("created_at"),
                                "entities": t.get("entities") or {},
                            } for t in jd.get("data", []) or [])
                            next_token = (jd.get("meta") or {}).get("next_token")
                        return out[:TWEET_SCAN_LIMIT]
        except Exception:
            pass

    # Fallback: parse Nitter HTML for ids, timestamps, and text
    nitter = _twitter_fetch_from_nitter(handle, max_results=min(50, max_results))
    if nitter:
        return nitter
    # Fallback 2: Nitter RSS
    try:
        import xml.etree.ElementTree as ET
        for base in ("https://nitter.net", "https://nitter.poast.org", "https://n.opnxng.com", "https://nitter.privacydev.net"):
            url = f"{base}/{handle}/rss"
            resp = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
            if not resp.ok or not resp.text:
                continue
            root = ET.fromstring(resp.text)
            items: List[Dict[str, Any]] = []
            for it in root.findall('.//item'):
                title_el = it.find('title'); link_el = it.find('link'); date_el = it.find('pubDate')
                text = title_el.text if title_el is not None else ''
                link = link_el.text if link_el is not None else ''
                ts = date_el.text if date_el is not None else None
                tid = None
                if link:
                    m = re.search(r"/status/(\d+)", link)
                    if m: tid = m.group(1)
                items.append({"id": tid, "text": text or '', "created_at": ts})
                if len(items) >= max_results:
                    break
            if items:
                return items
    except Exception:
        pass

    # Fallback 3: text proxy (Jina AI) on x.com profile
    try:
        for scheme in ("http", "https"):
            url = f"https://r.jina.ai/{scheme}://x.com/{handle}"
            resp = requests.get(url, timeout=12, headers={"User-Agent": "Mozilla/5.0"})
            if not resp.ok or not resp.text:
                continue
            txt = resp.text
            # Split the big blob into pseudo-tweets by double newlines
            chunks = [c.strip() for c in txt.split("\n\n") if c.strip()]
            items: List[Dict[str, Any]] = []
            for c in chunks[:max_results]:
                items.append({"id": None, "text": c, "created_at": None})
            if items:
                return items
    except Exception:
        pass
    return []


def _fetch_single_tweet(tweet_id: str, handle_hint: str | None = None) -> Dict[str, Any] | None:
    # Primary: Twitter API v2
    bearer = _get_bearer_token()
    if bearer:
        try:
            r = requests.get(
                f"https://api.twitter.com/2/tweets/{tweet_id}",
                params={"tweet.fields": "created_at"},
                headers={"Authorization": f"Bearer {bearer}"},
                timeout=10,
            )
            if r.ok:
                d = r.json().get("data", {})
                if d and d.get("id"):
                    return {"id": d.get("id"), "text": d.get("text", ""), "created_at": d.get("created_at")}
        except Exception:
            pass
    # Fallback: Nitter HTML
    instances = [
        "https://nitter.net",
        "https://nitter.poast.org",
        "https://n.opnxng.com",
        "https://nitter.privacydev.net",
    ]
    headers = {"User-Agent": "Mozilla/5.0"}
    for base in instances:
        for path in (f"/i/web/status/{tweet_id}", f"/{(handle_hint or '').lstrip('@')}/status/{tweet_id}"):
            try:
                url = base + path
                resp = requests.get(url, headers=headers, timeout=10)
                if not resp.ok or not resp.text:
                    continue
                soup = BeautifulSoup(resp.text, "html.parser")
                time_tag = soup.find('time')
                dt_iso = time_tag.get('datetime') if (time_tag and time_tag.has_attr('datetime')) else None
                content = soup.select_one('.tweet-content, .status-content, .content')
                text = content.get_text(" ", strip=True) if content else soup.get_text(" ", strip=True)[:400]
                if text:
                    return {"id": tweet_id, "text": text, "created_at": dt_iso}
            except Exception:
                continue
    return None


def _extract_coin_mentions_per_tweet(tweets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    for tw in tweets:
        text = tw.get("text") or ""
        ts = tw.get("created_at")
        tid = tw.get("id")
        entities = tw.get("entities") or {}
        # Prefer structured cashtags when available
        symbols = []
        try:
            for sym in (entities.get("cashtags") or []):
                val = (sym.get("tag") or "").upper()
                if val and val not in symbols and len(val) <= 15:
                    symbols.append(val)
            for h in (entities.get("hashtags") or []):
                val = (h.get("tag") or "").upper()
                if val and val not in symbols and len(val) <= 15:
                    symbols.append(val)
        except Exception:
            pass
        for sym in symbols:
            events.append({
                "kind": "symbol",
                "id": sym,
                "text": text,
                "created_at": ts,
                "tweet_id": tid,
                "source": "entity",
            })
        # Cashtags symbols
        for m in re.finditer(r"\$([A-Za-z][A-Za-z0-9]{1,15})\b", text):
            sym = m.group(1).upper()
            events.append({
                "kind": "symbol",
                "id": sym,
                "text": text,
                "created_at": ts,
                "tweet_id": tid,
                "source": "cashtag",
            })
        # Hashtags symbols (#TICKER)
        for m in re.finditer(r"#([A-Za-z][A-Za-z0-9]{1,15})\b", text):
            sym = m.group(1).upper()
            events.append({
                "kind": "symbol",
                "id": sym,
                "text": text,
                "created_at": ts,
                "tweet_id": tid,
                "source": "hashtag",
            })
        # Solana base58 mints
        for m in re.finditer(r"\b[1-9A-HJ-NP-Za-km-z]{32,44}\b", text):
            addr = m.group(0)
            events.append({
                "kind": "address",
                "id": addr,
                "text": text,
                "created_at": ts,
                "tweet_id": tid,
                "source": "mint",
            })
        # pump.fun links
        for m in re.finditer(r"pump\.fun/(?:token/)?([1-9A-HJ-NP-Za-km-z]{32,44})", text):
            addr = m.group(1)
            events.append({
                "kind": "address",
                "id": addr,
                "text": text,
                "created_at": ts,
                "tweet_id": tid,
                "source": "pumpfun",
            })
        # Dexscreener links: either token or pair address
        for m in re.finditer(r"dexscreener\.com/(?:[a-z\-]+/)?(?:token/)?([1-9A-HJ-NP-Za-km-z]{32,44})", text, re.IGNORECASE):
            ident = m.group(1)
            events.append({
                "kind": "pair",
                "id": ident,
                "text": text,
                "created_at": ts,
                "tweet_id": tid,
                "source": "dexscreener",
            })
        # Birdeye token links
        for m in re.finditer(r"birdeye\.so/(?:token|symbol)/([1-9A-HJ-NP-Za-km-z]{32,44})", text, re.IGNORECASE):
            ident = m.group(1)
            events.append({
                "kind": "address",
                "id": ident,
                "text": text,
                "created_at": ts,
                "tweet_id": tid,
                "source": "birdeye",
            })
    return events


def _dexscreener_lookup(ident: str, kind: str) -> Dict[str, Any] | None:
    try:
        ident_upper = (ident or "").upper()
        if kind == "address":
            r = requests.get(DEXSCREENER_TOKEN + ident, timeout=10)
        else:
            r = requests.get(DEXSCREENER_SEARCH + requests.utils.quote(ident), timeout=10)
        if not r.ok:
            return None
        data = r.json() if r.headers.get("content-type", "").startswith("application/json") else None
        if not isinstance(data, dict):
            return None
        pairs = data.get("pairs") or data.get("pairsByDexId") or data.get("pairs")
        if not pairs:
            return None
        # If looking up by symbol, require exact symbol match to avoid false matches (e.g., $AMD stock)
        if kind != "address":
            pairs_exact = []
            for p in pairs:
                sym = ((p.get("baseToken") or {}).get("symbol") or p.get("baseTokenSymbol") or "").upper()
                if sym == ident_upper:
                    pairs_exact.append(p)
            if pairs_exact:
                pairs = pairs_exact
            else:
                return None
        # Prefer highest liquidity among remaining, regardless of chain (all chains allowed)
        def _rank(p):
            liq = float((p.get("liquidity", {}) or {}).get("usd") or p.get("liquidity") or 0)
            return liq
        best = max(pairs, key=_rank)
        change = best.get("priceChange", {}) or {}
        return {
            "symbol": (best.get("baseToken", {}) or {}).get("symbol") or best.get("baseTokenSymbol") or ident,
            "address": (best.get("baseToken", {}) or {}).get("address") or best.get("baseTokenAddress"),
            "chain": best.get("chainId") or best.get("chainId"),
            "change1h": float(change.get("h1") or 0),
            "change6h": float(change.get("h6") or 0),
            "change24h": float(change.get("h24") or 0),
            "liquidity_usd": float((best.get("liquidity", {}) or {}).get("usd") or 0),
            "pair_address": best.get("pairAddress") or best.get("pairAddress"),
        }
    except Exception:
        return None
def _hours_since(dt_iso_str: str | None) -> float | None:
    if not dt_iso_str:
        return None
    try:
        # Normalize: 2025-01-01T12:34:56.000Z
        dt = datetime.fromisoformat(dt_iso_str.replace("Z", "+00:00"))
        return max(0.0, (datetime.now(timezone.utc) - dt).total_seconds() / 3600.0)
    except Exception:
        return None


def _approx_return_since_call_using_dex(change: Dict[str, float], hours_since: float | None) -> float | None:
    if hours_since is None:
        return None
    h1 = float(change.get("h1", 0) if isinstance(change, dict) else 0)
    h6 = float(change.get("h6", 0) if isinstance(change, dict) else 0)
    h24 = float(change.get("h24", 0) if isinstance(change, dict) else 0)
    if hours_since <= 1.0:
        return h1
    if hours_since <= 6.0:
        return h6
    if hours_since <= 24.0:
        return h24
    # Longer-term requires external history; return 24h as best-effort
    return h24


def _birdeye_history_returns(address: str, call_dt_iso: str) -> Dict[str, float] | None:
    api_key = os.environ.get("BIRDEYE_API_KEY")
    if not api_key or not address or not call_dt_iso:
        return None
    try:
        call_ts = int(datetime.fromisoformat(call_dt_iso.replace("Z", "+00:00")).timestamp())
    except Exception:
        return None
    horizons = {"1h": 3600, "24h": 86400, "7d": 7*86400, "30d": 30*86400}
    try:
        params = {
            "address": address,
            "time_from": str(call_ts - 3600),
            "time_to": str(call_ts + 30*86400 + 3600),
            "interval": "1h",
        }
        r = requests.get(BIRDEYE_HISTORY, params=params, headers={"X-API-KEY": api_key}, timeout=12)
        if not r.ok:
            return None
        data = r.json()
        prices = data.get("data", {}).get("items") or data.get("data", {}).get("price") or []
        if not prices:
            return None
        def _nearest(ts_target: int):
            nearest = None
            best_dt = 10**12
            for it in prices:
                ts_i = int(it.get("unixTime") or it.get("time"))
                dt = abs(ts_i - ts_target)
                if dt < best_dt:
                    best_dt = dt
                    nearest = it
            return nearest
        base = _nearest(call_ts)
        if not base:
            return None
        p0 = float(base.get("value") or base.get("price") or 0)
        if p0 <= 0:
            return None
        result: Dict[str, float] = {}
        for k, delta in horizons.items():
            tgt = _nearest(call_ts + delta)
            if tgt:
                p1 = float(tgt.get("value") or tgt.get("price") or 0)
                if p1 > 0:
                    result[k] = (p1 / p0 - 1.0) * 100.0
        return result if result else None
    except Exception:
        return None
def _yahoo_history_returns(symbol: str, call_dt_iso: str) -> Dict[str, float] | None:
    if not symbol or not call_dt_iso:
        return None
    try:
        call_ts = int(datetime.fromisoformat(call_dt_iso.replace("Z", "+00:00")).timestamp())
    except Exception:
        return None
    candidates = [symbol.upper(), f"{symbol.upper()}-USD"]
    for sym in candidates:
        try:
            r = requests.get(
                YAHOO_CHART + sym,
                params={"range": "35d", "interval": "1h"}, timeout=10,
            )
            if not r.ok:
                continue
            data = r.json().get("chart", {}).get("result", [None])[0]
            if not data:
                continue
            ts = data.get("timestamp") or []
            closes = ((data.get("indicators", {}) or {}).get("quote", [{}])[0]).get("close") or []
            if not ts or not closes or len(ts) != len(closes):
                continue
            # Build list of (timestamp, close)
            series = [(int(t), float(c)) for t, c in zip(ts, closes) if c is not None]
            if not series:
                continue
            def nearest(target: int):
                best = None
                best_dt = 10**12
                for t, c in series:
                    dt = abs(t - target)
                    if dt < best_dt:
                        best_dt = dt
                        best = (t, c)
                return best
            base = nearest(call_ts)
            if not base or base[1] <= 0:
                continue
            p0 = base[1]
            horizons = {"1h": 3600, "24h": 86400, "7d": 7*86400, "30d": 30*86400}
            out: Dict[str, float] = {}
            for k, delta in horizons.items():
                nxt = nearest(call_ts + delta)
                if nxt and nxt[1] > 0:
                    out[k] = (nxt[1] / p0 - 1.0) * 100.0
            if out:
                return out
        except Exception:
            continue
    return None


def grade_twitter(handle_or_url: str) -> Dict[str, Any]:
    handle = _extract_handle(handle_or_url or "")
    if not handle:
        return {"error": "Please provide a valid @handle or Twitter URL."}
    logger.info("grade_twitter:start handle=%s", handle)
    tweets: List[Dict[str, Any]] = []
    status_id = _extract_status_id(handle_or_url or "")
    if status_id:
        one = _fetch_single_tweet(status_id, handle)
        if one:
            tweets.append(one)
    tweets.extend(_twitter_fetch_recent(handle))
    # Deduplicate by id keeping the earliest occurrence (status first)
    seen: Set[str] = set()
    dedup: List[Dict[str, Any]] = []
    for t in tweets:
        tid = str(t.get("id") or "")
        if tid and tid in seen:
            continue
        if tid:
            seen.add(tid)
        dedup.append(t)
    tweets = dedup
    events = _extract_coin_mentions_per_tweet(tweets)
    # As a final fallback, extract uppercase tokens if entities missing
    if not events:
        for t in tweets:
            text = (t.get("text") or "")
            ts = t.get("created_at")
            tid = t.get("id")
            for m in re.finditer(r"\b([A-Z]{2,6})\b", text):
                sym = m.group(1)
                if sym in {"THE","AND","FOR","WITH","THIS","FROM","WHAT","WILL","HAVE","THAT","YOUR","JUST","LIKE"}:
                    continue
                events.append({
                    "kind": "symbol",
                    "id": sym,
                    "text": text,
                    "created_at": ts,
                    "tweet_id": tid,
                    "source": "uppercase",
                })
    # If still empty, try a second Nitter instance directly for robustness
    if not events:
        extra = _twitter_fetch_from_nitter(handle, max_results=50)
        if extra:
            events = _extract_coin_mentions_per_tweet(extra)
    if not events:
        return {
            "handle": handle,
            "score": 50,
            "label": human_label_from_score_twitter(50),
            "metrics": {
                "message": "No coin mentions detected in recent tweets",
                "unique_coins": 0,
                "total_calls": 0,
            },
            "coins": [],
        }

    # Aggregate by ident, enrich, and keep per-call timestamps
    calls_by_ident: Dict[str, List[Dict[str, Any]] ] = {}
    for ev in events:
        ident = ev["id"]
        calls_by_ident.setdefault(ident, []).append(ev)

    enriched: List[Dict[str, Any]] = []
    for ident, evs in list(calls_by_ident.items())[:60]:
        info = None
        # Route well-known tickers first
        if evs[0].get("kind") == "symbol":
            up = ident.upper()
            if up in YAHOO_STOCKS or up in YAHOO_CRYPTO:
                info = None  # force Yahoo for these symbols
            else:
                info = _dexscreener_lookup(ident, "symbol")
        else:
            info = _dexscreener_lookup(ident, evs[0].get("kind", "symbol"))
        # For symbols that aren't Solana tokens, also attempt Yahoo Finance series
        yahoo_long = None
        if (not info) and evs[0].get("kind") == "symbol":
            # Try Yahoo only if the symbol looks like a stock/crypto ticker
            for ev in evs[:2]:
                if ev.get("created_at"):
                    yahoo_long = _yahoo_history_returns(ident, ev.get("created_at"))
                    if yahoo_long:
                        break
        if not info and not yahoo_long:
            # Still include symbol with unknown performance so UI shows calls
            per_call_min: List[Dict[str, Any]] = []
            for ev in evs[:8]:
                per_call_min.append({
                    "created_at": ev.get("created_at"),
                    "source": ev.get("source"),
                    "tweet_id": ev.get("tweet_id"),
                    "approx_since_call_pct": None,
                    "longterm": None,
                })
            enriched.append({
                "id": ident,
                "kind": evs[0].get("kind", "symbol"),
                "symbol": ident,
                "address": None,
                "liquidity_usd": None,
                "change1h": None,
                "change6h": None,
                "change24h": None,
                "calls": per_call_min,
            })
            continue
        # Compute short-term approximations per call
        per_call: List[Dict[str, Any]] = []
        for ev in evs[:8]:  # cap calls per asset for perf
            hs = _hours_since(ev.get("created_at"))
            approx = None
            longterm = None
            if info:
                approx = _approx_return_since_call_using_dex({
                    "h1": info.get("change1h", 0),
                    "h6": info.get("change6h", 0),
                    "h24": info.get("change24h", 0),
                }, hs)
                longterm = _birdeye_history_returns(info.get("address"), ev.get("created_at")) if ev.get("created_at") else None
            if yahoo_long and ev.get("created_at"):
                # Prefer Yahoo long horizons for non-DEX assets
                longterm = yahoo_long
            per_call.append({
                "created_at": ev.get("created_at"),
                "source": ev.get("source"),
                "tweet_id": ev.get("tweet_id"),
                "approx_since_call_pct": approx,
                "longterm": longterm,
            })
        enriched.append({
            "id": ident,
            "kind": evs[0].get("kind", "symbol"),
            "symbol": (info or {}).get("symbol") or ident,
            "address": (info or {}).get("address"),
            "liquidity_usd": (info or {}).get("liquidity_usd"),
            "change1h": (info or {}).get("change1h"),
            "change6h": (info or {}).get("change6h"),
            "change24h": (info or {}).get("change24h"),
            "calls": per_call,
        })

    if not enriched:
        return {
            "handle": handle,
            "score": 45,
            "label": human_label_from_score_twitter(45),
            "metrics": {
                "message": "Mentions found but no Dexscreener data",
                "unique_coins": len(calls_by_ident),
                "total_calls": sum(len(v) for v in calls_by_ident.values()),
            },
            "coins": [],
        }

    total_calls = int(sum(len(c["calls"]) for c in enriched))
    unique_coins = len(enriched)

    # Compute composite performance score weighted by mentions and liquidity
    perf_vals: List[float] = []
    liq_low_calls = 0
    pump_calls = 0
    for c in enriched:
        liq = float(c.get("liquidity_usd") or 0)
        is_low_liq = liq < 30_000
        for ev in c["calls"]:
            if ev.get("source") == "pumpfun":
                pump_calls += 1
            if is_low_liq:
                liq_low_calls += 1
            approx = ev.get("approx_since_call_pct")
            if isinstance(approx, (int, float)):
                # Weight by liquidity bucket
                w = 1.0 if liq >= 100_000 else (0.7 if liq >= 30_000 else 0.4)
                perf_vals.append(w * float(approx))

    # Start from high and subtract for over-shilling
    score = 85
    # Over-shilling penalties
    if unique_coins > 3:
        score -= min(35, int(2 * (unique_coins - 3)))
    if total_calls > 25:
        score -= min(20, int(0.8 * (total_calls - 25)))
    # Illiquidity penalty
    if total_calls > 0:
        illiq_ratio = liq_low_calls / float(total_calls)
        score -= int(min(20, 25 * illiq_ratio))
    # Pump.fun penalty
    if total_calls > 0:
        pump_ratio = pump_calls / float(total_calls)
        score -= int(min(15, 20 * pump_ratio))
    # Performance impact (short-term)
    if perf_vals:
        mean_perf = sum(perf_vals) / len(perf_vals)
        # Consistency bonus/penalty
        import statistics as _stats
        try:
            stdev = _stats.pstdev(perf_vals)
        except Exception:
            stdev = 0.0
        sharpe_like = mean_perf / max(1.0, stdev)
        score += int(max(-25, min(25, mean_perf / 4)))  # scale
        score += int(max(-10, min(10, sharpe_like * 5)))
    # Clamp
    score = max(1, min(100, int(round(score))))

    label = human_label_from_score_twitter(score)

    # Summaries
    avg1h = sum(c.get("change1h", 0) or 0 for c in enriched) / max(1, unique_coins)
    avg6h = sum(c.get("change6h", 0) or 0 for c in enriched) / max(1, unique_coins)
    avg24h = sum(c.get("change24h", 0) or 0 for c in enriched) / max(1, unique_coins)
    best = max(enriched, key=lambda c: max(c.get("change1h", 0), c.get("change6h", 0), c.get("change24h", 0)))
    worst = min(enriched, key=lambda c: min(c.get("change1h", 0), c.get("change6h", 0), c.get("change24h", 0)))

    # Hit rate within ~24h approximation
    hits = [1 for c in enriched for ev in c["calls"] if isinstance(ev.get("approx_since_call_pct"), (int, float)) and ev.get("approx_since_call_pct", 0) >= 10.0]
    hit_rate = (sum(hits) / max(1, len(perf_vals))) if perf_vals else 0.0

    return {
        "handle": handle,
        "score": score,
        "label": label,
        "metrics": {
            "unique_coins": unique_coins,
            "total_calls": total_calls,
            "avg_change_1h": round(avg1h, 2),
            "avg_change_6h": round(avg6h, 2),
            "avg_change_24h": round(avg24h, 2),
            "hit_rate_approx_24h": round(hit_rate, 3),
            "low_liq_calls": int(liq_low_calls),
            "pumpfun_calls": int(pump_calls),
            "calls_with_perf": int(len(perf_vals)),
            "best_symbol": best.get("symbol"),
            "best_change_max": round(max(best.get("change1h", 0), best.get("change6h", 0), best.get("change24h", 0)), 2),
            "worst_symbol": worst.get("symbol"),
            "worst_change_min": round(min(worst.get("change1h", 0), worst.get("change6h", 0), worst.get("change24h", 0)), 2),
        },
        "coins": enriched,
    }


@app.post("/grade_twitter")
def grade_twitter_api():
    data = request.get_json(silent=True) or {}
    handle = (data.get("handle") or request.form.get("handle") or "").strip()
    if not handle:
        return jsonify({"error": "Please enter a Twitter handle (e.g., @name)."}), 400
    logger.info("grade_twitter_api:request handle=%s", handle)
    result = grade_twitter(handle)
    status = 200 if not result.get("error") else 400
    logger.info("grade_twitter_api:response handle=%s status=%s", handle, status)
    return jsonify(result), status


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=True)


