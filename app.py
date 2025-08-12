import os
import math
from datetime import datetime, timezone
from typing import Dict, Any, List, Set, Tuple
import time
import random

from flask import Flask, render_template, request, jsonify
import logging

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

# Single-scan tuning knobs (set via env for your RPC tier)
# Recommended for paid RPC (raise gradually): PAGE_LIMIT 500–1000, MAX_PAGES 2–5, MAX_ANALYZE_TX 150–400
PAGE_LIMIT = int(os.environ.get("SCAN_PAGE_LIMIT", "300"))  # signatures per page
MAX_PAGES = int(os.environ.get("SCAN_MAX_PAGES", "3"))     # pages of signatures
TIME_BUDGET_SEC = float(os.environ.get("SCAN_TIME_BUDGET_SEC", "12"))
MAX_ANALYZE_TX = int(os.environ.get("SCAN_MAX_ANALYZE_TX", "180"))


def get_solana_client() -> Client:
    rpc_url = os.environ.get("SOLANA_RPC_URL", "https://api.mainnet-beta.solana.com")
    global _LAST_RPC_URL
    if rpc_url != _LAST_RPC_URL:
        logger.info("rpc:selected url=%s", rpc_url)
        _LAST_RPC_URL = rpc_url
    return Client(rpc_url)


def human_label_from_score(score: int) -> str:
    if score >= 95:
        return "Based"
    if score >= 90:
        return "Cabal"
    if score >= 80:
        return "Chad"
    if score >= 70:
        return "Diamond Hands"
    if score >= 50:
        return "Normie"
    if score >= 30:
        return "Paper"
    if score >= 10:
        return "Jeet"
    return "Rug Victim"

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

            # Also include account keys, which may contain program IDs depending on encoding
            account_keys = msg.get("accountKeys", [])
            for key in account_keys:
                if isinstance(key, str):
                    program_ids.add(key)
                elif isinstance(key, dict) and "pubkey" in key:
                    program_ids.add(key["pubkey"])

            sampled_tx.append((int(block_time), program_ids))

            # Compute net SOL delta for this wallet excluding fee, if metadata is available
            try:
                meta = result_obj.get("meta", {})
                pre_balances = meta.get("preBalances", [])
                post_balances = meta.get("postBalances", [])
                fee_lamports = int(meta.get("fee", 0) or 0)
                # Find wallet index in account keys
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
    min_move_sol = 0.01  # ignore tiny moves

    class _Cycle:
        def __init__(self, start_ts: int, first_outflow: float) -> None:
            self.start_ts = int(start_ts)
            self.outflow = float(first_outflow)  # total SOL spent (cost basis)
            self.inflow = 0.0  # total SOL received

    realized: List[Dict[str, float | int]] = []
    open_cycle: _Cycle | None = None

    for ts, dsol in trade_deltas_sorted:
        if dsol < -min_move_sol:
            # Spend SOL (likely buy)
            if open_cycle is None:
                open_cycle = _Cycle(ts, abs(dsol))
            else:
                open_cycle.outflow += abs(dsol)
        elif dsol > min_move_sol:
            # Receive SOL (likely sell)
            if open_cycle is None:
                # Positive inflow without prior spend: ignore for realized PnL (could be deposit/airdrop)
                continue
            remaining_needed = max(0.0, open_cycle.outflow - open_cycle.inflow)
            consume = min(float(dsol), remaining_needed)
            open_cycle.inflow += consume
            # If cycle fully closed (recovered at least cost), realize PnL
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
                # Remainder of inflow (if any) is ignored; start fresh
                open_cycle = None
        # else small noise ignored

    # Aggregate metrics
    wins = [t for t in realized if t["pnl_sol"] > 0]
    losses = [t for t in realized if t["pnl_sol"] < 0]
    realized_trades = len(realized)
    wins_count = len(wins)
    losses_count = len(losses)
    total_profit_sol = float(sum(t["pnl_sol"] for t in wins)) if wins else 0.0
    total_loss_sol = float(sum(t["pnl_sol"] for t in losses)) if losses else 0.0  # negative
    net_profit_sol = total_profit_sol + total_loss_sol
    win_rate = round(wins_count / realized_trades, 4) if realized_trades > 0 else 0.0
    profit_factor = (total_profit_sol / abs(total_loss_sol)) if total_loss_sol < 0 else (99.0 if total_profit_sol > 0 else 0.0)

    best_trade_profit_sol = max([t["pnl_sol"] for t in wins], default=0.0)
    worst_trade_loss_sol = min([t["pnl_sol"] for t in losses], default=0.0)
    best_trade_roi = max([t["roi"] for t in realized], default=0.0)
    worst_trade_roi = min([t["roi"] for t in realized], default=0.0)

    # Quick wins by hold time thresholds (applied to winning trades only)
    win_hold_minutes = [t["hold_seconds"] / 60.0 for t in wins]
    median_win_hold_minutes = _median(win_hold_minutes)
    quick_30m = sum(1 for t in wins if t["hold_seconds"] <= 30 * 60)
    quick_2h = sum(1 for t in wins if 30 * 60 < t["hold_seconds"] <= 2 * 3600)
    quick_6h = sum(1 for t in wins if 2 * 3600 < t["hold_seconds"] <= 6 * 3600)

    # Big % profit counts
    roi_50 = sum(1 for t in realized if t["roi"] >= 0.5)
    roi_100 = sum(1 for t in realized if t["roi"] >= 1.0)
    roi_200 = sum(1 for t in realized if t["roi"] >= 2.0)

    # Tail loss count (very bad % losses)
    tail_loss_count = sum(1 for t in losses if t["roi"] <= -0.6)

    # Window-level proxy signals (fallback if no realized trades)
    window_positive_sol = float(sum(d for _, d in trade_deltas_sorted if d > 0))
    window_negative_sol = float(sum(abs(d) for _, d in trade_deltas_sorted if d < 0))
    window_net_sol = window_positive_sol - window_negative_sol
    positive_events = sum(1 for _, d in trade_deltas_sorted if d > 0)
    negative_events = sum(1 for _, d in trade_deltas_sorted if d < 0)
    best_positive_event_sol = max([d for _, d in trade_deltas_sorted if d > 0], default=0.0)

    # Score assembly (0-100), profit-first
    score = 0

    # Absolute profit size (0-35)
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

    # Big percentage profit bonus (0-15)
    pct_pts = 2 * roi_200 + 3 * roi_100 + 1 * roi_50
    score += min(15, pct_pts)

    # Big absolute win bonus (0-10)
    if best_trade_profit_sol >= 5:
        score += 10
    elif best_trade_profit_sol >= 2:
        score += 7
    elif best_trade_profit_sol >= 1:
        score += 5
    elif best_trade_profit_sol >= 0.5:
        score += 3

    # Loss magnitude penalty (up to -25)
    loss_mag = abs(total_loss_sol)
    if loss_mag >= 10:
        score -= 25
    elif loss_mag >= 5:
        score -= 18
    elif loss_mag >= 2:
        score -= 12
    elif loss_mag >= 1:
        score -= 7

    # Tail loss penalty (up to -10)
    tail_penalty = 2 * tail_loss_count
    if abs(worst_trade_loss_sol) >= 2.0:
        tail_penalty += 2
    score -= min(10, tail_penalty)

    # Fallback scoring if no realized trades were detected
    if realized_trades == 0:
        # Net gain over window
        if window_net_sol >= 5:
            score += 35
        elif window_net_sol >= 2:
            score += 25
        elif window_net_sol >= 1:
            score += 18
        elif window_net_sol >= 0.25:
            score += 10
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
        if window_negative_sol >= 5:
            score -= 15
        elif window_negative_sol >= 2:
            score -= 10
        elif window_negative_sol >= 1:
            score -= 6
        elif window_negative_sol >= 0.5:
            score -= 3

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
        address = request.form.get("address", "").strip()
        if not address:
            error = "Please enter a Solana wallet address."
        else:
            result = grade_wallet(address)
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


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=True)


