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

# Single-scan tuning knobs (set via env for your RPC tier)
# Recommended for paid RPC (raise gradually): PAGE_LIMIT 500–1000, MAX_PAGES 2–5, MAX_ANALYZE_TX 150–400
PAGE_LIMIT = int(os.environ.get("SCAN_PAGE_LIMIT", "300"))  # signatures per page
MAX_PAGES = int(os.environ.get("SCAN_MAX_PAGES", "3"))     # pages of signatures
TIME_BUDGET_SEC = float(os.environ.get("SCAN_TIME_BUDGET_SEC", "12"))
MAX_ANALYZE_TX = int(os.environ.get("SCAN_MAX_ANALYZE_TX", "180"))


def get_solana_client() -> Client:
    rpc_url = os.environ.get("SOLANA_RPC_URL", "https://api.mainnet-beta.solana.com")
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
                sleep_for = delay * (2 ** (attempts - 1)) + random.uniform(0, 0.2)
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

    # Build inter-transaction timing and simple churn features
    # Use signatures list for broader time coverage
    times_desc: List[int] = [s.get("blockTime") for s in signatures_json if s.get("blockTime")]
    times_asc: List[int] = sorted(times_desc)
    inter_tx_hours: List[float] = []
    for i in range(1, len(times_asc)):
        dt_sec = max(0, times_asc[i] - times_asc[i - 1])
        inter_tx_hours.append(dt_sec / 3600.0)
    median_inter_tx_hours = _median(inter_tx_hours)

    # Repeat sequences: adjacent sampled tx that share any program within 6 hours window
    sampled_tx_sorted = sorted(sampled_tx, key=lambda t: t[0])
    repeat_sequences_6h = 0
    for i in range(1, len(sampled_tx_sorted)):
        t_prev, progs_prev = sampled_tx_sorted[i - 1]
        t_curr, progs_curr = sampled_tx_sorted[i]
        if (t_curr - t_prev) <= 6 * 3600 and (progs_prev & progs_curr):
            repeat_sequences_6h += 1

    # Derive trade success heuristics from SOL balance deltas
    trade_deltas_sorted = sorted(trade_deltas, key=lambda t: t[0])
    min_move_sol = 0.01  # ignore tiny moves
    last_outflow_time: int | None = None
    last_outflow_amt: float = 0.0
    successful_roundtrips = 0
    big_profit_trades = 0
    best_profit_sol = 0.0
    total_profit_sol = 0.0
    positive_tx = 0
    negative_tx = 0
    for ts, dsol in trade_deltas_sorted:
        if dsol > min_move_sol:
            positive_tx += 1
            if last_outflow_time is not None and (ts - last_outflow_time) <= 12 * 3600 and dsol >= 1.05 * last_outflow_amt:
                profit = dsol - last_outflow_amt
                successful_roundtrips += 1
                total_profit_sol += profit
                if profit > best_profit_sol:
                    best_profit_sol = profit
                if profit >= 1.0:
                    big_profit_trades += 1
                last_outflow_time = None
                last_outflow_amt = 0.0
        elif dsol < -min_move_sol:
            negative_tx += 1
            last_outflow_time = ts
            last_outflow_amt = abs(dsol)
        else:
            # ignore noise
            pass

    win_rate = 0.0
    if (positive_tx + negative_tx) > 0:
        win_rate = round(positive_tx / float(positive_tx + negative_tx), 4)

    # New memecoin-oriented scoring (0-100) with profits focus
    score = 0

    # Profitable roundtrips (0-50)
    if successful_roundtrips >= 8:
        score += 50
    elif successful_roundtrips >= 5:
        score += 40
    elif successful_roundtrips >= 3:
        score += 30
    elif successful_roundtrips >= 2:
        score += 20
    elif successful_roundtrips >= 1:
        score += 10

    # Profit size bonus (0-20)
    if best_profit_sol >= 5 or total_profit_sol >= 10:
        score += 20
    elif best_profit_sol >= 2 or total_profit_sol >= 5:
        score += 15
    elif best_profit_sol >= 1 or total_profit_sol >= 2:
        score += 10
    elif best_profit_sol >= 0.25 or total_profit_sol >= 0.5:
        score += 5

    # Balance (0-5) — very light
    if sol_balance >= 100:
        score += 5
    elif sol_balance >= 10:
        score += 4
    elif sol_balance >= 1:
        score += 3
    elif sol_balance >= 0.1:
        score += 2
    elif sol_balance > 0:
        score += 1

    # Activity shape (0-10) — moderate activity best
    if tx_count == 0:
        activity_score = 0
    elif tx_count <= 5:
        activity_score = 2
    elif tx_count <= 15:
        activity_score = 6
    elif tx_count <= 60:
        activity_score = 10
    elif tx_count <= 120:
        activity_score = 7
    else:
        activity_score = 4
    score += activity_score

    # Account age (0-5) — reduced weight
    if days_old >= 365:
        score += 5
    elif days_old >= 180:
        score += 4
    elif days_old >= 90:
        score += 3
    elif days_old >= 30:
        score += 2

    # Holding quality (0-20) — reward longer median time between txs
    if median_inter_tx_hours >= 72:
        score += 20
    elif median_inter_tx_hours >= 36:
        score += 16
    elif median_inter_tx_hours >= 12:
        score += 10
    elif median_inter_tx_hours >= 4:
        score += 6
    elif median_inter_tx_hours >= 1:
        score += 2

    # Churn penalty (up to -15) — frequent in/out and very short holds suggest "jeet" behavior
    if repeat_sequences_6h >= 10:
        score -= 10
    elif repeat_sequences_6h >= 6:
        score -= 7
    elif repeat_sequences_6h >= 3:
        score -= 4

    if median_inter_tx_hours > 0 and median_inter_tx_hours < 1:
        score -= 5

    score = max(1, min(100, int(score)))

    label = human_label_from_score(score)

    # Derive simple profiles for display
    if median_inter_tx_hours >= 36:
        hold_profile = "long holds"
    elif median_inter_tx_hours >= 12:
        hold_profile = "swing holds"
    elif median_inter_tx_hours >= 1:
        hold_profile = "short holds"
    elif median_inter_tx_hours >= 0.25:
        hold_profile = "scalping"
    else:
        hold_profile = "high-speed trader"

    if tx_count <= 15:
        activity_profile = "low"
    elif tx_count <= 60:
        activity_profile = "moderate"
    elif tx_count <= 120:
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
            "median_inter_tx_hours": round(median_inter_tx_hours, 2),
            "repeat_sequences_6h": repeat_sequences_6h,
            "hold_profile": hold_profile,
            "activity_profile": activity_profile,
            "successful_trades": successful_roundtrips,
            "big_profit_trades": big_profit_trades,
            "best_profit_sol": round(best_profit_sol, 6),
            "total_profit_sol": round(total_profit_sol, 6),
            "win_rate": win_rate,
        },
    }
    logger.info("grade_wallet:done address=%s score=%s", address, score)
    return result_obj


def _analyze_signatures_full(client: Client, public_key: Any, signatures_json: List[Dict[str, Any]]) -> Dict[str, Any]:
    # This largely mirrors the logic inside grade_wallet but runs across all given signatures.
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

    # Times for inter-tx spacing
    times_desc: List[int] = [s.get("blockTime") for s in signatures_json if s.get("blockTime")]
    times_asc: List[int] = sorted(times_desc)
    inter_tx_hours: List[float] = []
    for i in range(1, len(times_asc)):
        dt_sec = max(0, times_asc[i] - times_asc[i - 1])
        inter_tx_hours.append(dt_sec / 3600.0)

    def _median(nums: List[float]) -> float:
        if not nums:
            return 0.0
        nums_sorted = sorted(nums)
        n = len(nums_sorted)
        mid = n // 2
        if n % 2 == 1:
            return float(nums_sorted[mid])
        return float((nums_sorted[mid - 1] + nums_sorted[mid]) / 2.0)

    median_inter_tx_hours = _median(inter_tx_hours)

    # Program overlap churn
    sample_for_churn = []
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
            program_ids: Set[str] = set()
            for ix in msg.get("instructions", []):
                pid = ix.get("programId") or ix.get("programIdIndex")
                if isinstance(pid, str):
                    program_ids.add(pid)
            account_keys = msg.get("accountKeys", [])
            for key in account_keys:
                if isinstance(key, str):
                    program_ids.add(key)
                elif isinstance(key, dict) and "pubkey" in key:
                    program_ids.add(key["pubkey"])
            sample_for_churn.append((int(block_time), program_ids))

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

    # Churn
    sample_for_churn_sorted = sorted(sample_for_churn, key=lambda t: t[0])
    repeat_sequences_6h = 0
    for i in range(1, len(sample_for_churn_sorted)):
        t_prev, progs_prev = sample_for_churn_sorted[i - 1]
        t_curr, progs_curr = sample_for_churn_sorted[i]
        if (t_curr - t_prev) <= 6 * 3600 and (progs_prev & progs_curr):
            repeat_sequences_6h += 1

    # Profits
    trade_deltas_sorted = sorted(trade_deltas, key=lambda t: t[0])
    min_move_sol = 0.01
    last_outflow_time: int | None = None
    last_outflow_amt: float = 0.0
    successful_roundtrips = 0
    big_profit_trades = 0
    best_profit_sol = 0.0
    total_profit_sol = 0.0
    positive_tx = 0
    negative_tx = 0
    for ts, dsol in trade_deltas_sorted:
        if dsol > min_move_sol:
            positive_tx += 1
            if last_outflow_time is not None and (ts - last_outflow_time) <= 12 * 3600 and dsol >= 1.05 * last_outflow_amt:
                profit = dsol - last_outflow_amt
                successful_roundtrips += 1
                total_profit_sol += profit
                if profit > best_profit_sol:
                    best_profit_sol = profit
                if profit >= 1.0:
                    big_profit_trades += 1
                last_outflow_time = None
                last_outflow_amt = 0.0
        elif dsol < -min_move_sol:
            negative_tx += 1
            last_outflow_time = ts
            last_outflow_amt = abs(dsol)

    # Age and counts
    days_old = 0
    if times_asc:
        first_tx_time = datetime.fromtimestamp(times_asc[0], tz=timezone.utc)
        days_old = max(0, (datetime.now(timezone.utc) - first_tx_time).days)
    tx_count = len(signatures_json)

    # Reuse the same scoring mix as grade_wallet after profits update
    # Build a faux signatures_resp-like object and call the profit-weighted path is complex; so re-embed key parts

    # Activity shape
    if tx_count == 0:
        activity_score = 0
    elif tx_count <= 5:
        activity_score = 2
    elif tx_count <= 15:
        activity_score = 6
    elif tx_count <= 60:
        activity_score = 10
    elif tx_count <= 120:
        activity_score = 7
    else:
        activity_score = 4

    # Holding quality buckets
    if median_inter_tx_hours >= 72:
        hold_points = 20
    elif median_inter_tx_hours >= 36:
        hold_points = 16
    elif median_inter_tx_hours >= 12:
        hold_points = 10
    elif median_inter_tx_hours >= 4:
        hold_points = 6
    elif median_inter_tx_hours >= 1:
        hold_points = 2
    else:
        hold_points = 0

    score = 0
    # Profits
    if successful_roundtrips >= 8:
        score += 50
    elif successful_roundtrips >= 5:
        score += 40
    elif successful_roundtrips >= 3:
        score += 30
    elif successful_roundtrips >= 2:
        score += 20
    elif successful_roundtrips >= 1:
        score += 10

    if best_profit_sol >= 5 or total_profit_sol >= 10:
        score += 20
    elif best_profit_sol >= 2 or total_profit_sol >= 5:
        score += 15
    elif best_profit_sol >= 1 or total_profit_sol >= 2:
        score += 10
    elif best_profit_sol >= 0.25 or total_profit_sol >= 0.5:
        score += 5

    if sol_balance >= 100:
        score += 5
    elif sol_balance >= 10:
        score += 4
    elif sol_balance >= 1:
        score += 3
    elif sol_balance >= 0.1:
        score += 2
    elif sol_balance > 0:
        score += 1

    score += activity_score
    score += hold_points
    if median_inter_tx_hours > 0 and median_inter_tx_hours < 1:
        score -= 5
    if repeat_sequences_6h >= 10:
        score -= 10
    elif repeat_sequences_6h >= 6:
        score -= 7
    elif repeat_sequences_6h >= 3:
        score -= 4

    score = max(1, min(100, int(score)))
    label = human_label_from_score(score)

    # Profiles
    if median_inter_tx_hours >= 36:
        hold_profile = "long holds"
    elif median_inter_tx_hours >= 12:
        hold_profile = "swing holds"
    elif median_inter_tx_hours >= 1:
        hold_profile = "short holds"
    elif median_inter_tx_hours >= 0.25:
        hold_profile = "scalping"
    else:
        hold_profile = "high-speed trader"

    if tx_count <= 15:
        activity_profile = "low"
    elif tx_count <= 60:
        activity_profile = "moderate"
    elif tx_count <= 120:
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
            "median_inter_tx_hours": round(median_inter_tx_hours, 2),
            "repeat_sequences_6h": repeat_sequences_6h,
            "hold_profile": hold_profile,
            "activity_profile": activity_profile,
            "successful_trades": successful_roundtrips,
            "big_profit_trades": big_profit_trades,
            "best_profit_sol": round(best_profit_sol, 6),
            "total_profit_sol": round(total_profit_sol, 6),
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


