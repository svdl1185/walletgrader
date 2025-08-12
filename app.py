import os
import math
from datetime import datetime, timezone
from typing import Dict, Any, List, Set, Tuple

from flask import Flask, render_template, request, jsonify

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
        balance_lamports = 0
    sol_balance = balance_lamports / 1_000_000_000

    # Signatures (recent history)
    signatures_resp = client.get_signatures_for_address(public_key, limit=200)
    signatures_json: List[Dict[str, Any]] = []
    try:
        if hasattr(signatures_resp, "value"):
            signatures_json = [
                {
                    "signature": getattr(s, "signature", None),
                    "blockTime": getattr(s, "block_time", None) or getattr(s, "blockTime", None),
                }
                for s in getattr(signatures_resp, "value", [])
            ]
        else:
            signatures_json = signatures_resp.get("result", []) if isinstance(signatures_resp, dict) else []
    except Exception:
        signatures_json = []
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

    # Inspect a sample of transactions to infer basic trading cadence features
    sample_sigs: List[str] = [s.get("signature") for s in signatures_json[:40] if s.get("signature")]
    sampled_tx: List[Tuple[int, Set[str]]] = []  # list of (timestamp, set(program_ids))

    for sig in sample_sigs:
        try:
            tx_resp = client.get_transaction(sig, max_supported_transaction_version=0)
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
        except Exception:
            continue

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

    # New memecoin-oriented scoring (0-100)
    score = 0

    # Balance (0-10) — lightly weighted
    if sol_balance >= 100:
        score += 10
    elif sol_balance >= 10:
        score += 8
    elif sol_balance >= 1:
        score += 6
    elif sol_balance >= 0.1:
        score += 3
    elif sol_balance > 0:
        score += 1

    # Activity shape (0-25) — moderate activity best, hyperactive penalized later via churn
    if tx_count == 0:
        activity_score = 0
    elif tx_count <= 5:
        activity_score = 5
    elif tx_count <= 15:
        activity_score = 15
    elif tx_count <= 60:
        activity_score = 25
    elif tx_count <= 120:
        activity_score = 18
    else:
        activity_score = 10
    score += activity_score

    # Account age (0-10) — reduced weight
    if days_old >= 365:
        score += 10
    elif days_old >= 180:
        score += 8
    elif days_old >= 90:
        score += 6
    elif days_old >= 30:
        score += 3

    # Holding quality (0-35) — reward longer median time between txs
    if median_inter_tx_hours >= 72:
        score += 35
    elif median_inter_tx_hours >= 36:
        score += 28
    elif median_inter_tx_hours >= 12:
        score += 18
    elif median_inter_tx_hours >= 4:
        score += 10
    elif median_inter_tx_hours >= 1:
        score += 4

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
    else:
        hold_profile = "scalping"

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
        },
    }


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
    result = grade_wallet(address)
    status = 200 if not result.get("error") else 400
    return jsonify(result), status


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=True)


