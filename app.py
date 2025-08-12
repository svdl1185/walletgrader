import os
import math
from datetime import datetime, timezone
from typing import Dict, Any, List, Set

from flask import Flask, render_template, request

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


KNOWN_DEFI_PROGRAMS: Dict[str, str] = {
    # Non-exhaustive sample of popular programs; used for small score bonuses
    "JUP3c2Uh3Mqn4hZ4rYd2G1iC2a9eGdQ5qS4s6c2iPdG": "Jupiter",
    "RVKd61ztZW9GUWhgSzQmfvuJraG7h1i3Xf7f5o7z88k": "Raydium v4",
    "orcaEKTdK7LKz57vaAYr9QeNsVEPfiu6QeMU1kektZE": "Orca",
    "MarBmsSgqKXSSJgh3W9a1fCtt7HSqJj1MqWw5qY8uX5": "Marinade",
    "JitoStakes3vLZ9M2vVwH7sH5G8fHrVh2kKfH2b2r1rs": "Jito Staking",
    "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA": "SPL Token",
}


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

    # Inspect a small sample of transactions to infer program diversity and DeFi interactions
    sample_sigs: List[str] = [s.get("signature") for s in signatures_json[:25] if s.get("signature")]
    unique_programs: Set[str] = set()
    defi_program_hits: Set[str] = set()

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
            # Collect program IDs from message account keys that are marked as program or from instructions
            msg = tx_json.get("result", {}).get("transaction", {}).get("message", {})
            account_keys = msg.get("accountKeys", [])
            for key in account_keys:
                # accountKeys entries can be strings or dicts depending on encoding
                if isinstance(key, str):
                    program_id = key
                elif isinstance(key, dict) and "pubkey" in key:
                    program_id = key["pubkey"]
                else:
                    continue
                unique_programs.add(program_id)
                if program_id in KNOWN_DEFI_PROGRAMS:
                    defi_program_hits.add(program_id)

            instrs = msg.get("instructions", [])
            for ix in instrs:
                program_id = ix.get("programId") or ix.get("programIdIndex")
                if isinstance(program_id, str):
                    unique_programs.add(program_id)
                    if program_id in KNOWN_DEFI_PROGRAMS:
                        defi_program_hits.add(program_id)
        except Exception:
            # Ignore per-tx failures; continue sampling
            continue

    # Basic scoring heuristic (0-100)
    score = 0

    # Balance weight (0-20)
    if sol_balance >= 100:
        score += 20
    elif sol_balance >= 10:
        score += 15
    elif sol_balance >= 1:
        score += 10
    elif sol_balance >= 0.1:
        score += 5
    elif sol_balance > 0:
        score += 2

    # Activity (0-25) within last 200 tx
    if tx_count >= 1000:
        score += 25
    elif tx_count >= 200:
        score += 20
    elif tx_count >= 50:
        score += 15
    elif tx_count >= 10:
        score += 10
    elif tx_count >= 1:
        score += 5

    # Account age (0-20)
    if days_old >= 365:
        score += 20
    elif days_old >= 180:
        score += 15
    elif days_old >= 90:
        score += 10
    elif days_old >= 30:
        score += 5

    # Program diversity (0-20)
    program_diversity = len(unique_programs)
    if program_diversity >= 15:
        score += 20
    elif program_diversity >= 8:
        score += 15
    elif program_diversity >= 4:
        score += 10
    elif program_diversity >= 2:
        score += 5

    # DeFi bonus (0-10)
    score += min(10, len(defi_program_hits) * 3)

    score = max(1, min(100, int(score)))

    label = human_label_from_score(score)

    return {
        "address": str(public_key),
        "score": score,
        "label": label,
        "metrics": {
            "sol_balance": sol_balance,
            "recent_tx_count": tx_count,
            "account_age_days": days_old,
            "unique_programs_sampled": program_diversity,
            "defi_program_hits": [KNOWN_DEFI_PROGRAMS.get(p, p) for p in sorted(defi_program_hits)],
        },
    }


@app.route("/", methods=["GET", "POST"])
def index():
    result: Dict[str, Any] | None = None
    error: str | None = None

    if request.method == "POST":
        address = request.form.get("address", "").strip()
        if not address:
            error = "Please enter a Solana wallet address."
        else:
            result = grade_wallet(address)
            error = result.get("error") if isinstance(result, dict) and result.get("error") else None

    return render_template("index.html", result=result, error=error)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=True)


