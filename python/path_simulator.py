from __future__ import annotations

"""Utility functions that evaluate paths on simplified AMM models."""

from typing import Dict, List, Tuple, Any, Union

from python.token_utils import normalize_amount


def build_market_map(markets: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Cache markets by address so lookups during simulation stay O(1)."""
    return {m.get("marketAddress"): m for m in markets}


Hop = Union[Dict[str, Any], Any]


def _get(hop: Hop, key: str) -> Any:
    """Support both dict and dataclass representations of a hop."""
    if isinstance(hop, dict):
        return hop.get(key)
    return getattr(hop, key)


def simulate_path(
    hops: List[Hop],
    amount: float,
    market_map: Dict[str, Dict[str, Any]],
) -> Tuple[float, float]:
    """Return (output_amount, slippage_estimate) for a sequence of hops."""
    current_amount = amount
    total_slippage = 0.0
    for hop in hops:
        pool_address = _get(hop, "poolAddress")
        from_token = _get(hop, "fromToken")
        to_token = _get(hop, "toToken")
        market = market_map.get(pool_address)
        if not market:
            current_amount *= 0.95  # Penalise missing pools to avoid crashes.
            total_slippage += amount * 0.05
            continue
        reserves = market.get("reserves") or {}
        from_reserve = normalize_amount(from_token, reserves.get(from_token))
        to_reserve = normalize_amount(to_token, reserves.get(to_token))
        fee = market.get("feeBps", 30) / 10000
        amount_in = current_amount * (1 - fee)
        if from_reserve > 0 and to_reserve > 0:
            output = (amount_in * to_reserve) / (from_reserve + amount_in)
            expected_price = to_reserve / from_reserve
            total_slippage += max(0.0, expected_price * current_amount - output)
            current_amount = output
        else:
            current_amount = amount_in * 0.99
            total_slippage += current_amount * 0.01
    return current_amount, total_slippage
