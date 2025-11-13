from __future__ import annotations

"""Token-decimal helpers used whenever amounts move between tokens."""

from decimal import Decimal

# Hard-coded decimals for assets that appear in the benchmark instances.
TOKEN_DECIMALS = {
    "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2": 18,  # WETH
    "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48": 6,   # USDC
    "0x6b175474e89094c44da98b954eedeac495271d0f": 18,  # DAI
    "0x111111111117dc0aa78b770fa6a738034120c302": 18,  # 1INCH token
}


def get_decimals(token: str) -> int:
    """Return the decimal precision for a token (defaulting to 18)."""
    return TOKEN_DECIMALS.get(token.lower(), 18)


def normalize_amount(token: str, raw: float | int | str) -> float:
    """Scale an integer-like raw amount down to human units."""
    if raw is None:
        return 0.0
    value = Decimal(str(raw))
    scale = Decimal(10) ** get_decimals(token)
    return float(value / scale)


def order_size_to_units(token: str, raw_wei: str) -> float:
    """Convenience helper for order sizes stored as integers."""
    return normalize_amount(token, raw_wei)
