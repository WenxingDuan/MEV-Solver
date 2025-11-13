from __future__ import annotations

"""Deterministic dual-decomposition baseline for arbitrage-style routing."""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Any

from python.token_utils import normalize_amount


@dataclass
class Market:
    marketAddress: str
    protocol: str
    tokens: List[str]
    feeBps: int
    reserves: Optional[Dict[str, str]] = None


@dataclass
class TradingEdge:
    fromToken: str
    toToken: str
    market: Market
    weight: float
    rate: float
    fee: float


@dataclass
class ArbitragePath:
    tokens: List[str]
    markets: List[Market]
    expectedProfit: float
    weight: float


@dataclass
class DualOptimizerConfig:
    maxIterations: int = 100
    maxPathLength: int = 4
    minProfitThreshold: float = 0.001


class DualDecompositionOptimizer:
    """Detects profitable walks using a Bellman-Ford style relaxation."""
    def __init__(self, config: Optional[DualOptimizerConfig] = None) -> None:
        self.config = config or DualOptimizerConfig()
        self.graph: Dict[str, List[TradingEdge]] = {}

    def find_profitable_paths(
        self,
        markets: Sequence[Dict[str, Any]],
        base_token: str,
        order_size: float,
    ) -> List[ArbitragePath]:
        """Return arbitrage paths whose profit clears the configured threshold."""
        self._build_graph(markets)
        paths = self._find_negative_cycles(base_token)
        return [
            path
            for path in paths
            if path.expectedProfit > self.config.minProfitThreshold
        ]

    def get_warm_start_seeds(
        self,
        markets: Sequence[Dict[str, Any]],
        base_token: str,
        count: int = 5,
    ) -> List[ArbitragePath]:
        """Expose the top-N profitable paths so the GA can be seeded."""
        paths = self.find_profitable_paths(markets, base_token, 1.0)
        paths.sort(key=lambda p: p.expectedProfit, reverse=True)
        return paths[:count]

    # Internal helpers -------------------------------------------------

    def _build_graph(self, markets: Sequence[Dict[str, Any]]) -> None:
        """Convert pool data into a directed graph with -log weights."""
        self.graph.clear()
        for market_dict in markets:
            market = Market(
                marketAddress=market_dict.get("marketAddress", ""),
                protocol=market_dict.get("protocol", ""),
                tokens=list(market_dict.get("tokens", [])),
                feeBps=int(market_dict.get("feeBps", 0)),
                reserves=market_dict.get("reserves"),
            )
            tokens = market.tokens
            for i, from_token in enumerate(tokens):
                for j, to_token in enumerate(tokens):
                    if i == j:
                        continue
                    rate = self._calculate_rate(market, from_token, to_token)
                    fee = market.feeBps / 10000
                    weight = -math.log(max(rate * (1 - fee), 1e-12))
                    edge = TradingEdge(
                        fromToken=from_token,
                        toToken=to_token,
                        market=market,
                        weight=weight,
                        rate=rate,
                        fee=fee,
                    )
                    self.graph.setdefault(from_token, []).append(edge)

    def _calculate_rate(self, market: Market, from_token: str, to_token: str) -> float:
        """Derive a rough exchange rate from pool reserves."""
        reserves = market.reserves or {}
        from_reserve = normalize_amount(from_token, reserves.get(from_token, "0"))
        to_reserve = normalize_amount(to_token, reserves.get(to_token, "0"))
        if from_reserve > 0 and to_reserve > 0:
            return to_reserve / from_reserve
        return 0.98 + 0.02

    def _find_negative_cycles(self, start_token: str) -> List[ArbitragePath]:
        """Run Bellman-Ford and collect negative cycles / profitable walks."""
        tokens = list(self.graph.keys())
        distance: Dict[str, float] = {token: math.inf for token in tokens}
        predecessor: Dict[str, Optional[TradingEdge]] = {token: None for token in tokens}
        if start_token in distance:
            distance[start_token] = 0.0
        else:
            return []

        for _ in range(len(tokens) - 1):
            updated = False
            for from_token, edges in self.graph.items():
                from_dist = distance.get(from_token, math.inf)
                if math.isinf(from_dist):
                    continue
                for edge in edges:
                    new_dist = from_dist + edge.weight
                    if new_dist < distance.get(edge.toToken, math.inf):
                        distance[edge.toToken] = new_dist
                        predecessor[edge.toToken] = edge
                        updated = True
            if not updated:
                break

        paths: List[ArbitragePath] = []
        for from_token, edges in self.graph.items():
            from_dist = distance.get(from_token, math.inf)
            if math.isinf(from_dist):
                continue
            for edge in edges:
                to_dist = distance.get(edge.toToken, math.inf)
                if from_dist + edge.weight < to_dist:
                    cycle = self._extract_cycle(edge, predecessor)
                    if cycle and cycle.tokens and cycle.tokens[0] == start_token:
                        paths.append(cycle)
        paths.extend(self._find_simple_paths(start_token, distance, predecessor))
        return paths

    def _extract_cycle(
        self,
        start_edge: TradingEdge,
        predecessor: Dict[str, Optional[TradingEdge]],
    ) -> Optional[ArbitragePath]:
        """Walk predecessor pointers to recover an explicit arbitrage cycle."""
        visited: set[str] = set()
        path_edges: List[TradingEdge] = []
        current: Optional[TradingEdge] = start_edge
        while current and current.fromToken not in visited:
            visited.add(current.fromToken)
            path_edges.append(current)
            current = predecessor.get(current.fromToken)
            if len(path_edges) > self.config.maxPathLength:
                return None
        if not path_edges:
            return None
        tokens: List[str] = []
        markets: List[Market] = []
        total_weight = 0.0
        for edge in reversed(path_edges):
            if not tokens:
                tokens.append(edge.fromToken)
            tokens.append(edge.toToken)
            markets.append(edge.market)
            total_weight += edge.weight
        expected_profit = math.exp(-total_weight) - 1
        return ArbitragePath(tokens=tokens, markets=markets, expectedProfit=expected_profit, weight=total_weight)

    def _find_simple_paths(
        self,
        start_token: str,
        distance: Dict[str, float],
        predecessor: Dict[str, Optional[TradingEdge]],
    ) -> List[ArbitragePath]:
        """Record profitable simple paths even if they are not full cycles."""
        paths: List[ArbitragePath] = []
        for end_token, dist in distance.items():
            if end_token == start_token or math.isinf(dist):
                continue
            path_edges: List[TradingEdge] = []
            current_token = end_token
            edge = predecessor.get(current_token)
            while edge and len(path_edges) < self.config.maxPathLength:
                path_edges.append(edge)
                current_token = edge.fromToken
                edge = predecessor.get(current_token)
                if current_token == start_token:
                    break
            if current_token != start_token or not path_edges:
                continue
            tokens = [start_token]
            markets: List[Market] = []
            total_weight = 0.0
            for e in reversed(path_edges):
                tokens.append(e.toToken)
                markets.append(e.market)
                total_weight += e.weight
            expected_profit = 1 - math.exp(total_weight)
            if expected_profit > self.config.minProfitThreshold:
                paths.append(
                    ArbitragePath(
                        tokens=tokens,
                        markets=markets,
                        expectedProfit=expected_profit,
                        weight=total_weight,
                    )
                )
        return paths
