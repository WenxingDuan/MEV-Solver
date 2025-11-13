from __future__ import annotations

"""NSGA-II based multi-objective router used for the GA experiments."""

import math
import time
from dataclasses import dataclass, asdict
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Any

from python.path_simulator import build_market_map, simulate_path


RandomFn = Callable[[], float]


@dataclass
class RouteHop:
    fromToken: str
    toToken: str
    poolAddress: str
    protocol: str


@dataclass
class FitnessVector:
    surplus: float
    gasUnits: float
    slippage: float


@dataclass
class Chromosome:
    id: str
    paths: List[List[RouteHop]]
    splitRatios: List[float]
    fitness: FitnessVector
    rank: Optional[int] = None
    crowdingDistance: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "paths": [
                [asdict(hop) for hop in path]
                for path in self.paths
            ],
            "splitRatios": self.splitRatios,
            "fitness": asdict(self.fitness),
            "rank": self.rank,
            "crowdingDistance": self.crowdingDistance,
        }


@dataclass
class GAConfig:
    populationSize: int
    maxGenerations: int
    crossoverRate: float
    mutationRate: float
    eliteCount: int
    maxPaths: int
    maxPathLength: int
    timeBudgetMs: int


def _default_random() -> float:
    import random
    return random.random()


class GeneticRouterEngine:
    """Minimal NSGA-II loop tailored to the Alpha-Router chromosome encoding."""

    def __init__(self, config: GAConfig, rand_fn: RandomFn = _default_random) -> None:
        self.config = config
        self._rand = rand_fn
        self.population: List[Chromosome] = []
        self.generation = 0
        self.start_time_ms = 0.0

    def optimize(
        self,
        start_token: str,
        end_token: str,
        amount: float,
        markets: Sequence[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Run the evolutionary loop until the generation or time budget expires."""
        self.start_time_ms = self._now_ms()
        self.generation = 0

        self._initialize_population(start_token, end_token, markets)

        while (
            self.generation < self.config.maxGenerations
            and self._now_ms() - self.start_time_ms < self.config.timeBudgetMs
        ):
            self._evaluate_fitness(amount, markets)
            self._nondominated_sort()
            self._calculate_crowding_distance()
            offspring = self._reproduce()
            self.population = self._environmental_selection(
                [*self.population, *offspring]
            )
            self.generation += 1

        self._evaluate_fitness(amount, markets)
        self._nondominated_sort()

        pareto_front = [c for c in self.population if c.rank == 1]
        best = self._select_best(pareto_front)

        return {
            "best": best,
            "paretoFront": pareto_front,
            "generations": self.generation,
        }

    # Internal helpers -----------------------------------------------------

    def _now_ms(self) -> float:
        """Return the current wall-clock time in milliseconds."""
        return time.time() * 1000.0

    def _initialize_population(
        self, start_token: str, end_token: str, markets: Sequence[Dict[str, Any]]
    ) -> None:
        """Seed the population with random paths and split ratios."""
        self.population = []
        for i in range(self.config.populationSize):
            num_paths = 1 + int(self._rand() * self.config.maxPaths)
            paths: List[List[RouteHop]] = []
            for _ in range(num_paths):
                path = self._generate_random_path(start_token, end_token, markets)
                if path:
                    paths.append(path)
            splits = self._generate_split_ratios(len(paths))
            chromosome = Chromosome(
                id=f"chr_{i}",
                paths=paths,
                splitRatios=splits,
                fitness=FitnessVector(0.0, 0.0, 0.0),
            )
            self.population.append(chromosome)

    def _generate_random_path(
        self, start_token: str, end_token: str, markets: Sequence[Dict[str, Any]]
    ) -> List[RouteHop]:
        """Attempt to build a path via random walk with a BFS fallback."""
        path = self._random_walk_path(start_token, end_token, markets)
        if path:
            return path
        return self._bfs_fallback_path(start_token, end_token, markets)

    def _random_walk_path(
        self, start_token: str, end_token: str, markets: Sequence[Dict[str, Any]], attempts: int = 5
    ) -> List[RouteHop]:
        """Try several stochastic walks to encourage path diversity."""
        path: List[RouteHop] = []
        for _ in range(attempts):
            path.clear()
            current = start_token
            visited = set()
            while (
                current != end_token
                and len(path) < self.config.maxPathLength
                and current not in visited
            ):
                visited.add(current)
                available = [m for m in markets if current in m.get("tokens", [])]
                if not available:
                    break
                market = available[int(self._rand() * len(available))]
                next_token = next((t for t in market.get("tokens", []) if t != current), None)
                if next_token is None:
                    break
                path.append(
                    RouteHop(
                        fromToken=current,
                        toToken=next_token,
                        poolAddress=market.get("marketAddress", ""),
                        protocol=market.get("protocol", ""),
                    )
                )
                current = next_token
            if current == end_token:
                return list(path)
        return []

    def _bfs_fallback_path(
        self, start_token: str, end_token: str, markets: Sequence[Dict[str, Any]]
    ) -> List[RouteHop]:
        """Find the shortest path via BFS when random walk fails."""
        from collections import deque

        if start_token == end_token:
            return []

        adjacency: Dict[str, List[Tuple[str, Dict[str, Any]]]] = {}
        for market in markets:
            tokens = market.get("tokens", [])
            for token in tokens:
                for other in tokens:
                    if token == other:
                        continue
                    adjacency.setdefault(token, []).append((other, market))

        queue = deque([start_token])
        prev: Dict[str, Tuple[str, Dict[str, Any]]] = {}
        visited = set([start_token])

        while queue:
            token = queue.popleft()
            for neighbor, market in adjacency.get(token, []):
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                prev[neighbor] = (token, market)
                if neighbor == end_token:
                    queue.clear()
                    break
                if len(prev) > self.config.maxPathLength * len(markets):
                    break
                queue.append(neighbor)

        if end_token not in prev:
            return []

        hops: List[RouteHop] = []
        current = end_token
        while current != start_token:
            parent, market = prev[current]
            hops.append(
                RouteHop(
                    fromToken=parent,
                    toToken=current,
                    poolAddress=market.get("marketAddress", ""),
                    protocol=market.get("protocol", ""),
                )
            )
            current = parent
            if len(hops) > self.config.maxPathLength:
                return []
        hops.reverse()
        if hops and hops[-1].toToken == end_token:
            return hops
        return []

    def _generate_split_ratios(self, num_paths: int) -> List[float]:
        """Draw random split ratios and normalise them to sum to one."""
        if num_paths == 0:
            return []
        ratios = [self._rand() for _ in range(num_paths)]
        total = sum(ratios)
        if total == 0:
            return [1.0 / num_paths] * num_paths
        return [r / total for r in ratios]

    def _evaluate_fitness(self, amount: float, markets: Sequence[Dict[str, Any]]) -> None:
        """Replay each chromosome through the AMM simulator to score objectives."""
        market_map = build_market_map(list(markets))
        for chromosome in self.population:
            total_surplus = 0.0
            total_gas = 21000.0
            total_slippage = 0.0
            for idx, path in enumerate(chromosome.paths):
                if idx >= len(chromosome.splitRatios):
                    break
                path_amount = amount * chromosome.splitRatios[idx]
                output, slippage = simulate_path(path, path_amount, market_map)
                total_surplus += output
                total_slippage += slippage
                total_gas += 150000.0 * len(path)
            chromosome.fitness = FitnessVector(
                surplus=total_surplus,
                gasUnits=total_gas,
                slippage=total_slippage,
            )

    def _nondominated_sort(self) -> None:
        """Assign Pareto ranks following the NSGA-II domination procedure."""
        n = len(self.population)
        domination_count = [0] * n
        dominated: List[List[int]] = [[] for _ in range(n)]
        fronts: List[List[int]] = [[]]

        for i in range(n):
            for j in range(i + 1, n):
                dom = self._dominates(self.population[i], self.population[j])
                if dom == 1:
                    dominated[i].append(j)
                    domination_count[j] += 1
                elif dom == -1:
                    dominated[j].append(i)
                    domination_count[i] += 1

        for i in range(n):
            if domination_count[i] == 0:
                self.population[i].rank = 1
                fronts[0].append(i)

        current = 0
        while fronts[current]:
            next_front: List[int] = []
            for i in fronts[current]:
                for j in dominated[i]:
                    domination_count[j] -= 1
                    if domination_count[j] == 0:
                        self.population[j].rank = current + 2
                        next_front.append(j)
            current += 1
            fronts.append(next_front)

    def _dominates(self, a: Chromosome, b: Chromosome) -> int:
        """Return 1 if a dominates b, -1 if b dominates a, or 0 otherwise."""
        a_better = False
        b_better = False

        if a.fitness.surplus > b.fitness.surplus:
            a_better = True
        elif b.fitness.surplus > a.fitness.surplus:
            b_better = True

        if a.fitness.gasUnits < b.fitness.gasUnits:
            a_better = True
        elif b.fitness.gasUnits < a.fitness.gasUnits:
            b_better = True

        if a.fitness.slippage < b.fitness.slippage:
            a_better = True
        elif b.fitness.slippage < a.fitness.slippage:
            b_better = True

        if a_better and not b_better:
            return 1
        if b_better and not a_better:
            return -1
        return 0

    def _calculate_crowding_distance(self) -> None:
        """Compute NSGA-II crowding distances within each rank."""
        fronts: Dict[int, List[Chromosome]] = {}
        for chromosome in self.population:
            rank = chromosome.rank or 1
            fronts.setdefault(rank, []).append(chromosome)

        for front in fronts.values():
            if len(front) <= 2:
                for chromosome in front:
                    chromosome.crowdingDistance = math.inf
                continue

            for chromosome in front:
                chromosome.crowdingDistance = 0.0

            objectives = ["surplus", "gasUnits", "slippage"]
            for obj in objectives:
                front.sort(key=lambda c: getattr(c.fitness, obj))
                front[0].crowdingDistance = math.inf
                front[-1].crowdingDistance = math.inf
                obj_range = getattr(front[-1].fitness, obj) - getattr(front[0].fitness, obj)
                if obj_range > 0:
                    for i in range(1, len(front) - 1):
                        next_val = getattr(front[i + 1].fitness, obj)
                        prev_val = getattr(front[i - 1].fitness, obj)
                        front[i].crowdingDistance += (next_val - prev_val) / obj_range

    def _tournament_selection(self) -> Chromosome:
        """Select a parent via dominance-aware tournament sampling."""
        tournament_size = 3
        best: Optional[Chromosome] = None
        for _ in range(tournament_size):
            candidate = self.population[int(self._rand() * len(self.population))]
            if (
                best is None
                or (candidate.rank or math.inf) < (best.rank or math.inf)
                or (
                    candidate.rank == best.rank
                    and (candidate.crowdingDistance or -math.inf)
                    > (best.crowdingDistance or -math.inf)
                )
            ):
                best = candidate
        return best  # type: ignore[return-value]

    def _crossover(self, parent1: Chromosome, parent2: Chromosome) -> List[Chromosome]:
        """Swap tail segments of the parent path lists to create offspring."""
        if self._rand() > self.config.crossoverRate:
            return [self._clone_chromosome(parent1), self._clone_chromosome(parent2)]

        child1_paths = list(parent1.paths)
        child2_paths = list(parent2.paths)

        if len(child1_paths) > 1 and len(child2_paths) > 1:
            point = int(self._rand() * min(len(child1_paths), len(child2_paths)))
            temp = child1_paths[point:]
            child1_paths[point:] = child2_paths[point:]
            child2_paths[point:] = temp

        now_id = int(time.time() * 1000)
        return [
            Chromosome(
                id=f"chr_{now_id}_1",
                paths=child1_paths,
                splitRatios=self._generate_split_ratios(len(child1_paths)),
                fitness=FitnessVector(0.0, 0.0, 0.0),
            ),
            Chromosome(
                id=f"chr_{now_id}_2",
                paths=child2_paths,
                splitRatios=self._generate_split_ratios(len(child2_paths)),
                fitness=FitnessVector(0.0, 0.0, 0.0),
            ),
        ]

    def _mutate(self, chromosome: Chromosome) -> None:
        """Perturb split ratios to maintain population diversity."""
        if self._rand() >= self.config.mutationRate:
            return
        if self._rand() < 0.5:
            chromosome.splitRatios = self._generate_split_ratios(len(chromosome.splitRatios))
        if chromosome.paths and self._rand() < 0.5:
            idx = int(self._rand() * len(chromosome.paths))
            if idx < len(chromosome.splitRatios):
                adjustment = (self._rand() - 0.5) * 0.1
                updated = max(0.01, min(0.99, chromosome.splitRatios[idx] + adjustment))
                chromosome.splitRatios[idx] = updated
                total = sum(chromosome.splitRatios)
                chromosome.splitRatios = [r / total for r in chromosome.splitRatios]

    def _reproduce(self) -> List[Chromosome]:
        """Create a new generation’s worth of offspring."""
        offspring: List[Chromosome] = []
        while len(offspring) < self.config.populationSize:
            parent1 = self._tournament_selection()
            parent2 = self._tournament_selection()
            children = self._crossover(parent1, parent2)
            for child in children:
                self._mutate(child)
                offspring.append(child)
        return offspring[: self.config.populationSize]

    def _environmental_selection(self, combined: List[Chromosome]) -> List[Chromosome]:
        """Select survivors by rank first, then by crowding distance."""
        combined.sort(
            key=lambda c: ((c.rank or 0), -(c.crowdingDistance or 0.0))
        )
        return combined[: self.config.populationSize]

    def _select_best(self, pareto_front: List[Chromosome]) -> Chromosome:
        """Return the Pareto-1 chromosome with the highest surplus."""
        return max(pareto_front, key=lambda c: c.fitness.surplus)

    def _clone_chromosome(self, chromosome: Chromosome) -> Chromosome:
        """Make a deep copy of a chromosome for elitism/crossover."""
        cloned_paths = [list(path) for path in chromosome.paths]
        return Chromosome(
            id=f"chr_clone_{int(time.time() * 1000)}",
            paths=cloned_paths,
            splitRatios=list(chromosome.splitRatios),
            fitness=FitnessVector(
                surplus=chromosome.fitness.surplus,
                gasUnits=chromosome.fitness.gasUnits,
                slippage=chromosome.fitness.slippage,
            ),
            rank=chromosome.rank,
            crowdingDistance=chromosome.crowdingDistance,
        )
