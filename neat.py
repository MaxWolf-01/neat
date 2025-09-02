from __future__ import annotations

import bisect
import copy
import math
import random
from collections import deque
from dataclasses import dataclass
from enum import StrEnum, auto
from itertools import accumulate
from typing import Callable


class NodeType(StrEnum):
    INPUT = auto()
    OUTPUT = auto()
    HIDDEN = auto()


@dataclass(frozen=True)
class NodeGene:
    id: int
    node_type: NodeType
    activation: str


@dataclass
class ConnectionGene:
    innovation: int
    in_node: int
    out_node: int
    weight: float
    enabled: bool


@dataclass
class Genome:
    species_id: int
    nodes: list[NodeGene]
    connections: list[ConnectionGene]
    fitness: float
    adjusted_fitness: float


@dataclass
class Species:
    id: int
    representative: Genome
    members: list[Genome]
    generations_without_improvement: int = 0
    best_fitness_ever: float = -float("inf")


@dataclass
class NEATConfig:
    n_inputs: int  # +1 for bias node
    n_outputs: int
    population_size: int
    activation_choices: list[str]
    output_activation: str  # activation for output nodes
    dense_initial_connectivity: bool
    allow_recurrent: bool  # whether to allow recurrent connections

    # Speciation parameters
    compatibility_threshold: float  # δt - distance threshold for species
    c1: float  # excess gene coefficient
    c2: float  # disjoint gene coefficient
    c3: float  # weight difference coefficient
    stagnation_generations: int  # remove species not improving for this many gens

    # mutation and reproduction parameters
    replacement_rate: float  # if mutating, probability of replacing a weight with a new random value vs adding noise
    mutation_std: float  # standard deviation of weight perturbation
    weight_range: float
    mutation_rate: float  # probability of mutating a weight
    speciate: bool  # whether to group genomes into species based on similarity
    survival_threshold: float  # fraction of species allowed to reproduce
    crossover_rate: float
    add_node_prob: float
    add_conn_prob: float
    interspecies_mating_rate: float
    champion_preservation_threshold: int  # species size above which to preserve champion
    inherited_disable_rate: float  # probability of disabling a gene disabled in either parent
    toggle_rate: float  # probability of toggling the enabled state of a connection

    # optional global stagnation cull (paper footnote). If not None and all species are stagnant for this many generations (after a starting gen), keep only the top-K species
    global_stagnation_generations: int | None = None
    global_cull_keep_top_k: int = 2
    global_cull_start_gen: int = 20


class NEAT:
    def __init__(self, config: NEATConfig):
        self.config = config

        self.generation: int = 0
        self.population: list[Genome] = []
        self.species: list[Species] = []  # Track species across generations
        self.next_species_id: int = 0

        self.innovations: dict[tuple[int, int], int] = {}
        self.next_innovations: int = 0

    @property
    def best(self) -> Genome:
        return max(self.population, key=lambda g: g.fitness)

    def ask(self) -> list[Genome]:
        if not self.population:
            self.population = [self._create_initial_genome() for _ in range(self.config.population_size)]
        return self.population

    def tell(self, fitnesses: list[float]) -> None:
        for genome, fitness in zip(self.population, fitnesses, strict=True):
            genome.fitness = fitness
        species = species = self._assign_species() if self.config.speciate else [self.population]
        self.population = self._reproduce(species)
        self.generation += 1

    def _create_initial_genome(self) -> Genome:
        nodes = []
        for i in range(self.config.n_inputs):
            nodes.append(NodeGene(i, NodeType.INPUT, "identity"))
        for i in range(self.config.n_outputs):
            nodes.append(
                NodeGene(
                    self.config.n_inputs + i,
                    NodeType.OUTPUT,
                    self.config.output_activation,
                )
            )
        connections = []
        if self.config.dense_initial_connectivity:
            for i in range(self.config.n_inputs):
                for j in range(self.config.n_outputs):
                    connections.append(
                        ConnectionGene(
                            innovation=self._get_or_create_innovations(i, self.config.n_inputs + j),
                            in_node=i,
                            out_node=self.config.n_inputs + j,
                            weight=random.uniform(-self.config.weight_range, self.config.weight_range),
                            enabled=True,
                        )
                    )
        return Genome(
            species_id=0,
            nodes=nodes,
            connections=connections,
            fitness=0.0,
            adjusted_fitness=0.0,
        )

    def _get_or_create_innovations(self, in_node: int, out_node: int) -> int:
        key = (in_node, out_node)
        if key not in self.innovations:
            self.innovations[key] = self.next_innovations
            self.next_innovations += 1
        return self.innovations[key]

    def _compatibility_distance(self, g1: Genome, g2: Genome) -> float:
        if not g1.connections and not g2.connections:
            return 0.0

        i, j = 0, 0
        matching, disjoint = 0, 0
        weight_diff_sum = 0.0
        while i < len(g1.connections) and j < len(g2.connections):
            conn1, conn2 = g1.connections[i], g2.connections[j]
            if conn1.innovation == conn2.innovation:
                matching += 1
                weight_diff_sum += abs(conn1.weight - conn2.weight)
                i += 1
                j += 1
            elif conn1.innovation < conn2.innovation:
                disjoint += 1
                i += 1
            else:
                disjoint += 1
                j += 1
        excess = len(g1.connections) - i + len(g2.connections) - j

        avg_weight_diff = weight_diff_sum / matching if matching > 0 else 0

        # normalize by larger genome size if > 20 genes
        n_genes = max(len(g1.connections), len(g2.connections))
        n = n_genes if n_genes > 20 else 1
        return self.config.c1 * excess / n + self.config.c2 * disjoint / n + self.config.c3 * avg_weight_diff

    def _assign_species(self) -> list[list[Genome]]:
        # assign each genome to a species based on similarity
        if not self.config.speciate:
            species = [self.population]
        else:
            for s in self.species:
                s.members = []

            # assign each genome to first compatible species
            for genome in self.population:
                placed = False
                for s in self.species:
                    if self._compatibility_distance(genome, s.representative) < self.config.compatibility_threshold:
                        s.members.append(genome)
                        genome.species_id = s.id
                        placed = True
                        break

                # create new species if no match
                if not placed:
                    new_species = Species(
                        id=self.next_species_id,
                        representative=genome,
                        members=[genome],
                    )
                    self.next_species_id += 1
                    self.species.append(new_species)
                    genome.species_id = new_species.id

            # remove empty species
            self.species = [s for s in self.species if s.members]

            # update stagnation
            for s in self.species:
                current_best = max(g.fitness for g in s.members)
                if current_best > s.best_fitness_ever:
                    s.best_fitness_ever = current_best
                    s.generations_without_improvement = 0
                else:
                    s.generations_without_improvement += 1

            # remove stagnant species (but keep at least 1 to continue evolution)
            non_stagnant = [s for s in self.species if s.generations_without_improvement < self.config.stagnation_generations]
            self.species = non_stagnant if non_stagnant else [max(self.species, key=lambda s: s.best_fitness_ever)]

            if (
                self.config.global_stagnation_generations is not None
                and self.generation > self.config.global_cull_start_gen
                and self.species
                and all(s.generations_without_improvement > self.config.global_stagnation_generations for s in self.species)
            ):
                self.species.sort(key=lambda s: s.best_fitness_ever, reverse=True)
                self.species = self.species[: max(1, self.config.global_cull_keep_top_k)]

            # set new representative for each species
            for s in self.species:
                if s.members:
                    s.representative = random.choice(s.members)

            species = [s.members for s in self.species]

        # adjust fitness by species size (fitness sharing)
        for s in species:
            for g in s:
                g.adjusted_fitness = g.fitness / len(s)
        return species

    def _reproduce(self, species: list[list[Genome]]) -> list[Genome]:
        offspring_counts = self._get_offspring_counts(species)
        new_population = []
        for s, count in zip(species, offspring_counts, strict=True):
            if count == 0:
                continue
            s.sort(key=lambda g: g.fitness, reverse=True)

            if len(s) > self.config.champion_preservation_threshold and count > 0:
                new_population.append(copy.deepcopy(s[0]))
                count -= 1

            survivors = s[: max(1, int(len(s) * self.config.survival_threshold))]  # Keep at least 1

            min_fitness = min(g.fitness for g in survivors)  # weights must be positive
            weights = [g.fitness - min_fitness + 1e-8 for g in survivors]
            pick_parent = lambda: random.choices(range(len(survivors)), weights=weights, k=1)[0]
            for _ in range(count):
                parent1_idx = pick_parent()
                parent1 = survivors[parent1_idx]
                if random.random() < self.config.crossover_rate and len(survivors) > 1:
                    if random.random() < self.config.interspecies_mating_rate and len(species) > 1:
                        all_others = [g for sp in species if sp != s for g in sp]
                        min_other = min(g.fitness for g in all_others)
                        other_weights = [g.fitness - min_other + 1e-8 for g in all_others]
                        parent2 = random.choices(all_others, weights=other_weights, k=1)[0]
                    else:
                        parent2_idx = pick_parent()
                        parent2_idx = (parent2_idx + 1) % len(survivors) if parent2_idx == parent1_idx else parent2_idx
                        parent2 = survivors[parent2_idx]
                    child = self._crossover(parent1, parent2)
                else:
                    child = copy.deepcopy(parent1)
                self._mutate(child)
                new_population.append(child)
        if len(new_population) != self.config.population_size:
            print(f"Population mismatch: got {len(new_population)}, expected {self.config.population_size}")
            print(f"Species: {len(species)}, offspring_counts: {offspring_counts}")
            for i, (s, count) in enumerate(zip(species, offspring_counts)):
                print(f"  Species {i}: {len(s)} members, {count} offspring requested")
        assert len(new_population) == self.config.population_size
        return new_population

    def _get_offspring_counts(self, species: list[list[Genome]]) -> list[int]:
        # mean raw fitness with sharing by species size
        scores = [sum(g.adjusted_fitness for g in s) for s in species]
        n_species = len(scores)
        if n_species == 0:
            return []

        # make scores non-negative via global min-shift
        min_score = min(scores)
        weights = [(s - min_score) if (s - min_score) > 0 else 0.0 for s in scores]
        total = sum(weights)

        if total <= 1e-12:
            # no signal: distribute equally
            base, remainder = divmod(self.config.population_size, n_species)
            counts = [base] * n_species
            for i in range(remainder):
                counts[i] += 1
            return counts

        # Stochastic Universal Sampling (SUS) for proportional allocation
        # could also use multinomial sampling, but SUS has no bias, lower variance => less risk of accidental extinction
        step = total / self.config.population_size
        start = random.uniform(0.0, step)
        cumulative = list(accumulate(weights))
        counts = [0] * n_species
        idx = 0
        for k in range(self.config.population_size):
            pointer = start + k * step
            while idx < n_species - 1 and cumulative[idx] < pointer:
                idx += 1
            counts[idx] += 1
        return counts

    def _crossover(self, parent1: Genome, parent2: Genome) -> Genome:
        fittest_parent = parent1 if parent1.fitness >= parent2.fitness else parent2
        equal_fitness = parent1.fitness == parent2.fitness
        i, j = 0, 0
        offspring_connections = []
        while i < len(parent1.connections) and j < len(parent2.connections):
            conn1 = parent1.connections[i]
            conn2 = parent2.connections[j]
            if conn1.innovation == conn2.innovation:
                # matching gene, inherit randomly
                inherited_conn = copy.deepcopy(random.choice([conn1, conn2]))
                inherited_conn.enabled = (
                    self.config.inherited_disable_rate < random.random() if not (conn1.enabled and conn2.enabled) else True
                )
                offspring_connections.append(inherited_conn)
                i += 1
                j += 1
            elif conn1.innovation < conn2.innovation:
                # disjoint gene from parent1
                if fittest_parent == parent1 or equal_fitness:
                    offspring_connections.append(copy.deepcopy(conn1))
                i += 1
            elif conn2.innovation < conn1.innovation:
                # disjoint gene from parent2
                if fittest_parent == parent2 or equal_fitness:
                    offspring_connections.append(copy.deepcopy(conn2))
                j += 1

        # take excess genes from the fitter parent
        if i < len(parent1.connections) and (fittest_parent == parent1 or equal_fitness):
            offspring_connections.extend(copy.deepcopy(conn) for conn in parent1.connections[i:])
        elif j < len(parent2.connections) and (fittest_parent == parent2 or equal_fitness):
            offspring_connections.extend(copy.deepcopy(conn) for conn in parent2.connections[j:])

        # always include input and output nodes
        offspring_node_ids = set()
        for node in parent1.nodes:
            if node.node_type in (NodeType.INPUT, NodeType.OUTPUT):
                offspring_node_ids.add(node.id)
        # add any nodes referenced by connections
        for conn in offspring_connections:
            offspring_node_ids.add(conn.in_node)
            offspring_node_ids.add(conn.out_node)

        return Genome(
            species_id=fittest_parent.species_id,
            nodes=list(set(n for n in parent1.nodes + parent2.nodes if n.id in offspring_node_ids)),
            connections=offspring_connections,
            fitness=0.0,
            adjusted_fitness=0.0,
        )

    def _mutate(self, genome: Genome) -> None:
        if random.random() < self.config.add_node_prob:
            self._add_node_mutation(genome)
        if random.random() < self.config.add_conn_prob:
            self._add_connection_mutation(genome)
        self._mutate_connections(genome)

    def _add_node_mutation(self, genome: Genome) -> bool:
        enabled_connections = [c for c in genome.connections if c.enabled]
        if not enabled_connections:
            return False
        connection = random.choice(enabled_connections)
        connection.enabled = False

        new_node = NodeGene(
            id=max(n.id for n in genome.nodes) + 1,
            node_type=NodeType.HIDDEN,
            activation=random.choice(self.config.activation_choices),
        )
        genome.nodes.append(new_node)

        genome.connections.extend(
            [
                ConnectionGene(
                    innovation=self._get_or_create_innovations(connection.in_node, new_node.id),
                    in_node=connection.in_node,
                    out_node=new_node.id,
                    weight=1.0,
                    enabled=True,
                ),
                ConnectionGene(
                    innovation=self._get_or_create_innovations(new_node.id, connection.out_node),
                    in_node=new_node.id,
                    out_node=connection.out_node,
                    weight=connection.weight,
                    enabled=True,
                ),
            ]
        )
        return True

    def _add_connection_mutation(self, genome: Genome) -> bool:
        # add a new connection between two unconnected nodes."""
        existing = {(c.in_node, c.out_node) for c in genome.connections}
        sources = [n for n in genome.nodes if n.node_type != NodeType.OUTPUT]
        targets = [n for n in genome.nodes if n.node_type not in (NodeType.INPUT)]

        valid_pairs = [
            (source.id, target.id)
            for source in sources
            for target in targets
            if (source.id, target.id) not in existing
            and (self.config.allow_recurrent or not self._would_create_cycle(genome, source.id, target.id))
        ]

        if not valid_pairs:
            return False

        in_node, out_node = random.choice(valid_pairs)
        innovation = self._get_or_create_innovations(in_node, out_node)
        weight = random.uniform(-self.config.weight_range, self.config.weight_range)

        # insert new connection in sorted order by innovation number
        new_connection = ConnectionGene(
            innovation=innovation,
            in_node=in_node,
            out_node=out_node,
            weight=weight,
            enabled=True,
        )
        bisect.insort(genome.connections, new_connection, key=lambda c: c.innovation)
        return True

    def _would_create_cycle(self, genome: Genome, from_node: int, to_node: int) -> bool:
        # DFS from to_node to from_node to see if we can reach from_node, which would create a cycle
        adjacency = {n.id: [] for n in genome.nodes}
        for conn in genome.connections:
            if conn.enabled:
                adjacency[conn.in_node].append(conn.out_node)

        visited = set()
        stack = [to_node]
        while stack:
            node = stack.pop()
            if node == from_node:
                return True
            if node not in visited:
                visited.add(node)
                stack.extend(adjacency.get(node, []))
        return False

    def _mutate_connections(self, genome: Genome) -> None:
        for connection in genome.connections:
            if random.random() > self.config.mutation_rate:
                continue
            if random.random() < self.config.replacement_rate:
                connection.weight = random.uniform(-self.config.weight_range, self.config.weight_range)
            else:
                connection.weight += random.gauss(0, self.config.mutation_std)
            if random.random() < self.config.toggle_rate:
                if not connection.enabled:
                    if self.config.allow_recurrent or not self._would_create_cycle(
                        genome, connection.in_node, connection.out_node
                    ):
                        connection.enabled = True
                else:
                    connection.enabled = False


class NEATNetwork:
    activation_map: dict[str, Callable] = {
        "identity": lambda x: x,
        # original paper uses 4.9 # clamp to prevent overflow
        "sigmoid": lambda x: 1 / (1 + math.exp(-4.9 * max(-20, min(20, x)))),
        "tanh": math.tanh,
        "sin": math.sin,
        "cos": math.cos,
        "gaussian": lambda x: math.exp(-x * x),
        "relu": lambda x: max(0, x),
    }

    def __init__(self, genome: Genome):
        self.genome = genome

        self.incoming = {n.id: [] for n in genome.nodes}
        for conn in genome.connections:
            if conn.enabled:
                self.incoming[conn.out_node].append((conn.in_node, conn.weight))

        self._topo_order = self._topological_sort()
        self._forward = self._feedforward if self._topo_order is not None else self._recurrent
        # for recurrent nets
        self.state = {n.id: 0.0 for n in genome.nodes} if self._topo_order is None else None

        self.node_activations = {n.id: self.activation_map[n.activation] for n in genome.nodes}
        self.input_nodes = [n.id for n in genome.nodes if n.node_type == NodeType.INPUT]
        self.output_nodes = [n.id for n in genome.nodes if n.node_type == NodeType.OUTPUT]

    def _topological_sort(self) -> list[int] | None:
        in_degree = {node_id: len(edges) for node_id, edges in self.incoming.items()}
        outgoing = {node_id: [] for node_id in self.incoming}
        for out_node, edges in self.incoming.items():
            for in_node, _ in edges:
                outgoing[in_node].append(out_node)
        queue = deque([n for n, d in in_degree.items() if d == 0])
        order = []
        while queue:
            node = queue.popleft()
            order.append(node)
            for neighbour in outgoing[node]:
                in_degree[neighbour] -= 1
                if in_degree[neighbour] == 0:
                    queue.append(neighbour)
        if len(order) != len(self.incoming):
            return None
        return order

    def reset_state(self) -> None:
        if self.state is not None:
            for node_id in self.state:
                if node_id not in self.input_nodes:
                    self.state[node_id] = 0.0

    def _recurrent(self, inputs: list[float], depth: int = 3) -> list[float]:
        assert self.state is not None
        for i, node_id in enumerate(self.input_nodes):
            self.state[node_id] = inputs[i]
        for _ in range(depth):
            new_state = self.state.copy()
            for node_id in (n for n in self.incoming if n not in self.input_nodes):
                new_state[node_id] = self.node_activations[node_id](
                    sum(self.state[in_node] * w for in_node, w in self.incoming[node_id])
                )
            self.state = new_state
        return [self.state[n] for n in self.output_nodes]

    def _feedforward(self, inputs: list[float]) -> list[float]:
        values: dict[int, float] = {n: value for n, value in zip(self.input_nodes, inputs)}
        for node in (n for n in self._topo_order if n not in self.input_nodes):  # type: ignore # topo_order never None
            values[node] = self.node_activations[node](sum(values[n] * w for n, w in self.incoming[node]))
        return [values[n] for n in self.output_nodes]

    def __call__(self, inputs: list[float], **kwargs) -> list[float]:
        return self._forward(inputs, **kwargs)
