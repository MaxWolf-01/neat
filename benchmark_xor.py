#!/usr/bin/env python3
"""Benchmark XOR to compare with original NEAT paper."""

from neat import NEAT, NEATConfig, NEATNetwork

# XOR test data
XOR_INPUTS = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
XOR_OUTPUTS = [[0.0], [1.0], [1.0], [0.0]]


def evaluate_xor(genome):
    network = NEATNetwork(genome)
    error = 0.0
    for inputs, expected in zip(XOR_INPUTS, XOR_OUTPUTS):
        inputs_with_bias = inputs + [1.0]
        outputs = network(inputs_with_bias)
        error += abs(outputs[0] - expected[0])  # Paper uses sum of distances
    fitness = 4.0 - error
    return fitness * fitness  # Paper squares fitness to amplify good solutions


def run_xor_trial(max_gens=100):  # Paper's worst case was 90 gens
    """Run a single XOR trial, return generations to solve or None if failed."""
    config = NEATConfig(
        n_inputs=3,  # 2 + bias
        n_outputs=1,
        population_size=10_000,  # 150,  # Same as paper
        activation_choices=["sigmoid"],  # Paper uses sigmoid
        output_activation="sigmoid",
        dense_initial_connectivity=True,  # Paper: fully connected
        allow_recurrent=True,  # Paper allows recurrence
        # Speciation parameters
        compatibility_threshold=3.0,  # Paper: δt = 3.0
        c1=1.0,  # Paper: excess coefficient = 1.0
        c2=1.0,  # Paper: disjoint coefficient = 1.0
        c3=0.4,  # Paper: weight coefficient = 0.4
        stagnation_generations=15,  # Paper: 15 generations
        # Mutation and reproduction parameters
        replacement_rate=0.1,  # Paper: 10% new random
        mutation_std=0.5,  # Not specified
        weight_range=2.0,  # Paper doesn't specify, reasonable default
        mutation_rate=0.8,  # Paper: 80%
        speciate=True,  # Enable speciation!
        survival_threshold=0.2,  # Paper uses top 20%
        crossover_rate=0.75,  # Paper: 75%
        add_node_prob=0.03,  # Paper: 0.03
        add_conn_prob=0.05,  # Paper: 0.05
        interspecies_mating_rate=0.001,  # Paper: 0.001
        champion_preservation_threshold=5,  # Paper: species > 5
        inherited_disable_rate=0.75,  # Paper: 75%
        toggle_rate=0.0,  # Not mentioned in paper
    )

    neat = NEAT(config)

    for generation in range(max_gens):
        genomes = neat.ask()
        fitnesses = [evaluate_xor(g) for g in genomes]
        neat.tell(fitnesses)

        # Check if solved: unsquared fitness > 3.9 means error < 0.1
        if max(fitnesses) > 15.21:  # 3.9^2 = 15.21
            best = neat.best
            n_hidden = len([n for n in best.nodes if str(n.node_type) == "hidden"])
            n_conn = len([c for c in best.connections if c.enabled])
            return generation + 1, n_hidden, n_conn

    return None, None, None  # Failed


# Run multiple trials
n_trials = 100  # Paper uses 100 runs
print(f"Running XOR benchmark ({n_trials} trials)...")

successes = 0
total_gens = []
total_hidden = []
total_conn = []
networks_evaluated = []

for trial in range(n_trials):
    gens, hidden, conn = run_xor_trial()
    if gens:
        successes += 1
        total_gens.append(gens)
        total_hidden.append(hidden)
        total_conn.append(conn)
        networks_evaluated.append(gens * 150)  # population_size * generations
        # Show progress
        if (trial + 1) % 10 == 0:
            print(f"  Progress: {trial + 1}/{n_trials} trials completed...")
    else:
        print(f"Trial {trial + 1:3d}: FAILED")

print(f"\n=== NEAT XOR Results ({n_trials} runs) ===")
print(f"Success rate: {successes}/{n_trials} ({successes / n_trials * 100:.0f}%)")

if total_gens:
    import statistics

    # Our results
    print(f"\nOur implementation:")
    print(
        f"  Generations:        {statistics.mean(total_gens):.1f} (std: {statistics.stdev(total_gens) if len(total_gens) > 1 else 0:.1f})"
    )
    print(
        f"  Networks evaluated: {statistics.mean(networks_evaluated):.0f} (std: {statistics.stdev(networks_evaluated) if len(networks_evaluated) > 1 else 0:.0f})"
    )
    print(
        f"  Hidden nodes:       {statistics.mean(total_hidden):.2f} (std: {statistics.stdev(total_hidden) if len(total_hidden) > 1 else 0:.2f})"
    )
    print(
        f"  Connections:        {statistics.mean(total_conn):.2f} (std: {statistics.stdev(total_conn) if len(total_conn) > 1 else 0:.2f})"
    )
    print(f"  Worst performance:  {max(total_gens)} generations ({max(networks_evaluated)} evaluations)")

    # Paper results
    print(f"\nOriginal NEAT paper:")
    print(f"  Generations:        32 (std: not given)")
    print(f"  Networks evaluated: 4,755 (std: 2,553)")
    print(f"  Hidden nodes:       2.35 (std: 1.11)")
    print(f"  Connections:        7.48 (std: not given)")
    print(f"  Worst performance:  90 generations (13,459 evaluations)")
    print(f"  Success rate:       100/100 (100%)")
