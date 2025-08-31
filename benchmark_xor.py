#!/usr/bin/env python3
"""Benchmark XOR to compare with original NEAT paper."""

import matplotlib.animation as animation
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import networkx as nx

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


def run_xor_trial(max_gens=100, return_best=False, collect_history=False):  # Paper's worst case was 90 gens
    """Run a single XOR trial, return generations to solve or None if failed."""
    config = NEATConfig(
        n_inputs=3,  # 2 + bias
        n_outputs=1,
        population_size=150,  # 150,  # Same as paper
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

    history = [] if collect_history else None

    for generation in range(max_gens):
        genomes = neat.ask()
        fitnesses = [evaluate_xor(g) for g in genomes]
        neat.tell(fitnesses)

        # Collect history if requested
        if collect_history:
            import copy

            best_genome = copy.deepcopy(neat.best)
            history.append((generation, best_genome, max(fitnesses)))

        # Check if solved: unsquared fitness > 3.9 means error < 0.1
        if max(fitnesses) > 15.21:  # 3.9^2 = 15.21
            best = neat.best
            n_hidden = len([n for n in best.nodes if str(n.node_type) == "hidden"])
            n_conn = len([c for c in best.connections if c.enabled])
            if collect_history:
                return generation + 1, n_hidden, n_conn, history
            if return_best:
                return generation + 1, n_hidden, n_conn, best
            return generation + 1, n_hidden, n_conn

    if collect_history:
        return None, None, None, history if history else None
    if return_best:
        return None, None, None, None
    return None, None, None  # Failed


def visualize_network(genome, title="NEAT Network"):
    """Visualize a NEAT genome as a neural network graph."""
    fig, ax = plt.subplots(figsize=(12, 8))

    # Create a directed graph
    G = nx.DiGraph()

    # Group nodes by type
    input_nodes = []
    output_nodes = []
    hidden_nodes = []

    for node in genome.nodes:
        G.add_node(node.id)
        if str(node.node_type) == "input":
            input_nodes.append(node.id)
        elif str(node.node_type) == "output":
            output_nodes.append(node.id)
        else:
            hidden_nodes.append(node.id)

    # Add edges from enabled connections
    edge_colors = []
    edge_widths = []
    for conn in genome.connections:
        if conn.enabled:
            G.add_edge(conn.in_node, conn.out_node, weight=conn.weight)
            # Color edges based on weight (blue for positive, red for negative)
            if conn.weight > 0:
                edge_colors.append("blue")
            else:
                edge_colors.append("red")
            # Width based on absolute weight
            edge_widths.append(min(abs(conn.weight) * 2, 5))

    # Position nodes in layers
    pos = {}

    # Input layer (including bias as the third input)
    for i, node_id in enumerate(input_nodes):
        pos[node_id] = (0, (len(input_nodes) - 1 - i) * 2)

    # Output layer
    for i, node_id in enumerate(output_nodes):
        pos[node_id] = (4, (len(output_nodes) - 1 - i) * 2)

    # Hidden layer(s) - arrange in middle columns
    if hidden_nodes:
        # Simple layout: put all hidden nodes in middle column
        for i, node_id in enumerate(hidden_nodes):
            pos[node_id] = (2, (len(hidden_nodes) - 1 - i) * 2)

    # Draw the graph
    nx.draw_networkx_nodes(G, pos, nodelist=input_nodes, node_color="lightgreen", node_size=700, label="Input", ax=ax)
    nx.draw_networkx_nodes(G, pos, nodelist=output_nodes, node_color="lightcoral", node_size=700, label="Output", ax=ax)
    if hidden_nodes:
        nx.draw_networkx_nodes(G, pos, nodelist=hidden_nodes, node_color="lightblue", node_size=700, label="Hidden", ax=ax)

    # Draw edges with colors and widths
    nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=edge_widths, alpha=0.7, arrows=True, arrowsize=20, ax=ax)

    # Draw labels
    labels = {}
    for i, node_id in enumerate(input_nodes):
        if i < 2:
            labels[node_id] = f"X{i + 1}"
        else:
            labels[node_id] = "Bias"
    for node_id in output_nodes:
        labels[node_id] = "Out"
    for i, node_id in enumerate(hidden_nodes):
        labels[node_id] = f"H{i + 1}"

    nx.draw_networkx_labels(G, pos, labels, font_size=10, ax=ax)

    # Add title and legend
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="upper right")
    ax.axis("off")

    # Add weight color legend
    blue_patch = patches.Patch(color="blue", label="Positive weight")
    red_patch = patches.Patch(color="red", label="Negative weight")
    ax.legend(handles=[blue_patch, red_patch], loc="lower right")

    plt.tight_layout()
    return fig


def animate_evolution(history, filename="xor_evolution.gif"):
    """Create an animation showing the evolution of the best network over generations."""
    fig, ax = plt.subplots(figsize=(12, 8))

    def update(frame):
        ax.clear()
        generation, genome, fitness = history[frame]

        # Create a directed graph
        G = nx.DiGraph()

        # Group nodes by type
        input_nodes = []
        output_nodes = []
        hidden_nodes = []

        for node in genome.nodes:
            G.add_node(node.id)
            if str(node.node_type) == "input":
                input_nodes.append(node.id)
            elif str(node.node_type) == "output":
                output_nodes.append(node.id)
            else:
                hidden_nodes.append(node.id)

        # Add edges from enabled connections
        edge_colors = []
        edge_widths = []
        for conn in genome.connections:
            if conn.enabled:
                G.add_edge(conn.in_node, conn.out_node, weight=conn.weight)
                if conn.weight > 0:
                    edge_colors.append("blue")
                else:
                    edge_colors.append("red")
                edge_widths.append(min(abs(conn.weight) * 2, 5))

        # Position nodes in layers
        pos = {}

        # Input layer
        for i, node_id in enumerate(input_nodes):
            pos[node_id] = (0, (len(input_nodes) - 1 - i) * 2)

        # Output layer
        for i, node_id in enumerate(output_nodes):
            pos[node_id] = (4, (len(output_nodes) - 1 - i) * 2)

        # Hidden layers
        if hidden_nodes:
            for i, node_id in enumerate(hidden_nodes):
                pos[node_id] = (2, (len(hidden_nodes) - 1 - i) * 2)

        # Draw the graph
        nx.draw_networkx_nodes(G, pos, nodelist=input_nodes, node_color="lightgreen", node_size=700, ax=ax)
        nx.draw_networkx_nodes(G, pos, nodelist=output_nodes, node_color="lightcoral", node_size=700, ax=ax)
        if hidden_nodes:
            nx.draw_networkx_nodes(G, pos, nodelist=hidden_nodes, node_color="lightblue", node_size=700, ax=ax)

        # Draw edges
        if edge_colors:
            nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=edge_widths, alpha=0.7, arrows=True, arrowsize=20, ax=ax)

        # Draw labels
        labels = {}
        for i, node_id in enumerate(input_nodes):
            if i < 2:
                labels[node_id] = f"X{i + 1}"
            else:
                labels[node_id] = "Bias"
        for node_id in output_nodes:
            labels[node_id] = "Out"
        for i, node_id in enumerate(hidden_nodes):
            labels[node_id] = f"H{i + 1}"

        nx.draw_networkx_labels(G, pos, labels, font_size=10, ax=ax)

        # Add title with generation and fitness info
        n_hidden = len(hidden_nodes)
        n_conn = len([c for c in genome.connections if c.enabled])
        ax.set_title(
            f"Generation {generation + 1} | Fitness: {fitness:.2f} | Hidden: {n_hidden} | Connections: {n_conn}",
            fontsize=14,
            fontweight="bold",
        )
        ax.axis("off")

        # Add legend
        blue_patch = patches.Patch(color="blue", label="Positive weight")
        red_patch = patches.Patch(color="red", label="Negative weight")
        ax.legend(handles=[blue_patch, red_patch], loc="lower right")

    # Create animation
    anim = animation.FuncAnimation(fig, update, frames=len(history), interval=100, repeat=True)

    # Save animation
    writer = animation.PillowWriter(fps=10)
    anim.save(filename, writer=writer)
    plt.close(fig)

    print(f"Animation saved as {filename}")


def main():
    # Run multiple trials
    n_trials = 100  # Paper uses 100 runs
    print(f"Running XOR benchmark ({n_trials} trials)...")

    successes = 0
    total_gens = []
    total_hidden = []
    total_conn = []
    networks_evaluated = []
    best_genome = None

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

        # Create animation of evolution
        print("\nGenerating animation of network evolution...")
        gens, hidden, conn, history = run_xor_trial(collect_history=True)
        if history:
            animate_evolution(history, "xor_evolution.gif")
            print(f"Evolution completed in {gens} generations with {hidden} hidden nodes and {conn} connections")


main()
