# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "simple-parsing>=0.1.5",
#   "gymnasium[box2d]>=0.29.1",
#   "numpy>=1.24",
#   "tqdm>=4.66",
# ]
# ///

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import gymnasium as gym
import numpy as np
from simple_parsing import ArgumentParser
from tqdm import trange

from neat import NEAT, NEATConfig, NEATNetwork
from neat_logging import log_generation, start_run


def make_env(seed: int | None = None) -> gym.Env:
    env = gym.make("LunarLander-v3")
    if seed is not None:
        env.reset(seed=seed)
    return env


def evaluate_population(
    genomes,  # type: ignore[no-untyped-def]
    episodes: int,
    seed: int,
) -> np.ndarray:
    env = make_env(seed)
    fitnesses = np.zeros(len(genomes), dtype=np.float32)
    nets = [NEATNetwork(g) for g in genomes]
    for idx, (g, net) in enumerate(zip(genomes, nets)):
        total = 0.0
        for ep in range(episodes):
            net.reset_state()
            obs, _ = env.reset(seed=seed + ep)
            done = False
            while not done:
                x = np.asarray(obs, dtype=np.float32)
                x = np.append(x, 1.0)
                logits = np.array(net(x.tolist()), dtype=np.float32)
                action = int(np.argmax(logits))
                obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                total += float(reward)
        fitnesses[idx] = total / episodes
    env.close()
    return fitnesses


@dataclass
class TrainCfg:
    # Evolution
    generations: int = 150
    population: int = 300
    episodes: int = 3
    seed: int = 42
    # Logging
    out_dir: str = "runs"
    notes: str = ""

    # NEAT hyperparameters (good starting point for LunarLander)
    dense_initial_connectivity: bool = True
    allow_recurrent: bool = False
    speciate: bool = True
    compatibility_threshold: float = 3.0
    c1: float = 1.0
    c2: float = 1.0
    c3: float = 0.4
    stagnation_generations: int = 25
    replacement_rate: float = 0.1
    mutation_std: float = 0.8
    weight_range: float = 2.0
    mutation_rate: float = 0.8
    survival_threshold: float = 0.4
    crossover_rate: float = 0.75
    add_node_prob: float = 0.03
    add_conn_prob: float = 0.08
    interspecies_mating_rate: float = 0.01
    champion_preservation_threshold: int = 5
    inherited_disable_rate: float = 0.75
    toggle_rate: float = 0.01
    # Global stagnation cull (paper footnote) configuration
    global_stagnation_generations: int = 50
    global_cull_keep_top_k: int = 2
    global_cull_start_gen: int = 20


def main() -> None:
    parser = ArgumentParser()
    parser.add_arguments(TrainCfg, dest="cfg")
    args = parser.parse_args()
    cfg: TrainCfg = args.cfg

    # Probe env for sizes
    probe_env = make_env(cfg.seed)
    obs, _ = probe_env.reset(seed=cfg.seed)
    obs_dim = int(np.array(obs).shape[0])
    n_actions = int(probe_env.action_space.n)  # type: ignore[attr-defined]
    probe_env.close()

    neat_cfg = NEATConfig(
        n_inputs=obs_dim + 1,
        n_outputs=n_actions,
        population_size=cfg.population,
        activation_choices=["tanh", "relu", "sin", "gaussian", "identity"],
        output_activation="identity",
        dense_initial_connectivity=cfg.dense_initial_connectivity,
        allow_recurrent=cfg.allow_recurrent,
        compatibility_threshold=cfg.compatibility_threshold,
        c1=cfg.c1,
        c2=cfg.c2,
        c3=cfg.c3,
        stagnation_generations=cfg.stagnation_generations,
        global_stagnation_generations=cfg.global_stagnation_generations,
        global_cull_keep_top_k=cfg.global_cull_keep_top_k,
        global_cull_start_gen=cfg.global_cull_start_gen,
        replacement_rate=cfg.replacement_rate,
        mutation_std=cfg.mutation_std,
        weight_range=cfg.weight_range,
        mutation_rate=cfg.mutation_rate,
        speciate=cfg.speciate,
        survival_threshold=cfg.survival_threshold,
        crossover_rate=cfg.crossover_rate,
        add_node_prob=cfg.add_node_prob,
        add_conn_prob=cfg.add_conn_prob,
        interspecies_mating_rate=cfg.interspecies_mating_rate,
        champion_preservation_threshold=cfg.champion_preservation_threshold,
        inherited_disable_rate=cfg.inherited_disable_rate,
        toggle_rate=cfg.toggle_rate,
    )

    neat = NEAT(neat_cfg)

    # Initialize log directory
    run_dir = start_run("LunarLander-v3", neat_cfg, out_dir=cfg.out_dir, notes=cfg.notes)

    best_fitness = -1e9

    t = trange(cfg.generations, desc="Evolving", unit="gen", leave=True)
    for gen in t:
        genomes = neat.ask()
        fitnesses = evaluate_population(genomes, cfg.episodes, cfg.seed + gen * 1337)
        neat.tell(fitnesses.tolist())

        # Extract species ids after speciation inside NEAT.tell
        species_ids = np.array([g.species_id for g in genomes], dtype=np.int32)

        # Log snapshot
        log_generation(run_dir, gen, genomes, fitnesses, species_ids, neat_cfg, save_delta=True)

        # Track best and update progress bar postfix
        gen_best = float(np.max(fitnesses))
        if gen_best > best_fitness:
            best_fitness = gen_best
        t.set_postfix(best=f"{best_fitness:.1f}")

    print(f"Logging complete in {run_dir}")


if __name__ == "__main__":
    main()
