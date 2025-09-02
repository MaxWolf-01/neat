# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "numpy>=1.24",
# ]
# ///

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict
from typing import Any, Iterable

import numpy as np

from neat import ConnectionGene, Genome, NEATConfig, NodeGene, NodeType

ActivationVocab = (
    "identity",
    "sigmoid",
    "tanh",
    "sin",
    "cos",
    "gaussian",
    "relu",
)


def ensure_dir(path: str) -> None:
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def start_run(env_id: str, cfg: NEATConfig, out_dir: str | None = None, notes: str | None = None) -> str:
    ts = time.strftime("%Y%m%d-%H%M%S")
    base = out_dir or os.path.join("runs", env_id)
    run_dir = os.path.join(base, ts)
    ensure_dir(run_dir)
    manifest = {
        "env_id": env_id,
        "created": ts,
        "log_version": 1,
        "neat_config": asdict(cfg),
        "notes": notes or "",
    }
    with open(os.path.join(run_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    with open(os.path.join(run_dir, "progress.json"), "w", encoding="utf-8") as f:
        json.dump({"last_gen": -1}, f)
    return run_dir


def _node_type_to_int(t: NodeType) -> int:
    if t == NodeType.INPUT:
        return 0
    if t == NodeType.OUTPUT:
        return 1
    return 2


def _int_to_node_type(v: int) -> NodeType:
    return [NodeType.INPUT, NodeType.OUTPUT, NodeType.HIDDEN][v]


def serialize_genome(genome: Genome) -> dict[str, Any]:
    nodes_ids = np.array([n.id for n in genome.nodes], dtype=np.int32)
    nodes_types = np.array([_node_type_to_int(n.node_type) for n in genome.nodes], dtype=np.int8)
    nodes_act = np.array([n.activation for n in genome.nodes], dtype="U16")
    conns = genome.connections
    conn_innov = np.array([c.innovation for c in conns], dtype=np.int32)
    conn_in = np.array([c.in_node for c in conns], dtype=np.int32)
    conn_out = np.array([c.out_node for c in conns], dtype=np.int32)
    conn_w = np.array([c.weight for c in conns], dtype=np.float32)
    conn_en = np.array([c.enabled for c in conns], dtype=np.bool_)
    return {
        "nodes_ids": nodes_ids,
        "nodes_types": nodes_types,
        "nodes_act": nodes_act,
        "conn_innov": conn_innov,
        "conn_in": conn_in,
        "conn_out": conn_out,
        "conn_w": conn_w,
        "conn_en": conn_en,
        "species_id": int(genome.species_id),
        "fitness": float(genome.fitness),
        "adjusted_fitness": float(genome.adjusted_fitness),
    }


def deserialize_genome(data: dict[str, Any]) -> Genome:
    node_ids = data["nodes_ids"].tolist()
    node_types = data["nodes_types"].tolist()
    node_act = data["nodes_act"].tolist()
    nodes = [NodeGene(i, _int_to_node_type(t), a) for i, t, a in zip(node_ids, node_types, node_act)]
    conns = []
    for inn, i, o, w, en in zip(
        data["conn_innov"].tolist(),
        data["conn_in"].tolist(),
        data["conn_out"].tolist(),
        data["conn_w"].tolist(),
        data["conn_en"].tolist(),
    ):
        conns.append(ConnectionGene(innovation=int(inn), in_node=int(i), out_node=int(o), weight=float(w), enabled=bool(en)))
    return Genome(
        species_id=int(data.get("species_id", 0)),
        nodes=nodes,
        connections=conns,
        fitness=float(data.get("fitness", 0.0)),
        adjusted_fitness=float(data.get("adjusted_fitness", 0.0)),
    )


def genome_features(genome: Genome) -> np.ndarray:
    conns = [c for c in genome.connections]
    if conns:
        weights = np.array([c.weight for c in conns], dtype=np.float32)
        en_mask = np.array([c.enabled for c in conns], dtype=np.bool_)
        mean_w = float(np.mean(weights))
        std_w = float(np.std(weights))
        frac_pos = float(np.mean(weights > 0))
        frac_neg = float(np.mean(weights < 0))
        enabled_ratio = float(np.mean(en_mask))
    else:
        mean_w = std_w = frac_pos = frac_neg = enabled_ratio = 0.0

    node_types = [_node_type_to_int(n.node_type) for n in genome.nodes]
    hidden_count = float(sum(1 for t in node_types if t == 2))
    conn_count = float(len(conns))

    # degree stats (enabled only)
    from collections import defaultdict

    indeg: dict[int, int] = defaultdict(int)
    outdeg: dict[int, int] = defaultdict(int)
    for c in conns:
        if c.enabled:
            outdeg[c.in_node] += 1
            indeg[c.out_node] += 1
    if genome.nodes:
        mean_in_deg = float(np.mean([indeg[n.id] for n in genome.nodes]))
        mean_out_deg = float(np.mean([outdeg[n.id] for n in genome.nodes]))
    else:
        mean_in_deg = mean_out_deg = 0.0

    # activation mix for hidden nodes over ActivationVocab
    hidden_acts = [n.activation for n in genome.nodes if _node_type_to_int(n.node_type) == 2]
    act_counts = np.zeros(len(ActivationVocab), dtype=np.float32)
    for a in hidden_acts:
        if a in ActivationVocab:
            act_counts[ActivationVocab.index(a)] += 1.0
    if len(hidden_acts) > 0:
        act_counts /= float(len(hidden_acts))

    feats = np.array(
        [
            hidden_count,
            conn_count,
            enabled_ratio,
            mean_w,
            std_w,
            frac_pos,
            frac_neg,
            mean_in_deg,
            mean_out_deg,
        ],
        dtype=np.float32,
    )
    return np.concatenate([feats, act_counts])


def compatibility_distance(
    g1: Genome,
    g2: Genome,
    *,
    c1: float,
    c2: float,
    c3: float,
) -> float:
    # Re-implement to avoid needing NEAT instance; assumes connections sorted by innovation
    conns1 = sorted(g1.connections, key=lambda c: c.innovation)
    conns2 = sorted(g2.connections, key=lambda c: c.innovation)
    if not conns1 and not conns2:
        return 0.0
    i, j = 0, 0
    matching = 0
    disjoint = 0
    weight_diff_sum = 0.0
    while i < len(conns1) and j < len(conns2):
        c1i = conns1[i]
        c2i = conns2[j]
        if c1i.innovation == c2i.innovation:
            matching += 1
            weight_diff_sum += abs(c1i.weight - c2i.weight)
            i += 1
            j += 1
        elif c1i.innovation < c2i.innovation:
            disjoint += 1
            i += 1
        else:
            disjoint += 1
            j += 1
    excess = (len(conns1) - i) + (len(conns2) - j)
    avg_weight_diff = (weight_diff_sum / matching) if matching > 0 else 0.0
    n_genes = max(len(conns1), len(conns2))
    n = n_genes if n_genes > 20 else 1
    return c1 * excess / n + c2 * disjoint / n + c3 * avg_weight_diff


def log_generation(
    run_dir: str,
    gen_idx: int,
    genomes: Iterable[Genome],
    fitnesses: np.ndarray,
    species_ids: np.ndarray,
    cfg: NEATConfig,
    *,
    save_delta: bool = True,
) -> None:
    genomes_list = list(genomes)
    fitnesses = np.asarray(fitnesses, dtype=np.float32)
    species_ids = np.asarray(species_ids, dtype=np.int32)
    assert len(genomes_list) == len(fitnesses) == len(species_ids)

    # compute features
    feats = np.stack([genome_features(g) for g in genomes_list], axis=0)

    # best
    best_idx = int(np.argmax(fitnesses))

    # species stats
    species = {}
    for sid in np.unique(species_ids):
        mask = species_ids == sid
        subfit = fitnesses[mask]
        species[int(sid)] = {
            "size": int(np.sum(mask)),
            "best": float(np.max(subfit)) if subfit.size > 0 else 0.0,
            "mean": float(np.mean(subfit)) if subfit.size > 0 else 0.0,
        }

    save: dict[str, Any] = {
        "fitness": fitnesses,
        "species_id": species_ids,
        "features": feats,
        "best_index": best_idx,
        "species_stats": json.dumps(species),  # JSON string for dict of small size
    }

    # inline best genome arrays (no pickle required)
    best_ser = serialize_genome(genomes_list[best_idx])
    for k, v in best_ser.items():
        save[f"best_{k}"] = v

    if save_delta:
        n = len(genomes_list)
        delta = np.zeros((n, n), dtype=np.float32)
        for i in range(n):
            gi = genomes_list[i]
            for j in range(i + 1, n):
                dj = compatibility_distance(gi, genomes_list[j], c1=cfg.c1, c2=cfg.c2, c3=cfg.c3)
                delta[i, j] = delta[j, i] = dj
        save["delta"] = delta

    np.savez_compressed(os.path.join(run_dir, f"gen{gen_idx:04d}.npz"), **save)

    with open(os.path.join(run_dir, "progress.json"), "w", encoding="utf-8") as f:
        json.dump({"last_gen": gen_idx}, f)
