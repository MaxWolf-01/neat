# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "simple-parsing>=0.1.5",
#   "numpy>=1.24",
#   "matplotlib>=3.7",
#   "imageio>=2.31",
#   "imageio-ffmpeg>=0.4.9",
#   "pillow>=10.0",
#   "tqdm>=4.66",
#   "scikit-learn>=1.2",
#   "gymnasium[box2d]>=0.29.1",
# ]
# ///

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import gymnasium as gym
import imageio
import numpy as np
from PIL import Image
from simple_parsing import ArgumentParser
from sklearn.manifold import MDS
from tqdm import tqdm

from neat import Genome, NEATNetwork, NodeType
from neat_logging import deserialize_genome

# -----------------------------
# Helpers
# -----------------------------


def ensure_dir(path: str) -> None:
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def fig_to_image(fig) -> np.ndarray:  # type: ignore[no-untyped-def]
    import matplotlib

    matplotlib.use("Agg")
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    rgba = buf.reshape(height, width, 4)
    return rgba[..., :3].copy()


def resize_img(img: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    w, h = size
    pil = Image.fromarray(img)
    pil = pil.resize((w, h), Image.BILINEAR)
    return np.asarray(pil)


def make_env(env_id: str, seed: int | None = None, render: bool = True) -> gym.Env:
    env = gym.make(env_id, render_mode=("rgb_array" if render else None))
    if seed is not None:
        env.reset(seed=seed)
    return env


# -----------------------------
# Panels
# -----------------------------


def draw_network_panel(
    genome: Genome,
    size: Tuple[int, int],
    title: str = "",
    values: Dict[int, float] | None = None,
    highlight: set[int] | None = None,
) -> np.ndarray:
    import matplotlib.pyplot as plt

    # Lay out nodes by topological depth for readability
    nn = NEATNetwork(genome)
    incoming = nn.incoming
    indeg = {n: len(incoming[n]) for n in incoming}
    outgoing: Dict[int, List[int]] = {n: [] for n in incoming}
    for out_node, edges in incoming.items():
        for in_node, _ in edges:
            outgoing[in_node].append(out_node)
    order: List[int] = []
    queue = [n for n, d in indeg.items() if d == 0]
    while queue:
        v = queue.pop(0)
        order.append(v)
        for m in outgoing[v]:
            indeg[m] -= 1
            if indeg[m] == 0:
                queue.append(m)
    depths: Dict[int, int] = {n: 0 for n in nn.input_nodes}
    for n in order:
        if n in nn.input_nodes:
            depths[n] = 0
            continue
        preds = incoming[n]
        depths[n] = 0 if not preds else 1 + max(depths.get(p, 0) for p, _ in preds)
    max_d = max(depths.values(), default=0)
    for n in nn.output_nodes:
        depths[n] = max(depths.get(n, 0), max_d + 1)

    # positions
    by_layer: Dict[int, List[int]] = {}
    for n, d in depths.items():
        by_layer.setdefault(d, []).append(n)
    max_layer = max(by_layer.keys(), default=0)
    pos: Dict[int, Tuple[float, float]] = {}
    for layer, nodes in by_layer.items():
        nodes_sorted = sorted(nodes)
        x = 0.05 + 0.9 * (layer / max(1, max_layer))
        count = len(nodes_sorted)
        for i, n in enumerate(nodes_sorted):
            y = 0.05 + (0.9 * (i + 1) / (count + 1)) if count > 1 else 0.5
            pos[n] = (x, y)

    # draw
    w, h = size
    fig = plt.figure(figsize=(w / 100, h / 100), dpi=100, facecolor="#0f0f13")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # edges
    for c in genome.connections:
        if c.in_node in pos and c.out_node in pos:
            x1, y1 = pos[c.in_node]
            x2, y2 = pos[c.out_node]
            color = "#2f9df4" if c.weight >= 0 else "#ff4d6d"
            alpha = 0.9 if c.enabled else 0.25
            style = "-" if c.enabled else "--"
            lw = max(0.5, min(5.0, abs(c.weight)))
            ax.plot([x1, x2], [y1, y2], style, color=color, alpha=alpha, lw=lw)

    # nodes
    for n in genome.nodes:
        x, y = pos.get(n.id, (0.5, 0.5))
        if n.node_type == NodeType.INPUT:
            base_fc, ec = "#ffd166", "#b38600"
        elif n.node_type == NodeType.OUTPUT:
            base_fc, ec = "#06d6a0", "#027a62"
        else:
            base_fc, ec = "#cdb4db", "#7a5ea8"
        # shade by activation value if provided
        if values is not None and n.id in values:
            v = float(values[n.id])
            v_norm = max(0.0, min(1.0, (v + 1.0) / 2.0))  # map [-1,1] -> [0,1]
            import matplotlib.colors as mcolors

            fc = mcolors.to_rgb(base_fc)
            # stronger contrast: blend toward near-black at low activation
            fc = tuple(v_norm * c + (1 - v_norm) * 0.05 for c in fc)
        else:
            fc = base_fc
        lw = 1.5
        if highlight is not None and n.id in highlight:
            lw = 3.0
            ec = "#ffffff"
        circ = plt.Circle((x, y), 12 / 640, facecolor=fc, edgecolor=ec, linewidth=lw)
        ax.add_patch(circ)
        label = f"{n.id}" if n.node_type != NodeType.HIDDEN else f"{n.id}\n{n.activation}"
        ax.text(x, y, label, ha="center", va="center", fontsize=7, color="#111")

    if title:
        ax.text(0.02, 0.98, title, va="top", ha="left", color="white", fontsize=10)

    img = fig_to_image(fig)
    import matplotlib.pyplot as plt  # noqa

    plt.close(fig)
    return img


def draw_species_spindle_panel(
    species_order: List[int],
    sizes_per_gen: List[Dict[int, int]],
    best_per_gen: List[Dict[int, float]],
    mean_per_gen: List[float],
    std_per_gen: List[float],
    upto_gen: int,
    size: Tuple[int, int],
) -> np.ndarray:
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt

    w, h = size
    fig = plt.figure(figsize=(w / 100, h / 100), dpi=100, facecolor="#0f0f13")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_facecolor("#0f0f13")

    # Time along x
    T = max(1, upto_gen + 1)
    xs = np.linspace(0.06, 0.96, T)

    # Stable color per species using golden-angle HSV
    def species_color(sid: int) -> Tuple[float, float, float]:
        hue = (sid * 0.6180339887498949) % 1.0
        s, v = 0.65, 0.9
        return mcolors.hsv_to_rgb((hue, s, v))

    K = max(1, len(species_order))
    # Build fractions matrix (K x T): share per species per gen
    fractions = np.zeros((K, T), dtype=np.float32)
    for g in range(T):
        sizes = sizes_per_gen[g]
        total = sum(sizes.values())
        if total <= 0:
            continue
        for k, sid in enumerate(species_order):
            fractions[k, g] = sizes.get(sid, 0) / total

    # Cumulative stacks
    cum = np.cumsum(fractions, axis=0)
    lower = cum - fractions
    upper = cum

    # Draw stacked areas species by species
    for k, sid in enumerate(species_order):
        col = species_color(sid)
        ax.fill_between(xs, lower[k], upper[k], color=(*col, 0.75), linewidth=0)

        # Performance centerline (midline) segments
        mid = (lower[k] + upper[k]) * 0.5
        prom_lo = []
        prom_hi = []
        for g in range(T):
            best = best_per_gen[g].get(sid, -1e9)
            mean = mean_per_gen[g] if g < len(mean_per_gen) else 0.0
            std = std_per_gen[g] if g < len(std_per_gen) else 1.0
            prom_lo.append(best >= mean + 1 * std)
            prom_hi.append(best >= mean + 2 * std)

        def plot_mask(mask: List[bool], lw: float, color_line: Tuple[float, float, float]):
            seg_x: List[float] = []
            seg_y: List[float] = []
            for i, m in enumerate(mask):
                if m and fractions[k, i] > 0:
                    seg_x.append(xs[i])
                    seg_y.append(mid[i])
                else:
                    if seg_x:
                        ax.plot(seg_x, seg_y, color=(*color_line, 1.0), lw=lw)
                        seg_x, seg_y = [], []
            if seg_x:
                ax.plot(seg_x, seg_y, color=(*color_line, 1.0), lw=lw)

        plot_mask(prom_lo, lw=1.4, color_line=tuple(0.5 + 0.5 * c for c in col))
        plot_mask(prom_hi, lw=2.4, color_line=(1.0, 1.0, 1.0))

    # X-axis
    ax.plot([0.06, 0.96], [0.05, 0.05], color="#aaa", lw=1.0, alpha=0.6)
    tick_idx = [0, max(0, T // 2), max(0, T - 1)]
    for g in tick_idx:
        x = xs[g]
        ax.plot([x, x], [0.045, 0.055], color="#aaa", lw=1.0, alpha=0.6)
        ax.text(x, 0.02, str(g), color="#ddd", ha="center", va="top", fontsize=8)
    ax.text(0.51, 0.0, "generation", color="#ddd", ha="center", va="top", fontsize=9)

    # Caption
    ax.text(
        0.06,
        0.98,
        "Species stacked area — thickness=size share; centerline: >1σ (light), >2σ (white)",
        va="top",
        ha="left",
        color="white",
        fontsize=9,
        alpha=0.9,
    )

    fig.subplots_adjust(0, 0, 1, 1)
    img = fig_to_image(fig)
    import matplotlib.pyplot as plt  # noqa

    plt.close(fig)
    return img


def draw_metrics_panel(history: Dict[str, List[float]], size: Tuple[int, int]) -> np.ndarray:
    import matplotlib.pyplot as plt

    w, h = size
    fig = plt.figure(figsize=(w / 100, h / 100), dpi=100, facecolor="#0f0f13")
    ax = fig.add_subplot(2, 1, 1)
    ax.set_facecolor("#0f0f13")
    ax.plot(history.get("best", []), label="best", color="#06d6a0")
    ax.plot(history.get("mean", []), label="mean", color="#2f9df4")
    ax.plot(history.get("median", []), label="median", color="#cdb4db")
    leg1 = ax.legend(loc="lower right", frameon=False)
    for txt in leg1.get_texts():
        txt.set_color("white")
    ax.grid(True, alpha=0.2)
    ax.tick_params(colors="#ddd")
    for spine in ax.spines.values():
        spine.set_color("#444")
    ax.set_title("Fitness over time", color="#ddd")

    ax2 = fig.add_subplot(2, 1, 2)
    ax2.set_facecolor("#0f0f13")
    ax2.plot(history.get("species_count", []), label="#species", color="#ffd166")
    ax2.plot(history.get("nodes", []), label="nodes (best)", color="#ff4d6d")
    ax2.plot(history.get("conns", []), label="conns (best)", color="#8ecae6")
    leg2 = ax2.legend(loc="upper left", frameon=False)
    for txt in leg2.get_texts():
        txt.set_color("white")
    ax2.grid(True, alpha=0.2)
    ax2.tick_params(colors="#ddd")
    for spine in ax2.spines.values():
        spine.set_color("#444")
    ax2.set_title("Topology & species", color="#ddd")

    ax.set_xlabel("")
    ax2.set_xlabel("generation", color="#ddd")
    for a in (ax, ax2):
        a.set_ylabel("")

    fig.patch.set_facecolor("#0f0f13")
    # space subplots to avoid overlap further
    fig.subplots_adjust(hspace=0.55, left=0.08, right=0.98, top=0.9, bottom=0.18)
    img = fig_to_image(fig)
    import matplotlib.pyplot as plt  # noqa

    plt.close(fig)
    return img


def draw_network_firing_frame(
    genome: Genome,
    obs: np.ndarray,
    size: Tuple[int, int],
) -> np.ndarray:
    # compute per-node activations using feedforward order
    nn = NEATNetwork(genome)
    values: Dict[int, float] = {nid: float(v) for nid, v in zip(nn.input_nodes, obs.tolist())}
    topo = [n for n in nn._topo_order if n not in nn.input_nodes] if nn._topo_order is not None else []
    for node in topo:
        act = nn.node_activations[node]
        s = sum(values.get(src, 0.0) * w for src, w in nn.incoming[node])
        values[node] = float(act(s))
    # reuse static network drawing but add overlay intensities
    out_vals = np.array([values.get(n, 0.0) for n in nn.output_nodes], dtype=np.float32)
    if out_vals.size > 0:
        amax = int(np.argmax(out_vals))
        highlight = {nn.output_nodes[amax]}
    else:
        highlight = set()
    base = draw_network_panel(genome, size, values=values, highlight=highlight)
    return base


# -----------------------------
# Renderer
# -----------------------------


@dataclass
class VizCfg:
    run_dir: str
    output: str = "lunar_lander_evolution.mp4"
    width: int = 1600
    height: int = 900
    fps: int = 30
    stride: int = 2  # env frames stride
    render_every_gen: bool = True


def main() -> None:
    ap = ArgumentParser()
    ap.add_arguments(VizCfg, dest="cfg")
    args = ap.parse_args()
    cfg: VizCfg = args.cfg

    # load manifest
    with open(os.path.join(cfg.run_dir, "manifest.json"), "r", encoding="utf-8") as f:
        manifest = json.load(f)
    env_id = manifest["env_id"]

    # gather gens
    gen_files = sorted([p for p in os.listdir(cfg.run_dir) if p.startswith("gen") and p.endswith(".npz")])
    if not gen_files:
        raise SystemExit("No generation logs found.")

    writer = imageio.get_writer(
        cfg.output,
        fps=cfg.fps,
        codec="libx264",
        quality=8,
        pixelformat="yuv420p",
        ffmpeg_params=["-preset", "slow", "-crf", "18"],
    )

    W, H = cfg.width, cfg.height
    panel_w, panel_h = W // 2, H // 2

    history: Dict[str, List[float]] = {"best": [], "mean": [], "median": [], "species_count": [], "nodes": [], "conns": []}
    # For species timeline panel
    species_order: List[int] = []
    sizes_per_gen: List[Dict[int, int]] = []
    best_per_gen: List[Dict[int, float]] = []
    mean_per_gen: List[float] = []
    std_per_gen: List[float] = []

    try:
        for gf in tqdm(gen_files, desc="Rendering gens"):
            data = np.load(os.path.join(cfg.run_dir, gf), allow_pickle=False)
            species_id = data["species_id"]
            fitness = data["fitness"]
            feats = data["features"]
            delta = data["delta"] if "delta" in data else None
            best_index = int(data["best_index"])  # type: ignore[index]
            best_data = {
                "nodes_ids": data["best_nodes_ids"],
                "nodes_types": data["best_nodes_types"],
                "nodes_act": data["best_nodes_act"],
                "conn_innov": data["best_conn_innov"],
                "conn_in": data["best_conn_in"],
                "conn_out": data["best_conn_out"],
                "conn_w": data["best_conn_w"],
                "conn_en": data["best_conn_en"],
                "species_id": 0,
                "fitness": 0.0,
                "adjusted_fitness": 0.0,
            }
            best_genome = deserialize_genome(best_data)
            species_stats = json.loads(str(data["species_stats"].item()))  # type: ignore[index]

            gen_best = float(np.max(fitness))
            gen_mean = float(np.mean(fitness))
            gen_median = float(np.median(fitness))
            gen_std = float(np.std(fitness))
            history["best"].append(gen_best)
            history["mean"].append(gen_mean)
            history["median"].append(gen_median)
            std_per_gen.append(gen_std)
            history["species_count"].append(float(len(np.unique(species_id))))
            history["nodes"].append(float(len(best_genome.nodes)))
            history["conns"].append(float(len(best_genome.connections)))

            # Update species timeline data structures
            stats = species_stats  # dict as stored
            sizes: Dict[int, int] = {}
            best_by_sid: Dict[int, float] = {}
            for sid_str, rec in stats.items():
                sid = int(sid_str)
                sizes[sid] = int(rec.get("size", 0))
                best_by_sid[sid] = float(rec.get("best", 0.0))
                if sid not in species_order:
                    species_order.append(sid)
            sizes_per_gen.append(sizes)
            best_per_gen.append(best_by_sid)
            mean_per_gen.append(gen_mean)

            # Pre-render static panels
            net_static = draw_network_panel(best_genome, (panel_w, panel_h), title=f"Best net (gen {len(history['best']) - 1})")
            # Species evolution stacked area panel (cumulative up to current gen)
            spec_img = draw_species_spindle_panel(
                species_order,
                sizes_per_gen,
                best_per_gen,
                mean_per_gen,
                std_per_gen,
                upto_gen=len(history["best"]) - 1,
                size=(panel_w, panel_h),
            )
            metrics_img = draw_metrics_panel(history, (panel_w, panel_h))

            improved = len(history["best"]) == 1 or history["best"][-1] > max(history["best"][:-1])
            if not cfg.render_every_gen and not improved:
                # Emit a single composite static frame and skip rollout
                blank = np.zeros_like(net_static)
                top = np.hstack([net_static, blank])
                bottom = np.hstack([spec_img, metrics_img])
                full = np.vstack([top, bottom])
                writer.append_data(full)
                continue

            # Rollout env for this best genome
            env = make_env(env_id, render=True)
            nn = NEATNetwork(best_genome)
            nn.reset_state()
            obs, _ = env.reset()
            done = False
            step = 0
            while not done:
                # top-right env
                frame = env.render()
                assert isinstance(frame, np.ndarray)
                # compute inputs for action and (maybe) panel
                x = np.asarray(obs, dtype=np.float32)
                x = np.append(x, 1.0)
                if step % cfg.stride == 0:
                    # top-left firing (use current obs)
                    firing = draw_network_firing_frame(best_genome, x, (panel_w, panel_h))
                    env_img = resize_img(frame, (panel_w, panel_h))
                    # compose
                    top = np.hstack([firing, env_img])
                    bottom = np.hstack([spec_img, metrics_img])
                    full = np.vstack([top, bottom])
                    writer.append_data(full)
                # step env
                logits = np.array(nn(x.tolist()), dtype=np.float32)
                action = int(np.argmax(logits))
                obs, _, term, trunc, _ = env.step(action)
                done = term or trunc
                step += 1
            env.close()
    finally:
        writer.close()


if __name__ == "__main__":
    main()
