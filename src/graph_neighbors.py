"""src/graph_neighbors.py — neighborhood subgraph around selected AS + HTML viz.

Loads the CSR snapshot written by src/graph.py::save_graph
(graph.npz + edge_types.npy + asn_ids.npy), runs a multi-source BFS up to
--depth hops, builds an undirected subgraph whose edge type is one of
{peer, upstream, downstream, mixed}, and renders it with pyvis.

Usage:
    python -m src.graph_neighbors --as 15169 13335 --depth 1
    python -m src.graph_neighbors --as-file my_as.txt --depth 2
    python -m src.graph_neighbors --as 15169 --dir data/processed --out out/
    python -m src.graph_neighbors --as 15169 --depth 2 --verbose
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections import deque
from pathlib import Path

import numpy as np
import scipy.sparse as sparse
from pyvis.network import Network

log = logging.getLogger("graph_neighbors")

# --- Constants kept in sync with src/graph.py (UPSTREAM/DOWNSTREAM/PEER) ---
UPSTREAM = 1    # customer -> provider
DOWNSTREAM = 2  # provider -> customer
PEER = 3        # peer -> peer

# Edit this list to change the default seed AS without using --as.
DEFAULT_ASNS: list[int] = [
    42385,
]

# Colors / widths for the HTML output.
NODE_SEED_COLOR = "#e74c3c"
NODE_OTHER_COLOR = "#95a5a6"
EDGE_COLORS = {
    "peer": "#2e86de",
    "upstream": "#27ae60",
    "downstream": "#e67e22",
    "mixed": "#8e44ad",
}
EDGE_WIDTHS = {"peer": 2, "upstream": 3, "downstream": 3, "mixed": 5}


class StageTimer:
    """Context manager that logs how long a stage took."""

    def __init__(self, name: str):
        self.name = name

    def __enter__(self):
        self.t0 = time.perf_counter()
        log.info("=== Stage: %s ===", self.name)
        return self

    def __exit__(self, exc_type, exc, tb):
        dt = time.perf_counter() - self.t0
        if exc_type is None:
            log.info("Stage '%s' done in %.2fs", self.name, dt)
        else:
            log.error("Stage '%s' failed in %.2fs: %s", self.name, dt, exc)
        return False


def setup_logging(verbose: bool, quiet: bool) -> None:
    level = logging.INFO
    if verbose:
        level = logging.DEBUG
    elif quiet:
        level = logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def load_snapshot(directory: Path):
    """Mirror src/graph.py::load_graph without importing the module."""
    graph = sparse.load_npz(directory / "graph.npz").tocsr()
    log.debug("Loaded graph.npz: shape=%s, nnz=%d", graph.shape, graph.nnz)
    kinds = np.load(directory / "edge_types.npy", allow_pickle=False)
    log.debug("Loaded edge_types.npy: len=%d", len(kinds))
    asn_ids = np.load(directory / "asn_ids.npy", allow_pickle=False)
    log.debug("Loaded asn_ids.npy: len=%d", len(asn_ids))
    if graph.nnz != len(kinds) or graph.shape[0] != len(asn_ids):
        raise ValueError("Graph files are incompatible")
    return graph, kinds, asn_ids


def build_reverse(graph: sparse.csr_matrix, kinds: np.ndarray):
    """Build the CSR of graph.T with kinds re-ordered to match (dst, src)."""
    n = graph.shape[0]
    # Reconstruct the flat (src, dst) arc list in CSR order, then re-sort by (dst, src).
    row_idx = np.repeat(np.arange(n, dtype=np.int64), np.diff(graph.indptr))
    log.debug("Built row_idx: %d rows", len(row_idx))
    col_idx = graph.indices.astype(np.int64)
    order = np.lexsort((col_idx, row_idx))
    src_t = col_idx[order]
    dst_t = row_idx[order]
    kinds_rev = kinds[order]
    data_t = np.ones(graph.nnz, dtype=np.uint8)
    indptr_t = np.empty(n + 1, dtype=np.int32)
    indptr_t[0] = 0
    indptr_t[1:] = np.cumsum(np.bincount(src_t, minlength=n), dtype=np.int32)
    reverse = sparse.csr_matrix((data_t, dst_t, indptr_t), shape=(n, n))
    log.debug("Reverse CSR built: nnz=%d", reverse.nnz)
    return reverse, kinds_rev


def resolve_seeds(args) -> list[int]:
    seeds: list[int] = []
    as_list = args.as_list or []
    seeds.extend(int(x) for x in as_list)
    if args.as_file:
        log.debug("Reading seeds from %s", args.as_file)
        with Path(args.as_file).open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                seeds.append(int(line))
    if not seeds:
        seeds = list(DEFAULT_ASNS)
        log.debug("Using DEFAULT_ASNS: %s", seeds)
    else:
        log.debug("Resolved seeds from args: %s", seeds)
    seen: set[int] = set()
    out: list[int] = []
    for s in seeds:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def bfs_selected(graph: sparse.csr_matrix, reverse: sparse.csr_matrix,
                 seed_idx: np.ndarray, depth: int) -> np.ndarray:
    """Multi-source BFS up to `depth` hops. Returns sorted selected vertex indices."""
    n = graph.shape[0]
    log.info("BFS: n=%d, seeds=%d, depth=%d", n, len(seed_idx), depth)
    dist = np.full(n, -1, dtype=np.int32)
    dist[seed_idx] = 0
    frontier = seed_idx.copy()
    for step in range(1, depth + 1):
        t0 = time.perf_counter()
        log.debug("BFS step %d: frontier size=%d", step, len(frontier))
        if len(frontier) == 0:
            break
        neighbors = np.empty(0, dtype=np.int64)
        for v in frontier:
            neighbors = np.concatenate([
                neighbors,
                graph.indices[graph.indptr[v]: graph.indptr[v + 1]],
                reverse.indices[reverse.indptr[v]: reverse.indptr[v + 1]],
            ])
        if neighbors.size == 0:
            log.debug("BFS step %d: no neighbors, stopping", step)
            break
        new = np.unique(neighbors[dist[neighbors] < 0])
        if len(new) == 0:
            log.debug("BFS step %d: no new nodes, stopping", step)
            break
        dist[new] = step
        log.debug("BFS step %d: +%d new nodes (cumulative=%d), %.2fs",
                  step, len(new), int((dist >= 0).sum()),
                  time.perf_counter() - t0)
        frontier = new
    selected = np.flatnonzero(dist >= 0)
    log.info("BFS finished: selected=%d AS", len(selected))
    return selected


def classify_edge(
    graph: sparse.csr_matrix,
    reverse: sparse.csr_matrix,
    kinds: np.ndarray,
    kinds_rev: np.ndarray,
    u: int,
    v: int,
) -> str:
    """Classify the undirected edge (u, v) from its directed arc types."""
    has: set[str] = set()

    def look(csr, k, row, col):
        idx = csr.indices[csr.indptr[row]: csr.indptr[row + 1]]
        p = np.searchsorted(idx, col)
        if p < len(idx) and idx[p] == col:
            kind = k[csr.indptr[row] + p]
            if kind == UPSTREAM:
                return "upstream"
            if kind == DOWNSTREAM:
                return "downstream"
            return "peer"
        return None

    t1 = look(graph, kinds, u, v)
    if t1:
        has.add(t1)
    t2 = look(reverse, kinds_rev, v, u)
    if t2:
        has.add(t2)

    if has == {"peer"}:
        return "peer"
    if has == {"upstream"}:
        return "upstream"
    if has == {"downstream"}:
        return "downstream"
    return "mixed"


def collect_edges(
    graph: sparse.csr_matrix,
    reverse: sparse.csr_matrix,
    kinds: np.ndarray,
    kinds_rev: np.ndarray,
    selected: np.ndarray,
    asn_ids: np.ndarray,
) -> list[tuple[int, int, str]]:
    """Return undirected edges (asn_u, asn_v, type) within the selected set."""
    log.info("Collecting edges within %d selected nodes...", len(selected))
    selected_set = set(selected.tolist())
    edges: list[tuple[int, int, str]] = []
    seen: set[tuple[int, int]] = set()
    for i, v in enumerate(selected):
        if i % 5000 == 0 and i > 0:
            log.debug("  ...processed %d/%d nodes, %d edges so far",
                      i, len(selected), len(edges))
        for w in graph.indices[graph.indptr[v]: graph.indptr[v + 1]]:
            if w == v or w not in selected_set:
                continue
            a, b = (v, w) if v < w else (w, v)
            key = (a, b)
            if key in seen:
                continue
            seen.add(key)
            etype = classify_edge(graph, reverse, kinds, kinds_rev, a, b)
            edges.append((int(asn_ids[a]), int(asn_ids[b]), etype))
    log.info("Collected %d undirected edges", len(edges))
    return edges


def bfs_levels(seed_idx: np.ndarray, graph: sparse.csr_matrix,
               reverse: sparse.csr_matrix, n: int) -> np.ndarray:
    """Distance (in hops) from the nearest seed for every vertex, -1 if unreachable."""
    dist = np.full(n, -1, dtype=np.int32)
    dist[seed_idx] = 0
    frontier = seed_idx.copy()
    while frontier.size:
        nbrs = np.concatenate([
            np.concatenate([graph.indices[graph.indptr[v]: graph.indptr[v + 1]]
                            for v in frontier]) if frontier.size else np.empty(0, dtype=np.int64),
            np.concatenate([reverse.indices[reverse.indptr[v]: reverse.indptr[v + 1]]
                            for v in frontier]) if frontier.size else np.empty(0, dtype=np.int64),
        ])
        nbrs = nbrs[nbrs >= 0]
        new = np.unique(nbrs[dist[nbrs] < 0]) if nbrs.size else np.empty(0, dtype=np.int64)
        if new.size == 0:
            break
        cur = int(dist[frontier[0]])
        dist[new] = cur + 1
        frontier = new
    return dist


def static_layout(levels: np.ndarray, node_ids: list, seed_asns: list) -> dict:
    """Deterministic concentric-ring layout: seed at center, each BFS level
    on its own ring. Returns {node_id: (x, y)}."""
    seed_set = set(seed_asns)
    by_level: dict[int, list] = {}
    for asn, lvl in zip(node_ids, levels):
        if asn in seed_set:
            lvl = 0  # seeds always at center
        by_level.setdefault(int(lvl), []).append(asn)

    pos: dict = {}
    for lvl, members in by_level.items():
        radius = lvl * 160
        k = len(members)
        if k == 0:
            continue
        # deterministic order: ASN value (stable across runs)
        for i, asn in enumerate(sorted(members, key=lambda a: int(a))):
            angle = 2.0 * np.pi * i / k
            # jitter angle per level so rings don't align into spokes
            angle += lvl * 0.35
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            pos[str(asn)] = (float(round(x, 2)), float(round(y, 2)))
    return pos


def render_html(edges, seed_asns, out_path: Path,
                static: bool = False, levels: np.ndarray | None = None) -> None:
    node_asns_count = len({a for a, b, _ in edges} | {b for a, b, _ in edges})
    log.info("Rendering HTML: %d nodes, %d edges (static=%s)",
             node_asns_count, len(edges), static)
    # cdn_resources="in_line": vis.js + bindings + данные вшиваются в один файл.
    # Не зависит от расположения lib/ рядом с HTML и от сети в браузере.
    net = Network(
        directed=False, notebook=False,
        height="100vh", width="100vw",
        bgcolor="#ffffff", font_color="#222",
        cdn_resources="in_line",
    )

    # Вся настройка — один JSON. Это обходит incompatibility между
    # set_options/show_buttons/barnes_hut в разных версиях pyvis.
    # Панель "configure" отключена: в vis-network 9.1.2 фильтр
    # {"filter": {"physics": true}} не валиден (ошибка "Invalid type for filter"),
    # а физика задаётся явно ниже — панель функционально ничего не даёт.
    seed_set = set(seed_asns)
    node_asns = set()
    for a, _, _ in edges:
        node_asns.add(a)
    for _, b, _ in edges:
        node_asns.add(b)
    node_asns_list = list(node_asns)

    if static:
        # No physics: nodes stay exactly where we place them.
        net.set_options("""{
            "physics": {"enabled": false},
            "interaction": {
                "hover": true,
                "navigationButtons": true
            }
        }""")
        positions = static_layout(levels, sorted(node_asns_list), seed_asns) \
            if levels is not None else {}
    else:
        net.set_options("""{
            "physics": {
                "barnesHut": {
                    "theta": 0.9,
                    "gravitationalConstant": -5000,
                    "centralGravity": 0.05,
                    "springLength": 50
                }
            },
            "interaction": {
                "hover": true,
                "navigationButtons": true
            }
        }""")
        positions = {}

    log.debug("Adding %d nodes...", len(node_asns))
    for idx, asn in enumerate(sorted(node_asns)):
        if idx % 5000 == 0 and idx > 0:
            log.debug("  ...added %d/%d nodes", idx, len(node_asns))
        is_seed = asn in seed_set
        label = f"AS{asn}" + ("  (seed)" if is_seed else "")
        kwargs = dict(label=label,
                      color=NODE_SEED_COLOR if is_seed else NODE_OTHER_COLOR,
                      title=f"AS{asn}",
                      size=18 if is_seed else 8)
        if str(asn) in positions:
            x, y = positions[str(asn)]
            kwargs["x"] = x
            kwargs["y"] = y
            kwargs["fixed"] = True
        net.add_node(str(asn), **kwargs)

    log.debug("Adding %d edges...", len(edges))
    for idx, (a, b, etype) in enumerate(edges):
        if idx % 5000 == 0 and idx > 0:
            log.debug("  ...added %d/%d edges", idx, len(edges))
        net.add_edge(str(a), str(b),
                     color=EDGE_COLORS[etype],
                     width=EDGE_WIDTHS[etype],
                     title=f"{etype}: AS{a} - AS{b}",
                     physics=not static)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    log.debug("Writing HTML to %s", out_path)
    net.write_html(str(out_path), notebook=False, open_browser=False)
    log.info("HTML written: %s (%.1f KB)", out_path,
             out_path.stat().st_size / 1024)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build neighborhood subgraph around selected AS and render HTML."
    )
    parser.add_argument("--as", dest="as_list", type=int, nargs="+",
                        help="One or more ASN values (overrides DEFAULT_ASNS).")
    parser.add_argument("--as-file", type=Path,
                        help="File with ASN, one per line; # for comments.")
    parser.add_argument("--dir", type=Path, default=Path("data/processed"),
                        help="Snapshot directory (default: data/processed).")
    parser.add_argument("--depth", type=int, default=1,
                        help="BFS hops from the nearest seed (default: 1).")
    parser.add_argument("--out", type=Path, default=Path("out"),
                        help="Output directory for HTML (default: out).")
    parser.add_argument("--html", default="as_neighbors.html",
                        help="HTML filename (default: as_neighbors.html).")
    parser.add_argument("--edge-cutoff", type=int, default=0,
                        help="Keep only edges whose endpoint degree sum >= K "
                             "(0 = keep all; useful for large --depth).")
    parser.add_argument("--static", action="store_true",
                        help="Disable physics and lay out nodes on concentric "
                             "rings by BFS level (seed at center). Image is "
                             "static immediately, no stabilization jitter.")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable debug-level logging.")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress info logs (warnings only).")
    args = parser.parse_args()
    if args.depth < 1:
        raise SystemExit("--depth must be >= 1")

    setup_logging(args.verbose, args.quiet)
    t_start = time.perf_counter()

    # --- Stage 1: Load snapshot ---
    with StageTimer("Load snapshot"):
        seeds = resolve_seeds(args)
        log.info("Seeds: %s", seeds)
        graph, kinds, asn_ids = load_snapshot(args.dir)
        log.info("Snapshot: n_vertices=%d, nnz=%d",
                 graph.shape[0], graph.nnz)

    # --- Stage 2: Build reverse CSR ---
    with StageTimer("Build reverse CSR"):
        reverse, kinds_rev = build_reverse(graph, kinds)

    # --- Stage 3: Resolve seed indices ---
    with StageTimer("Resolve seed indices"):
        seed_idx = np.searchsorted(asn_ids, np.asarray(seeds, dtype=np.int64))
        missing = [a for a, i in zip(seeds, seed_idx)
                   if i >= len(asn_ids) or int(asn_ids[i]) != a]
        if missing:
            raise SystemExit(f"Seeds not found in snapshot: {missing}")
        log.info("Seed indices resolved: %s", seed_idx.tolist())

    # --- Stage 4: BFS ---
    with StageTimer("BFS expansion"):
        selected = bfs_selected(graph, reverse, seed_idx, args.depth)

    # --- Stage 5: Collect edges ---
    with StageTimer("Collect subgraph edges"):
        edges = collect_edges(graph, reverse, kinds, kinds_rev, selected, asn_ids)

    # --- Stage 6: Edge cutoff filter ---
    if args.edge_cutoff > 0:
        with StageTimer(f"Edge cutoff (>= {args.edge_cutoff})"):
            degree: dict[int, int] = {}
            for a, b, _ in edges:
                degree[a] = degree.get(a, 0) + 1
                degree[b] = degree.get(b, 0) + 1
            before = len(edges)
            edges = [(a, b, t) for a, b, t in edges
                     if degree.get(a, 0) + degree.get(b, 0) >= args.edge_cutoff]
            log.info("Edge cutoff: %d -> %d edges", before, len(edges))

    # --- Stage 6.5: BFS levels (for --static ring layout) ---
    levels = None
    if args.static:
        with StageTimer("BFS levels (static layout)"):
            levels = bfs_levels(seed_idx, graph, reverse, graph.shape[0])

    # --- Stage 7: Render HTML ---
    with StageTimer("Render HTML"):
        out_html = args.out / args.html
        render_html(edges, seeds, out_html, static=args.static, levels=levels)

    # --- Summary ---
    total_dt = time.perf_counter() - t_start
    type_counts: dict[str, int] = {}
    for _, _, t in edges:
        type_counts[t] = type_counts.get(t, 0) + 1

    info = {
        "seeds": seeds,
        "depth": args.depth,
        "n_selected_as": int(len(selected)),
        "n_undirected_edges": int(len(edges)),
        "edge_types": type_counts,
        "html": str(out_html),
        "total_seconds": round(total_dt, 2),
    }
    log.info("Total time: %.2fs", total_dt)
    print(json.dumps(info, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
