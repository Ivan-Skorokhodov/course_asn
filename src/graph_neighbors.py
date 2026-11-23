"""Explore CAIDA AS neighborhoods as a BFS discovery tree.

Reads the existing CSR snapshot produced by src.graph. During BFS, an edge is
saved only when it discovers a new AS: cycles and redundant links never enter
this working tree. The original CSR files are not modified.

Examples:
    python -m src.graph_neighbors --as 43832 --depth 1 --edge-cutoff 25 --static
    python -m src.graph_neighbors --as 42385 --depth 2 --max-nodes 1000 --static
    python -m src.graph_neighbors --as 15169 13335 --depth 2 --static

--edge-cutoff now checks the sum of FULL CSR degrees of edge endpoints BEFORE
adding a node (not degrees in a fully collected neighborhood). This avoids
building the dense induced subgraph just to calculate its degree counts.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections import deque
from pathlib import Path

import numpy as np
from scipy import sparse
from pyvis.network import Network

log = logging.getLogger("graph_neighbors")

# Codes in src.graph; arcs are stored in both directions.
UPSTREAM = 1    # customer -> provider
DOWNSTREAM = 2  # provider -> customer
PEER = 3        # peer -> peer
TYPE_NAMES = {UPSTREAM: "upstream", DOWNSTREAM: "downstream", PEER: "peer"}

DEFAULT_ASNS = [42385]

NODE_SEED_COLOR = "#e74c3c"
NODE_OTHER_COLOR = "#95a5a6"
EDGE_COLORS = {
    "peer": "#2e86de",
    "upstream": "#27ae60",
    "downstream": "#e67e22",
}
EDGE_WIDTHS = {"peer": 2, "upstream": 3, "downstream": 3}


class StageTimer:
    def __init__(self, name: str):
        self.name = name

    def __enter__(self):
        self.started = time.perf_counter()
        log.info("=== %s ===", self.name)
        return self

    def __exit__(self, exc_type, exc, tb):
        duration = time.perf_counter() - self.started
        if exc_type is None:
            log.info("%s: %.2f s", self.name, duration)
        else:
            log.error("%s: %.2f s (%s)", self.name, duration, exc)
        return False


def setup_logging(verbose: bool, quiet: bool) -> None:
    level = logging.DEBUG if verbose else logging.WARNING if quiet else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def load_snapshot(directory: Path):
    graph = sparse.load_npz(directory / "graph.npz").tocsr()
    kinds = np.load(directory / "edge_types.npy", allow_pickle=False)
    asn_ids = np.load(directory / "asn_ids.npy", allow_pickle=False)
    if graph.nnz != len(kinds) or graph.shape[0] != len(asn_ids):
        raise ValueError("Graph files are incompatible")
    if not graph.has_sorted_indices:
        raise ValueError("CSR indices must be sorted: edge types are position-aligned")
    if np.any(asn_ids[1:] <= asn_ids[:-1]):
        raise ValueError("asn_ids.npy must contain sorted unique ASN values")
    return graph, kinds, asn_ids


def resolve_seeds(args) -> list[int]:
    seeds = list(args.as_list or [])
    if args.as_file:
        with args.as_file.open("r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if line and not line.startswith("#"):
                    seeds.append(int(line))
    if not seeds:
        seeds = list(DEFAULT_ASNS)
    return list(dict.fromkeys(seeds))


def seed_indices(seeds: list[int], asn_ids: np.ndarray) -> list[int]:
    indices = np.searchsorted(asn_ids, np.asarray(seeds, dtype=np.int64))
    missing = [
        asn for asn, idx in zip(seeds, indices)
        if idx >= len(asn_ids) or int(asn_ids[idx]) != asn
    ]
    if missing:
        raise ValueError(f"Seeds not found in snapshot: {missing}")
    return [int(i) for i in indices]


def bfs_tree(
    graph: sparse.csr_matrix,
    kinds: np.ndarray,
    asn_ids: np.ndarray,
    roots: list[int],
    depth: int,
    edge_cutoff: int = 0,
    max_nodes: int = 2000,
) -> tuple[list[tuple[int, int, str]], dict[int, int], bool]:
    """Explore original CSR once, keeping only edges that discover new nodes.

    Returns (tree_edges, levels_by_asn, truncated). For multiple roots, the
    result is a BFS forest. max_nodes=0 disables the node cap.
    """
    if depth < 1:
        raise ValueError("depth must be >= 1")
    if edge_cutoff < 0 or max_nodes < 0:
        raise ValueError("edge_cutoff and max_nodes must be >= 0")
    if max_nodes and len(roots) > max_nodes:
        raise ValueError("max_nodes must be >= number of roots")

    n = graph.shape[0]
    visited = np.zeros(n, dtype=np.bool_)
    levels: dict[int, int] = {}
    queue: deque[tuple[int, int]] = deque()

    for i in roots:
        visited[i] = True
        asn = int(asn_ids[i])
        levels[asn] = 0
        queue.append((i, 0))

    # Each original CAIDA relationship has two directed arcs, so scanning a
    # single CSR row gives all neighbors; reverse CSR is unnecessary.
    degrees = np.diff(graph.indptr) if edge_cutoff else None
    edges: list[tuple[int, int, str]] = []
    truncated = False

    while queue:
        parent, parent_depth = queue.popleft()
        if parent_depth >= depth:
            continue

        parent_asn = int(asn_ids[parent])
        for offset in range(int(graph.indptr[parent]), int(graph.indptr[parent + 1])):
            child = int(graph.indices[offset])
            if visited[child]:
                continue
            if degrees is not None and int(degrees[parent]) + int(degrees[child]) < edge_cutoff:
                continue

            if max_nodes and len(levels) >= max_nodes:
                truncated = True
                break

            child_asn = int(asn_ids[child])
            edge_type = TYPE_NAMES.get(int(kinds[offset]))
            if edge_type is None:
                raise ValueError(f"Unknown edge type {int(kinds[offset])}")

            visited[child] = True
            levels[child_asn] = parent_depth + 1
            edges.append((parent_asn, child_asn, edge_type))
            queue.append((child, parent_depth + 1))

        if truncated:
            break

    log.info("BFS tree: %d AS, %d edges%s",
             len(levels), len(edges), ", node limit reached" if truncated else "")
    return edges, levels, truncated


def render_html(
    edges: list[tuple[int, int, str]],
    levels: dict[int, int],
    seed_asns: list[int],
    out_path: Path,
    static: bool,
) -> None:
    net = Network(
        directed=False,
        notebook=False,
        height="100vh",
        width="100vw",
        bgcolor="#ffffff",
        font_color="#222",
        cdn_resources="in_line",
    )

    # Let vis-network arrange the BFS discovery tree from top to bottom.
    # The levels come directly from BFS; no manual coordinates are needed.
    options = {
        "layout": {
            "hierarchical": {
                "enabled": True,
                "direction": "UD",
                "sortMethod": "directed",
                "levelSeparation": 150,
                "nodeSpacing": 180,
                "treeSpacing": 220,
                "blockShifting": True,
                "edgeMinimization": True,
                "parentCentralization": True,
            }
        },
        "physics": {"enabled": not static},
        "edges": {"smooth": False},
        "interaction": {"hover": True, "navigationButtons": True},
    }
    if not static:
        options["physics"].update({
            "solver": "hierarchicalRepulsion",
            "hierarchicalRepulsion": {"nodeDistance": 180},
        })
    net.set_options(json.dumps(options))

    root_set = set(seed_asns)
    for asn in sorted(levels):
        is_root = asn in root_set
        options = {
            "label": f"AS{asn}" + ("  (seed)" if is_root else ""),
            "title": f"AS{asn}, depth={levels[asn]}",
            "color": NODE_SEED_COLOR if is_root else NODE_OTHER_COLOR,
            "size": 18 if is_root else 8,
            "level": levels[asn],
        }
        net.add_node(str(asn), **options)

    for parent, child, edge_type in edges:
        net.add_edge(
            str(parent), str(child),
            color=EDGE_COLORS[edge_type],
            width=EDGE_WIDTHS[edge_type],
            title=f"AS{parent} → AS{child}: {edge_type}",
            physics=not static,
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    net.write_html(str(out_path), notebook=False, open_browser=False)
    log.info("HTML saved: %s (%.1f KiB)", out_path,
             out_path.stat().st_size / 1024)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a BFS tree of an AS neighborhood and render HTML."
    )
    parser.add_argument("--as", dest="as_list", type=int, nargs="+",
                        help="Seed ASNs (defaults to DEFAULT_ASNS)")
    parser.add_argument("--as-file", type=Path,
                        help="Path with one seed ASN per line")
    parser.add_argument("--dir", type=Path, default=Path("data/processed"),
                        help="Directory with graph.npz, edge_types.npy and asn_ids.npy")
    parser.add_argument("--depth", type=int, default=1,
                        help="BFS depth (default: 1)")
    parser.add_argument("--out", type=Path, default=Path("out"),
                        help="HTML output directory")
    parser.add_argument("--html", default="as_neighbors.html",
                        help="HTML filename")
    parser.add_argument("--edge-cutoff", type=int, default=0,
                        help="Ignore edges if endpoint FULL-graph degree sum < K")
    parser.add_argument("--max-nodes", type=int, default=2000,
                        help="Maximum AS in tree (default: 2000; 0 = no limit)")
    parser.add_argument("--static", action="store_true",
                        help="Show a top-down hierarchical tree without physics")
    parser.add_argument("--verbose", action="store_true",
                        help="Verbose logs")
    parser.add_argument("--quiet", action="store_true",
                        help="Only warnings and errors")
    args = parser.parse_args()

    if args.depth < 1 or args.edge_cutoff < 0 or args.max_nodes < 0:
        parser.error("--depth must be >= 1; --edge-cutoff and --max-nodes >= 0")

    setup_logging(args.verbose, args.quiet)
    started = time.perf_counter()

    with StageTimer("Load CSR"):
        graph, kinds, asn_ids = load_snapshot(args.dir)
        seeds = resolve_seeds(args)
        try:
            roots = seed_indices(seeds, asn_ids)
        except ValueError as exc:
            parser.error(str(exc))
        log.info("Graph: %d AS, %d directed arcs; seeds: %s",
                 graph.shape[0], graph.nnz, seeds)

    with StageTimer("Build BFS tree"):
        edges, levels, truncated = bfs_tree(
            graph, kinds, asn_ids, roots, args.depth,
            args.edge_cutoff, args.max_nodes,
        )

    with StageTimer("Render HTML"):
        output = args.out / args.html
        render_html(edges, levels, seeds, output, args.static)

    counts = {edge_type: sum(t == edge_type for _, _, t in edges)
              for edge_type in EDGE_COLORS}
    info = {
        "seeds": seeds,
        "depth": args.depth,
        "n_selected_as": len(levels),
        "n_undirected_edges": len(edges),
        "edge_types": counts,
        "truncated": truncated,
        "max_nodes": args.max_nodes,
        "html": str(output),
        "total_seconds": round(time.perf_counter() - started, 2),
    }
    print(json.dumps(info, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
