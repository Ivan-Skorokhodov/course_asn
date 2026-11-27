"""Remove isolated ASNs from an existing CSR graph without modifying its source."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
from scipy import sparse

from src.graph import DOWNSTREAM, PEER, TYPE_NAMES, UPSTREAM, file_sha256, load_graph


OUTPUT_FILES = ("graph.npz", "edge_types.npy", "asn_ids.npy", "metadata.json")


def prune_isolated(graph, edge_types, asn_ids):
    """Return a new CSR, aligned types, ASN IDs, and removed zero-degree ASNs.

    Isolated means no incoming AND no outgoing edges. The edge types are put
    into sparse-matrix data during slicing to preserve their CSR alignment.
    """
    if graph.shape != (len(asn_ids), len(asn_ids)) or graph.nnz != len(edge_types):
        raise ValueError("Incompatible CSR graph, edge types and ASN IDs")

    out_degree = np.diff(graph.indptr)
    in_degree = np.bincount(graph.indices, minlength=len(asn_ids))
    keep = (out_degree + in_degree) > 0
    removed_asns = asn_ids[~keep].copy()
    kept_asns = asn_ids[keep].copy()
    if not len(kept_asns):
        raise ValueError("Source graph contains no non-isolated vertices")

    typed_graph = sparse.csr_matrix(
        (edge_types.copy(), graph.indices.copy(), graph.indptr.copy()),
        shape=graph.shape,
    )
    kept_indices = np.flatnonzero(keep)
    subset = typed_graph[kept_indices, :][:, kept_indices].tocsr()
    subset.sort_indices()
    kept_types = subset.data.astype(np.uint8, copy=True)
    clean_graph = sparse.csr_matrix(
        (np.ones(subset.nnz, dtype=np.uint8),
         subset.indices.copy(), subset.indptr.copy()),
        shape=subset.shape,
    )

    # Deleting truly isolated vertices must never delete an edge.
    if clean_graph.nnz != graph.nnz:
        raise ValueError("Unexpected edge loss: a supposedly isolated vertex has edges")
    if np.any(np.diff(clean_graph.indptr) + np.bincount(
        clean_graph.indices, minlength=len(kept_asns)
    ) == 0):
        raise ValueError("The result still contains isolated vertices")
    return clean_graph, kept_types, kept_asns, removed_asns


def prune_graph(source_dir: Path, output_dir: Path) -> dict:
    """Save a separate CSR snapshot and JSON metadata; never overwrite a graph."""
    if source_dir.resolve() == output_dir.resolve():
        raise ValueError("Source and output directories must differ")
    if output_dir.exists() and any(
        entry.name != ".gitkeep" for entry in output_dir.iterdir()
    ):
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}. "
            "Choose a new --out path to keep earlier graph snapshots."
        )

    graph, edge_types, asn_ids = load_graph(source_dir)
    parent_path = source_dir / "metadata.json"
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    clean, clean_types, clean_asns, removed_asns = prune_isolated(
        graph, edge_types, asn_ids
    )

    upstream = int(np.count_nonzero(clean_types == UPSTREAM))
    downstream = int(np.count_nonzero(clean_types == DOWNSTREAM))
    peers = int(np.count_nonzero(clean_types == PEER))
    if upstream != downstream or peers % 2 or clean.nnz % 2:
        raise ValueError("Unexpected asymmetric AS relationship counts")

    metadata = {
        "format": "csr_aligned_v1",
        "operation": "prune_isolated_vertices",
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_filename": parent.get("input_filename"),
        "input_sha256": parent.get("input_sha256"),
        "asn_list_filename": parent.get("asn_list_filename"),
        "asn_list_sha256": parent.get("asn_list_sha256"),
        "source_graph_dir": str(source_dir),
        "source_graph_operation": parent.get("operation", "build"),
        "source_metadata_sha256": file_sha256(parent_path),
        "source_files_sha256": {
            name: file_sha256(source_dir / name)
            for name in ("graph.npz", "edge_types.npy", "asn_ids.npy")
        },
        "source_graph_vertices": int(len(asn_ids)),
        "source_graph_directed_arcs": int(graph.nnz),
        "source_graph_relationships": int(graph.nnz // 2),
        "removed_isolated_vertices": int(len(removed_asns)),
        "removed_isolated_asns": [int(asn) for asn in removed_asns],
        "n_vertices": int(len(clean_asns)),
        "n_directed_arcs": int(clean.nnz),
        "n_unique_as_relationships": int(clean.nnz // 2),
        "provider_customer_relationships": upstream,
        "peer_relationships": peers // 2,
        "directed_arc_types": {
            TYPE_NAMES[t]: int(np.count_nonzero(clean_types == t))
            for t in (UPSTREAM, DOWNSTREAM, PEER)
        },
        "type_codes": {TYPE_NAMES[t]: t for t in (UPSTREAM, DOWNSTREAM, PEER)},
        "isolated_vertices": 0,
        "note": (
            "Only zero-degree vertices were removed. All arcs and relation types "
            "were preserved. This does not guarantee a single connected component. "
            "Original CAIDA edge source provenance is not in CSR."
        ),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    sparse.save_npz(output_dir / "graph.npz", clean)
    np.save(output_dir / "edge_types.npy", clean_types)
    np.save(output_dir / "asn_ids.npy", clean_asns)
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return metadata


def main():
    parser = argparse.ArgumentParser(
        description="Remove isolated vertices from an existing CAIDA CSR snapshot"
    )
    parser.add_argument("--dir", type=Path, default=Path("data/filtered"),
                        help="Source filtered CSR directory")
    parser.add_argument("--out", type=Path, default=Path("data/pruned"),
                        help="New directory for the cleaned CSR (never overwritten)")
    args = parser.parse_args()
    info = prune_graph(args.dir, args.out)
    print(json.dumps(info, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
