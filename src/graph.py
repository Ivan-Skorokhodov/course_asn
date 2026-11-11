"""Parse CAIDA serial-2 AS relationships and store an aligned CSR graph."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy import sparse

# Types for directed arcs: customer -> provider, provider -> customer, peer -> peer.
UPSTREAM = 1
DOWNSTREAM = 2
PEER = 3
TYPE_NAMES = {UPSTREAM: "upstream", DOWNSTREAM: "downstream", PEER: "peer"}


def open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("rt", encoding="utf-8")


def read_relationships(path: Path):
    """Return directed arcs (ASN, ASN, type) and counts; reject malformed input."""
    arcs = []
    counts = Counter()
    with open_text(path) as file:
        for line_number, line in enumerate(file, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split("|")
            if len(fields) < 3:
                raise ValueError(f"Line {line_number}: expected at least 3 columns")
            try:
                a, b = int(fields[0]), int(fields[1])
            except ValueError as exc:
                raise ValueError(f"Line {line_number}: invalid ASN") from exc
            rel = fields[2]
            if a <= 0 or b <= 0 or a == b:
                raise ValueError(f"Line {line_number}: invalid AS pair {a}, {b}")
            if rel == "-1":
                # CAIDA: provider|customer|-1; add both directions.
                arcs.extend(((a, b, DOWNSTREAM), (b, a, UPSTREAM)))
                counts["provider_customer_rows"] += 1
            elif rel == "0":
                arcs.extend(((a, b, PEER), (b, a, PEER)))
                counts["peer_rows"] += 1
            else:
                raise ValueError(f"Line {line_number}: unsupported relationship {rel!r}")
    if not arcs:
        raise ValueError("Input contains no relationships")
    return np.asarray(arcs, dtype=np.int64), counts


def build_graph(arcs: np.ndarray):
    """Return (CSR topology, aligned edge types, ASN mapping).

    Duplicate identical arcs are removed; conflicting relation types raise an error.
    """
    asn_ids = np.unique(arcs[:, :2])
    src = np.searchsorted(asn_ids, arcs[:, 0]).astype(np.int32)
    dst = np.searchsorted(asn_ids, arcs[:, 1]).astype(np.int32)
    kinds = arcs[:, 2].astype(np.uint8)
    order = np.lexsort((dst, src))
    src, dst, kinds = src[order], dst[order], kinds[order]

    same_pair = (src[1:] == src[:-1]) & (dst[1:] == dst[:-1])
    conflict = same_pair & (kinds[1:] != kinds[:-1])
    if np.any(conflict):
        i = int(np.flatnonzero(conflict)[0])
        raise ValueError(
            f"Conflicting relationships for AS{asn_ids[src[i]]} -> AS{asn_ids[dst[i]]}"
        )
    keep = np.ones(len(src), dtype=bool)
    keep[1:] = ~same_pair
    src, dst, kinds = src[keep], dst[keep], kinds[keep]

    n = len(asn_ids)
    indptr = np.empty(n + 1, dtype=np.int32)
    indptr[0] = 0
    indptr[1:] = np.cumsum(np.bincount(src, minlength=n), dtype=np.int32)
    graph = sparse.csr_matrix(
        (np.ones(len(dst), dtype=np.uint8), dst, indptr), shape=(n, n)
    )
    # Explicitly keep the same sorted order in both arrays.
    assert graph.has_sorted_indices and graph.nnz == len(kinds)
    return graph, kinds, asn_ids


def file_sha256(path: Path):
    sha = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def save_graph(input_file: Path, output_dir: Path):
    arcs, counts = read_relationships(input_file)
    graph, kinds, asn_ids = build_graph(arcs)
    output_dir.mkdir(parents=True, exist_ok=True)
    sparse.save_npz(output_dir / "graph.npz", graph)
    np.save(output_dir / "edge_types.npy", kinds)
    np.save(output_dir / "asn_ids.npy", asn_ids)
    metadata = {
        "input_filename": input_file.name,
        "input_sha256": file_sha256(input_file),
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "format": "csr_aligned_v1",
        "n_vertices": int(graph.shape[0]),
        "n_directed_arcs": int(graph.nnz),
        "n_unique_as_relationships": int(graph.nnz // 2),
        "input_rows": dict(counts),
        "directed_arc_types": {TYPE_NAMES[t]: int((kinds == t).sum()) for t in TYPE_NAMES},
        "type_codes": {TYPE_NAMES[t]: t for t in TYPE_NAMES},
        "note": "Peer/source provenance column is not stored; types aligned with CSR indices.",
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return metadata


def load_graph(directory: Path):
    graph = sparse.load_npz(directory / "graph.npz").tocsr()
    kinds = np.load(directory / "edge_types.npy", allow_pickle=False)
    asn_ids = np.load(directory / "asn_ids.npy", allow_pickle=False)
    if graph.nnz != len(kinds) or graph.shape[0] != len(asn_ids):
        raise ValueError("Graph files are incompatible")
    return graph, kinds, asn_ids


def make_weighted_graph(graph, kinds, alpha: float):
    """Create directed weighted CSR; zero-cost edges are not used."""
    if alpha < 0 or not np.isfinite(alpha):
        raise ValueError("alpha must be finite and nonnegative")
    weights = np.ones(graph.nnz, dtype=np.float64)
    weights[kinds == UPSTREAM] += alpha
    return sparse.csr_matrix(
        (weights, graph.indices.copy(), graph.indptr.copy()), shape=graph.shape
    )


def main():
    parser = argparse.ArgumentParser(description="Build/read a CAIDA serial-2 CSR graph")
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build", help="Parse CAIDA AS relationships and save CSR")
    build.add_argument("input", type=Path, help="*.as-rel2.txt or *.as-rel2.txt.gz")
    build.add_argument("--out", type=Path, default=Path("data/processed"))
    inspect = sub.add_parser("inspect", help="Verify CSR files and print basic stats")
    inspect.add_argument("--dir", type=Path, default=Path("data/processed"))
    args = parser.parse_args()
    if args.command == "build":
        info = save_graph(args.input, args.out)
        print(json.dumps(info, indent=2, ensure_ascii=False))
    else:
        graph, types, asn = load_graph(args.dir)
        print(f"AS: {len(asn)}; directed arcs: {graph.nnz}")
        print("Arc types:", {TYPE_NAMES[t]: int(np.sum(types == t)) for t in TYPE_NAMES})
        meta = args.dir / "metadata.json"
        if meta.exists():
            print("Snapshot:", json.loads(meta.read_text(encoding="utf-8"))["input_filename"])


if __name__ == "__main__":
    main()
