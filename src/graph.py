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


def read_asn_list(path: Path) -> np.ndarray:
    """Read unique positive ASNs, one decimal number per line."""
    numbers = []
    seen = set()
    with path.open("r", encoding="utf-8") as file:
        for line_no, line in enumerate(file, 1):
            value = line.strip()
            if not value:
                continue
            if not value.isascii() or not value.isdecimal():
                raise ValueError(f"{path}:{line_no}: expected a positive ASN, got {value!r}")
            asn = int(value)
            if asn < 0 or asn > 4294967295:
                raise ValueError(f"{path}:{line_no}: ASN out of range: {asn}")
            if asn in seen:
                raise ValueError(f"{path}:{line_no}: duplicate ASN {asn}")
            seen.add(asn)
            numbers.append(asn)
    if not numbers:
        raise ValueError(f"ASN list is empty: {path}")
    return np.asarray(sorted(numbers), dtype=np.int64)


def induced_subgraph(graph, kinds, asn_ids, wanted_asns):
    """Keep selected CAIDA ASNs and ONLY arcs with both endpoints selected.

    Use edge types as temporary sparse-matrix data during slicing, so that
    graph.indices and edge_types stay aligned even if SciPy reorders entries.
    Isolated selected vertices remain in the result.
    """
    if graph.nnz != len(kinds) or graph.shape[0] != len(asn_ids):
        raise ValueError("Incompatible CSR arrays")
    keep = np.isin(asn_ids, wanted_asns)
    indices = np.flatnonzero(keep)
    if not indices.size:
        raise ValueError("None of the requested ASNs occur in the CAIDA graph")
    retained_asns = asn_ids[indices].copy()
    missing_asns = np.setdiff1d(wanted_asns, retained_asns, assume_unique=True)

    typed_graph = sparse.csr_matrix(
        (kinds.copy(), graph.indices.copy(), graph.indptr.copy()),
        shape=graph.shape,
    )
    subset = typed_graph[indices, :][:, indices].tocsr()
    subset.sort_indices()
    subset_types = subset.data.astype(np.uint8, copy=True)
    subset_graph = sparse.csr_matrix(
        (np.ones(subset.nnz, dtype=np.uint8),
         subset.indices.copy(), subset.indptr.copy()),
        shape=subset.shape,
    )
    return subset_graph, subset_types, retained_asns, missing_asns


def filter_graph(source_dir: Path, asn_file: Path, output_dir: Path) -> dict:
    """Filter an existing CAIDA CSR and save a separate CSR plus JSON metadata."""
    if source_dir.resolve() == output_dir.resolve():
        raise ValueError("Output directory must differ from the original CAIDA CSR directory")
    requested = read_asn_list(asn_file)
    graph, kinds, asn_ids = load_graph(source_dir)
    subgraph, subtypes, subasns, missing = induced_subgraph(
        graph, kinds, asn_ids, requested
    )
    if subgraph.nnz % 2 or (subtypes == UPSTREAM).sum() != (subtypes == DOWNSTREAM).sum() or (subtypes == PEER).sum() % 2:
        raise ValueError("Unexpected asymmetric AS relationship counts in filtered graph")

    # Every relationship is stored in both directions. Count the boundary once.
    inside = np.isin(asn_ids, subasns)
    source_is_inside = np.repeat(inside, np.diff(graph.indptr))
    destination_is_inside = inside[graph.indices]
    boundary_relationships = int(np.count_nonzero(
        source_is_inside != destination_is_inside
    ) // 2)
    total_rels = int(graph.nnz // 2)
    retained_rels = int(subgraph.nnz // 2)
    outside_rels = total_rels - retained_rels - boundary_relationships

    source_metadata_file = source_dir / "metadata.json"
    parent = json.loads(source_metadata_file.read_text(encoding="utf-8")) if source_metadata_file.exists() else {}
    metadata = {
        "format": "csr_aligned_v1",
        "operation": "induced_subgraph_by_asn_list",
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_filename": parent.get("input_filename"),
        "input_sha256": parent.get("input_sha256"),
        "source_graph_dir": str(source_dir),
        "source_graph_vertices": int(len(asn_ids)),
        "source_graph_directed_arcs": int(graph.nnz),
        "source_graph_relationships": total_rels,
        "asn_list_filename": asn_file.name,
        "asn_list_sha256": file_sha256(asn_file),
        "requested_asns": int(len(requested)),
        "found_asns": int(len(subasns)),
        "missing_asns_count": int(len(missing)),
        "missing_asns": [int(asn) for asn in missing],
        "n_vertices": int(len(subasns)),
        "n_directed_arcs": int(subgraph.nnz),
        "n_unique_as_relationships": retained_rels,
        "provider_customer_relationships": int((subtypes == UPSTREAM).sum()),
        "peer_relationships": int((subtypes == PEER).sum() // 2),
        "directed_arc_types": {
            TYPE_NAMES[t]: int((subtypes == t).sum()) for t in TYPE_NAMES
        },
        "type_codes": {TYPE_NAMES[t]: t for t in TYPE_NAMES},
        "boundary_relationships_removed": boundary_relationships,
        "outside_outside_relationships_removed": outside_rels,
        "isolated_vertices": int(np.count_nonzero(np.diff(subgraph.indptr) == 0)),
        "note": "Only original CAIDA arcs between retained ASNs; no new edges. Original edge source field is not preserved.",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    sparse.save_npz(output_dir / "graph.npz", subgraph)
    np.save(output_dir / "edge_types.npy", subtypes)
    np.save(output_dir / "asn_ids.npy", subasns)
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return metadata


def main():
    parser = argparse.ArgumentParser(description="Build/read a CAIDA serial-2 CSR graph")
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build", help="Parse CAIDA AS relationships and save CSR")
    build.add_argument("input", type=Path, help="*.as-rel2.txt or *.as-rel2.txt.gz")
    build.add_argument("--out", type=Path, default=Path("data/processed"))
    filter_cmd = sub.add_parser("filter", help="Build an induced subgraph from an ASN list")
    filter_cmd.add_argument("--asns", type=Path, required=True, help="Text file: one ASN per line")
    filter_cmd.add_argument("--dir", type=Path, default=Path("data/processed"), help="Source CAIDA CSR directory")
    filter_cmd.add_argument("--out", type=Path, default=Path("data/filtered"), help="Filtered CSR directory")
    inspect = sub.add_parser("inspect", help="Verify CSR files and print basic stats")
    inspect.add_argument("--dir", type=Path, default=Path("data/processed"))
    args = parser.parse_args()
    if args.command == "build":
        info = save_graph(args.input, args.out)
        print(json.dumps(info, indent=2, ensure_ascii=False))
    elif args.command == "filter":
        info = filter_graph(args.dir, args.asns, args.out)
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
