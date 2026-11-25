"""Extract a rank-threshold induced AS subgraph from the existing CAIDA CSR.

Run: python -m src.asrank_subgraph --as 42385 43832

The official CAIDA AS Rank threshold is the WORSE (larger numeric) rank of
both requested ASNs; all ASNs with rank <= threshold are retained. This is
an induced subgraph: it keeps every original CAIDA relationship between the
retained ASNs, preserving aligned arc types and both directions.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components

from src.graph import DOWNSTREAM, PEER, UPSTREAM, load_graph

API_URL = "https://api.asrank.caida.org/v2/graphql"
log = logging.getLogger("asrank_subgraph")


def graphql(query: str, timeout: float = 40.0, retries: int = 2) -> dict:
    """Call AS Rank GraphQL with bounded retry; raise on API/HTTP errors."""
    request = Request(
        API_URL,
        data=json.dumps({"query": query}).encode("utf-8"),
        headers={"Content-Type": "application/json", "User-Agent": "course-asn-research/1.0"},
        method="POST",
    )
    for attempt in range(retries + 1):
        try:
            with urlopen(request, timeout=timeout) as response:
                payload = json.load(response)
            if payload.get("errors"):
                raise ValueError(f"AS Rank GraphQL error: {payload['errors']}")
            if not isinstance(payload.get("data"), dict):
                raise ValueError(f"Invalid AS Rank response: {str(payload)[:200]}")
            return payload["data"]
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            if isinstance(exc, HTTPError) and exc.code not in (408, 429, 500, 502, 503, 504):
                raise RuntimeError(f"AS Rank HTTP {exc.code}: {exc.reason}") from exc
            if attempt == retries:
                raise RuntimeError(f"AS Rank unavailable after {retries + 1} attempts: {exc}") from exc
            pause = 2 ** attempt
            log.warning("AS Rank request failed: %s; retry in %d s", exc, pause)
            time.sleep(pause)
    raise AssertionError("unreachable")


def cached_query(cache_file: Path, query: str, refresh: bool, timeout: float, retries: int) -> dict:
    if cache_file.exists() and not refresh:
        saved = json.loads(cache_file.read_text(encoding="utf-8"))
        if saved.get("query") == query and isinstance(saved.get("data"), dict):
            return saved["data"]
    data = graphql(query, timeout=timeout, retries=retries)
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps({
        "query": query, "retrieved_utc": datetime.now(timezone.utc).isoformat(), "data": data,
    }, ensure_ascii=False), encoding="utf-8")
    return data


def fetch_targets(asns: list[int], cache: Path, refresh: bool, timeout: float, retries: int) -> dict[int, dict]:
    fields = "asn rank asnName cone { numberAsns }"
    query = "{ " + " ".join(
        f'a{i}: asn(asn:"{asn}") {{ {fields} }}' for i, asn in enumerate(asns)
    ) + " }"
    data = cached_query(cache / ("targets_" + "_".join(map(str, asns)) + ".json"),
                        query, refresh, timeout, retries)
    targets = {}
    for i, asn in enumerate(asns):
        row = data.get(f"a{i}")
        if row is None or row.get("rank") is None:
            raise ValueError(f"No AS Rank record/rank for AS{asn}; cannot set an exact threshold")
        if int(row["asn"]) != asn:
            raise ValueError(f"AS Rank returned incorrect ASN for AS{asn}: {row}")
        targets[asn] = row
    return targets


def parse_rank_page(data: dict) -> list[dict]:
    edges = (data.get("asns") or {}).get("edges")
    if not isinstance(edges, list):
        raise ValueError("Invalid AS Rank page: missing asns.edges")
    out = []
    for item in edges:
        row = item.get("node") if isinstance(item, dict) else None
        if not row or row.get("asn") is None or row.get("rank") is None:
            raise ValueError("AS Rank page contains ASN without a rank")
        out.append(row)
    if any(int(a["rank"]) > int(b["rank"]) for a, b in zip(out, out[1:])):
        raise ValueError("AS Rank results are not sorted by ascending rank")
    return out


def fetch_ranked_asns(threshold: int, cache: Path, refresh: bool,
                      timeout: float, retries: int, page_size: int = 200) -> dict[int, dict]:
    """Page ranked ASN list until the first rank strictly worse than threshold."""
    chosen: dict[int, dict] = {}
    offset = 0
    last_rank = 0
    while True:
        query = ("{ asns(first: %d, offset: %d) { edges { node { "
                 "asn rank asnName cone { numberAsns } } } } }" % (page_size, offset))
        page_data = cached_query(cache / f"page_{page_size}_{offset:06d}.json",
                                 query, refresh, timeout, retries)
        rows = parse_rank_page(page_data)
        if not rows:
            log.info("AS Rank: reached end of ranking at offset=%d", offset)
            break
        if int(rows[0]["rank"]) < last_rank:
            raise ValueError("AS Rank rank order differs across pages; retry with --refresh")
        last_rank = int(rows[-1]["rank"])
        for row in rows:
            rank = int(row["rank"])
            if rank > threshold:
                log.info("AS Rank: completed at rank %d > threshold %d", rank, threshold)
                return chosen
            asn = int(row["asn"])
            if asn in chosen:
                raise ValueError(f"Repeated AS{asn} in paginated AS Rank; retry with --refresh")
            chosen[asn] = row
        offset += len(rows)
        if offset % 1000 < page_size:
            log.info("AS Rank: loaded %d ASNs; last rank=%d; threshold=%d",
                     len(chosen), last_rank, threshold)
        if len(rows) < page_size:
            break
    return chosen


def induced_rank_subgraph(graph: sparse.csr_matrix, kinds: np.ndarray,
                         asn_ids: np.ndarray, ranked: dict[int, dict]):
    """Return CSR, aligned arc kinds, ASN IDs, original indices, and boundary arc count."""
    keep_mask = np.fromiter((int(asn) in ranked for asn in asn_ids),
                            dtype=np.bool_, count=len(asn_ids))
    selected = np.flatnonzero(keep_mask)
    # Slice arc types AS A SEPARATE CSR, not a raw kind array. Both slices
    # thus have identical row/column ordering even if SciPy changes layout.
    kinds_csr = sparse.csr_matrix((kinds, graph.indices, graph.indptr), shape=graph.shape)
    subgraph = graph[selected, :][:, selected].tocsr()
    subtypes = kinds_csr[selected, :][:, selected].tocsr()
    if not (np.array_equal(subgraph.indptr, subtypes.indptr) and
            np.array_equal(subgraph.indices, subtypes.indices)):
        raise AssertionError("Subgraph CSR and edge types are misaligned")
    if subgraph.nnz != len(subtypes.data):
        raise AssertionError("Unexpected arc count after induced CSR extraction")
    # The source CAIDA snapshot stores both directions of each relationship.
    total_outgoing = int(np.diff(graph.indptr)[selected].sum())
    cross_boundary = total_outgoing - int(subgraph.nnz)
    return subgraph, subtypes.data.astype(np.uint8), asn_ids[selected], selected, cross_boundary


def validate_bidirectional(graph: sparse.csr_matrix, kinds: np.ndarray) -> None:
    """Check that reverse CAIDA arcs and their relationship types are consistent."""
    types_csr = sparse.csr_matrix((kinds, graph.indices, graph.indptr), shape=graph.shape)
    reverse = types_csr.transpose().tocsr()
    if not (types_csr != 0).nnz == (reverse != 0).nnz:
        raise ValueError("Unexpected number of reverse arcs")
    # UPSTREAM=1, DOWNSTREAM=2 should mirror to 2 and 1, PEER=3 to 3.
    # Verify by checking the matrix of encoded reciprocal pairs.
    # Vectorized: types_csr + reverse should be 3 for p2c arcs or 6 for peers.
    pairs = (types_csr + reverse).data
    if np.any((pairs != 3) & (pairs != 6)):
        raise ValueError("Non-reciprocal or inconsistent relationship types")


def statistics(subgraph: sparse.csr_matrix, kinds: np.ndarray, ids: np.ndarray,
               full_graph: sparse.csr_matrix, boundary_arcs: int) -> dict:
    n = int(subgraph.shape[0]); arcs = int(subgraph.nnz)
    components, labels = connected_components(subgraph, directed=True, connection="weak") if n else (0, np.array([], dtype=int))
    counts = np.bincount(labels, minlength=components) if n else np.array([], dtype=int)
    pc_up = int(np.count_nonzero(kinds == UPSTREAM))
    pc_down = int(np.count_nonzero(kinds == DOWNSTREAM))
    peer_arcs = int(np.count_nonzero(kinds == PEER))
    if pc_up != pc_down or peer_arcs % 2 or pc_up + pc_down + peer_arcs != arcs:
        raise ValueError("Unexpected relationship type counts in the induced subgraph")
    degrees = np.diff(subgraph.indptr)
    return {
        "vertices": n, "directed_arcs": arcs, "undirected_relationships": arcs // 2,
        "provider_customer_relationships": pc_up, "peer_relationships": peer_arcs // 2,
        "boundary_relationships_removed": int(boundary_arcs),
        "weak_components": int(components),
        "largest_component_vertices": int(counts.max()) if counts.size else 0,
        "isolated_vertices": int(np.count_nonzero(degrees == 0)),
        "avg_undirected_degree": round(float(degrees.mean()), 4) if n else 0.0,
        "max_undirected_degree": int(degrees.max()) if n else 0,
        "pct_full_vertices": round(100 * n / full_graph.shape[0], 3),
        "pct_full_relationships": round(100 * arcs / full_graph.nnz, 3),
    }


def save_experiment(output: Path, graph: sparse.csr_matrix, kinds: np.ndarray,
                    asn_ids: np.ndarray, rows: dict[int, dict], summary: dict) -> None:
    output.mkdir(parents=True, exist_ok=True)
    sparse.save_npz(output / "graph.npz", graph)
    np.save(output / "edge_types.npy", kinds)
    np.save(output / "asn_ids.npy", asn_ids)
    with (output / "selected_as.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["asn", "asrank_rank", "asrank_cone_asns", "asrank_name"])
        for asn in asn_ids:
            row = rows[int(asn)]
            writer.writerow([int(asn), row["rank"], (row.get("cone") or {}).get("numberAsns", ""), row.get("asnName") or ""])
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                                         encoding="utf-8")
    # Same format as src.graph.save_graph, so src.graph inspect works.
    metadata = {
        "format": "csr_aligned_v1", "source": "rank_threshold_induced_subgraph",
        "n_vertices": int(graph.shape[0]), "n_directed_arcs": int(graph.nnz),
        "n_unique_as_relationships": int(graph.nnz // 2),
        "type_codes": {"upstream": UPSTREAM, "downstream": DOWNSTREAM, "peer": PEER},
        "rank_threshold": summary["rank_threshold"],
        "targets": summary["targets"],
        "note": "Original CAIDA directed relationship types preserved; rank source is live/cached AS Rank",
    }
    metadata["input_filename"] = "CAIDA CSR (rank-threshold induced subgraph)"
    (output / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
                                          encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build induced CAIDA subgraph of all ASNs with AS Rank >= target quality")
    parser.add_argument("--as", dest="targets", nargs="+", type=int, default=[42385, 43832],
                        help="Target ASNs (default: 42385 43832)")
    parser.add_argument("--dir", type=Path, default=Path("data/processed"),
                        help="Existing full CAIDA CSR directory")
    parser.add_argument("--out", type=Path, default=Path("out/asrank_subgraph"),
                        help="Output subgraph directory (must differ from --dir)")
    parser.add_argument("--cache", type=Path, default=Path("data/analysis/.cache/asrank_subgraph"),
                        help="Cache of GraphQL responses, reused on subsequent runs")
    parser.add_argument("--ranks-only", action="store_true",
                        help="Print the target AS Rank values without downloading/building the subgraph")
    parser.add_argument("--page-size", type=int, default=200,
                        help="Number of AS Rank records per GraphQL call (default 200)")
    parser.add_argument("--timeout", type=float, default=40,
                        help="HTTPS request timeout in seconds")
    parser.add_argument("--retries", type=int, default=2,
                        help="Number of network retries")
    parser.add_argument("--refresh", action="store_true", help="Ignore cached AS Rank responses")
    args = parser.parse_args()
    if not args.targets or any(a <= 0 for a in args.targets):
        parser.error("--as must contain positive ASNs")
    if args.page_size < 1 or args.timeout <= 0 or args.retries < 0:
        parser.error("--page-size > 0, --timeout > 0 and --retries >= 0 are required")
    if args.out.resolve() == args.dir.resolve():
        parser.error("--out must differ from --dir to avoid overwriting the complete CSR")
    if args.out.resolve() == args.cache.resolve() or args.out.resolve() in args.cache.resolve().parents:
        parser.error("--cache must not be within --out")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s",
                        datefmt="%H:%M:%S")
    start = time.perf_counter()
    target_asns = list(dict.fromkeys(args.targets))
    graph, kinds, asn_ids = load_graph(args.dir)
    missing = [asn for asn in target_asns if asn not in set(asn_ids)]
    if missing:
        parser.error(f"Target ASNs missing from full CAIDA CSR: {missing}")
    log.info("Full CAIDA CSR: %d ASNs, %d directed arcs", len(asn_ids), graph.nnz)

    targets = fetch_targets(target_asns, args.cache, args.refresh, args.timeout, args.retries)
    ranks = {asn: int(row["rank"]) for asn, row in targets.items()}
    threshold = max(ranks.values())
    for asn in target_asns:
        log.info("Target AS%d: AS Rank=%d, customer cone=%s", asn, ranks[asn],
                 (targets[asn].get("cone") or {}).get("numberAsns"))
    log.info("Threshold: rank <= %d (the worse of the target ranks)", threshold)
    if args.ranks_only:
        print(json.dumps({"targets": {str(k): ranks[k] for k in target_asns},
                          "rank_threshold": threshold}, ensure_ascii=False, indent=2))
        return
    if threshold > 10000:
        log.warning("Rank threshold is >10000: the induced subgraph may remain very large")
    ranked = fetch_ranked_asns(threshold, args.cache, args.refresh,
                              args.timeout, args.retries, args.page_size)
    for asn in target_asns:
        if asn not in ranked:
            raise ValueError(f"Target AS{asn} missing from AS Rank paginated results; no graph exported")
        if int(ranked[asn]["rank"]) != ranks[asn]:
            raise ValueError(f"Inconsistent rank for AS{asn}; use --refresh to get a consistent snapshot")
    sub, subtypes, sub_asns, indices, boundary = induced_rank_subgraph(graph, kinds, asn_ids, ranked)
    validate_bidirectional(sub, subtypes)
    stats = statistics(sub, subtypes, sub_asns, graph, boundary)
    _, labels = connected_components(sub, directed=True, connection="weak")
    target_comp = {asn: int(labels[int(np.searchsorted(sub_asns, asn))]) for asn in target_asns}
    summary = {
        "targets": {str(asn): {"rank": ranks[asn],
                 "customer_cone_asns": (targets[asn].get("cone") or {}).get("numberAsns")}
                    for asn in target_asns},
        "rank_threshold": threshold,
        "ranked_asns_in_asrank": len(ranked),
        "ranked_asns_absent_from_caida": len(ranked) - len(sub_asns),
        "targets_same_component": len(set(target_comp.values())) == 1,
        "target_component_ids": {str(k): v for k, v in target_comp.items()},
        "full_caida_vertices": int(graph.shape[0]),
        "full_caida_directed_arcs": int(graph.nnz),
        **stats,
        "seconds": round(time.perf_counter() - start, 3),
        "asrank_endpoint": API_URL,
        "note": "AS Rank is fetched at execution time (or from cache), which may not match CAIDA CSR snapshot date",
    }
    save_experiment(args.out, sub, subtypes, sub_asns, ranked, summary)
    log.info("Selected AS: %d (%.2f%%), relationships: %d (%.2f%%)",
             stats["vertices"], stats["pct_full_vertices"],
             stats["undirected_relationships"], stats["pct_full_relationships"])
    log.info("Peer: %d, provider/customer: %d; components: %d, largest: %d; isolated: %d",
             stats["peer_relationships"], stats["provider_customer_relationships"],
             stats["weak_components"], stats["largest_component_vertices"], stats["isolated_vertices"])
    log.info("Targets in same component: %s", summary["targets_same_component"])
    log.info("Output: %s", args.out)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
