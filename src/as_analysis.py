"""Build AS hierarchy metrics and a candidate shortlist from CAIDA CSR.

Optional AS Rank enrichment uses live CAIDA data; it is not inferred from CSR.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
from scipy import sparse

UPSTREAM = 1
DOWNSTREAM = 2
PEER = 3
ASRANK_URL = "https://api.asrank.caida.org/v2/graphql"

METRIC_FIELDS = [
    "asn", "customer_degree", "provider_degree", "peer_degree", "total_degree",
    "asrank_rank", "asrank_cone_asns", "asrank_name", "asrank_org", "asrank_country",
]
CANDIDATE_FIELDS = [
    "asn", "groups", "customer_degree", "provider_degree", "peer_degree",
    "total_degree", "asrank_rank", "asrank_cone_asns", "asrank_name",
    "asrank_org", "asrank_country", "facility_count", "ix_count", "countries",
    "facility_cities", "facility_countries", "ix_registered_cities",
]

log = logging.getLogger(__name__)


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def load_snapshot(directory: Path):
    graph = sparse.load_npz(directory / "graph.npz").tocsr()
    kinds = np.load(directory / "edge_types.npy", allow_pickle=False)
    asn_ids = np.load(directory / "asn_ids.npy", allow_pickle=False)
    if graph.shape[0] != graph.shape[1] or graph.nnz != len(kinds) or graph.shape[0] != len(asn_ids):
        raise ValueError("Incompatible CSR snapshot")
    if not graph.has_sorted_indices or np.any(asn_ids[1:] <= asn_ids[:-1]):
        raise ValueError("CSR indices and ASN IDs must be sorted")
    if not np.isin(kinds, [UPSTREAM, DOWNSTREAM, PEER]).all():
        raise ValueError("Unknown relationship type in CSR")
    return graph, kinds, asn_ids


def csr_metrics(graph: sparse.csr_matrix, kinds: np.ndarray, asn_ids: np.ndarray) -> list[dict]:
    """Count per-AS relationships; each neighbor is represented by one CSR arc."""
    counts = np.diff(graph.indptr)
    rows = np.repeat(np.arange(graph.shape[0]), counts)
    degrees = {
        name: np.bincount(rows[kinds == code], minlength=graph.shape[0])
        for name, code in [
            ("customer_degree", DOWNSTREAM),
            ("provider_degree", UPSTREAM),
            ("peer_degree", PEER),
        ]
    }
    result = []
    for i, asn in enumerate(asn_ids):
        result.append({
            "asn": int(asn),
            **{key: int(values[i]) for key, values in degrees.items()},
            "total_degree": int(counts[i]),
            "asrank_rank": "", "asrank_cone_asns": "", "asrank_name": "",
            "asrank_org": "", "asrank_country": "",
        })
    return result


def fetch_asrank_top(limit: int, cache: Path, refresh: bool = False) -> dict[int, dict]:
    """Get top N AS Rank entries via CAIDA's GraphQL API and cache the response."""
    if limit <= 0:
        return {}
    if cache.exists() and not refresh:
        raw = json.loads(cache.read_text(encoding="utf-8"))
        if raw.get("limit", 0) >= limit:
            return {int(row["asn"]): row for row in raw["rows"][:limit]}

    result = []
    offset = 0
    while offset < limit:
        size = min(200, limit - offset)
        query = """{
          asns(first: %d, offset: %d) {
            edges { node {
              asn rank asnName
              cone { numberAsns }
              organization { orgName }
              country { iso }
            } }
          }
        }""" % (size, offset)
        request = Request(
            ASRANK_URL,
            data=json.dumps({"query": query}).encode("utf-8"),
            headers={"Content-Type": "application/json", "User-Agent": "course-asn-research/1.0"},
            method="POST",
        )
        with urlopen(request, timeout=45) as response:
            payload = json.load(response)
        if payload.get("errors"):
            raise RuntimeError(f"AS Rank API error: {payload['errors']}")
        edges = payload.get("data", {}).get("asns", {}).get("edges", [])
        if not edges:
            break
        result.extend(entry["node"] for entry in edges)
        offset += len(edges)
        log.info("AS Rank: %s / %s", offset, limit)
        if len(edges) < size:
            break

    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({"limit": len(result), "rows": result}, ensure_ascii=False), encoding="utf-8")
    return {int(row["asn"]): row for row in result}


def add_asrank(metrics: list[dict], rankings: dict[int, dict]) -> None:
    for row in metrics:
        ranked = rankings.get(row["asn"])
        if ranked is None:
            continue
        cone = ranked.get("cone") or {}
        org = ranked.get("organization") or {}
        country = ranked.get("country") or {}
        row.update({
            "asrank_rank": ranked.get("rank") or "",
            "asrank_cone_asns": cone.get("numberAsns") if cone.get("numberAsns") is not None else "",
            "asrank_name": ranked.get("asnName") or "",
            "asrank_org": org.get("orgName") or "",
            "asrank_country": country.get("iso") or "",
        })


def select_candidates(metrics: list[dict], large: int, middle: int, peering: int,
                      sites: list[dict] | None = None) -> list[dict]:
    """Shortlist by AS Rank cone when available, plus an independent peer-degree group."""
    selected: dict[int, set[str]] = {}

    with_rank = [r for r in metrics if str(r["asrank_rank"]).strip()]
    if with_rank:
        ordered = sorted(with_rank, key=lambda r: (int(r["asrank_rank"]), int(r["asn"])))
        first, second = "large_cone", "middle_cone"
    else:
        # Without AS Rank this is a DIRECT CUSTOMER DEGREE ranking, not a cone.
        ordered = sorted(metrics, key=lambda r: (-int(r["customer_degree"]), int(r["asn"])))
        first, second = "high_customer_degree", "middle_customer_degree"
    for group, segment in [(first, ordered[:large]), (second, ordered[large:large + middle])]:
        for row in segment:
            selected.setdefault(int(row["asn"]), set()).add(group)

    by_peers = sorted(metrics, key=lambda r: (-int(r["peer_degree"]), int(r["asn"])))
    for row in [r for r in by_peers if int(r["peer_degree"]) > 0][:peering]:
        selected.setdefault(int(row["asn"]), set()).add("high_peering")

    site_stats: dict[int, dict[str, set]] = {}
    for site in sites or []:
        asn = int(site["asn"])
        stats = site_stats.setdefault(asn, {
            "facility": set(), "ix": set(), "countries": set(),
            "facility_cities": set(), "facility_countries": set(),
            "ix_registered_cities": set(),
        })
        kind = site.get("site_type", "")
        if kind in {"facility", "ix"}:
            stats[kind].add(site["site_id"])
            if site.get("country"):
                stats["countries"].add(site["country"])
        if kind == "facility":
            if site.get("city"):
                stats["facility_cities"].add(site["city"])
            if site.get("country"):
                stats["facility_countries"].add(site["country"])
        elif kind == "ix" and site.get("city"):
            stats["ix_registered_cities"].add(site["city"])

    result = []
    for row in metrics:
        asn = int(row["asn"])
        if asn not in selected:
            continue
        stats = site_stats.get(asn)
        result.append({
            **row,
            "groups": "|".join(sorted(selected[asn])),
            "facility_count": len(stats["facility"]) if stats is not None else "",
            "ix_count": len(stats["ix"]) if stats is not None else "",
            "countries": "|".join(sorted(stats["countries"])) if stats is not None else "",
            "facility_cities": "|".join(sorted(stats["facility_cities"])) if stats is not None else "",
            "facility_countries": "|".join(sorted(stats["facility_countries"])) if stats is not None else "",
            "ix_registered_cities": "|".join(sorted(stats["ix_registered_cities"])) if stats is not None else "",
        })
    return sorted(result, key=lambda r: (int(r["asrank_rank"]) if str(r["asrank_rank"]).strip() else 10**9,
                                          -int(r["customer_degree"]), int(r["asn"])))


def customer_hierarchy(graph: sparse.csr_matrix, kinds: np.ndarray, asn_ids: np.ndarray,
                       root_asn: int, depth: int = 2, max_nodes: int = 60) -> list[str]:
    """Show a small p2c-only discovery tree (not all BGP paths or an AS Rank cone)."""
    if depth < 0 or max_nodes < 1:
        raise ValueError("depth must be >= 0 and max_nodes must be >= 1")
    root = int(np.searchsorted(asn_ids, root_asn))
    if root >= len(asn_ids) or int(asn_ids[root]) != root_asn:
        raise ValueError(f"AS{root_asn} is not in the CSR snapshot")

    customer_counts = np.bincount(
        np.repeat(np.arange(graph.shape[0]), np.diff(graph.indptr))[kinds == DOWNSTREAM],
        minlength=graph.shape[0],
    )
    seen = set()
    output = []

    def visit(v: int, level: int):
        if v in seen or len(seen) >= max_nodes:
            return
        seen.add(v)
        output.append(f"{'  ' * level}AS{int(asn_ids[v])}  (direct customers: {int(customer_counts[v])})")
        if level >= depth:
            return
        begin, end = int(graph.indptr[v]), int(graph.indptr[v + 1])
        children = [int(graph.indices[i]) for i in range(begin, end)
                    if int(kinds[i]) == DOWNSTREAM and int(graph.indices[i]) not in seen]
        children.sort(key=lambda u: (-int(customer_counts[u]), int(asn_ids[u])))
        for child in children:
            if len(seen) >= max_nodes:
                break
            visit(child, level + 1)

    visit(root, 0)
    if len(seen) == max_nodes:
        output.append(f"... reached max-nodes={max_nodes}; increase the limit to expand")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze AS topology and build candidate shortlists")
    sub = parser.add_subparsers(dest="command", required=True)
    metrics = sub.add_parser("metrics", help="Generate as_metrics.csv from the full CAIDA CSR")
    metrics.add_argument("--dir", type=Path, default=Path("data/processed"))
    metrics.add_argument("--out", type=Path, default=Path("data/analysis/as_metrics.csv"))
    metrics.add_argument("--asrank-top", type=int, default=0, help="Enrich top N CAIDA AS Rank entries (network)")
    metrics.add_argument("--asrank-cache", type=Path, default=Path("data/analysis/.cache/asrank.json"))
    metrics.add_argument("--refresh", action="store_true", help="Refresh cached AS Rank response")

    candidates = sub.add_parser("candidates", help="Generate candidates.csv from metrics")
    candidates.add_argument("--metrics", type=Path, default=Path("data/analysis/as_metrics.csv"))
    candidates.add_argument("--sites", type=Path, help="Optional asn_sites.csv to add site counts")
    candidates.add_argument("--out", type=Path, default=Path("data/analysis/candidates.csv"))
    candidates.add_argument("--large", type=int, default=40)
    candidates.add_argument("--middle", type=int, default=80)
    candidates.add_argument("--peering", type=int, default=80)
    hierarchy = sub.add_parser("hierarchy", help="Explore provider -> customer hierarchy from an ASN")
    hierarchy.add_argument("--dir", type=Path, default=Path("data/processed"))
    hierarchy.add_argument("--as", dest="root_asn", type=int, required=True)
    hierarchy.add_argument("--depth", type=int, default=2)
    hierarchy.add_argument("--max-nodes", type=int, default=60)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if args.command == "metrics":
        if args.asrank_top < 0:
            parser.error("--asrank-top must be >= 0")
        graph, kinds, asn_ids = load_snapshot(args.dir)
        rows = csr_metrics(graph, kinds, asn_ids)
        if args.asrank_top:
            add_asrank(rows, fetch_asrank_top(args.asrank_top, args.asrank_cache, args.refresh))
        write_csv(args.out, METRIC_FIELDS, rows)
        print(f"Wrote {len(rows)} AS metrics: {args.out}")
    elif args.command == "hierarchy":
        graph, kinds, asn_ids = load_snapshot(args.dir)
        print("\n".join(customer_hierarchy(graph, kinds, asn_ids,
                                           args.root_asn, args.depth, args.max_nodes)))
    else:
        if min(args.large, args.middle, args.peering) < 0:
            parser.error("candidate group sizes must be >= 0")
        rows = read_csv(args.metrics)
        site_rows = read_csv(args.sites) if args.sites else None
        selected = select_candidates(rows, args.large, args.middle, args.peering, site_rows)
        write_csv(args.out, CANDIDATE_FIELDS, selected)
        print(f"Wrote {len(selected)} candidates: {args.out}")


if __name__ == "__main__":
    main()
