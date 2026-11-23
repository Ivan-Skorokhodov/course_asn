"""Enrich shortlisted ASNs with PeeringDB facility and IX presence.

Facility rows identify reported physical sites. An IX's registered city is
not proof that a network has equipment in that city (remote peering exists).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

BASE_URL = "https://www.peeringdb.com/api"
SITE_FIELDS = [
    "asn", "site_type", "site_id", "site_name", "city", "country",
    "latitude", "longitude", "location_scope", "peeringdb_url", "exported_utc",
]


def batches(values: list[int], size: int = 100):
    for i in range(0, len(values), size):
        yield values[i:i + size]


class PeeringDBClient:
    def __init__(
        self, cache_dir: Path, refresh: bool = False, min_interval: float = 3.1,
        timeout: float = 15.0, retries: int = 2,
    ):
        self.cache_dir = cache_dir
        self.refresh = refresh
        self.min_interval = min_interval
        self.timeout = timeout
        self.retries = retries
        self.last_request = 0.0

    def fetch(self, url: str) -> dict:
        """Request JSON with bounded retries for temporary network failures."""
        headers = {"Accept": "application/json", "User-Agent": "course-asn-research/1.0"}
        if os.environ.get("PEERINGDB_API_KEY"):
            headers["Authorization"] = "Api-Key " + os.environ["PEERINGDB_API_KEY"]
        request = Request(url, headers=headers)

        for attempt in range(self.retries + 1):
            elapsed = time.monotonic() - self.last_request
            if self.last_request and elapsed < self.min_interval:
                time.sleep(self.min_interval - elapsed)
            try:
                with urlopen(request, timeout=self.timeout) as response:
                    return json.load(response)
            except HTTPError as exc:
                # Only temporary server errors and rate limits should be retried.
                retryable = exc.code in (408, 429, 500, 502, 503, 504)
                if not retryable:
                    raise RuntimeError(
                        f"PeeringDB returned HTTP {exc.code}. "
                        "Check the API URL and API key (if supplied)."
                    ) from exc
                error = f"HTTP {exc.code}"
                retry_after = exc.headers.get("Retry-After") if exc.headers else None
            except (URLError, TimeoutError, ConnectionError, OSError) as exc:
                error = str(exc)
                retry_after = None
            finally:
                self.last_request = time.monotonic()

            if attempt >= self.retries:
                raise RuntimeError(
                    f"Could not connect to PeeringDB after {attempt + 1} attempts: {error}. "
                    "Check with: curl -4 -v --connect-timeout 10 "
                    "'https://www.peeringdb.com/api/net?limit=1'"
                )
            wait = 3 * (2 ** attempt)
            if retry_after:
                try:
                    wait = max(wait, float(retry_after))
                except ValueError:
                    pass
            print(
                f"PeeringDB request failed ({error}); retry "
                f"{attempt + 1}/{self.retries} in {wait:g} s...",
                file=sys.stderr,
            )
            time.sleep(wait)

        raise AssertionError("unreachable")

    def get(self, endpoint: str, **params) -> list[dict]:
        """Query an endpoint with pagination and a file cache for every page."""
        results = []
        page_size = 500
        skip = 0
        while True:
            options = {**params, "limit": page_size, "skip": skip}
            url = f"{BASE_URL}/{endpoint}?{urlencode(sorted(options.items()))}"
            cache_name = hashlib.sha256(url.encode("utf-8")).hexdigest() + ".json"
            cache_file = self.cache_dir / cache_name
            if cache_file.exists() and not self.refresh:
                payload = json.loads(cache_file.read_text(encoding="utf-8"))
            else:
                payload = self.fetch(url)
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                cache_file.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

            page = payload.get("data")
            if not isinstance(page, list):
                raise ValueError(f"Invalid PeeringDB response from {endpoint}")
            results.extend(page)
            if len(page) < page_size:
                return results
            skip += len(page)


def indexed(client: PeeringDBClient, endpoint: str, ids: list[int]) -> dict[int, dict]:
    output = {}
    for group in batches(sorted(set(ids))):
        for item in client.get(endpoint, id__in=",".join(map(str, group))):
            output[int(item["id"])] = item
    return output


def query_sites(client: PeeringDBClient, asns: list[int], collected_at: str) -> list[dict]:
    """Return facility and IX membership records for ASNs present in PeeringDB."""
    networks = {}
    for group in batches(sorted(set(asns))):
        for item in client.get("net", asn__in=",".join(map(str, group))):
            networks[int(item["id"])] = int(item["asn"])
    net_ids = sorted(networks)
    facilities = []
    exchanges = []
    for group in batches(net_ids):
        ids = ",".join(map(str, group))
        facilities.extend(client.get("netfac", net_id__in=ids))
        exchanges.extend(client.get("netixlan", net_id__in=ids))

    facility_data = indexed(client, "fac", [int(row["fac_id"]) for row in facilities])
    exchange_data = indexed(client, "ix", [int(row["ix_id"]) for row in exchanges])

    output = []
    seen = set()
    for item in facilities:
        asn = networks.get(int(item["net_id"]))
        if asn is None:
            continue
        fac_id = int(item["fac_id"])
        key = (asn, "facility", fac_id)
        if key in seen:
            continue
        seen.add(key)
        fac = facility_data.get(fac_id, {})
        output.append({
            "asn": asn, "site_type": "facility", "site_id": fac_id,
            "site_name": fac.get("name") or item.get("name") or "",
            "city": fac.get("city") or item.get("city") or "",
            "country": fac.get("country") or item.get("country") or "",
            "latitude": fac.get("latitude") if fac.get("latitude") is not None else "",
            "longitude": fac.get("longitude") if fac.get("longitude") is not None else "",
            "location_scope": "facility_presence",
            "peeringdb_url": f"https://www.peeringdb.com/fac/{fac_id}",
            "exported_utc": collected_at,
        })

    for item in exchanges:
        asn = networks.get(int(item["net_id"]))
        if asn is None:
            continue
        ix_id = int(item["ix_id"])
        key = (asn, "ix", ix_id)
        if key in seen:
            continue
        seen.add(key)
        ix = exchange_data.get(ix_id, {})
        output.append({
            "asn": asn, "site_type": "ix", "site_id": ix_id,
            "site_name": ix.get("name") or item.get("name") or "",
            "city": ix.get("city") or "",
            "country": ix.get("country") or "",
            "latitude": "", "longitude": "",
            "location_scope": "ix_registered_city_not_verified_pop",
            "peeringdb_url": f"https://www.peeringdb.com/ix/{ix_id}",
            "exported_utc": collected_at,
        })
    return sorted(output, key=lambda row: (int(row["asn"]), row["site_type"], int(row["site_id"])))


def main() -> None:
    parser = argparse.ArgumentParser(description="Resolve candidate AS geographic presence in PeeringDB")
    parser.add_argument("--candidates", type=Path, default=Path("data/analysis/candidates.csv"))
    parser.add_argument("--out", type=Path, default=Path("data/analysis/asn_sites.csv"))
    parser.add_argument("--cache", type=Path, default=Path("data/analysis/.cache/peeringdb"))
    parser.add_argument("--refresh", action="store_true", help="Ignore cached PeeringDB responses")
    parser.add_argument("--timeout", type=float, default=15.0,
                        help="Timeout per HTTPS request, seconds (default: 15)")
    parser.add_argument("--retries", type=int, default=2,
                        help="Retries after a connection error (default: 2)")
    parser.add_argument("--limit-asns", type=int, default=0,
                        help="Process only the first N ASNs for a trial run (0 = all)")
    args = parser.parse_args()
    if args.timeout <= 0 or args.retries < 0 or args.limit_asns < 0:
        parser.error("--timeout must be > 0; --retries and --limit-asns must be >= 0")

    with args.candidates.open("r", encoding="utf-8", newline="") as file:
        asns = sorted({int(row["asn"]) for row in csv.DictReader(file)})
    if args.limit_asns:
        asns = asns[:args.limit_asns]
    client = PeeringDBClient(args.cache, refresh=args.refresh,
                             timeout=args.timeout, retries=args.retries)
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    try:
        sites = query_sites(client, asns, timestamp)
    except RuntimeError as exc:
        parser.exit(1, f"Error: {exc}\n")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, SITE_FIELDS)
        writer.writeheader()
        writer.writerows(sites)
    print(f"Wrote {len(sites)} site records for {len(asns)} candidates: {args.out}")


if __name__ == "__main__":
    main()
