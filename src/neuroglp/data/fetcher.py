"""
scripts/fetch_curated_dataset.py

Acquisition script to fetch 5,000-10,000 genuine OpenFDA adverse event records
for GLP-1 receptor agonists (Semaglutide/Tirzepatide) with psychiatric endpoints.
Provides structured schema pruning, rate-limit backoff, checkpoint resumption,
and gzipped JSON output.
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
import urllib.parse
import urllib.request
import urllib.error

# Ensure src/ is on path
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from neuroglp.data.openfda_client import (
    OPENFDA_EVENT_ENDPOINT,
    build_openfda_query,
    load_curated_dataset,
)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("NeuroGLP-Curator")


def prune_record(rec: Dict[str, Any]) -> Dict[str, Any]:
    """
    Prune voluminous packaging NDC / SPL arrays while strictly preserving
    the exact OpenFDA results array schema hierarchy for pharmacovigilance.
    """
    patient_raw = rec.get("patient") or {}

    # Clean drug entries
    cleaned_drugs = []
    for d in patient_raw.get("drug", []):
        openfda_raw = d.get("openfda") or {}
        # Keep essential clinical drug entity fields
        cleaned_openfda = {
            k: v
            for k, v in openfda_raw.items()
            if k in [
                "generic_name",
                "brand_name",
                "substance_name",
                "pharm_class_epc",
                "route",
            ]
        }
        cleaned_drugs.append(
            {
                "drugcharacterization": d.get("drugcharacterization"),
                "medicinalproduct": d.get("medicinalproduct"),
                "openfda": cleaned_openfda,
            }
        )

    # Clean reaction entries
    cleaned_reactions = [
        {
            "reactionmeddrapt": r.get("reactionmeddrapt"),
            "reactionoutcome": r.get("reactionoutcome"),
        }
        for r in patient_raw.get("reaction", [])
        if r.get("reactionmeddrapt")
    ]

    return {
        "safetyreportversion": str(rec.get("safetyreportversion", "1")),
        "safetyreportid": str(rec.get("safetyreportid", "")),
        "primarysourcecountry": rec.get("primarysourcecountry"),
        "occurcountry": rec.get("occurcountry"),
        "transmissiondate": rec.get("transmissiondate"),
        "reporttype": rec.get("reporttype"),
        "serious": rec.get("serious"),
        "seriousnessdeath": rec.get("seriousnessdeath"),
        "seriousnesslifethreatening": rec.get("seriousnesslifethreatening"),
        "seriousnesshospitalization": rec.get("seriousnesshospitalization"),
        "seriousnessdisabling": rec.get("seriousnessdisabling"),
        "seriousnessother": rec.get("seriousnessother"),
        "receivedate": rec.get("receivedate"),
        "receiptdate": rec.get("receiptdate"),
        "patient": {
            "patientonsetage": patient_raw.get("patientonsetage"),
            "patientonsetageunit": patient_raw.get("patientonsetageunit"),
            "patientsex": patient_raw.get("patientsex"),
            "drug": cleaned_drugs,
            "reaction": cleaned_reactions,
        },
    }


def execute_request_with_retry(
    url: str,
    max_retries: int = 5,
    backoff_factor: float = 2.0,
) -> Dict[str, Any]:
    """Execute HTTP GET with exponential backoff on 429 and 503."""
    headers = {"User-Agent": "NeuroGLP-Curator/1.0 (Pharmacovigilance Research)"}
    req = urllib.request.Request(url, headers=headers)

    for attempt in range(max_retries):
        try:
            with urllib.request.urlopen(req, timeout=25) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code == 429:
                sleep_sec = (backoff_factor ** attempt) * 2.5
                logger.warning(
                    f"HTTP 429 Rate Limit encountered. Backing off for {sleep_sec:.1f}s..."
                )
                time.sleep(sleep_sec)
            elif e.code in (500, 502, 503, 504):
                sleep_sec = (backoff_factor ** attempt) * 1.5
                logger.warning(
                    f"HTTP {e.code} Server Error. Retrying in {sleep_sec:.1f}s..."
                )
                time.sleep(sleep_sec)
            elif e.code == 404:
                logger.info("HTTP 404: End of results or 0 records found.")
                return {"results": [], "meta": {"results": {"total": 0}}}
            else:
                logger.error(f"HTTP Error {e.code}: {e.reason}")
                raise
        except Exception as e:
            sleep_sec = (backoff_factor ** attempt) * 1.5
            logger.warning(f"Network error: {e}. Retrying in {sleep_sec:.1f}s...")
            time.sleep(sleep_sec)

    raise RuntimeError(f"Failed to fetch {url} after {max_retries} attempts.")


def fetch_dataset(
    output_path: Path,
    max_records: Optional[int] = None,
    batch_size: int = 100,
    delay_sec: float = 0.5,
    prune: bool = True,
    compress: bool = True,
    checkpoint_dir: Optional[Path] = None,
) -> Path:
    """Fetch complete benchmark dataset from OpenFDA."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if checkpoint_dir:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    query = build_openfda_query(include_extended_reactions=True)

    # 1. Probe total count
    probe_url = f"{OPENFDA_EVENT_ENDPOINT}?search={query}&limit=1"
    logger.info("Probing OpenFDA for total matching reports...")
    probe_data = execute_request_with_retry(probe_url)
    total_available = probe_data.get("meta", {}).get("results", {}).get("total", 0)
    logger.info(f"OpenFDA live total available records: {total_available:,}")

    target_records = min(total_available, max_records) if max_records else total_available
    total_batches = math.ceil(target_records / batch_size)
    logger.info(
        f"Targeting {target_records:,} records across {total_batches} batches (batch_size={batch_size})"
    )

    all_records: List[Dict[str, Any]] = []

    # 2. Iterate batches
    for batch_idx in range(total_batches):
        skip = batch_idx * batch_size
        current_limit = min(batch_size, target_records - skip)
        if current_limit <= 0:
            break

        checkpoint_file = (
            checkpoint_dir / f"batch_{skip}.json" if checkpoint_dir else None
        )
        if checkpoint_file and checkpoint_file.exists():
            with open(checkpoint_file, "r", encoding="utf-8") as f:
                batch_records = json.load(f)
            logger.info(
                f"Loaded batch {batch_idx+1}/{total_batches} (skip={skip}) from checkpoint"
            )
        else:
            batch_url = (
                f"{OPENFDA_EVENT_ENDPOINT}?search={query}&limit={current_limit}&skip={skip}"
            )
            data = execute_request_with_retry(batch_url)
            batch_records = data.get("results", [])

            if prune:
                batch_records = [prune_record(r) for r in batch_records]

            if checkpoint_file:
                with open(checkpoint_file, "w", encoding="utf-8") as f:
                    json.dump(batch_records, f, separators=(",", ":"))

            logger.info(
                f"Fetched batch {batch_idx+1}/{total_batches} (skip={skip}): {len(batch_records)} records"
            )
            if delay_sec > 0:
                time.sleep(delay_sec)

        all_records.extend(batch_records)

    logger.info(f"Successfully collected {len(all_records):,} total adverse event records.")

    # 3. Assemble OpenFDA-compliant payload
    final_payload = {
        "meta": {
            "disclaimer": "Do not use openFDA to make decisions regarding medical care. Preserved benchmark.",
            "terms": "https://open.fda.gov/terms/",
            "license": "https://open.fda.gov/license/",
            "last_updated": time.strftime("%Y-%m-%d"),
            "results": {
                "skip": 0,
                "limit": len(all_records),
                "total": len(all_records),
            },
        },
        "results": all_records,
    }

    # 4. Save to destination
    if compress or str(output_path).endswith(".gz"):
        final_path = (
            output_path
            if str(output_path).endswith(".gz")
            else output_path.with_suffix(output_path.suffix + ".gz")
        )
        with gzip.open(final_path, "wt", encoding="utf-8") as f:
            json.dump(final_payload, f, separators=(",", ":"))
        logger.info(
            f"Saved gzipped benchmark dataset to: {final_path} ({final_path.stat().st_size / 1024 / 1024:.2f} MB)"
        )
    else:
        final_path = output_path
        with open(final_path, "w", encoding="utf-8") as f:
            json.dump(final_payload, f, separators=(",", ":"))
        logger.info(
            f"Saved JSON benchmark dataset to: {final_path} ({final_path.stat().st_size / 1024 / 1024:.2f} MB)"
        )

    return final_path


def validate_dataset(filepath: Path) -> bool:
    """Validate resulting dataset against acceptance criteria."""
    logger.info(f"Validating dataset: {filepath}...")
    results = load_curated_dataset(filepath)
    total = len(results)
    logger.info(f"Validation: Total records loaded = {total:,}")

    assert 5000 <= total <= 10000, f"Record count {total} outside required range [5000, 10000]"

    # Check key fields across sample
    for idx, rec in enumerate(results[:20]):
        assert "safetyreportid" in rec, f"Record {idx} missing safetyreportid"
        assert "patient" in rec, f"Record {idx} missing patient block"
        assert "drug" in rec["patient"], f"Record {idx} missing patient.drug"
        assert "reaction" in rec["patient"], f"Record {idx} missing patient.reaction"

    # Compute quick clinical stats
    females = sum(1 for r in results if str(r.get("patient", {}).get("patientsex")) == "2")
    hospitalized = sum(1 for r in results if str(r.get("seriousnesshospitalization")) == "1")
    logger.info(
        f"Validation passed: Female share = {females/total*100:.1f}%, Hospitalization rate = {hospitalized/total*100:.1f}%"
    )
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch GLP-1 psychiatric dataset from OpenFDA.")
    parser.add_argument(
        "--output",
        type=str,
        default="data/curated_glp1_psychiatric.json.gz",
        help="Output path",
    )
    parser.add_argument(
        "--compress",
        action="store_true",
        default=True,
        help="Compress with gzip (.json.gz)",
    )
    parser.add_argument(
        "--no-prune",
        action="store_true",
        help="Disable schema pruning",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=".cache/fda_batches",
        help="Checkpoint directory",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.4,
        help="Delay between requests in seconds",
    )
    args = parser.parse_args()

    out_file = Path(args.output)
    ckpt = Path(args.checkpoint_dir)
    saved_file = fetch_dataset(
        output_path=out_file,
        prune=not args.no_prune,
        compress=args.compress,
        checkpoint_dir=ckpt,
        delay_sec=args.delay,
    )
    validate_dataset(saved_file)
