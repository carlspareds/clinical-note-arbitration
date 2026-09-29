#!/usr/bin/env python3
"""
Automated downloader for public clinical NLP benchmarks:
- ACI-Bench (Ambient Clinical Intelligence dialogue-to-note benchmark from Hugging Face)
- MTS-Dialog (Doctor-patient consultation section summaries)
- HealthBench (Physician-written clinical evaluation rubrics from OpenAI simple-evals)

Compliance notice:
Raw datasets are strictly cached in data/raw/ and NEVER committed to version control.
Only de-identified synthetic samples in data/samples/ are tracked.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import httpx

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("download_datasets")

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"
SAMPLES_DIR = DATA_DIR / "samples"

HEALTHBENCH_RAW_URL = "https://openaipublic.blob.core.windows.net/simple-evals/healthbench/consensus_2025-05-09-20-00-46.jsonl"
MTS_DIALOG_GITHUB_URL = "https://raw.githubusercontent.com/abachaa/MTS-Dialog/main/Main-Dataset/MTS-Dialog-ValidationSet.csv"
ACI_BENCH_HF_URL_TEST1 = "https://datasets-server.huggingface.co/rows?dataset=mkieffer%2FACI-Bench&config=aci&split=test1&offset=0&limit=25"
ACI_BENCH_HF_URL_VALID = "https://datasets-server.huggingface.co/rows?dataset=mkieffer%2FACI-Bench&config=aci&split=valid&offset=0&limit=15"


def ensure_directories():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)


def download_healthbench(limit: int = 100) -> List[Dict[str, Any]]:
    """Download real HealthBench physician-rubric evaluations from OpenAI simple-evals."""
    output_path = RAW_DIR / "healthbench.jsonl"
    logger.info("Attempting to fetch HealthBench from OpenAI simple-evals consensus...")
    items = []
    try:
        with httpx.stream("GET", HEALTHBENCH_RAW_URL, timeout=45.0) as stream:
            if stream.status_code == 200:
                with open(output_path, "w", encoding="utf-8") as f:
                    for line in stream.iter_lines():
                        if line.strip():
                            item = json.loads(line)
                            items.append(item)
                            f.write(line + "\n")
                            if len(items) >= limit:
                                break
                logger.info(
                    f"Successfully downloaded {len(items)} real HealthBench items to {output_path}"
                )
                return items
    except Exception as e:
        logger.warning(
            f"Could not reach remote HealthBench endpoint ({e}). Using verified fallback fixtures."
        )

    sample_path = SAMPLES_DIR / "healthbench_samples.json"
    if sample_path.exists():
        with open(sample_path, "r", encoding="utf-8") as f:
            items = json.load(f)
        logger.info(f"Loaded {len(items)} HealthBench sample items from local fixtures.")
    return items


def download_mts_dialog() -> List[Dict[str, Any]]:
    """Download MTS-Dialog doctor-patient summaries."""
    output_path = RAW_DIR / "mts_dialog.csv"
    logger.info("Attempting to fetch MTS-Dialog validation set...")
    items = []
    try:
        with httpx.Client(timeout=30.0) as client:
            resp = client.get(MTS_DIALOG_GITHUB_URL)
            if resp.status_code == 200:
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(resp.text)
                logger.info(f"Successfully downloaded MTS-Dialog to {output_path}")
    except Exception as e:
        logger.warning(f"Could not reach remote MTS-Dialog endpoint ({e}). Using sample fixtures.")

    sample_path = SAMPLES_DIR / "mts_dialog_samples.json"
    if sample_path.exists():
        with open(sample_path, "r", encoding="utf-8") as f:
            items = json.load(f)
    return items


def download_aci_bench(limit: int = 35) -> List[Dict[str, Any]]:
    """Download real ACI-Bench clinical encounters from Hugging Face."""
    output_path = RAW_DIR / "aci_bench.json"
    logger.info("Fetching real ACI-Bench ambient clinical encounters from Hugging Face...")
    encounters = []
    try:
        with httpx.Client(timeout=45.0) as client:
            for url in [ACI_BENCH_HF_URL_TEST1, ACI_BENCH_HF_URL_VALID]:
                resp = client.get(url)
                if resp.status_code == 200:
                    rows = resp.json().get("rows", [])
                    for row_item in rows:
                        row = row_item.get("row", {})
                        if row.get("dialogue") and row.get("note"):
                            encounters.append(
                                {
                                    "encounter_id": row.get(
                                        "encounter_id", f"ACI-{len(encounters) + 1:03d}"
                                    ),
                                    "dialogue": row.get("dialogue"),
                                    "gold_note": row.get("note"),
                                }
                            )
                            if len(encounters) >= limit:
                                break
                if len(encounters) >= limit:
                    break

        if encounters:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(encounters, f, indent=2)
            logger.info(
                f"Successfully downloaded {len(encounters)} real ACI-Bench encounters to {output_path}"
            )
            return encounters
    except Exception as e:
        logger.warning(f"Could not reach Hugging Face endpoint ({e}). Using sample encounters.")

    sample_path = SAMPLES_DIR / "aci_bench_samples.json"
    if sample_path.exists():
        with open(sample_path, "r", encoding="utf-8") as f:
            encounters = json.load(f)
        logger.info(f"Loaded {len(encounters)} ACI-Bench clinical encounters from samples.")
    return encounters


def main():
    ensure_directories()
    hb = download_healthbench(limit=100)
    mts = download_mts_dialog()
    aci = download_aci_bench(limit=35)

    manifest = {
        "healthbench_items_available": len(hb),
        "mts_dialog_items_available": len(mts),
        "aci_bench_items_available": len(aci),
        "storage_policy": "Strict isolation in data/raw/ (gitignored). No PHI.",
    }
    manifest_path = RAW_DIR / "dataset_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    logger.info(f"Dataset download completed. Manifest saved to {manifest_path}: {manifest}")


if __name__ == "__main__":
    main()
