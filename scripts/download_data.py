#!/usr/bin/env python3
"""Reproducible download of the MetroPT-3 dataset (UCI ML Repository, ID 791).

The dataset is NEVER committed to this repository. This script fetches it from the
official UCI endpoint into ``data/raw/`` and verifies the resulting archive with a
SHA-256 checksum recorded in ``data/raw/manifest.json``.

Usage
-----
    python scripts/download_data.py            # download if not present
    python scripts/download_data.py --force    # re-download
    python scripts/download_data.py --verify-only

Exit codes: 0 ok, 1 failure, 2 dataset already present (no-op).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import urllib.error
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

UCI_DATASET_ID = 791
UCI_PAGE = "https://archive.ics.uci.edu/dataset/791/metropt+3+dataset"
DOWNLOAD_URL = "https://archive.ics.uci.edu/static/public/791/metropt+3+dataset.zip"
EXPECTED_MEMBER = "MetroPT3(AirCompressor).csv"
#: SHA-256 of the official archive used for every reported experiment.
EXPECTED_SHA256 = "aab991a970e58210de853bb8078ce0e63abb4d9412fdc5c79792dae3d8e1721a"

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / "data" / "raw"
ARCHIVE = RAW_DIR / "metropt3.zip"
EXTRACT_DIR = RAW_DIR / "metropt3"
MANIFEST = RAW_DIR / "manifest.json"


def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"[download] {url}")
    request = urllib.request.Request(url, headers={"User-Agent": "AnomaliRadar/0.1"})
    try:
        with urllib.request.urlopen(request, timeout=120) as response, tmp.open("wb") as out:
            total = int(response.headers.get("Content-Length") or 0)
            seen = 0
            while True:
                block = response.read(1 << 20)
                if not block:
                    break
                out.write(block)
                seen += len(block)
                if total:
                    pct = 100.0 * seen / total
                    print(f"\r[download] {seen / 1e6:7.1f} / {total / 1e6:7.1f} MB ({pct:5.1f}%)", end="")
            print()
    except urllib.error.URLError as exc:
        tmp.unlink(missing_ok=True)
        raise SystemExit(f"[error] download failed: {exc}") from exc
    tmp.replace(dest)


def extract(archive: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as bundle:
        names = bundle.namelist()
        print(f"[extract] {len(names)} member(s)")
        for name in names:
            print(f"          - {name}")
        bundle.extractall(out_dir)


def write_manifest(archive: Path, checksum: str) -> None:
    payload = {
        "dataset": "MetroPT-3",
        "uci_id": UCI_DATASET_ID,
        "source_page": UCI_PAGE,
        "download_url": DOWNLOAD_URL,
        "archive": archive.name,
        "archive_bytes": archive.stat().st_size,
        "sha256": checksum,
        "expected_sha256": EXPECTED_SHA256,
        "retrieved_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "expected_member": EXPECTED_MEMBER,
    }
    MANIFEST.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"[manifest] {MANIFEST.relative_to(REPO_ROOT)}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true", help="re-download even if the archive exists")
    parser.add_argument("--verify-only", action="store_true", help="check an existing archive against the manifest")
    args = parser.parse_args(argv)

    if args.verify_only:
        if not ARCHIVE.exists():
            print("[error] archive missing; run without --verify-only first")
            return 1
        recorded = None
        if MANIFEST.exists():
            recorded = json.loads(MANIFEST.read_text())["sha256"]
        actual = sha256_of(ARCHIVE)
        ok = actual == EXPECTED_SHA256 and (recorded is None or recorded == actual)
        print(
            f"[verify] pinned ={EXPECTED_SHA256}\n"
            f"[verify] recorded={recorded}\n"
            f"[verify] actual  ={actual}\n"
            f"[verify] {'OK' if ok else 'MISMATCH'}"
        )
        return 0 if ok else 1

    if ARCHIVE.exists() and not args.force:
        print(f"[skip] archive already present: {ARCHIVE.relative_to(REPO_ROOT)} (use --force to refresh)")
        return 2

    download(DOWNLOAD_URL, ARCHIVE)
    checksum = sha256_of(ARCHIVE)
    print(f"[sha256] {checksum}")
    if checksum != EXPECTED_SHA256:
        print(
            "[error] archive checksum does not match the pinned SHA-256.\n"
            f"        expected={EXPECTED_SHA256}\n"
            f"        actual  ={checksum}\n"
            "        The upstream artefact may have changed; do not run the "
            "frozen experiment against this file."
        )
        return 1

    if EXTRACT_DIR.exists() and args.force:
        shutil.rmtree(EXTRACT_DIR)
    extract(ARCHIVE, EXTRACT_DIR)

    target = EXTRACT_DIR / EXPECTED_MEMBER
    if not target.exists():
        print(f"[error] expected member not found: {target}")
        return 1
    write_manifest(ARCHIVE, checksum)
    print(f"[done] {target.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
