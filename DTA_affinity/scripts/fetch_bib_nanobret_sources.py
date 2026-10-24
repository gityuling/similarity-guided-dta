#!/usr/bin/env python3
"""Archive primary NanoBRET literature and its original supplementary files."""

import argparse
import json
import time
import zipfile
from datetime import datetime, timezone
from io import BytesIO

import requests

from bib_common import PROJECT, record, write_once


OUT = PROJECT / "reports/bib_nanobret_20261018"
API = "https://www.ebi.ac.uk/europepmc/webservices/rest"


def retrieve(url, path, params=None):
    if path.exists():
        return record(path)
    errors = []
    for attempt in range(3):
        try:
            response = requests.get(url, params=params, timeout=(15, 90))
            response.raise_for_status()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(response.content)
            return {**record(path), "url": response.url,
                    "retrieved_utc": datetime.now(timezone.utc).isoformat()}
        except requests.RequestException as error:
            errors.append(str(error))
            if attempt < 2:
                time.sleep(2)
    raise RuntimeError(f"Source unavailable: {url}: {errors}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pmcid", default="PMC5814754")
    args = parser.parse_args()
    destination = OUT / "sources" / args.pmcid
    sources = [retrieve(f"{API}/{args.pmcid}/fullTextXML", destination / "article.xml")]
    archive = destination / "supplementary.zip"
    sources.append(retrieve(f"{API}/{args.pmcid}/supplementaryFiles", archive))
    with zipfile.ZipFile(archive) as handle:
        for name in handle.namelist():
            if name.lower().endswith((".xlsx", ".xls", ".csv", ".tsv", ".pdf")):
                if "/" in name or "\\" in name:
                    raise RuntimeError(f"Unexpected archive member: {name}")
                path = destination / name
                contents = handle.read(name)
                if path.exists() and path.read_bytes() != contents:
                    raise RuntimeError(f"Archived file changed: {path}")
                path.write_bytes(contents)
                sources.append(record(path))
    manifest = destination / "download_manifest.json"
    if not manifest.exists():
        write_once(manifest, json.dumps({"pmcid": args.pmcid, "files": sources}, indent=2) + "\n")
    print(json.dumps({"pmcid": args.pmcid, "files": sources}, indent=2), flush=True)


if __name__ == "__main__":
    main()
