#!/usr/bin/env python3
"""Archive primary documentation used to define the cellular case."""

import json
from html.parser import HTMLParser

import requests

from bib_common import PROJECT, record, write_once


class TextParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []
        self.hidden = 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self.hidden += 1

    def handle_endtag(self, tag):
        if tag in ("script", "style"):
            self.hidden = max(0, self.hidden - 1)

    def handle_data(self, data):
        if not self.hidden and data.strip():
            self.parts.append(data.strip())


def main():
    out = PROJECT / "reports/bib_cellular_case_20261018/sources"
    out.mkdir(parents=True, exist_ok=True)
    sources = {
        "clue_perturbagen_types.html": "https://clue.io/connectopedia/perturbagen_types_and_controls",
        "cellosaurus_A375.txt": "https://www.cellosaurus.org/CVCL_0132.txt",
        "zenodo_unipert_g2cp.json": "https://zenodo.org/api/records/20355906",
        "uniprot_BRAF.json": "https://rest.uniprot.org/uniprotkb/P15056.json",
    }
    records = []
    for name, url in sources.items():
        path = out / name
        if not path.exists():
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            write_once(path, response.text)
        records.append({"url": url, **record(path)})
    write_once(out / "manifest.json", json.dumps(records, indent=2) + "\n")
    parser = TextParser()
    parser.feed((out / "clue_perturbagen_types.html").read_text())
    text = " ".join(parser.parts)
    for keyword in ("trt_xpr", "CRISPR", "overexpression"):
        p = text.find(keyword)
        print(keyword, p, text[max(0, p - 150):p + 500] if p >= 0 else "absent")
    for line in (out / "cellosaurus_A375.txt").read_text().splitlines():
        if "Sequence variation" in line and "BRAF" in line:
            print(line)
    gene = json.loads((out / "uniprot_BRAF.json").read_text())
    print("target", gene["primaryAccession"], gene["genes"][0]["geneName"]["value"])
    meta = json.loads((out / "zenodo_unipert_g2cp.json").read_text())
    print("G2CP weights", next(f for f in meta["files"] if f["key"] == "G2CP:saved_models.zip"))


if __name__ == "__main__":
    main()
