#!/usr/bin/env python3
"""Package the matched comparison with pooled features and trained checkpoints."""
import hashlib
import importlib
import importlib.metadata
import json
from pathlib import Path
import platform
import tarfile

PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT.parent
REPORT = PROJECT / "reports/knn_shared_20261025"
RUNS = PROJECT / "results/knn_shared_20261025"


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def package_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        # Some conda packages expose their version without wheel metadata.
        return importlib.import_module(name).__version__


def main():
    analysis = json.loads((REPORT / "analysis.json").read_text())
    replay = json.loads((REPORT / "independent_replay.json").read_text())
    if not analysis["complete"] or analysis["runs"] != 30 or not replay["passed"] or replay["runs"] != 30:
        raise RuntimeError("All planned comparisons and independent checks must complete before packaging")
    runtime = {"python": platform.python_version(),
               "packages": {name: package_version(name) for name in ("numpy", "pandas", "scipy", "torch", "rdkit")}}
    (REPORT / "runtime.json").write_text(json.dumps(runtime, indent=2) + "\n")
    files = set()
    for folder in (RUNS, REPORT, PROJECT / "reports/knn_comparison_20261025/inputs"):
        for path in folder.rglob("*"):
            if path.is_file() and "archive_before_index_memory_fix" not in path.parts and not path.name.endswith(".tmp"):
                files.add(path)
    names = ["knn_dta_reference.py", "run_knn_shared.py", "prepare_knn_comparison.py", "summarize_knn_shared.py",
             "verify_knn_shared.py", "test_knn_dta_reference.py", "test_knn_shared.py", "package_knn_shared.py",
             "build_knn_comparison_tables.py"]
    files.update(PROJECT / "scripts" / name for name in names)
    files.update(PROJECT / "configs" / name for name in ("knn_shared_20261025.json", "knn_comparison_20261025.json"))
    files.update(PROJECT / "reports" / name for name in
                 ("knn_comparison_20261025/prepared_inputs.json", "bib_protocol_20261017/protocol.json",
                  "davis_target_id_20261025/amendment.json", "davis_target_id_20261025/progress_summary.json"))
    for dataset in ("davis", "kiba"):
        files.add(PROJECT / f"data/processed/{dataset}_dtiam_protocol/table.csv")
        for fold in range(5):
            files.add(PROJECT / f"data/processed/{dataset}_dtiam_protocol/folds/drug_coldstart/fold_{fold}.csv")
            stem = (f"bib_davis_target_id_f{fold}_frozen_prior" if dataset == "davis"
                    else f"kiba_dtiam_drugcold_f{fold}_frozen_morgan_prior")
            files.add(PROJECT / f"results/predictions/{stem}_predictions.csv")
    destination = PROJECT / "release/assets/knn_shared_comparison_20261025.tar.gz"
    destination.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = destination.with_suffix("").with_suffix(".manifest.json")
    manifest = [{"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size, "sha256": sha(path)}
                for path in sorted(files)]
    manifest_path.write_text(json.dumps({"files": manifest}, indent=2) + "\n")
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with tarfile.open(temporary, "w:gz") as archive:
        for path in sorted(files):
            archive.add(path, arcname=str(path.relative_to(ROOT)), recursive=False)
        archive.add(manifest_path, arcname="manifest.json", recursive=False)
    temporary.replace(destination)
    print(json.dumps({"archive": str(destination), "files": len(files), "bytes": destination.stat().st_size,
                      "sha256": sha(destination)}, indent=2))


if __name__ == "__main__":
    main()
