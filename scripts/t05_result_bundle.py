"""Collect only top-level result metadata/images into one lossless feedback JSON.

Standard library only. No model/cache/trajectory/video loading, no inference.
Inputs remain untouched; exports go into a fresh independent directory.
"""
import argparse
import base64
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
FORMAT = "ls_imagine_result_bundle_v1"
MAX_FILE_BYTES = 32 * 1024**2
MAX_TOTAL_BYTES = 64 * 1024**2
EXTENSIONS = {".json", ".csv", ".jsonl", ".png"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def label(value):
    require(isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_-]{1,100}", value),
            "label must contain only letters, digits, underscores or hyphens")
    return value


def sources_for_run(directory, depth=2):
    """Follow only report directories; never read a referenced checkpoint."""
    require(type(depth) is int and 0 <= depth <= 3, "related depth must be 0..3")
    pending, seen, result, missing = [(Path(directory).resolve(), 0)], set(), [], []
    while pending:
        current, level = pending.pop(0)
        if current in seen:
            continue
        seen.add(current)
        report_path = current / "report.json"
        require(report_path.is_file() and not report_path.is_symlink(), f"missing report: {report_path}")
        record = json.loads(report_path.read_text(encoding="utf-8"))
        alias = label(f"stage_{len(result):02d}")
        result.append((alias, current))
        if level >= depth:
            continue
        arguments = record.get("arguments", {})
        for key in ("eval_dir", "check_dir", "repair_verify_dir"):
            value = arguments.get(key)
            if not value:
                continue
            candidate = Path(value)
            candidate = (candidate if candidate.is_absolute() else ROOT / candidate).resolve()
            if (candidate / "report.json").is_file():
                pending.append((candidate, level + 1))
            else:
                missing.append(dict(source=str(current), argument=key, directory=str(candidate)))
        checkpoint = arguments.get("checkpoint")
        if checkpoint:
            candidate = Path(checkpoint)
            candidate = (candidate if candidate.is_absolute() else ROOT / candidate).resolve().parent
            if (candidate / "report.json").is_file():
                pending.append((candidate, level + 1))
    return result, missing


def selected_files(directory):
    require(directory.is_dir() and (directory / "report.json").is_file(), f"not a result directory: {directory}")
    files = sorted(p for p in directory.iterdir() if p.is_file() and
                   (p.suffix.lower() in EXTENSIONS or p.name == "error.txt") and
                   not p.name.startswith(("T05_", "result_bundle", "bundle_index")))
    require(all(not p.is_symlink() and p.resolve().parent == directory for p in files),
            "result symlinks are not exported")
    return files


def build_bundle(sources, output_directory, name="feedback", missing_related=None):
    sources = [(label(alias), Path(path).resolve()) for alias, path in sources]
    require(sources and len({alias for alias, _ in sources}) == len(sources) and
            len({path for _, path in sources}) == len(sources), "source aliases/directories must be unique")
    output = Path(output_directory).resolve()
    require(all(output != path and path not in output.parents and output not in path.parents
                for _, path in sources), "export must be independent of all input directories")
    label(name)
    records, blobs, inventory, snapshots = [], [], [], {}
    total = 0
    for alias, directory in sources:
        report = json.loads((directory / "report.json").read_text(encoding="utf-8"))
        # Preserve passed, failed and interrupted results alike, without reclassifying acceptance.
        inventory.append(dict(alias=alias, directory=str(directory), command=report.get("command"),
            status=report.get("status", "unfinished"), worker_version=report.get("worker_version"),
            numerical_gate=report.get("numerical_gate"), behavior_accepted=report.get("behavior_accepted"),
            t06_approved=report.get("t06_approved"), report_sha256=sha((directory / "report.json").read_bytes())))
        manifest_path = directory / "evaluation_manifest.json"
        bound = json.loads(manifest_path.read_text(encoding="utf-8")).get("summary_artifacts", {}) if manifest_path.is_file() else {}
        diagnosis_path = directory / "diagnosis_manifest.json"
        if diagnosis_path.is_file():
            bound.update(json.loads(diagnosis_path.read_text(encoding="utf-8")).get("artifacts", {}))
        for path in selected_files(directory):
            stat = path.stat()
            require(stat.st_size <= MAX_FILE_BYTES, f"metadata file exceeds limit; nothing truncated: {path}")
            raw = path.read_bytes()
            digest = sha(raw)
            require(path.stat().st_size == stat.st_size and path.stat().st_mtime_ns == stat.st_mtime_ns,
                    f"result is changing during collection: {path}")
            require(path.name not in bound or digest == bound[path.name], f"manifest SHA256 mismatch: {path}")
            total += len(raw)
            require(total <= MAX_TOTAL_BYTES, "result metadata exceeds total limit; nothing truncated")
            encoding = "base64" if path.suffix.lower() == ".png" else "utf-8"
            content = base64.b64encode(raw).decode("ascii") if encoding == "base64" else raw.decode("utf-8")
            if encoding == "base64":
                require(raw.startswith(b"\x89PNG\r\n\x1a\n"), f"invalid PNG: {path}")
            elif path.suffix.lower() == ".json":
                json.loads(content)  # Parse for validity, but retain exact bytes/text and original formatting.
            records.append(dict(source=alias, name=path.name, bytes=len(raw), sha256=digest,
                                encoding=encoding, content=content))
            blobs.append((f"sources/{alias}/{path.name}", raw))
            snapshots[path] = (stat.st_size, stat.st_mtime_ns, digest)
        exported_report = next(item for item in records if item["source"] == alias and item["name"] == "report.json")
        require(exported_report["sha256"] == inventory[-1]["report_sha256"],
                f"report changed while collecting inventory: {directory}")
    bundle = dict(format=FORMAT, created_utc=datetime.now(timezone.utc).isoformat(),
        source_inventory=inventory, files=records, file_count=len(records), original_bytes=total,
        missing_related=missing_related or [], images_embedded=True, source_inputs_unchanged=True,
        excluded_artifacts=["models", "checkpoints", "cache arrays", "trajectories", "videos", "nested trial directories"],
        interpretation="Each UTF-8 content is the exact source text; PNG content is base64. Decode and verify SHA256. "
                       "Packaging preserves source acceptance, including failures; it is not a new scientific acceptance.")
    raw_bundle = (json.dumps(bundle, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n").encode("utf-8")
    require(len(raw_bundle) <= MAX_TOTAL_BYTES * 2, "feedback JSON exceeds limit")
    for path, (size, modified, digest) in snapshots.items():
        stat = path.stat()
        require((stat.st_size, stat.st_mtime_ns, sha(path.read_bytes())) == (size, modified, digest),
                f"source changed before export: {path}")
    output.mkdir(parents=True, exist_ok=False)
    stem = f"T05_{name}_results"
    json_path, zip_path = output / f"{stem}.json", output / f"{stem}.zip"
    json_path.write_bytes(raw_bundle)
    with zipfile.ZipFile(zip_path, "x", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(json_path.name, raw_bundle)
        for relative, content in blobs:
            archive.writestr(relative, content)
    # Validate transport bytes and every source entry before exposing successful export.
    with zipfile.ZipFile(zip_path, "r") as archive:
        require(archive.testzip() is None and archive.read(json_path.name) == raw_bundle and
                all(archive.read(relative) == content for relative, content in blobs), "ZIP verification failed")
    index = dict(format=FORMAT, json_path=str(json_path), json_sha256=sha(raw_bundle),
                 zip_path=str(zip_path), zip_sha256=sha(zip_path.read_bytes()), file_count=len(records),
                 source_count=len(sources), original_bytes=total, source_inputs_unchanged=True)
    (output / "bundle_index.json").write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    return index


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", help="one run; associated check/verify/train reports are found automatically")
    parser.add_argument("--input", action="append", default=[], help="explicit alias=/absolute/result-directory")
    parser.add_argument("--related-depth", type=int, default=2)
    parser.add_argument("--output-dir")
    parser.add_argument("--name")
    args = parser.parse_args()
    try:
        require(bool(args.run_dir) != bool(args.input), "choose --run-dir OR explicit --input entries")
        if args.run_dir:
            sources, missing = sources_for_run(args.run_dir, args.related_depth)
            name = args.name or Path(args.run_dir).name
        else:
            require(all("=" in item for item in args.input), "--input needs alias=directory")
            sources, missing = [item.split("=", 1) for item in args.input], []
            name = args.name or "feedback"
        tag = datetime.now().strftime("%Y%m%dT%H%M%S_%f")
        output = Path(args.output_dir) if args.output_dir else ROOT / "relevance_map/t05_feedback" / f"{name}_{tag}"
        index = build_bundle(sources, output, name, missing)
        print(json.dumps(index, ensure_ascii=False), flush=True)
        print(f"FEEDBACK_FILE={index['json_path']}", flush=True)
        print(f"FEEDBACK_ZIP={index['zip_path']}", flush=True)
        return 0
    except (OSError, ValueError, KeyError, TypeError, UnicodeError, zipfile.BadZipFile) as error:
        print(f"[FAIL] result_bundle: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    sys.exit(main())
