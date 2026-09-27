"""Fetch a published BioM3 weights bundle from an OCI registry."""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys

DEFAULT_REGISTRY = "ghcr.io/natural-machine/biom3-weights"
TEST_WEIGHTS_REGISTRY = "ghcr.io/natural-machine/biom3-test-weights"
TEST_WEIGHTS_TAG = "latest"
WEIGHTS_PREFIX = "weights/"

_ORAS_MISSING = (
    "oras not found on PATH. It ships in the BioM3 container images; on a bare "
    "host install it from https://oras.land/docs/installation"
)


def _sha256_file(path, chunk_size=1 << 22):
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _require_oras():
    if shutil.which("oras") is None:
        raise RuntimeError(_ORAS_MISSING)


def _run(cmd):
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"{' '.join(cmd[:3])} failed:\n{result.stderr.strip()}"
        )
    return result.stdout


def list_bundle_files(ref):
    """Return [(path, digest, size)] for every file in the bundle.

    Read from the OCI manifest, which is a few KB, so nothing large is
    transferred to find out what a bundle holds.
    """
    manifest = json.loads(_run(["oras", "manifest", "fetch", ref]))
    files = []
    for layer in manifest.get("layers", []):
        title = (layer.get("annotations") or {}).get(
            "org.opencontainers.image.title"
        )
        if title:
            files.append((title, layer["digest"], layer.get("size", 0)))
    return files


def _plan(files, output_dir, include_configs):
    """Split bundle files into (to_fetch, present, conflicting, ignored)."""
    to_fetch, present, conflicting, ignored = [], [], [], []
    for path, digest, size in files:
        if path.startswith(WEIGHTS_PREFIX):
            rel = path[len(WEIGHTS_PREFIX):]
        elif include_configs and path != "MANIFEST.json":
            rel = path
        else:
            ignored.append(path)
            continue
        dest = os.path.join(output_dir, rel)
        if os.path.isfile(dest):
            if _sha256_file(dest) == digest:
                present.append(rel)
            else:
                conflicting.append(rel)
        else:
            to_fetch.append((rel, dest, digest, size))
    return to_fetch, present, conflicting, ignored


def fetch_bundle(bundle, output_dir, registry=DEFAULT_REGISTRY,
                 include_configs=False, force=False, dry_run=False):
    """Fetch ``registry:bundle`` into ``output_dir``, skipping files already there.

    Only ``weights/`` entries are taken unless *include_configs*. Each file is
    matched against the registry's own content digest, so an existing file is
    re-downloaded only when its bytes differ.
    """
    _require_oras()
    ref = f"{registry}:{bundle}"
    repo = registry

    files = list_bundle_files(ref)
    if not files:
        raise RuntimeError(f"{ref}: manifest lists no files")

    dest_root = os.path.abspath(output_dir)
    to_fetch, present, conflicting, ignored = _plan(
        files, dest_root, include_configs
    )

    if conflicting and not force:
        raise RuntimeError(
            "these files differ from the published bundle; pass --force to "
            "replace them:\n  " + "\n  ".join(conflicting)
        )
    if conflicting:
        for rel in conflicting:
            digest = next(d for p, d, _ in files if p.endswith(rel))
            size = next(s for p, _, s in files if p.endswith(rel))
            to_fetch.append((rel, os.path.join(dest_root, rel), digest, size))

    total = sum(size for _, _, _, size in to_fetch)
    print(f"{ref}", file=sys.stderr)
    print(f"  {len(present)} already present, {len(to_fetch)} to fetch "
          f"({total / 1e9:.2f} GB)", file=sys.stderr)
    if ignored:
        print(f"  {len(ignored)} non-weight file(s) skipped "
              f"(--include_configs to take them)", file=sys.stderr)
    if dry_run:
        for rel, _, _, size in to_fetch:
            print(f"  would fetch {rel} ({size / 1e6:.1f} MB)", file=sys.stderr)
        return dest_root
    if not to_fetch:
        print("  nothing to do", file=sys.stderr)
        return dest_root

    for i, (rel, dest, digest, size) in enumerate(to_fetch, 1):
        os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
        tmp = dest + ".partial"
        print(f"  [{i}/{len(to_fetch)}] {rel} ({size / 1e6:.1f} MB)",
              file=sys.stderr)
        _run(["oras", "blob", "fetch", "--output", tmp, f"{repo}@{digest}"])
        got = _sha256_file(tmp)
        if got != digest:
            os.remove(tmp)
            raise RuntimeError(f"{rel}: digest {got}, expected {digest}")
        os.replace(tmp, dest)

    print(f"  done -> {dest_root}", file=sys.stderr)
    return dest_root


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(
        prog="biom3_fetch_weights",
        description="Fetch a published BioM3 weights bundle from an OCI registry.",
        epilog=(
            "examples:\n"
            "  biom3_fetch_weights run1_base -o ./weights\n"
            "  biom3_fetch_weights --test_weights -o ./weights   # everything the tests need\n"
            "  biom3_fetch_weights run1_base -o ./weights --dry_run\n"
            "\nlist available bundles:\n"
            f"  oras repo tags {DEFAULT_REGISTRY}"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("bundle", nargs="?",
                        help="bundle tag to fetch, e.g. run1_base")
    parser.add_argument("-o", "--output_dir", required=True,
                        help="weights root to merge into, e.g. ./weights")
    parser.add_argument("--registry", default=DEFAULT_REGISTRY,
                        help=f"OCI repository holding the bundles (default: {DEFAULT_REGISTRY})")
    parser.add_argument("--test_weights", action="store_true",
                        help="shortcut for the published set of weights the test suite "
                             f"needs ({TEST_WEIGHTS_REGISTRY})")
    parser.add_argument("--include_configs", action="store_true",
                        help="also take the bundle's non-weight files")
    parser.add_argument("--force", action="store_true",
                        help="replace local files whose bytes differ from the bundle")
    parser.add_argument("--dry_run", action="store_true",
                        help="report what would be fetched and exit")
    args = parser.parse_args(argv)

    if args.test_weights:
        if args.bundle:
            parser.error("give either a bundle tag or --test_weights, not both")
        args.bundle = TEST_WEIGHTS_TAG
        if args.registry == DEFAULT_REGISTRY:
            args.registry = TEST_WEIGHTS_REGISTRY
    elif not args.bundle:
        parser.error("a bundle tag is required (or pass --test_weights)")
    return args


def main(args=None):
    if args is None:
        args = parse_arguments()
    try:
        fetch_bundle(args.bundle, args.output_dir, registry=args.registry,
                     include_configs=args.include_configs, force=args.force,
                     dry_run=args.dry_run)
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0
