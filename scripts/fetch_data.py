"""Download and verify the datasets this project needs.

Usage
-----
    python scripts/fetch_data.py                 # download anything missing, then verify
    python scripts/fetch_data.py --verify-only   # just check the files you already have
    python scripts/fetch_data.py --base-url URL  # download from somewhere else

By default the files come from this repo's GitHub Release tagged ``data-v1``.
Only the standard library is used, so this runs before ``pip install``.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

REPO = "definitelynotironman/BuildingClassifier"
RELEASE_TAG = "data-v1"
DEFAULT_BASE_URL = f"https://github.com/{REPO}/releases/download/{RELEASE_TAG}"

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# filename -> expected SHA-256
FILES = {
    "madrid_train.parquet": "0e106a07044d867f8c68e77de8c36826c491d0f26692b8cde753d1046ce24c9c",
    "amsterdam_data.parquet": "32388f3a9792b01d9831ca4d0fc4bfadba132a1c23c06e372d7a82e985be0719",
}


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def download(url: str, dest: Path) -> None:
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"  downloading {url}")
    with urllib.request.urlopen(url) as resp, tmp.open("wb") as out:
        total = int(resp.headers.get("Content-Length") or 0)
        done = 0
        while True:
            block = resp.read(1 << 20)
            if not block:
                break
            out.write(block)
            done += len(block)
            if total:
                print(f"\r  {done / 1e6:7.1f} / {total / 1e6:.1f} MB", end="", flush=True)
    print()
    tmp.replace(dest)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default=DEFAULT_BASE_URL, help="folder URL that holds the files")
    ap.add_argument("--verify-only", action="store_true", help="do not download, only check")
    args = ap.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    ok = True

    for name, expected in FILES.items():
        path = DATA_DIR / name
        print(f"{name}")
        if not path.exists():
            if args.verify_only:
                print("  missing")
                ok = False
                continue
            try:
                download(f"{args.base_url.rstrip('/')}/{name}", path)
            except Exception as exc:  # network errors, 404s, etc.
                print(f"  could not download: {exc}")
                print("  see data/README.md for other ways to get this file")
                ok = False
                continue

        actual = sha256(path)
        if actual == expected:
            print("  OK (checksum matches)")
        else:
            print("  WARNING: checksum does not match the version used for the reported results")
            print(f"    expected {expected}\n    got      {actual}")
            ok = False

    print("\nAll good. Next: run 3Notebook_Processing.ipynb" if ok else "\nSome files need attention (see above).")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
