#!/usr/bin/env python
"""Targeted fetch of the nuScenes CAM_FRONT frames the CODA-clips benchmark needs.

The full nuScenes trainval blobs are ~400 GB. We only need the CAM_FRONT keyframes
(``samples/CAM_FRONT/``) for the 134 CODA anchors plus the surrounding 12 fps sweeps
(``sweeps/CAM_FRONT/``) within each clip's history window -- ~2.5k files, ~350 MB.

Both live as range-readable ZIP64 archives on the Hugging Face mirror, so this script
opens each zip over HTTP, reads only the central directory, and extracts just the
filenames the resolver asks for (buffered range reads -> ~1 request per file). It writes
them into the canonical nuScenes layout under ``nuscenes.dataroot`` so the builder picks
them up unchanged.

    python scripts/fetch_nuscenes_camfront.py            # resolve needed files + fetch
    python scripts/fetch_nuscenes_camfront.py --dry-run  # just report counts
"""

from __future__ import annotations

import argparse
import io
import os
import sys
import time
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed

_TIMEOUT = 30   # seconds per HTTP request -- urllib hangs forever without this
_RETRIES = 5

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from vista_umad.coda_clips.config import load_config  # noqa: E402
from vista_umad.coda_clips.index import load_scene_index  # noqa: E402
from vista_umad.coda_clips.sources import nuscenes as nu  # noqa: E402

# HF mirror (Xiaodong): CAM_FRONT keyframes + sweeps, as single ZIP64 archives.
REPOS = {
    "samples/CAM_FRONT/": "https://huggingface.co/datasets/Xiaodong/Nuscenes-v1.0-trainval-CAM_FRONT/resolve/main/samples.zip",
    "sweeps/CAM_FRONT/": "https://huggingface.co/datasets/Xiaodong/Nuscenes-v1.0-trainval-CAM_FRONT_Sweeps/resolve/main/sweeps.zip",
}


class BufferedHTTPRangeFile(io.RawIOBase):
    """Seekable read-only file over an HTTP range-serving URL, with a read-ahead cache.

    zipfile issues many small reads per entry; a contiguous read-ahead window collapses
    those into ~1 HTTP request per file (a JPEG entry + its local header fit one window).
    """

    def __init__(self, url: str, window: int = 1 << 18):
        self.url = url
        self.pos = 0
        self.window = window
        self._buf = b""
        self._buf_start = -1
        with urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=_TIMEOUT) as r:
            self.size = int(r.headers["Content-Length"])

    def seekable(self) -> bool:
        return True

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else (self.pos + off if whence == 1 else self.size + off)
        return self.pos

    def tell(self) -> int:
        return self.pos

    def _fetch(self, start: int, length: int) -> bytes:
        end = min(start + length, self.size) - 1
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={start}-{end}"})
        last = None
        for attempt in range(_RETRIES):
            try:
                with urllib.request.urlopen(req, timeout=_TIMEOUT) as r:
                    return r.read()
            except Exception as e:  # noqa: BLE001 -- network flakiness; retry with backoff
                last = e
                time.sleep(1.5 * (attempt + 1))
        raise OSError(f"range fetch failed after {_RETRIES} tries: {last}")

    def read(self, n=-1) -> bytes:
        if n is None or n < 0:
            n = self.size - self.pos
        if n == 0:
            return b""
        if not (self._buf_start <= self.pos and self.pos + n <= self._buf_start + len(self._buf)):
            length = max(n, self.window)
            self._buf = self._fetch(self.pos, length)
            self._buf_start = self.pos
        off = self.pos - self._buf_start
        data = self._buf[off : off + n]
        self.pos += len(data)
        return data


def needed_files(cfg) -> set[str]:
    """Resolve the relative CAM_FRONT filenames for all nuScenes scenes' clips."""
    scenes = [s for s in load_scene_index(cfg) if s.source == "nuscenes"]
    meta = nu.load_meta(cfg)
    sds = meta["sample_data"]
    files: set[str] = set()
    for s in scenes:
        seq = nu.resolve(cfg, s)
        if seq is None:
            continue
        for fr in seq.frames:
            files.add(sds[fr.key]["filename"])
    return files


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--workers", type=int, default=12, help="parallel range-read workers")
    args = ap.parse_args()

    cfg = load_config(args.config)
    dataroot = cfg.sources["nuscenes"].dataroot
    want = needed_files(cfg)
    by_prefix = {p: sorted(f for f in want if f.startswith(p)) for p in REPOS}
    for p, fs in by_prefix.items():
        print(f"need {len(fs):5d} files under {p}")
    if args.dry_run:
        return

    total_ok = total_missing = 0
    for prefix, url in REPOS.items():
        targets = [n for n in by_prefix[prefix]
                   if not (os.path.exists(os.path.join(dataroot, n)) and os.path.getsize(os.path.join(dataroot, n)) > 0)]
        already = len(by_prefix[prefix]) - len(targets)
        if not targets:
            print(f"[{prefix}] all {already} files already present")
            continue
        print(f"\n[{prefix}] {len(targets)} to fetch ({already} present); {args.workers} workers ...")

        # Partition across workers; each worker opens its own remote zip handle (zipfile
        # read is not thread-safe on a shared handle).
        chunks = [targets[i :: args.workers] for i in range(args.workers)]
        done = [0]

        def worker(names):
            zf = zipfile.ZipFile(BufferedHTTPRangeFile(url))
            present = set(zf.namelist())
            ok = miss = 0
            for name in names:
                if name not in present:
                    miss += 1
                    continue
                out = os.path.join(dataroot, name)
                os.makedirs(os.path.dirname(out), exist_ok=True)
                data = zf.read(name)
                with open(out, "wb") as fh:
                    fh.write(data)
                ok += 1
                done[0] += 1
                if done[0] % 250 == 0:
                    print(f"  {done[0]}/{len(targets)} fetched")
            return ok, miss

        ok = miss = 0
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            for fut in as_completed([ex.submit(worker, c) for c in chunks if c]):
                a, b = fut.result()
                ok += a
                miss += b
        print(f"[{prefix}] done: written {ok}, already-present {already}, missing {miss}")
        total_ok += ok
        total_missing += miss
    print(f"\nTOTAL: written {total_ok}, missing {total_missing}")


if __name__ == "__main__":
    main()
