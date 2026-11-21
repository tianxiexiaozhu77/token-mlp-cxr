"""Read only missing NFS token-cache regions; never modify training sources.

mincore is a residency snapshot, not a promise that pages stay resident. Reads
are limited to 1 MiB-aligned blocks containing a missing page at scan time.
Fragmented holes can expand the read volume; the JSON records both byte counts.
This independent helper does not form part of the frozen training program.
"""
import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import ctypes
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.baseline_token_classification.common import save, sha256

ALIGNMENT = 1024 ** 2
SCAN_WINDOW = 256 * ALIGNMENT
MAX_SPAN = 8 * ALIGNMENT
READ_CHUNK = ALIGNMENT


def missing_blocks(fd, size, offset, length):
    """Return aligned blocks from one bounded, read-only mincore mapping."""
    page_size = os.sysconf("SC_PAGE_SIZE")
    if offset % page_size or ALIGNMENT % page_size:
        raise ValueError("Mapping/block offsets must be page-aligned")
    libc = ctypes.CDLL(None, use_errno=True)
    libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int,
                          ctypes.c_int, ctypes.c_int, ctypes.c_long]
    libc.mmap.restype = ctypes.c_void_p
    libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t,
                             ctypes.POINTER(ctypes.c_ubyte)]
    libc.mincore.restype = ctypes.c_int
    libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    libc.munmap.restype = ctypes.c_int
    # PROT_READ=1, MAP_SHARED=1. No write permission or page touching is used.
    address = libc.mmap(None, length, 1, 1, fd, offset)
    if address == ctypes.c_void_p(-1).value:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    try:
        count = (length + page_size - 1) // page_size
        residency = (ctypes.c_ubyte * count)()
        if libc.mincore(address, length, residency):
            error = ctypes.get_errno()
            raise OSError(error, os.strerror(error))
        blocks, pages, missing_bytes = set(), 0, 0
        for index, present in enumerate(residency):
            if not (present & 1):
                position = offset + index * page_size
                blocks.add(position // ALIGNMENT)
                pages += 1
                missing_bytes += min(page_size, size - position)
        return sorted(blocks), pages, missing_bytes
    finally:
        if libc.munmap(address, length):
            error = ctypes.get_errno()
            raise OSError(error, os.strerror(error))


def merged_spans(blocks, size):
    """Coalesce adjacent aligned blocks, at most 8 MiB per work item."""
    start = previous = None
    for block in blocks:
        if start is not None and (block != previous + 1 or (block - start + 1) * ALIGNMENT > MAX_SPAN):
            yield start * ALIGNMENT, min((previous + 1) * ALIGNMENT, size) - start * ALIGNMENT
            start = None
        if start is None:
            start = block
        previous = block
    if start is not None:
        yield start * ALIGNMENT, min((previous + 1) * ALIGNMENT, size) - start * ALIGNMENT


def read_span(fd, offset, length):
    done = 0
    while done < length:
        value = os.pread(fd, min(READ_CHUNK, length - done), offset + done)
        if not value:
            raise IOError("Unexpected EOF during read-only missing-page warming")
        done += len(value)
    return done


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--models", required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--scan-only", action="store_true")
    parser.add_argument("--max-read-gib", type=float,
                        help="Optional maximum aligned read volume; stop with read_budget_reached status")
    args = parser.parse_args()
    if not 1 <= args.workers <= 4:
        parser.error("workers must be between 1 and 4 (default: 2)")
    if args.max_read_gib is not None and args.max_read_gib <= 0:
        parser.error("max-read-gib must be positive")
    run = args.run_dir.resolve()
    protocol_path = run / "PROTOCOL.json"
    protocol_hash = sha256(protocol_path)
    protocol = json.loads(protocol_path.read_text())
    model_names = args.models.split(",")
    if not all(model_names) or len(set(model_names)) != len(model_names):
        parser.error("models must be a comma-separated list without empty/duplicate names")
    sources, seen = [], set()
    for model in model_names:
        for spec in protocol["caches"][model + "/" + args.dataset]:
            if sha256(spec["path"]) != spec["sha256"]:
                raise ValueError("Cache record hash changed: " + spec["path"])
            cache = json.loads(Path(spec["path"]).read_text())
            filenames = ([value["path"] for value in cache["token_shards"]] if "token_shards" in cache
                         else [cache["files"]["tokens.npy"]["path"]])
            for filename in filenames:
                if filename in seen:
                    continue
                seen.add(filename)
                expected = spec["files"][filename]
                stat = Path(filename).stat()
                if stat.st_size != expected["bytes"] or stat.st_mtime_ns != expected["mtime_ns"]:
                    raise ValueError("Token file metadata changed: " + filename)
                sources.append(dict(path=filename, bytes=stat.st_size, mtime_ns=stat.st_mtime_ns,
                    cache_record=spec["path"], cache_record_sha256=spec["sha256"],
                    device=stat.st_dev, inode=stat.st_ino, scanned_bytes=0, missing_pages=0,
                    missing_bytes_at_scan=0, aligned_candidate_bytes=0, scheduled_bytes=0, read_bytes=0))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output = run / "io_warming" / ("missing_" + args.dataset + "_" + "_".join(model_names) + "_" + stamp + ".json")
    started = last = time.time()
    state = dict(status="running", read_only=True, scan_only=args.scan_only, workers=args.workers,
        pid=os.getpid(), helper_sha256=sha256(__file__), protocol_sha256=protocol_hash,
        sources=sources, alignment_bytes=ALIGNMENT, scan_window_bytes=SCAN_WINDOW,
        max_span_bytes=MAX_SPAN, read_chunk_bytes=READ_CHUNK, max_pending_tasks=2 * args.workers,
        mincore_is_snapshot=True, scan_complete=False, files_unchanged=None,
        max_read_gib=args.max_read_gib, scheduled_bytes=0, read_bytes=0, seconds=0.)
    save(output, state)
    stop = {"signal": None}
    def on_stop(signum, frame):
        stop["signal"] = signum
    signal.signal(signal.SIGTERM, on_stop)
    signal.signal(signal.SIGINT, on_stop)
    budget = int(args.max_read_gib * 1024 ** 3) if args.max_read_gib is not None else None
    pending, descriptors, budget_reached = {}, [], False
    def progress(force=False):
        nonlocal last
        if force or time.time() - last >= 20:
            state["seconds"] = time.time() - started
            save(output, state)
            last = time.time()
            print(f"MISSING_WARM read={state['read_bytes']/1024**3:.3f} GiB "
                  f"scanned={sum(s['scanned_bytes'] for s in sources)/1024**3:.2f} GiB "
                  f"seconds={state['seconds']:.1f}", flush=True)
    def collect():
        completed, _ = wait(pending, timeout=5, return_when=FIRST_COMPLETED)
        for future in completed:
            source = pending.pop(future)
            count = future.result()
            source["read_bytes"] += count
            state["read_bytes"] += count
        progress()
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for source in sources:
                fd = os.open(source["path"], os.O_RDONLY)
                descriptors.append(fd)
                stat = os.fstat(fd)
                if (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns) != (
                        source["device"], source["inode"], source["bytes"], source["mtime_ns"]):
                    raise ValueError("Opened file no longer matches validated path: " + source["path"])
                for offset in range(0, source["bytes"], SCAN_WINDOW):
                    if stop["signal"] or budget_reached:
                        break
                    length = min(SCAN_WINDOW, source["bytes"] - offset)
                    blocks, pages, missing_bytes = missing_blocks(fd, source["bytes"], offset, length)
                    source["scanned_bytes"] += length
                    source["missing_pages"] += pages
                    source["missing_bytes_at_scan"] += missing_bytes
                    spans = list(merged_spans(blocks, source["bytes"]))
                    source["aligned_candidate_bytes"] += sum(length for _, length in spans)
                    if not args.scan_only:
                        for start, count in spans:
                            if stop["signal"]:
                                break
                            if budget is not None and state["scheduled_bytes"] + count > budget:
                                budget_reached = True
                                break
                            while len(pending) >= 2 * args.workers:
                                collect()
                                if stop["signal"]:
                                    break
                            if stop["signal"]:
                                break
                            pending[pool.submit(read_span, fd, start, count)] = source
                            source["scheduled_bytes"] += count
                            state["scheduled_bytes"] += count
                    progress()
                if stop["signal"] or budget_reached:
                    break
            while pending:
                collect()
        for source in sources:
            stat = Path(source["path"]).stat()
            if stat.st_size != source["bytes"] or stat.st_mtime_ns != source["mtime_ns"]:
                raise ValueError("Token file changed during warming: " + source["path"])
        if sha256(protocol_path) != protocol_hash:
            raise ValueError("Frozen protocol changed during warming")
        state.update(files_unchanged=True, scan_complete=all(s["scanned_bytes"] == s["bytes"] for s in sources),
                     status="stopped" if stop["signal"] else "read_budget_reached" if budget_reached else "complete",
                     stop_signal=stop["signal"], aligned_read_amplification=(
                         sum(s["aligned_candidate_bytes"] for s in sources) /
                         max(1, sum(s["missing_bytes_at_scan"] for s in sources))))
        progress(force=True)
        print(json.dumps(dict(status=state["status"], output=str(output), read_bytes=state["read_bytes"],
                              scan_complete=state["scan_complete"])), flush=True)
    except BaseException as error:
        state.update(status="failed", error=repr(error), seconds=time.time() - started)
        save(output, state)
        raise
    finally:
        for fd in descriptors:
            os.close(fd)


if __name__ == "__main__":
    main()
