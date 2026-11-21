import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def read_dataset(path):
    meta = json.loads(Path(path).read_text())
    if sha256(meta["samples_file"]) != meta["samples_sha256"]:
        raise ValueError("Dataset sample checksum changed")
    rows = json.loads(Path(meta["samples_file"]).read_text())
    return meta, rows
