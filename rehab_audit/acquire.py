"""Fetch the public ScienceDB files declared in a versioned manifest."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import tempfile
import time
from urllib.parse import urlencode
from urllib.request import Request, urlopen


def digest(path: Path, algorithm: str = "sha256") -> str:
    result = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def fetch_files(manifest_path: Path, destination: Path) -> list[dict]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = manifest["files"]
    names = set()
    # Validate every destination before opening a connection or creating files.
    for entry in entries:
        name = entry["fileName"]
        if not isinstance(name, str) or not re.fullmatch(r"[0-9]{3}_[12]\.npy", name) or name in names:
            raise ValueError("unsafe or duplicate source filename")
        if type(entry["size"]) is not int or entry["size"] <= 0:
            raise ValueError("source size must be a positive integer")
        if not isinstance(entry["md5"], str) or not re.fullmatch(r"[a-fA-F0-9]{32}", entry["md5"]):
            raise ValueError("source MD5 must have 32 hexadecimal digits")
        if not isinstance(entry["id"], str) or not re.fullmatch(r"[a-zA-Z0-9]+", entry["id"]):
            raise ValueError("invalid public source file ID")
        names.add(name)
    destination.mkdir(parents=True, exist_ok=True)
    results = []
    for entry in entries:
        name = entry["fileName"]
        target = destination / name
        expected_md5 = entry["md5"].lower()
        valid = target.is_file() and target.stat().st_size == entry["size"] and digest(target, "md5") == expected_md5
        url = "https://china.scidb.cn/download?" + urlencode({"fileId": entry["id"]})
        if not valid:
            for attempt in range(3):
                temporary = None
                try:
                    request = Request(url, headers={"User-Agent": "REHAB-record-audit/0.1 academic reuse"})
                    with urlopen(request, timeout=90) as response, tempfile.NamedTemporaryFile(
                        mode="wb", dir=destination, prefix=name + ".", suffix=".part", delete=False
                    ) as stream:
                        temporary = Path(stream.name)
                        received = 0
                        for block in iter(lambda: response.read(1024 * 1024), b""):
                            received += len(block)
                            if received > entry["size"]:
                                raise ValueError(f"ScienceDB response exceeds declared size for {name}")
                            stream.write(block)
                    if temporary.stat().st_size != entry["size"] or digest(temporary, "md5") != expected_md5:
                        raise ValueError(f"ScienceDB size/MD5 verification failed for {name}")
                    temporary.replace(target)
                    break
                except OSError:
                    if attempt == 2:
                        raise
                    time.sleep(2 * (attempt + 1))
                finally:
                    if temporary is not None:
                        temporary.unlink(missing_ok=True)
        results.append({"file": name, "bytes": target.stat().st_size,
                        "md5": expected_md5, "sha256": digest(target), "url": url})
        print(f"Verified {name} ({target.stat().st_size} bytes)", flush=True)
    return results
