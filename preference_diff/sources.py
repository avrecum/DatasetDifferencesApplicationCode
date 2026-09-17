"""Pinned metadata first; no execution of custom dataset loaders."""

from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import os
from pathlib import Path, PurePosixPath
import time
import zipfile

import requests
from PIL import Image

from .io import file_hash, read_rows, write_json, clean


def default_cache():
    if os.environ.get("HF_HOME"):
        return Path(os.environ["HF_HOME"])
    if os.environ.get("PROJECT"):
        return Path(os.environ["PROJECT"]) / "avrecum/.hf"
    return Path.home() / ".cache/huggingface"


def hub_file(source, filename, cache):
    from huggingface_hub import hf_hub_download

    try:
        return Path(
            hf_hub_download(
                source["repo"],
                filename,
                repo_type=source["type"],
                revision=source["revision"],
                cache_dir=str(Path(cache) / "hub"),
                local_files_only=True,
            )
        )
    except FileNotFoundError:
        pass
    return Path(
        hf_hub_download(
            source["repo"],
            filename,
            repo_type=source["type"],
            revision=source["revision"],
            cache_dir=str(Path(cache) / "hub"),
        )
    )


def prefetch_model(source, cache):
    from huggingface_hub import snapshot_download

    return snapshot_download(
        source["repo"],
        revision=source["revision"],
        cache_dir=str(Path(cache) / "hub"),
        allow_patterns=["*.json", "*.py", "*.model", "*.safetensors"],
        max_workers=4,
    )


def inspect_sources(lock, cache, output):
    """Download all metadata (small) and model Python/config for review, not weights/images."""
    from huggingface_hub import HfApi

    api = HfApi()
    result = {"sources": {}, "metadata": {}}
    for name, source in lock.items():
        info = api.repo_info(
            source["repo"],
            repo_type=source["type"],
            revision=source["revision"],
            files_metadata=True,
        )
        if info.sha != source["revision"]:
            raise ValueError("Source lock must contain immutable commit IDs")
        files = [dict(path=f.rfilename, size=f.size) for f in info.siblings]
        result["sources"][name] = dict(
            **source,
            files=files,
            url=f"https://huggingface.co/{'datasets/' if source['type'] == 'dataset' else ''}{source['repo']}/tree/{source['revision']}",
        )
        for f in files:
            filename = f["path"]
            if filename.endswith(
                (".parquet", ".py", ".json", "README.md")
            ) and not filename.endswith("tokenizer.json"):
                path = hub_file(source, filename, cache)
                entry = dict(path=str(path), sha256=file_hash(path))
                if filename.endswith(".parquet"):
                    import pyarrow.parquet as pq

                    table = pq.read_table(path)
                    entry.update(
                        rows=table.num_rows,
                        schema=str(table.schema),
                        example=clean(table.slice(0, 2).to_pylist()),
                    )
                result["metadata"][f"{name}/{filename}"] = entry
    write_json(Path(output) / "source_inspection.json", result)
    write_json(Path(output) / "sources.lock.json", lock)
    return result


def load_metadata(dataset, lock, cache, inspection=None):
    if dataset == "imagereward":
        rows = []
        for split in ("train", "validation", "test"):
            for row in read_rows(
                hub_file(lock[dataset], f"metadata-{split}.parquet", cache)
            ):
                row["original_split"] = split
                rows.append(row)
        return rows, None
    from huggingface_hub import HfApi

    result = []
    for key in ("pickapic_rankings", "pickapic_images"):
        src = lock[key]
        files = HfApi().list_repo_files(
            src["repo"], repo_type="dataset", revision=src["revision"]
        )
        rows = []
        for filename in sorted(f for f in files if f.endswith(".parquet")):
            rows.extend(read_rows(hub_file(src, filename, cache)))
        result.append(rows)
    return tuple(result)


def validate_image(path):
    """Hash both original bytes and decoded RGB pixels (catches re-encoded exact copies)."""
    with Image.open(path) as im:
        im.load()
        rgb = im.convert("RGB")
        h = hashlib.sha256()
        h.update(f"RGB:{rgb.width}:{rgb.height}:".encode())
        h.update(rgb.tobytes())
        return dict(
            content_hash=file_hash(path),
            pixel_hash=h.hexdigest(),
            width=rgb.width,
            height=rgb.height,
            aspect_ratio=rgb.width / rgb.height,
            image_format=im.format,
        )


def download_url(url, path, retries=2, timeout=30, max_bytes=50 * 1024 * 1024):
    path = Path(path)
    if path.exists():
        validate_image(path)
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    for attempt in range(retries + 1):
        try:
            with requests.get(url, timeout=(10, timeout), stream=True) as response:
                response.raise_for_status()
                size = 0
                with tmp.open("wb") as f:
                    for block in response.iter_content(1024 * 128):
                        size += len(block)
                        if size > max_bytes:
                            raise ValueError("Image exceeds download byte limit")
                        f.write(block)
            validate_image(tmp)
            os.replace(tmp, path)
            return path
        except (requests.RequestException, OSError, ValueError) as exc:
            tmp.unlink(missing_ok=True)
            if (
                isinstance(exc, requests.HTTPError)
                and exc.response is not None
                and exc.response.status_code in (400, 401, 403, 404)
            ):
                raise  # Permanent source/access errors cannot be fixed by identical retries.
            if attempt == retries:
                raise
            time.sleep(min(2**attempt, 4))


def safe_member(name):
    p = PurePosixPath(name)
    if p.is_absolute() or ".." in p.parts or "\\" in name or ":" in name:
        raise ValueError(f"Unsafe archive path: {name}")
    return p


class HTTPRangeReader(io.RawIOBase):
    """Seekable bounded HTTP reader for ZIP members, avoiding whole 600–700 MB shards."""

    def __init__(self, url, size, block_size=1024 * 1024, timeout=30, retries=2):
        self.url, self.size, self.block_size = url, size, block_size
        self.timeout, self.retries, self.position = timeout, retries, 0
        self.blocks = {}
        self.bytes_downloaded = 0

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        self.position = (
            offset
            if whence == 0
            else self.position + offset
            if whence == 1
            else self.size + offset
        )
        if self.position < 0:
            raise ValueError("Negative seek")
        return self.position

    def read(self, size=-1):
        size = (
            self.size - self.position
            if size < 0
            else min(size, self.size - self.position)
        )
        if size <= 0:
            return b""
        result = []
        while size:
            start = self.position // self.block_size * self.block_size
            end = min(start + self.block_size, self.size) - 1
            if start not in self.blocks:
                for attempt in range(self.retries + 1):
                    try:
                        sep = "&" if "?" in self.url else "?"
                        # Distinct query prevents intermediaries replaying a different Range.
                        with requests.get(
                            f"{self.url}{sep}range_start={start}&range_end={end}",
                            headers={"Range": f"bytes={start}-{end}"},
                            timeout=(10, self.timeout),
                            stream=True,
                        ) as r:
                            r.raise_for_status()
                            if (
                                r.status_code != 206
                                or r.headers.get("Content-Range")
                                != f"bytes {start}-{end}/{self.size}"
                            ):
                                raise ValueError(
                                    "Server did not honor exact HTTP byte range; no unbounded fallback"
                                )
                            data = r.content
                        if len(data) != end - start + 1:
                            raise ValueError("Truncated HTTP range")
                        self.bytes_downloaded += len(data)
                        if len(self.blocks) >= 8:
                            self.blocks.pop(next(iter(self.blocks)))
                        self.blocks[start] = data
                        break
                    except (requests.RequestException, ValueError):
                        if attempt == self.retries:
                            raise
                        time.sleep(min(2**attempt, 4))
            offset = self.position - start
            chunk = self.blocks[start][offset : offset + size]
            result.append(chunk)
            self.position += len(chunk)
            size -= len(chunk)
        return b"".join(result)


def extract_required(zip_path, source_paths, destination, errors=None):
    """Extract explicitly requested, uniquely matched members; never extractall."""
    destination = Path(destination)
    resolved = {}
    with zipfile.ZipFile(zip_path) as archive:
        by_base = {}
        for info in archive.infolist():
            safe_member(info.filename)
            if (info.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError("Archive symlinks are not allowed")
            by_base.setdefault(PurePosixPath(info.filename).name, []).append(info)
        for source in source_paths:
            safe_member(source)
            candidates = by_base.get(PurePosixPath(source).name, [])
            if len(candidates) != 1:
                continue
            info = candidates[0]
            if info.file_size > 50 * 1024 * 1024:
                raise ValueError("Unreasonably large image archive member")
            target = destination / hashlib.sha256(source.encode()).hexdigest()
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                tmp = target.with_suffix(".part")
                with archive.open(info) as src, tmp.open("wb") as dst:
                    import shutil

                    shutil.copyfileobj(src, dst)
                try:
                    validate_image(tmp)
                except (OSError, ValueError) as exc:
                    if errors is not None:
                        errors[source] = (
                            f"archive image decode: {type(exc).__name__}: {exc}"
                        )
                    tmp.unlink(missing_ok=True)
                    continue
                os.replace(tmp, target)
            resolved[source] = str(target)
    return resolved


def resolve_images(
    manifest, lock, cache, workers=4, retries=2, timeout=30, local_root=None
):
    required = {
        i
        for e in manifest["events"]
        if e["sampled"] and not e["exclusion_reason"]
        for i in e["candidate_ids"]
    }
    selected = [i for i in manifest["images"] if i["image_id"] in required]
    base = Path(cache) / "preference_diff/images" / manifest["audit"]["dataset"]
    if manifest["audit"]["dataset"] == "imagereward" and local_root is None:
        # Metadata paths identify the exact archive: train/train_1/filename, etc.
        by_archive = {}
        for im in selected:
            source = im["original_path_or_url"]
            parts = safe_member(source).parts
            shard = next(
                (
                    p
                    for p in parts
                    if p.startswith(("train_", "validation_", "test_"))
                    and p.rsplit("_", 1)[-1].isdigit()
                ),
                None,
            )
            if shard is None:
                raise ValueError(
                    f"Cannot infer ImageReward archive from metadata path: {source}"
                )
            split = shard.rsplit("_", 1)[0]
            by_archive.setdefault(f"images/{split}/{shard}.zip", []).append(im)

        def resolve_archive(item):
            filename, ims = item
            print(f"Resolving {len(ims)} required images from {filename}", flush=True)
            try:
                destination = base / lock["imagereward"]["revision"]
                paths = {}
                missing = []
                for im in ims:
                    src = im["original_path_or_url"]
                    target = destination / hashlib.sha256(src.encode()).hexdigest()
                    if target.exists():
                        paths[src] = str(target)
                    else:
                        missing.append(src)
                if missing:
                    from huggingface_hub import HfApi, hf_hub_url

                    source = lock["imagereward"]
                    info = HfApi().get_paths_info(
                        source["repo"],
                        [filename],
                        repo_type="dataset",
                        revision=source["revision"],
                    )[0]
                    url = hf_hub_url(
                        source["repo"],
                        filename,
                        repo_type="dataset",
                        revision=source["revision"],
                    )
                    reader = HTTPRangeReader(
                        url, info.size, timeout=timeout, retries=retries
                    )
                    member_errors = {}
                    paths.update(
                        extract_required(reader, missing, destination, member_errors)
                    )
                    for im in ims:
                        if im["original_path_or_url"] in member_errors:
                            im["error"] = member_errors[im["original_path_or_url"]]
                    print(
                        f"Fetched {reader.bytes_downloaded / 1e6:.1f} MB by ZIP byte ranges (archive {info.size / 1e6:.1f} MB)",
                        flush=True,
                    )
                for im in ims:
                    im["local_path"] = paths.get(im["original_path_or_url"])
            except Exception as exc:
                for im in ims:
                    im["error"] = f"archive: {type(exc).__name__}: {exc}"

        # Shards own disjoint canonical paths. Each worker keeps only eight range
        # blocks and writes only requested images; no whole-archive fallback.
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            list(pool.map(resolve_archive, sorted(by_archive.items())))

    def resolve(im):
        try:
            if local_root:
                source = im["original_path_or_url"]
                if im["dataset"] == "pickapic":
                    uid = im["image_id"].split(":", 1)[1]
                    safe_member(uid)
                    matches = [
                        p
                        for ext in (".png", ".jpg", ".jpeg", ".webp", ".image")
                        if (p := Path(local_root) / (uid + ext)).is_file()
                    ]
                    if len(matches) != 1:
                        raise ValueError(
                            "Local PickaPic cache needs exactly one UID-named image"
                        )
                    im["local_path"] = str(matches[0])
                    im["retrieval_provenance"] = {
                        "kind": "user_supplied_original_uid_cache",
                        "image_uid": uid,
                    }
                else:
                    safe_member(source)
                    im["local_path"] = str(Path(local_root) / source)
            if not im["local_path"] and im["dataset"] == "pickapic":
                url = im["original_path_or_url"]
                if not isinstance(url, str) or not url.startswith(
                    ("https://", "http://")
                ):
                    raise ValueError("Missing source image URL")
                # UUID in source URL is auditable; never substitute another release.
                uid = im["image_id"].split(":", 1)[1]
                if uid not in url:
                    raise ValueError("Image UID is not verifiable from original URL")
                target = base / im["source_revision"] / (uid + ".image")
                im["local_path"] = str(download_url(url, target, retries, timeout))
            if not im["local_path"]:
                raise ValueError(
                    im.get("error") or "Image absent from expected archive"
                )
            im.update(validate_image(im["local_path"]), status="valid", error=None)
        except Exception as exc:
            im.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        return im

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        list(pool.map(resolve, selected))
    manifest["audit"]["image_availability"] = dict(
        required=len(selected),
        statuses=dict(Counter(i["status"] for i in selected)),
        failures=[
            dict(
                image_id=i["image_id"], url=i["original_path_or_url"], error=i["error"]
            )
            for i in selected
            if i["status"] != "valid"
        ],
    )
    return selected
