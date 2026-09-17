"""Recovery tools must preserve verifiable identities and avoid image downloads."""

import io
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scripts.find_pickapic_copy import inspect_shard, source_uid


def test_source_uid_requires_uuid_and_handles_original_image_urls():
    uid = "e815662d-9709-4380-b4c1-96be11f07574"
    assert source_uid(uid) == uid
    assert source_uid(f"https://example.com/images/{uid}.png?download=1") == uid
    assert source_uid(None) is None
    assert source_uid("none") is None
    assert source_uid("arbitrary-image-label") is None


def test_scan_records_locations_and_resumes_without_reading_image_column(
    tmp_path, monkeypatch
):
    from scripts import find_pickapic_copy as module

    uid = "e815662d-9709-4380-b4c1-96be11f07574"
    sink = io.BytesIO()
    pq.write_table(pa.table({"image_0_uid": [uid], "jpg_0": [b"not-needed"]}), sink)
    calls = []

    class FakeReader(io.BytesIO):
        bytes_downloaded = 123

        def __init__(self, *args, **kwargs):
            calls.append(args)
            super().__init__(sink.getvalue())

    monkeypatch.setattr(module, "HTTPRangeReader", FakeReader)
    file = {"rfilename": "data/test.parquet", "size": len(sink.getvalue())}
    a = inspect_shard("test/repo", "pinned-revision", file, tmp_path)
    b = inspect_shard("test/repo", "pinned-revision", file, tmp_path)
    assert a == b and len(calls) == 1
    assert a["selected_columns"] == ["image_0_uid"]
    assert a["uid_locations"] == [dict(uid=uid, row=0, column="image_0_uid")]
    assert a["embedded_image_columns"] == ["jpg_0"]
    inspect_shard("test/repo", "different-revision", file, tmp_path)
    assert len(calls) == 2


def test_archive_identity_checks_prompt_and_generation_metadata():
    from preference_diff.recovery import verify_identity

    uid = "e815662d-9709-4380-b4c1-96be11f07574"
    original = dict(
        image_id=f"pickapic:{uid}",
        original_prompt="A cat",
        generation_metadata=dict(seed=42, model_id="generator"),
    )
    row = dict(
        url=f"https://example.com/{uid}.png",
        prompt="A cat",
        seed=42,
        model_id="generator",
    )
    verify_identity(row, original)
    with pytest.raises(ValueError, match="prompt"):
        verify_identity(dict(row, prompt="A dog"), original)
    with pytest.raises(ValueError, match="seed"):
        verify_identity(dict(row, seed=43), original)
    with pytest.raises(ValueError, match="UID"):
        verify_identity(dict(row, url="https://example.com/unverifiable.png"), original)


def test_local_archive_provenance_rejects_changed_image_bytes(tmp_path):
    from PIL import Image
    from preference_diff.io import write_json, file_hash
    from preference_diff.sources import resolve_images

    image = tmp_path / "original.png"
    Image.new("RGB", (4, 4), "red").save(image)
    write_json(
        tmp_path / "retrieval_provenance.json",
        dict(
            images={
                "original": dict(
                    content_hash=file_hash(image), source_revision="pinned"
                )
            }
        ),
    )
    manifest = dict(
        images=[
            dict(
                dataset="pickapic",
                image_id="pickapic:original",
                source_revision="original-metadata",
                original_path_or_url="https://example.com/original.png",
                local_path=None,
            )
        ],
        events=[
            dict(
                sampled=True, exclusion_reason=None, candidate_ids=["pickapic:original"]
            )
        ],
        audit=dict(dataset="pickapic"),
    )
    resolve_images(manifest, {}, tmp_path, local_root=tmp_path)
    assert manifest["images"][0]["status"] == "valid"
    assert manifest["images"][0]["retrieval_provenance"]["kind"] == "pinned_hf_archive"
    Image.new("RGB", (4, 4), "blue").save(image)
    resolve_images(manifest, {}, tmp_path, local_root=tmp_path)
    assert manifest["images"][0]["status"] == "failed"
    assert "differ from recorded archive" in manifest["images"][0]["error"]
