from pathlib import Path

from PIL import Image

from scripts.capture_nine_tables import save_or_reuse_crop, timed_crop


def test_save_or_reuse_crop_reuses_identical_pixels(tmp_path: Path):
    image = Image.new("RGB", (20, 20), (10, 20, 30))
    box = (0, 0, 10, 10)
    first_path = tmp_path / "first.png"
    first_digest, first_pixel_sha, reused = save_or_reuse_crop(image, box, first_path, None)

    assert reused is False
    assert first_path.is_file()

    second_path = tmp_path / "second.png"
    second_digest, second_pixel_sha, reused = save_or_reuse_crop(
        image,
        box,
        second_path,
        {
            "pixel_sha256": first_pixel_sha,
            "crop_sha256": first_digest,
            "relative": Path("first.png"),
            "frame_id": 0,
        },
    )

    assert reused is True
    assert second_digest == first_digest
    assert second_pixel_sha == first_pixel_sha
    assert not second_path.exists()


def test_save_or_reuse_crop_saves_changed_pixels(tmp_path: Path):
    original = Image.new("RGB", (20, 20), (10, 20, 30))
    changed = Image.new("RGB", (20, 20), (11, 20, 30))
    box = (0, 0, 10, 10)
    first_path = tmp_path / "first.png"
    first_digest, first_pixel_sha, _ = save_or_reuse_crop(original, box, first_path, None)

    second_path = tmp_path / "second.png"
    second_digest, second_pixel_sha, reused = save_or_reuse_crop(
        changed,
        box,
        second_path,
        {
            "pixel_sha256": first_pixel_sha,
            "crop_sha256": first_digest,
            "relative": Path("first.png"),
            "frame_id": 0,
        },
    )

    assert reused is False
    assert second_path.is_file()
    assert second_digest != first_digest
    assert second_pixel_sha != first_pixel_sha


def test_parallel_capture_retains_the_same_png_bytes(tmp_path: Path):
    from concurrent.futures import ThreadPoolExecutor

    image = Image.new("RGB", (20, 20), (10, 20, 30))
    box = (0, 0, 10, 10)
    old_hash, old_pixel, _ = save_or_reuse_crop(image, box, tmp_path / "old.png", None)
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(timed_crop, image, box, tmp_path / f"new_{i}.png", None)
                   for i in range(4)]
        for future in futures:
            digest, pixels, reused, _, _ = future.result()
            assert (digest, pixels, reused) == (old_hash, old_pixel, False)
