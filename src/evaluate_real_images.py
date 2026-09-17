"""
Qualitative evaluation of the real damaged-image dataset.

Runs the restoration pipeline on all real damaged images and saves:

1. A CSV summary containing degradation analysis, applied operations,
   processing time, and success/failure status.
2. Original and restored PNG images for visual inspection.
3. Refined artifact masks when artifact inpainting is applied.

No MSE, PSNR, or SSIM metrics are calculated because the real damaged
images do not have confirmed ground-truth images.
"""

from __future__ import annotations

import csv
import time
from pathlib import Path

import cv2
import numpy as np

from src.dataset_loader import DatasetLoader, load_image_rgb
from src.pipeline import RestorationPipeline
from src.artifact_removal import refine_artifact_mask


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_ROOT = PROJECT_ROOT / "dataset"

RESULTS_DIR = PROJECT_ROOT / "results" / "qualitative"
IMAGE_DIR = RESULTS_DIR / "images"

CSV_PATH = RESULTS_DIR / "real_image_evaluation.csv"


# ---------------------------------------------------------------------------
# Image saving helpers
# ---------------------------------------------------------------------------

def save_rgb_image(path: Path, image: np.ndarray) -> None:
    """Save an RGB uint8 NumPy image as a PNG."""

    if image.dtype != np.uint8:
        raise ValueError(
            f"Expected uint8 image, got {image.dtype}"
        )

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(
            f"Expected RGB image with shape (H, W, 3), got {image.shape}"
        )

    path.parent.mkdir(parents=True, exist_ok=True)

    # OpenCV expects BGR when writing.
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    success = cv2.imwrite(str(path), bgr)

    if not success:
        raise IOError(f"Failed to save image: {path}")


def save_mask(path: Path, mask: np.ndarray) -> None:
    """Save a binary artifact mask as a PNG."""

    if mask.ndim != 2:
        raise ValueError(
            f"Expected 2D mask, got {mask.shape}"
        )

    path.parent.mkdir(parents=True, exist_ok=True)

    mask_uint8 = (mask > 0).astype(np.uint8) * 255

    success = cv2.imwrite(str(path), mask_uint8)

    if not success:
        raise IOError(f"Failed to save mask: {path}")


def format_operations(operations: list[str]) -> str:
    """Convert pipeline operations into a CSV-friendly string."""

    if not operations:
        return "none"

    return "; ".join(operations)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_real_images() -> None:
    """Run the restoration pipeline on every real damaged image."""

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    # DatasetLoader expects a string path.
    loader = DatasetLoader(str(DATASET_ROOT))

    pipeline = RestorationPipeline()

    # This returns a list of file paths, not custom image objects.
    real_image_paths = loader.load_real_damaged()

    print("=" * 72)
    print("REAL DAMAGED IMAGE QUALITATIVE EVALUATION")
    print("=" * 72)
    print(f"Dataset root : {DATASET_ROOT}")
    print(f"Images       : {len(real_image_paths)}")
    print(f"Output CSV   : {CSV_PATH}")
    print(f"Output images: {IMAGE_DIR}")
    print()

    rows: list[dict] = []

    success_count = 0
    failure_count = 0
    total_runtime = 0.0

    # -----------------------------------------------------------------------
    # Process every real damaged image.
    # -----------------------------------------------------------------------

    for index, image_path in enumerate(real_image_paths, start=1):

        filename = Path(image_path).name

        print(
            f"[{index:02d}/{len(real_image_paths)}] "
            f"{filename}",
            end=" ... "
        )

        start_time = time.perf_counter()

        try:
            # ---------------------------------------------------------------
            # Load image using the standalone loader function.
            # ---------------------------------------------------------------

            image = load_image_rgb(image_path)

            height, width = image.shape[:2]

            # ---------------------------------------------------------------
            # Run restoration pipeline.
            # ---------------------------------------------------------------

            result = pipeline.process(image)

            elapsed = time.perf_counter() - start_time
            total_runtime += elapsed

            report = result.degradation_report

            # ---------------------------------------------------------------
            # Save original and restored images.
            # ---------------------------------------------------------------

            stem = Path(filename).stem

            original_path = IMAGE_DIR / f"{stem}_original.png"
            restored_path = IMAGE_DIR / f"{stem}_restored.png"

            save_rgb_image(
                original_path,
                result.original_image
            )

            save_rgb_image(
                restored_path,
                result.restored_image
            )

            # ---------------------------------------------------------------
            # Save refined artifact mask if artifact removal was applied.
            # ---------------------------------------------------------------

            artifact_mask_path = ""

            if "artifact_inpainting" in result.operations_applied:

                if report.artifact_mask is not None:

                    refined_mask = refine_artifact_mask(
                        report.artifact_mask,
                        min_area=pipeline.artifact_min_area,
                        max_area=pipeline.artifact_max_area,
                    )

                    mask_path = IMAGE_DIR / f"{stem}_artifact_mask.png"

                    save_mask(
                        mask_path,
                        refined_mask
                    )

                    artifact_mask_path = str(
                        mask_path.relative_to(PROJECT_ROOT)
                    )

            # ---------------------------------------------------------------
            # Store evaluation row.
            #
            # These field names exactly match DegradationReport:
            #   contrast_std
            #   mean_brightness
            #   artifact_count
            # ---------------------------------------------------------------

            operations = format_operations(
                result.operations_applied
            )

            rows.append(
                {
                    "filename": filename,
                    "width": width,
                    "height": height,

                    "noise_score": report.noise_score,
                    "blur_score": report.blur_score,
                    "contrast_std": report.contrast_std,
                    "dynamic_range": report.dynamic_range,
                    "mean_brightness": report.mean_brightness,

                    "artifact_ratio": report.artifact_ratio,
                    "artifact_count": report.artifact_count,

                    "is_noisy": report.is_noisy,
                    "is_blurry": report.is_blurry,
                    "is_low_contrast": report.is_low_contrast,
                    "has_artifacts": report.has_artifacts,

                    "operations_applied": operations,

                    "processing_time_seconds": elapsed,

                    "original_image": str(
                        original_path.relative_to(PROJECT_ROOT)
                    ),

                    "restored_image": str(
                        restored_path.relative_to(PROJECT_ROOT)
                    ),

                    "artifact_mask": artifact_mask_path,

                    "success": True,
                    "error": "",
                }
            )

            success_count += 1

            print(f"OK ({elapsed:.2f}s)")

        except Exception as exc:

            elapsed = time.perf_counter() - start_time
            total_runtime += elapsed

            rows.append(
                {
                    "filename": filename,
                    "width": "",
                    "height": "",

                    "noise_score": "",
                    "blur_score": "",
                    "contrast_std": "",
                    "dynamic_range": "",
                    "mean_brightness": "",

                    "artifact_ratio": "",
                    "artifact_count": "",

                    "is_noisy": "",
                    "is_blurry": "",
                    "is_low_contrast": "",
                    "has_artifacts": "",

                    "operations_applied": "",

                    "processing_time_seconds": elapsed,

                    "original_image": "",
                    "restored_image": "",
                    "artifact_mask": "",

                    "success": False,
                    "error": str(exc),
                }
            )

            failure_count += 1

            print(f"FAILED ({exc})")

    # -----------------------------------------------------------------------
    # Write CSV.
    # -----------------------------------------------------------------------

    fieldnames = [
        "filename",
        "width",
        "height",

        "noise_score",
        "blur_score",
        "contrast_std",
        "dynamic_range",
        "mean_brightness",

        "artifact_ratio",
        "artifact_count",

        "is_noisy",
        "is_blurry",
        "is_low_contrast",
        "has_artifacts",

        "operations_applied",

        "processing_time_seconds",

        "original_image",
        "restored_image",
        "artifact_mask",

        "success",
        "error",
    ]

    with CSV_PATH.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as csv_file:

        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(rows)

    # -----------------------------------------------------------------------
    # Calculate summary statistics.
    # -----------------------------------------------------------------------

    successful_rows = [
        row
        for row in rows
        if row["success"]
    ]

    if successful_rows:
        average_runtime = (
            sum(
                float(row["processing_time_seconds"])
                for row in successful_rows
            )
            / len(successful_rows)
        )
    else:
        average_runtime = 0.0

    print()
    print("=" * 72)
    print("EVALUATION COMPLETE")
    print("=" * 72)

    print(f"Total images    : {len(real_image_paths)}")
    print(f"Successful      : {success_count}")
    print(f"Failed          : {failure_count}")

    print(
        f"Total runtime   : "
        f"{total_runtime:.2f} seconds"
    )

    print(
        f"Average runtime : "
        f"{average_runtime:.2f} seconds/image"
    )

    print(f"CSV             : {CSV_PATH}")
    print(f"Images          : {IMAGE_DIR}")

    print()

    # -----------------------------------------------------------------------
    # Operation summary.
    # -----------------------------------------------------------------------

    operation_counts: dict[str, int] = {}

    for row in successful_rows:

        operations = row["operations_applied"]

        if operations == "none":
            operation_counts["none"] = (
                operation_counts.get("none", 0) + 1
            )
            continue

        for operation in operations.split("; "):

            operation_counts[operation] = (
                operation_counts.get(operation, 0) + 1
            )

    print("Operation usage:")

    for operation, count in sorted(operation_counts.items()):
        print(f"  {operation}: {count}")

    print()

    # -----------------------------------------------------------------------
    # Degradation flag summary.
    # -----------------------------------------------------------------------

    flag_fields = [
        ("is_noisy", "Noisy"),
        ("is_blurry", "Blurry"),
        ("is_low_contrast", "Low contrast"),
        ("has_artifacts", "Artifacts"),
    ]

    print("Degradation flags:")

    for field, label in flag_fields:

        count = sum(
            1
            for row in successful_rows
            if row[field] is True
        )

        print(
            f"  {label}: "
            f"{count}/{len(successful_rows)}"
        )

    print()
    print(
        "Use the saved original/restored pairs "
        "for qualitative inspection."
    )

    print(
        "The degradation flags are diagnostic estimates, "
        "not ground-truth labels."
    )

    print("=" * 72)


if __name__ == "__main__":
    evaluate_real_images()