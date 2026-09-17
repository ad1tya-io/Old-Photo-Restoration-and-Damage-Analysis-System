"""Command-line interface for the Old Photo Restoration system.

This module provides a terminal-only execution path for restoring a single
image using the same RestorationPipeline used by the Streamlit application.

Example:
    uv run python -m src.cli --input photo.jpg --output restored.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

from .dataset_loader import load_image_rgb
from .pipeline import RestorationPipeline


def build_parser() -> argparse.ArgumentParser:
    """Create and configure the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Analyze and restore a degraded photograph using the "
            "classical computer vision restoration pipeline."
        )
    )

    parser.add_argument(
        "--input",
        "-i",
        required=True,
        type=Path,
        help="Path to the input JPEG or PNG image.",
    )

    parser.add_argument(
        "--output",
        "-o",
        required=True,
        type=Path,
        help="Path where the restored image will be saved.",
    )

    return parser


def save_image_rgb(image, output_path: Path) -> None:
    """Save an RGB uint8 image using OpenCV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # OpenCV writes images in BGR order, while the pipeline uses RGB.
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    success = cv2.imwrite(str(output_path), image_bgr)

    if not success:
        raise OSError(f"Failed to save restored image: {output_path}")


def format_bool(value: bool) -> str:
    """Format a boolean value for terminal output."""
    return "Yes" if value else "No"


def run(input_path: Path, output_path: Path) -> int:
    """Run the restoration pipeline and return a process exit code."""
    if not input_path.exists():
        print(f"Error: input file does not exist: {input_path}", file=sys.stderr)
        return 1

    if not input_path.is_file():
        print(f"Error: input path is not a file: {input_path}", file=sys.stderr)
        return 1

    try:
        image = load_image_rgb(str(input_path))

        pipeline = RestorationPipeline()
        result = pipeline.process(image)

        save_image_rgb(result.restored_image, output_path)

        report = result.degradation_report

        print("=" * 60)
        print("Old Photo Restoration and Damage Analysis System")
        print("=" * 60)

        print(f"Input image : {input_path}")
        print(f"Output image: {output_path}")
        print()

        print("Degradation Analysis")
        print("-" * 60)
        print(f"Noise score      : {report.noise_score:.2f}")
        print(f"Blur score       : {report.blur_score:.2f}")
        print(f"Contrast std     : {report.contrast_std:.2f}")
        print(f"Dynamic range    : {report.dynamic_range:.2f}")
        print(f"Mean brightness  : {report.mean_brightness:.2f}")
        print(f"Artifact ratio   : {report.artifact_ratio:.4f}")
        print(f"Artifact count   : {report.artifact_count}")
        print()

        print("Detected Degradation")
        print("-" * 60)
        print(f"Noisy            : {format_bool(report.is_noisy)}")
        print(f"Blurry           : {format_bool(report.is_blurry)}")
        print(f"Low contrast     : {format_bool(report.is_low_contrast)}")
        print(f"Artifacts        : {format_bool(report.has_artifacts)}")
        print()

        print("Restoration Operations")
        print("-" * 60)

        for operation in result.operations_applied:
            print(f"- {operation}")

        print()
        print(f"Restored image saved successfully: {output_path}")
        print("=" * 60)

        return 0

    except (FileNotFoundError, ValueError, TypeError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    except Exception as exc:
        # Keep unexpected failures visible while returning a non-zero exit
        # code so terminal-based evaluation can detect the failure.
        print(f"Unexpected error: {exc}", file=sys.stderr)
        return 1


def main() -> int:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args()

    return run(args.input, args.output)


if __name__ == "__main__":
    raise SystemExit(main())

