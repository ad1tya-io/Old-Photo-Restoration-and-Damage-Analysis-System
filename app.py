"""
Old Photo Restoration & Damage Analysis System
Streamlit application — classical Computer Vision frontend.

All CV logic lives in src/. This file handles only:
  - User interaction
  - Image I/O conversion
  - Calling backend modules
  - Displaying results
"""

import io
import numpy as np
import streamlit as st
from PIL import Image

# ─── Backend imports (never duplicate CV logic here) ──────────────────────────
from src.pipeline import RestorationPipeline, RestorationResult
from src.degradation_analysis import DegradationAnalyzer
from src.frequency_analysis import compute_fft
from src.artifact_removal import refine_artifact_mask
from src.metrics import calculate_mse, calculate_psnr, calculate_ssim
from src.dataset_loader import DatasetLoader

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Old Photo Restoration & Damage Analysis",
    page_icon="🖼️",
    layout="wide",
)

# ─── Helpers: image I/O ───────────────────────────────────────────────────────

def uploaded_file_to_rgb(uploaded_file) -> np.ndarray:
    """
    Converts a Streamlit UploadedFile to a uint8 RGB NumPy array.
    PIL handles JPEG, PNG, and other common formats transparently.
    We drop alpha channels from RGBA PNGs and treat grayscale as RGB.
    """
    pil_img = Image.open(uploaded_file)

    # Normalise mode to RGB so the backend always receives 3-channel uint8
    if pil_img.mode == "RGBA":
        # Composite onto white background to preserve visual intent
        background = Image.new("RGB", pil_img.size, (255, 255, 255))
        background.paste(pil_img, mask=pil_img.split()[3])
        pil_img = background
    elif pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")

    return np.array(pil_img, dtype=np.uint8)


def rgb_to_pil(image: np.ndarray) -> Image.Image:
    """Converts an RGB uint8 NumPy array back to a PIL Image."""
    return Image.fromarray(image.astype(np.uint8))


def image_to_png_bytes(image: np.ndarray) -> bytes:
    """Encodes an RGB uint8 NumPy array to PNG bytes for download."""
    buf = io.BytesIO()
    rgb_to_pil(image).save(buf, format="PNG")
    return buf.getvalue()


def spectrum_to_pil(spectrum: np.ndarray) -> Image.Image:
    """
    Normalises a float log-magnitude spectrum to [0, 255] uint8 for display.
    """
    s_min, s_max = spectrum.min(), spectrum.max()
    if s_max > s_min:
        normalised = ((spectrum - s_min) / (s_max - s_min) * 255).astype(np.uint8)
    else:
        normalised = np.zeros_like(spectrum, dtype=np.uint8)
    return Image.fromarray(normalised, mode="L")

# ─── Helpers: display blocks ──────────────────────────────────────────────────

def display_image_info(image: np.ndarray):
    h, w, c = image.shape
    st.caption(f"**{w} × {h} px** · {c} channels · uint8")


def status_badge(detected: bool) -> str:
    return "🔴 Detected" if detected else "🟢 Not detected"


def display_degradation_report(report):
    """Renders the DegradationReport in a readable two-column table."""
    st.subheader("🔎 Degradation Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Noise**")
        st.write(status_badge(report.is_noisy))
        st.caption(f"Immerkær score: `{report.noise_score:.4f}` (threshold ≥ 3.5)")

        st.markdown("**Blur**")
        st.write(status_badge(report.is_blurry))
        st.caption(f"Laplacian variance: `{report.blur_score:.2f}` (threshold < 100)")

        st.markdown("**Low Contrast**")
        st.write(status_badge(report.is_low_contrast))
        st.caption(
            f"Intensity σ: `{report.contrast_std:.2f}` · "
            f"Dynamic range: `{report.dynamic_range}` (threshold σ < 45)"
        )

    with col2:
        st.markdown("**Brightness / Fading**")
        brightness_pct = report.mean_brightness / 255 * 100
        st.write(f"Mean brightness: `{report.mean_brightness:.1f}` / 255 ({brightness_pct:.1f}%)")
        if report.mean_brightness > 180:
            st.caption("⚠️ Image appears unusually bright / possibly faded.")
        elif report.mean_brightness < 50:
            st.caption("⚠️ Image appears very dark.")

        st.markdown("**Scratches / Dust / Artifacts**")
        st.write(status_badge(report.has_artifacts))
        st.caption(
            f"Artifact pixel ratio: `{report.artifact_ratio:.4%}` · "
            f"Connected regions: `{report.artifact_count}` (threshold ≥ 0.5%)"
        )


def display_operations(operations: list[str]):
    """Displays the pipeline operations in a clear list."""
    st.subheader("⚙️ Restoration Operations Applied")
    op_labels = {
        "bilateral_denoise":      "🔵 Bilateral Denoising — edge-preserving spatial filter",
        "clahe_enhancement":      "🟡 CLAHE Contrast Enhancement — adaptive histogram equalisation",
        "artifact_inpainting":    "🟣 Artifact Inpainting — Fast-Marching inpainting of scratch/dust mask",
        "unsharp_mask":           "🟠 Unsharp Masking — high-frequency sharpening",
    }
    any_real = False
    for op in operations:
        label = op_labels.get(op)
        if label:
            st.markdown(f"- {label}")
            any_real = True
        elif op.startswith("none"):
            st.info("ℹ️ No degradation was detected above the configured thresholds. "
                    "The image was returned without modification.")
        else:
            # Catch-all for skip messages
            st.markdown(f"- ⚠️ `{op}`")


def display_metrics_row(label: str, mse: float, psnr: float, ssim: float):
    col1, col2, col3 = st.columns(3)
    col1.metric(f"MSE ({label})", f"{mse:.2f}", help="Lower is better")
    col2.metric(f"PSNR ({label})", f"{psnr:.2f} dB", help="Higher is better")
    col3.metric(f"SSIM ({label})", f"{ssim:.4f}", help="Higher is better (max 1.0)")

# ─── Cached backend objects (created once per session) ────────────────────────

@st.cache_resource
def get_pipeline() -> RestorationPipeline:
    """Creates a single shared pipeline instance."""
    return RestorationPipeline()


@st.cache_resource
def get_dataset_loader() -> DatasetLoader:
    return DatasetLoader(dataset_dir="dataset")

# ─── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🖼️ Old Photo Restoration")
    st.caption("Classical Computer Vision — MSc / BSc Project")
    st.divider()

    st.header("Controls")

    uploaded = st.file_uploader(
        "Upload a degraded photograph",
        type=["jpg", "jpeg", "png"],
        help="JPEG and PNG formats are supported.",
    )

    st.divider()
    st.subheader("Display Options")
    show_fft       = st.checkbox("Show Fourier Spectrum", value=True)
    show_masks     = st.checkbox("Show Artifact Masks",   value=True)
    show_steps     = st.checkbox("Show Intermediate Steps", value=False)

    st.divider()
    st.subheader("🔬 Evaluation Demo Mode")
    demo_mode = st.checkbox(
        "Ground-truth evaluation (synthetic dataset)",
        value=False,
        help="Select a synthetic degraded image that has a clean ground truth. "
             "Enables MSE / PSNR / SSIM comparison.",
    )

# ─── Main page header ─────────────────────────────────────────────────────────

st.title("Old Photo Restoration & Damage Analysis")
st.markdown(
    "_Classical Computer Vision based analysis and restoration of degraded photographs_"
)
st.divider()

# ─── Evaluation Demo Mode ─────────────────────────────────────────────────────

if demo_mode:
    st.info(
        "**Ground-truth evaluation mode** — Select a synthetic image pair from the "
        "dataset. Full-reference metrics (MSE / PSNR / SSIM) will be calculated against "
        "the clean reference image. This is different from the upload workflow, where "
        "no ground truth is available."
    )

    loader = get_dataset_loader()
    pairs  = loader.load_synthetic_pairs()

    # Build a compact selector: show source_id + degradation type
    options = {f"{p.source_id}  [{p.degradation_type}]": p for p in sorted(pairs, key=lambda x: x.source_id)}
    chosen_label = st.selectbox("Select degraded image pair", list(options.keys()))
    chosen_pair  = options[chosen_label]

    col_run, _ = st.columns([1, 3])
    run_demo = col_run.button("▶ Run Evaluation", type="primary")

    if run_demo:
        from src.dataset_loader import load_image_rgb
        with st.spinner("Loading images and running pipeline…"):
            try:
                clean_img    = load_image_rgb(chosen_pair.clean_path)
                degraded_img = load_image_rgb(chosen_pair.degraded_path)
                pipeline     = get_pipeline()
                result       = pipeline.process(degraded_img, return_intermediates=show_steps)

                restored_img = result.restored_image

                # Display images
                st.subheader("Images")
                c1, c2, c3 = st.columns(3)
                c1.image(rgb_to_pil(degraded_img), caption="Degraded Input", use_container_width=True)
                c2.image(rgb_to_pil(restored_img), caption="Restored Output", use_container_width=True)
                c3.image(rgb_to_pil(clean_img),    caption="Ground Truth (Clean)", use_container_width=True)

                # Degradation report
                display_degradation_report(result.degradation_report)

                # Operations
                display_operations(result.operations_applied)

                # Metrics
                st.subheader("📊 Quantitative Evaluation")
                st.caption(
                    "Metrics are calculated against the clean ground-truth image. "
                    "MSE: lower = better. PSNR/SSIM: higher = better."
                )
                if clean_img.shape == degraded_img.shape:
                    mse_before  = calculate_mse(clean_img, degraded_img)
                    psnr_before = calculate_psnr(clean_img, degraded_img)
                    ssim_before = calculate_ssim(clean_img, degraded_img)
                    mse_after   = calculate_mse(clean_img, restored_img)
                    psnr_after  = calculate_psnr(clean_img, restored_img)
                    ssim_after  = calculate_ssim(clean_img, restored_img)

                    display_metrics_row("Before restoration", mse_before, psnr_before, ssim_before)
                    display_metrics_row("After restoration",  mse_after,  psnr_after,  ssim_after)

                    delta_col1, delta_col2, delta_col3 = st.columns(3)
                    delta_col1.metric("ΔMSE",  f"{mse_before  - mse_after:+.2f}",  help="+ve = improved")
                    delta_col2.metric("ΔPSNR", f"{psnr_after  - psnr_before:+.2f} dB", help="+ve = improved")
                    delta_col3.metric("ΔSSIM", f"{ssim_after  - ssim_before:+.4f}", help="+ve = improved")

                    st.caption(
                        "⚠️ Note: Full-reference metrics measure pixel-level similarity to the clean "
                        "reference. A decrease in PSNR does not necessarily mean the image looks worse "
                        "to a human viewer. These are objective measurements, not perceptual quality scores."
                    )
                else:
                    st.warning(
                        f"Dimension mismatch between clean `{clean_img.shape}` and "
                        f"degraded `{degraded_img.shape}`. Metrics cannot be calculated."
                    )

                # Download
                st.download_button(
                    "⬇️ Download Restored Image",
                    data=image_to_png_bytes(restored_img),
                    file_name=f"restored_{chosen_pair.source_id}.png",
                    mime="image/png",
                )

            except Exception as exc:
                st.error(f"An error occurred during evaluation: {exc}")

    st.stop()   # Don't show the upload section in demo mode

# ─── User Upload Workflow ─────────────────────────────────────────────────────

if uploaded is None:
    st.markdown(
        """
        ### Getting Started
        Upload a degraded old photograph using the sidebar panel.

        The system will automatically:
        1. **Analyse** the image for common degradation types
        2. **Apply** appropriate classical Computer Vision restoration techniques
        3. **Display** a before / after comparison

        #### Techniques Used
        | Stage | Algorithm | Syllabus Module |
        |-------|-----------|-----------------|
        | Noise reduction | Bilateral Filter | Module 1 – Filtering |
        | Contrast enhancement | CLAHE (Luminance channel) | Module 1 – Histogram Processing |
        | Scratch / dust removal | Morphological Top-Hat + Telea Inpainting | Module 3 – Morphology |
        | Sharpening | Unsharp Masking | Module 1 – Filtering / Enhancement |
        | Frequency visualisation | 2D DFT magnitude spectrum | Module 1 – Fourier Transform |
        """
    )
    st.stop()

# ─── Load uploaded image ──────────────────────────────────────────────────────

try:
    image_rgb = uploaded_file_to_rgb(uploaded)
except Exception as exc:
    st.error(f"Could not open the uploaded file: {exc}")
    st.stop()

# Quick sanity-check (backend validation will also raise, but give a UI-friendly message first)
if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
    st.error("The uploaded file could not be converted to a 3-channel RGB image. Please try another file.")
    st.stop()

# ─── Section 1: Original Image ────────────────────────────────────────────────

st.subheader("📷 Original Image")
col_img, col_meta = st.columns([3, 1])
with col_img:
    st.image(rgb_to_pil(image_rgb), caption="Uploaded photograph", use_container_width=True)
with col_meta:
    h, w, c = image_rgb.shape
    st.markdown("**Image Properties**")
    st.write(f"Width: `{w}` px")
    st.write(f"Height: `{h}` px")
    st.write(f"Channels: `{c}`")
    st.write(f"Data type: `{image_rgb.dtype}`")
    size_kb = image_rgb.nbytes / 1024
    st.write(f"Array size: `{size_kb:.1f}` KB")

st.divider()

# ─── Section 2: Fourier Spectrum (optional) ───────────────────────────────────

if show_fft:
    with st.expander("📡 Fourier Magnitude Spectrum", expanded=False):
        st.caption(
            "The 2D Discrete Fourier Transform shows image content in the frequency domain. "
            "The **bright centre** is the DC component (low frequencies — overall brightness / "
            "smooth regions). Bright areas towards the **edges** represent high-frequency content "
            "(sharp edges, fine texture, noise)."
        )
        with st.spinner("Computing FFT…"):
            try:
                spectrum = compute_fft(image_rgb)
                spectrum_pil = spectrum_to_pil(spectrum)
                fft_col1, fft_col2 = st.columns(2)
                fft_col1.image(rgb_to_pil(image_rgb), caption="Spatial Domain", use_container_width=True)
                fft_col2.image(spectrum_pil, caption="Frequency Domain (log magnitude)", use_container_width=True)
            except Exception as exc:
                st.warning(f"Fourier spectrum could not be computed: {exc}")

st.divider()

# ─── Section 3: Run Pipeline ──────────────────────────────────────────────────

st.subheader("🔬 Degradation Analysis & Restoration")

pipeline = get_pipeline()

with st.spinner("Analysing and restoring image — this may take a moment for large photographs…"):
    try:
        result: RestorationResult = pipeline.process(
            image_rgb, return_intermediates=show_steps
        )
    except (TypeError, ValueError) as exc:
        st.error(f"Image validation failed: {exc}")
        st.stop()
    except Exception as exc:
        st.error(f"An unexpected error occurred during restoration: {exc}")
        st.stop()

report      = result.degradation_report
restored    = result.restored_image
operations  = result.operations_applied

# ─── Section 4: Degradation Report ───────────────────────────────────────────

display_degradation_report(report)
st.divider()

# ─── Section 5: Operations Applied ───────────────────────────────────────────

display_operations(operations)
st.divider()

# ─── Section 6: Artifact Masks (optional) ────────────────────────────────────

if show_masks and report.has_artifacts:
    with st.expander("🩹 Artifact Mask Detail", expanded=True):
        st.caption(
            "The **raw mask** (morphological Top-Hat + Black-Hat) highlights potential "
            "scratch/dust regions. The **refined mask** has connected components filtered "
            "by size (5–500 px) to suppress false positives from natural edges and textures."
        )
        raw_mask     = report.artifact_mask
        refined_mask = refine_artifact_mask(raw_mask)

        m1, m2 = st.columns(2)
        m1.image(Image.fromarray(raw_mask,     mode="L"), caption="Raw Artifact Mask",     use_container_width=True)
        m2.image(Image.fromarray(refined_mask, mode="L"), caption="Refined Artifact Mask", use_container_width=True)

        raw_px     = int(np.count_nonzero(raw_mask))
        refined_px = int(np.count_nonzero(refined_mask))
        total_px   = raw_mask.size
        st.caption(
            f"Raw mask: `{raw_px}` pixels flagged (`{raw_px/total_px:.2%}`). "
            f"After refinement: `{refined_px}` pixels (`{refined_px/total_px:.2%}`). "
            f"Reduction: `{raw_px - refined_px}` pixels removed as false positives."
        )
    st.divider()

# ─── Section 7: Intermediate Steps (optional) ────────────────────────────────

if show_steps and result.intermediate_steps:
    with st.expander("🔄 Intermediate Restoration Steps", expanded=False):
        for step_name, step_img in result.intermediate_steps:
            op_label = step_name.replace("_", " ").title()
            st.image(rgb_to_pil(step_img), caption=f"After: {op_label}", use_container_width=True)
    st.divider()

# ─── Section 8: Before / After Comparison ────────────────────────────────────

st.subheader("✅ Before / After Comparison")
before_col, after_col = st.columns(2)
before_col.image(rgb_to_pil(image_rgb), caption="Original (Degraded)",  use_container_width=True)
after_col.image(rgb_to_pil(restored),   caption="Restored",             use_container_width=True)

st.caption(
    "ℹ️ For user-uploaded photographs, no clean ground-truth reference exists, so "
    "full-reference metrics (PSNR / SSIM) are not calculated. Use the "
    "**Ground-truth evaluation mode** in the sidebar to see quantitative metrics "
    "on synthetic paired images."
)
st.divider()

# ─── Section 9: Download ──────────────────────────────────────────────────────

st.subheader("⬇️ Download Restored Image")
st.download_button(
    label="Download Restored Image (PNG)",
    data=image_to_png_bytes(restored),
    file_name="restored_photo.png",
    mime="image/png",
    type="primary",
)
