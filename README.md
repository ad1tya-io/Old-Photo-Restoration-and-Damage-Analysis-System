# Old Photo Restoration and Damage Analysis System

A classical computer vision system for analyzing and restoring degraded historical photographs using image processing, frequency-domain analysis, morphological operations, and adaptive restoration techniques.

The project is designed as a practical Computer Vision course project and deliberately avoids deep learning. It focuses on interpretable and explainable restoration operations derived from classical image processing concepts.

---

## Overview

Old photographs commonly suffer from multiple forms of degradation, including:

* Noise
* Blur
* Fading and low contrast
* Scratches
* Dust-like artifacts
* JPEG degradation
* Combined degradation

This project provides an automated pipeline that first analyzes the degradation present in an image and then applies appropriate restoration operations.

The system supports both:

1. **Real damaged photographs** for qualitative evaluation
2. **Synthetic degraded photographs with clean ground truth** for quantitative evaluation

---

## Key Features

* Automated degradation analysis
* Noise estimation
* Blur detection
* Contrast and brightness analysis
* Scratch and artifact detection
* Morphological artifact-mask refinement
* Telea-based image inpainting
* Median, Gaussian, and bilateral filtering
* CLAHE contrast enhancement
* Histogram equalization
* Unsharp-mask sharpening
* Laplacian sharpening
* FFT-based frequency analysis
* Adaptive restoration pipeline
* MSE, PSNR, and SSIM evaluation
* Streamlit interactive interface
* Ground-truth evaluation mode
* Real-image qualitative evaluation
* Automated unit testing
* Input validation and error handling
* Modular Python architecture

---

## System Architecture

```text
                     +----------------------+
                     |     Input Image      |
                     +----------+-----------+
                                |
                                v
                 +--------------+---------------+
                 |      Input Validation        |
                 +--------------+---------------+
                                |
                                v
                 +--------------+---------------+
                 |   Degradation Analyzer      |
                 |------------------------------|
                 | Noise   | Blur | Contrast    |
                 | Brightness | Artifacts       |
                 +--------------+---------------+
                                |
                                v
                 +--------------+---------------+
                 |    Adaptive Decision Layer   |
                 +--------------+---------------+
                                |
             +------------------+------------------+
             |                  |                  |
             v                  v                  v
       Noise Removal      Contrast Recovery   Artifact Removal
       Bilateral Filter       CLAHE             Inpainting
             |                  |                  |
             +------------------+------------------+
                                |
                                v
                       Blur Compensation
                         Unsharp Mask
                                |
                                v
                 +--------------+---------------+
                 |     Restored Image          |
                 +--------------+---------------+
                                |
                 +--------------+---------------+
                 |       Evaluation            |
                 | MSE | PSNR | SSIM | Runtime |
                 +------------------------------+
```

---

## Workflow

```text
Load Image
    |
    v
Validate RGB / uint8 / dimensions
    |
    v
Analyze Degradation
    |
    +----> Noise detected? ------> Bilateral Denoising
    |
    +----> Low contrast? --------> CLAHE Enhancement
    |
    +----> Artifacts detected? --> Mask Refinement
    |                                  |
    |                                  v
    |                              Inpainting
    |
    +----> Blur detected? --------> Unsharp Mask
    |
    v
Restored Image
    |
    +----> Optional FFT Analysis
    |
    +----> Optional Artifact Mask
    |
    +----> Quantitative Evaluation
    |
    v
Final Output
```

---

# Dataset

The project uses the Kaggle dataset:

**Old Vintage Degraded Image (Synthetic + Real)**

Dataset source:

`https://www.kaggle.com/datasets/shrutimandaokar2301/vintage-degraded-image-synthetic-real`

The dataset is not included in the Git repository because of repository size considerations.

## Dataset Structure

```text
dataset/
├── 01_Clean_Candidates_GT/
│   └── 146 clean images
│
├── 02_Damaged_Testing_Set/
│   └── 54 real damaged images
│
├── 03_Synthetic_Dataset/
│   ├── Train_GT_Clean/
│   │   └── 146 clean images
│   │
│   └── Train_Input_Degraded/
│       ├── Complex_All/
│       │   └── 146 degraded images
│       ├── Faded_Scratches_Blur/
│       │   └── 146 degraded images
│       └── Noise_JPEG_Blur/
│           └── 146 degraded images
│
└── image_classification_results.csv
```

### Dataset Statistics

| Dataset Component         | Images |
| ------------------------- | -----: |
| Clean Candidates          |    146 |
| Real Damaged Testing Set  |     54 |
| Synthetic Clean Images    |    146 |
| Synthetic Degraded Images |    438 |
| Total Images              |    874 |

The synthetic dataset contains **438 confirmed clean/degraded pairs** with no missing pairs.

---

## Synthetic Degradation Categories

### `Faded_Scratches_Blur`

Contains synthetic degradation involving combinations of:

* Fading
* Scratches
* Blur

### `Noise_JPEG_Blur`

Contains combinations of:

* Noise
* JPEG degradation
* Blur

### `Complex_All`

Contains a more complex combination of degradation types.

---

# Classical Computer Vision Approach

The system uses techniques covered by the Computer Vision syllabus, primarily from low-level image processing, enhancement, restoration, filtering, Fourier analysis, morphology, and edge/artifact analysis.

No deep learning model is used.

## Restoration Techniques

### Noise Reduction

The project implements:

* Median filtering
* Gaussian filtering
* Bilateral filtering

The adaptive pipeline primarily uses bilateral filtering because it provides smoothing while attempting to preserve image edges.

### Contrast Enhancement

Two approaches are implemented:

* Histogram equalization
* CLAHE

CLAHE is used by the adaptive pipeline when the image is detected as having low contrast.

### Sharpening

The system implements:

* Unsharp masking
* Laplacian sharpening

Unsharp masking is used for blur compensation in the adaptive pipeline.

### Artifact Removal

Artifacts such as scratches and dust-like regions are detected using morphological operations.

The process is:

```text
Input Image
     |
     v
Morphological Top-Hat / Black-Hat
     |
     v
Candidate Artifact Mask
     |
     v
Connected Component Filtering
     |
     v
Refined Artifact Mask
     |
     v
OpenCV Telea Inpainting
     |
     v
Artifact-Reduced Image
```

Large or unreliable regions are filtered to reduce the risk of destroying legitimate image structures.

---

# Degradation Analysis

The `DegradationAnalyzer` estimates several image properties.

## Noise

Noise is estimated using the Immerkær-style noise estimation method based on a zero-sum 3×3 operator.

The resulting score is compared against a configurable threshold.

Default:

```text
noise_threshold = 3.5
```

---

## Blur

Blur is estimated using the variance of the Laplacian.

A lower variance generally indicates weaker high-frequency structure and stronger blur.

Default:

```text
blur_threshold = 100.0
```

---

## Contrast

Contrast is measured using the standard deviation of grayscale intensity.

The system also records the image's dynamic range.

Default:

```text
contrast_threshold = 45.0
```

---

## Brightness

Mean grayscale intensity is used as a basic indicator of image brightness and fading.

---

## Artifact Detection

Morphological Top-Hat and Black-Hat operations are used to detect small bright and dark structures that may correspond to scratches or dust.

The system records:

* Artifact ratio
* Connected component count
* Binary artifact mask

Default artifact threshold:

```text
artifact_ratio_threshold = 0.005
```

---

# Adaptive Restoration Pipeline

The main pipeline is implemented in:

```text
src/pipeline.py
```

The pipeline dynamically chooses restoration operations based on the degradation report.

## Decision Logic

```text
                     Degradation Report
                             |
              +--------------+--------------+
              |              |              |
            Noisy?       Low Contrast?   Artifacts?
              |              |              |
             Yes            Yes            Yes
              |              |              |
        Bilateral Filter   CLAHE        Inpainting
              |              |              |
              +--------------+--------------+
                             |
                          Blurry?
                             |
                            Yes
                             |
                       Unsharp Mask
                             |
                             v
                       Restored Image
```

The default operation order is:

```text
Denoising
    ↓
Contrast Enhancement
    ↓
Artifact Processing
    ↓
Sharpening
```

---

# Frequency-Domain Analysis

The project also includes Fourier-domain analysis.

Implemented functionality includes:

* 2D FFT computation
* FFT shift
* Log-magnitude spectrum
* Low-pass filtering
* High-pass filtering
* Frequency-domain denoising
* Frequency-domain sharpening

The frequency-domain processing operates primarily on image luminance to reduce unnecessary color-channel artifacts.

Example conceptual representation:

```text
Spatial Image
     |
     v
     FFT
     |
     v
Frequency Spectrum
     |
     +----> Low-Pass Mask
     |          |
     |          v
     |      Denoising
     |
     +----> High-Pass Mask
                |
                v
             Sharpening
```

A limitation of binary frequency masks is the possibility of Gibbs-type ringing near strong transitions.

---

# Project Structure

```text
Old-Photo-Restoration/
│
├── app.py
├── pyproject.toml
├── uv.lock
├── README.md
├── .gitignore
│
├── src/
│   ├── __init__.py
│   ├── dataset_loader.py
│   ├── metrics.py
│   ├── restoration.py
│   ├── degradation_analysis.py
│   ├── artifact_removal.py
│   ├── frequency_analysis.py
│   ├── pipeline.py
│   ├── evaluate_experiments.py
│   └── evaluate_real_images.py
│
├── tests/
│   ├── __init__.py
│   ├── test_dataset_loader.py
│   ├── test_metrics.py
│   ├── test_restoration.py
│   ├── test_degradation_analysis.py
│   ├── test_artifact_removal.py
│   ├── test_frequency_analysis.py
│   └── test_pipeline.py
│
├── docs/
│   └── dataset.md
│
├── dataset/
│   └── ...
│
└── results/
    ├── metrics/
    └── qualitative/
```

> `dataset/` and generated evaluation results are excluded from Git using `.gitignore`.

---

# Installation

## Requirements

* Windows 11
* Python 3.13+
* `uv`
* Git

The project was developed and tested with:

```text
Python       3.13.14
OpenCV       5.0.0.93
NumPy        2.5.3
SciPy        1.18.1
scikit-image 0.26.0
scikit-learn 1.9.1
Pandas       3.0.5
Matplotlib   3.11.2
Streamlit    1.64.0
```

---

## Setup

Clone the repository:

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd Old-Photo-Restoration
```

Create the environment:

```bash
uv venv
```

Activate the environment on Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
uv sync
```

---

# Dataset Setup

Download the Kaggle dataset:

**Old Vintage Degraded Image (Synthetic + Real)**

Place the extracted dataset in:

```text
dataset/
```

The expected directory structure is:

```text
dataset/
├── 01_Clean_Candidates_GT/
├── 02_Damaged_Testing_Set/
├── 03_Synthetic_Dataset/
└── image_classification_results.csv
```

The dataset should not be committed to the repository.

---

# Running the Application

Launch the Streamlit application:

```bash
uv run streamlit run app.py
```

The application provides two main modes.

## 1. Upload Workflow

Upload a JPEG or PNG image.

The application performs:

```text
Upload
  ↓
Validation
  ↓
Degradation Analysis
  ↓
Adaptive Restoration
  ↓
Visualization
  ↓
Optional FFT / Artifact Analysis
  ↓
PNG Output
```

The interface displays:

* Original image
* Restored image
* Degradation measurements
* Detected degradation flags
* Operations applied
* Optional artifact mask
* Optional frequency spectrum
* Restored image download

---

## 2. Ground-Truth Evaluation Demo

This mode allows a synthetic degraded image to be selected together with its known clean ground truth.

The application reports:

* MSE before restoration
* MSE after restoration
* PSNR before restoration
* PSNR after restoration
* SSIM before restoration
* SSIM after restoration

This provides an interactive demonstration of quantitative evaluation.

---

# Evaluation Methodology

The project uses two evaluation strategies.

## Synthetic Quantitative Evaluation

The 438 paired synthetic degraded images are evaluated against their corresponding clean ground-truth images.

Metrics:

### Mean Squared Error

```text
MSE = (1 / N) Σ(I - K)²
```

Lower MSE indicates smaller pixel-level error.

### Peak Signal-to-Noise Ratio

```text
PSNR = 10 log10(MAX² / MSE)
```

Higher PSNR generally indicates lower reconstruction error.

### Structural Similarity Index

SSIM measures structural similarity between images.

Higher SSIM indicates greater structural similarity.

---

## Real-Image Qualitative Evaluation

The 54 real damaged images do not have confirmed clean ground truth.

Therefore, MSE, PSNR, and SSIM are not reported for the real damaged set.

Instead, the system records:

* Degradation measurements
* Operations applied
* Processing time
* Original image
* Restored image
* Artifact mask when applicable

All 54 real images were successfully processed.

---

# Quantitative Results

The baseline adaptive pipeline was evaluated on all 438 synthetic pairs.

## Overall Results

| Metric | Before |  After | Mean Change |
| ------ | -----: | -----: | ----------: |
| MSE    | 664.11 | 676.04 |      +11.93 |
| PSNR   |  22.24 |  21.94 |       -0.30 |
| SSIM   | 0.7810 | 0.7798 |     -0.0012 |

At the individual-image level:

* 70.5% of images improved according to MSE and PSNR
* 71.7% improved according to SSIM

The aggregate mean metrics decreased slightly because a smaller number of images experienced relatively large regressions.

This demonstrates why both aggregate statistics and per-image behavior are useful when evaluating restoration algorithms.

---

# Results by Degradation Category

## Faded + Scratches + Blur

| Metric | Before |  After |
| ------ | -----: | -----: |
| MSE    | 919.28 | 910.44 |
| PSNR   |  18.93 |  18.96 |
| SSIM   | 0.7784 | 0.7861 |

Improvement:

* MSE/PSNR: 74.0% of images
* SSIM: 86.3% of images

---

## Noise + JPEG + Blur

| Metric | Before |  After |
| ------ | -----: | -----: |
| MSE    | 165.19 | 207.10 |
| PSNR   |  28.77 |  27.86 |
| SSIM   | 0.8011 | 0.7939 |

Improvement:

* MSE/PSNR: 77.4% of images
* SSIM: 64.4% of images

---

## Complex All

| Metric | Before |  After |
| ------ | -----: | -----: |
| MSE    | 907.86 | 910.59 |
| PSNR   |  19.01 |  18.99 |
| SSIM   | 0.7636 | 0.7595 |

Improvement:

* MSE/PSNR: 60.3% of images
* SSIM: 64.4% of images

---

# Operation Usage

Across the 438 synthetic images:

| Operation           | Images | Percentage |
| ------------------- | -----: | ---------: |
| Unsharp Mask        |    438 |     100.0% |
| Artifact Inpainting |    109 |      24.9% |
| CLAHE               |     40 |       9.1% |
| Bilateral Denoising |      0 |       0.0% |

The results also revealed useful limitations in the baseline thresholds.

For example, the blur threshold resulted in sharpening being triggered for all synthetic images, while the noise threshold did not trigger bilateral denoising on the synthetic evaluation set.

These observations are retained as experimental findings rather than automatically tuning the system to the test results.

---

# Real-Image Evaluation

The system was evaluated on all 54 real damaged photographs.

| Measure          |             Result |
| ---------------- | -----------------: |
| Images processed |                 54 |
| Successful       |                 54 |
| Failed           |                  0 |
| Total runtime    |      18.29 seconds |
| Average runtime  | 0.34 seconds/image |

Detected degradation:

| Degradation  | Images |
| ------------ | -----: |
| Noisy        |     14 |
| Blurry       |     15 |
| Low Contrast |     15 |
| Artifacts    |     35 |

Operation usage:

| Operation                   | Images |
| --------------------------- | -----: |
| Artifact Inpainting         |     32 |
| Bilateral Denoising         |     14 |
| CLAHE                       |     15 |
| Unsharp Mask                |     15 |
| No Operation                |      5 |
| Artifact Processing Skipped |      3 |

The real-image results are used primarily for qualitative inspection because no reliable clean reference image is available.

---

# Testing

The project contains unit tests covering:

* Dataset loading
* Synthetic image pairing
* Metric calculations
* Restoration functions
* Degradation analysis
* Artifact mask refinement
* Frequency-domain operations
* Adaptive pipeline behavior
* Input validation
* Output properties

Run the complete test suite:

```bash
uv run pytest
```

Current result:

```text
51 passed
```

The complete test suite passes successfully.

---

# Functional Requirements

The system provides the following major functional modules.

| Module               | Input                    | Output                 |
| -------------------- | ------------------------ | ---------------------- |
| Dataset Loader       | Dataset directory        | Validated image pairs  |
| Degradation Analyzer | RGB image                | Degradation report     |
| Restoration Pipeline | RGB image + report       | Restored image         |
| Artifact Removal     | Image + artifact mask    | Artifact-reduced image |
| Frequency Analysis   | Image                    | FFT / filtered image   |
| Evaluation           | Original + restored + GT | MSE / PSNR / SSIM      |
| Streamlit UI         | User image               | Interactive result     |

---

# Non-Functional Requirements

## Performance

The system should process ordinary photographs within a reasonable interactive time.

The real-image evaluation averaged approximately:

```text
0.34 seconds/image
```

## Reliability

Invalid image shapes, data types, and incompatible image dimensions are rejected instead of silently modified.

## Maintainability

The system is divided into independent Python modules with focused responsibilities.

## Usability

The Streamlit interface provides visual before/after comparisons and exposes the detected degradation and applied operations.

## Resource Efficiency

Images are processed individually rather than loading the complete dataset into memory simultaneously.

## Reproducibility

The Python environment and dependencies are managed through `uv` and `uv.lock`.

## Error Handling

Dataset loading, image validation, metric calculation, and pipeline processing include explicit validation and exception handling.

---

# Design Decisions

## Classical Computer Vision Instead of Deep Learning

The project intentionally uses classical techniques because the objective is to demonstrate concepts from the Computer Vision course syllabus.

This also makes the restoration decisions interpretable.

---

## Adaptive Processing

Instead of applying every restoration operation to every image, the pipeline first estimates degradation and then selects operations.

This reduces unnecessary processing and makes the system more modular.

---

## LAB Luminance Processing

Contrast enhancement is applied to the luminance component rather than independently modifying RGB channels.

This helps reduce unwanted color shifts.

---

## Conservative Artifact Removal

Artifact masks are refined using connected-component size filtering.

The objective is to reduce the possibility of treating large legitimate structures as scratches or dust.

---

## No Silent Resizing

Metric calculations require compatible dimensions.

Images are not silently resized because doing so could hide dataset or processing errors.

---

# Limitations

The current system has several limitations.

### No Ground Truth for Real Damaged Images

The real photographs cannot be evaluated using full-reference metrics because clean reference images are unavailable.

### Fixed Thresholds

The degradation detector uses fixed thresholds that may not generalize perfectly to every image collection.

### Conservative Artifact Detection

Small artifacts can be detected successfully, but large tears or holes may be ignored.

The current artifact filtering uses:

```text
minimum area = 5 pixels
maximum area = 500 pixels
```

### Texture Confusion

Strong image textures may sometimes resemble scratches or small artifacts.

### Frequency Mask Ringing

Binary frequency-domain masks may introduce ringing artifacts around strong edges.

### Classical Restoration Limits

Severely damaged regions cannot always be reconstructed accurately using local filtering and inpainting alone.

---

# Future Enhancements

Potential future improvements include:

* Adaptive threshold estimation
* Multi-scale artifact detection
* Improved scratch orientation analysis
* More selective blur estimation
* Better handling of large damaged regions
* Edge-aware inpainting
* Automatic parameter optimization
* More extensive perceptual evaluation
* Human evaluation studies
* Additional real-world restoration datasets
* Optional comparison with learning-based approaches

These improvements are intentionally outside the current baseline implementation.

---

# Technology Stack

| Technology   | Purpose                               |
| ------------ | ------------------------------------- |
| Python       | Core implementation                   |
| OpenCV       | Image processing and restoration      |
| NumPy        | Numerical computation                 |
| SciPy        | Scientific computation                |
| scikit-image | Image analysis and SSIM               |
| scikit-learn | Scientific/utility dependency         |
| Pandas       | Evaluation result handling            |
| Matplotlib   | Visualization                         |
| Streamlit    | Interactive interface                 |
| pytest       | Automated testing                     |
| uv           | Environment and dependency management |
| Git          | Version control                       |

---

# Module Mapping to Computer Vision Syllabus

| Course Concept                         | Project Usage                            |
| -------------------------------------- | ---------------------------------------- |
| Image Formation / Low-Level Processing | Image loading and validation             |
| Convolution / Filtering                | Gaussian, median, bilateral filtering    |
| Image Enhancement                      | Histogram equalization, CLAHE            |
| Image Restoration                      | Denoising, artifact removal, sharpening  |
| Histogram Processing                   | Histogram equalization                   |
| Fourier Transform                      | FFT analysis and frequency filtering     |
| Morphological Operations               | Top-Hat / Black-Hat artifact detection   |
| Edge Information                       | Laplacian-based blur/sharpening analysis |
| Image Quality Evaluation               | MSE, PSNR, SSIM                          |

The project focuses on concepts that directly support image restoration rather than forcing unrelated techniques into the system.

---

# Reproducibility

To reproduce the evaluation:

```bash
uv sync
```

Run tests:

```bash
uv run pytest
```

Run the Streamlit application:

```bash
uv run streamlit run app.py
```

Run synthetic evaluation:

```bash
uv run python src/evaluate_experiments.py
```

Run real-image evaluation:

```bash
uv run python src/evaluate_real_images.py
```

Evaluation outputs are generated locally under:

```text
results/
```

and are excluded from Git.

---

# Version Control

Git is used for source-code version control.

The repository tracks:

* Source code
* Tests
* Documentation
* Configuration
* Dependency lock file

Large datasets and generated evaluation results are intentionally excluded.

The working tree was verified clean after the final test run.

---

# Conclusion

This project implements an end-to-end classical computer vision pipeline for old photograph damage analysis and restoration.

The system combines:

```text
Image Validation
       ↓
Degradation Analysis
       ↓
Adaptive Decision Making
       ↓
Denoising
       ↓
Contrast Enhancement
       ↓
Artifact Detection
       ↓
Inpainting
       ↓
Sharpening
       ↓
Frequency Analysis
       ↓
Quantitative / Qualitative Evaluation
```

The project demonstrates how classical computer vision techniques can be combined into a modular restoration system without relying on deep learning.

The synthetic dataset provides controlled quantitative evaluation through clean/degraded image pairs, while the real damaged dataset provides a practical qualitative evaluation scenario.

---

# License

This project is intended for academic and educational use.

The dataset remains subject to the licensing and usage terms specified by its original provider.

---

# Author

**[Your Name]**

Computer Vision Course Project

**Old Photo Restoration and Damage Analysis System**
