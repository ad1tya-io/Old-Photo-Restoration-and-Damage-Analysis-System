# Project Statement

## Old Photo Restoration and Damage Analysis System

### 1. Project Title

**Old Photo Restoration and Damage Analysis System**

---

## 2. Problem Statement

Old photographs often contain visible degradation caused by age, storage conditions, repeated handling, printing processes, and image digitization. Common problems include noise, blur, fading, low contrast, scratches, dust-like artifacts, and compression-related degradation.

Manual restoration can be time-consuming and requires specialized image-editing knowledge. At the same time, applying the same restoration operation to every photograph is not always appropriate because different images can contain different types and levels of degradation.

This project addresses the problem by developing a **classical computer vision-based system that analyzes image degradation and adaptively applies suitable restoration techniques**.

The system does not rely on deep learning. Instead, it combines image filtering, histogram-based enhancement, morphological processing, Fourier analysis, artifact detection, inpainting, and image-quality metrics into a modular restoration pipeline.

---

## 3. Motivation

Historical photographs contain valuable visual information, but degradation can make important details difficult to see.

A restoration system should therefore be able to:

* Identify common forms of image degradation
* Select appropriate restoration operations
* Reduce noise without unnecessarily destroying edges
* Improve faded or low-contrast photographs
* Detect and remove small scratches and artifacts
* Compensate for blur
* Provide measurable restoration results
* Allow users to visually inspect the restoration process

The project also provides an opportunity to apply concepts from the Computer Vision syllabus to a complete practical problem rather than implementing individual algorithms in isolation.

---

## 4. Aim

The primary aim of this project is:

> **To design and implement a modular classical computer vision system that automatically analyzes degradation in old photographs and applies adaptive image restoration techniques to improve their visual quality.**

---

## 5. Objectives

The project has the following objectives:

1. Develop a reliable image-loading and validation module.

2. Analyze photographs for common degradation characteristics such as:

   * Noise
   * Blur
   * Low contrast
   * Brightness/fading
   * Scratches and other small artifacts

3. Implement classical image restoration techniques including:

   * Median filtering
   * Gaussian filtering
   * Bilateral filtering
   * Histogram equalization
   * CLAHE
   * Unsharp masking
   * Laplacian sharpening
   * Morphological artifact detection
   * Image inpainting

4. Develop an adaptive restoration pipeline that selects operations according to the detected degradation.

5. Implement frequency-domain analysis using the Fourier Transform.

6. Evaluate restoration quality using:

   * Mean Squared Error (MSE)
   * Peak Signal-to-Noise Ratio (PSNR)
   * Structural Similarity Index (SSIM)

7. Evaluate the system on both synthetic paired images and real damaged photographs.

8. Develop an interactive Streamlit interface for image restoration and demonstration.

9. Create automated tests to verify the correctness and reliability of the implementation.

---

## 6. Scope

### In Scope

The project covers:

* RGB photograph processing
* Image validation
* Classical image filtering
* Image enhancement
* Degradation analysis
* Morphological artifact detection
* Artifact-mask refinement
* Image inpainting
* Image sharpening
* Fourier-domain analysis
* Adaptive restoration
* Quantitative evaluation on paired synthetic data
* Qualitative evaluation on real damaged images
* Interactive visualization
* Automated testing

### Out of Scope

The project does not include:

* Deep learning-based restoration
* Generative AI restoration
* Facial reconstruction
* Semantic image understanding
* Automatic reconstruction of large missing regions
* Colorization of black-and-white photographs
* 3D reconstruction
* Stereo vision
* Optical flow
* Photometric stereo

These areas are outside the selected scope of the project and are not required for the core restoration objective.

---

## 7. Computer Vision Concepts Used

The project primarily applies concepts from low-level image processing, enhancement, restoration, filtering, morphology, and frequency-domain analysis.

| Computer Vision Concept  | Application in Project                    |
| ------------------------ | ----------------------------------------- |
| Image Filtering          | Median, Gaussian, and bilateral filtering |
| Convolution              | Spatial filtering and sharpening          |
| Histogram Processing     | Histogram equalization                    |
| Contrast Enhancement     | CLAHE                                     |
| Image Restoration        | Noise, artifact, and blur reduction       |
| Fourier Transform        | Frequency-domain analysis                 |
| Low-Pass Filtering       | Frequency-domain denoising                |
| High-Pass Filtering      | Frequency-domain sharpening               |
| Morphological Operations | Scratch and artifact detection            |
| Laplacian Operator       | Blur estimation and sharpening            |
| Image Quality Metrics    | MSE, PSNR, and SSIM                       |

The project intentionally focuses on relevant syllabus concepts instead of adding unrelated algorithms solely to increase the number of techniques.

---

## 8. Proposed System

The proposed system follows an adaptive workflow.

```text
                 Input Photograph
                        |
                        v
                Image Validation
                        |
                        v
              Degradation Analysis
                        |
       +----------------+----------------+
       |                |                |
       v                v                v
     Noise            Contrast       Artifacts
       |                |                |
       v                v                v
   Bilateral          CLAHE          Mask Refinement
   Filtering                           |
                                       v
                                  Inpainting
       |                |                |
       +----------------+----------------+
                        |
                        v
                 Blur Compensation
                   Unsharp Mask
                        |
                        v
                Restored Photograph
                        |
              +---------+---------+
              |                   |
              v                   v
       Frequency Analysis    Quality Metrics
                            MSE / PSNR / SSIM
```

---

## 9. Major Functional Modules

### Module 1: Dataset Loading and Validation

**Input:** Dataset directory and image files

**Output:** Validated image objects and synthetic clean/degraded pairs

Responsibilities include:

* Loading RGB images
* Validating image shape and type
* Pairing synthetic degraded images with ground truth
* Identifying real damaged images
* Detecting missing or invalid files

---

### Module 2: Degradation Analysis

**Input:** RGB image

**Output:** Degradation report

The analyzer estimates:

* Noise score
* Blur score
* Contrast standard deviation
* Dynamic range
* Mean brightness
* Artifact ratio
* Artifact component count

It then produces Boolean degradation flags for:

* Noise
* Blur
* Low contrast
* Artifacts

---

### Module 3: Restoration Pipeline

**Input:** Validated RGB image

**Output:** Restored RGB image and processing report

The pipeline uses the degradation report to determine which operations should be applied.

Possible operations include:

* Bilateral denoising
* CLAHE enhancement
* Artifact inpainting
* Unsharp-mask sharpening

The system also records the operations applied to each image.

---

### Module 4: Artifact Removal

**Input:** Image and artifact candidate mask

**Output:** Refined mask and artifact-reduced image

The module uses:

* Morphological processing
* Connected-component analysis
* Area filtering
* OpenCV Telea inpainting

The approach is deliberately conservative to reduce the risk of removing legitimate image structures.

---

### Module 5: Frequency Analysis

**Input:** Image

**Output:** Frequency spectrum and optionally filtered image

The module provides:

* FFT computation
* Shifted frequency spectrum
* Log-magnitude visualization
* Low-pass filtering
* High-pass filtering
* Frequency-domain denoising
* Frequency-domain sharpening

---

### Module 6: Evaluation

**Input:** Degraded image, restored image, and clean ground truth

**Output:** MSE, PSNR, and SSIM

The evaluation module supports quantitative comparison of restoration results on the synthetic paired dataset.

---

### Module 7: Interactive Application

**Input:** User-uploaded photograph or synthetic test image

**Output:** Interactive restoration results

The Streamlit application provides:

* Image upload
* Degradation analysis
* Restoration
* Before/after comparison
* Operation summary
* Artifact-mask visualization
* FFT visualization
* Ground-truth evaluation
* PNG output

---

## 10. Dataset

The project uses the Kaggle dataset:

**Old Vintage Degraded Image (Synthetic + Real)**

The dataset contains both synthetic degraded images with clean ground truth and real damaged photographs.

### Dataset Composition

| Component                 | Number of Images |
| ------------------------- | ---------------: |
| Clean Candidate Images    |              146 |
| Real Damaged Images       |               54 |
| Synthetic Clean Images    |              146 |
| Synthetic Degraded Images |              438 |
| Total                     |              874 |

The synthetic portion contains **438 confirmed clean/degraded image pairs**.

The three synthetic degradation categories are:

* `Faded_Scratches_Blur`
* `Noise_JPEG_Blur`
* `Complex_All`

The real damaged photographs are used for qualitative evaluation because confirmed clean ground-truth images are not available for them.

---

## 11. Evaluation Strategy

The project uses two evaluation approaches.

### Synthetic Evaluation

Synthetic degraded images are compared against their corresponding clean images.

The following metrics are calculated:

* MSE
* PSNR
* SSIM

This provides objective full-reference evaluation.

### Real-Image Evaluation

The real damaged set does not have reliable clean reference images.

Therefore, evaluation focuses on:

* Degradation analysis
* Operations selected
* Processing success
* Processing time
* Before/after visual inspection
* Artifact-mask inspection where applicable

No MSE, PSNR, or SSIM values are reported for real images.

---

## 12. Baseline Evaluation Results

The adaptive pipeline was evaluated on all 438 synthetic image pairs.

### Overall Results

| Metric | Before |  After |
| ------ | -----: | -----: |
| MSE    | 664.11 | 676.04 |
| PSNR   |  22.24 |  21.94 |
| SSIM   | 0.7810 | 0.7798 |

At the individual-image level:

* 70.5% improved according to MSE/PSNR
* 71.7% improved according to SSIM

The mean aggregate metrics decreased slightly because a smaller number of images experienced relatively large regressions.

This result highlights an important aspect of restoration evaluation: average metrics alone do not fully describe the behavior of an adaptive image-processing system.

---

## 13. Category-Level Results

### Faded + Scratches + Blur

| Metric | Before |  After |
| ------ | -----: | -----: |
| MSE    | 919.28 | 910.44 |
| PSNR   |  18.93 |  18.96 |
| SSIM   | 0.7784 | 0.7861 |

Individual-image improvement:

* MSE/PSNR: 74.0%
* SSIM: 86.3%

### Noise + JPEG + Blur

| Metric | Before |  After |
| ------ | -----: | -----: |
| MSE    | 165.19 | 207.10 |
| PSNR   |  28.77 |  27.86 |
| SSIM   | 0.8011 | 0.7939 |

Individual-image improvement:

* MSE/PSNR: 77.4%
* SSIM: 64.4%

### Complex All

| Metric | Before |  After |
| ------ | -----: | -----: |
| MSE    | 907.86 | 910.59 |
| PSNR   |  19.01 |  18.99 |
| SSIM   | 0.7636 | 0.7595 |

Individual-image improvement:

* MSE/PSNR: 60.3%
* SSIM: 64.4%

---

## 14. Real-Image Evaluation

All 54 real damaged images were successfully processed.

| Measure          |             Result |
| ---------------- | -----------------: |
| Images Processed |                 54 |
| Successful       |                 54 |
| Failed           |                  0 |
| Total Runtime    |      18.29 seconds |
| Average Runtime  | 0.34 seconds/image |

Detected degradation:

| Degradation  | Images |
| ------------ | -----: |
| Noise        |     14 |
| Blur         |     15 |
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

---

## 15. Software Architecture

The project follows a modular architecture.

```text
                       +----------------+
                       |   Streamlit UI |
                       +-------+--------+
                               |
                               v
                       +---------------+
                       |    Pipeline   |
                       +-------+-------+
                               |
             +-----------------+-----------------+
             |                 |                 |
             v                 v                 v
      Dataset Loader    Degradation Analyzer   Restoration
                                                 |
                              +------------------+----------------+
                              |                  |                |
                              v                  v                v
                         Denoising          Enhancement      Artifact Removal
                              |                  |                |
                              +------------------+----------------+
                                                 |
                                                 v
                                            Sharpening
                                                 |
                                                 v
                                          Restored Image
                                                 |
                                                 v
                                            Evaluation
```

The architecture separates:

* Data management
* Analysis
* Restoration
* Evaluation
* User interface

This separation improves maintainability and testing.

---

## 16. Non-Functional Requirements

### Performance

The system should process individual photographs within a practical interactive time.

### Reliability

Invalid images and incompatible metric inputs should be rejected explicitly.

### Maintainability

The implementation should remain modular, with individual responsibilities separated into dedicated source files.

### Usability

The application should provide clear visual feedback and expose the detected degradation and applied restoration operations.

### Resource Efficiency

Images should be processed individually rather than requiring the entire dataset to remain in memory.

### Reproducibility

Dependencies and the development environment should be reproducible using `uv` and `uv.lock`.

### Error Handling

The system should validate inputs and handle failures without silently producing invalid results.

---

## 17. Testing

Automated tests cover the major modules of the system.

The test suite includes:

* Dataset loader tests
* Metric tests
* Restoration tests
* Degradation analysis tests
* Artifact removal tests
* Frequency analysis tests
* Pipeline tests

The final test run produced:

```text
51 passed
```

No test failures were present in the final implementation.

---

## 18. Implementation Constraints

The project intentionally follows several constraints:

1. No deep learning models are used.
2. No pretrained restoration networks are used.
3. The restoration pipeline is based on deterministic classical computer vision operations.
4. Synthetic paired data is used for quantitative evaluation.
5. Real damaged images are evaluated qualitatively.
6. Input images are not silently resized during metric calculation.
7. Dataset files and generated results are not committed to the Git repository.

---

## 19. Known Limitations

The system has several known limitations.

### Fixed Thresholds

The degradation analyzer uses fixed thresholds. These values may not generalize equally well across every photograph collection.

### Blur Detection

The baseline blur threshold resulted in sharpening being applied broadly across the synthetic evaluation set.

### Noise Detection

The selected noise threshold did not trigger bilateral denoising on the synthetic evaluation set, although it did trigger on a subset of real images.

### Artifact Detection

Texture and fine image structures can sometimes resemble scratches or dust.

### Large Damage

The artifact-removal stage is designed primarily for relatively small regions. Large tears and missing areas cannot reliably be reconstructed with the current approach.

### Full-Reference Metrics

MSE, PSNR, and SSIM do not necessarily correspond perfectly to human visual preferences for historical photo restoration.

### Frequency-Domain Filtering

Binary frequency masks can introduce ringing artifacts around strong image transitions.

---

## 20. Future Work

Potential improvements include:

* Adaptive threshold selection
* Multi-scale artifact detection
* Improved scratch orientation analysis
* Better blur estimation
* Improved artifact classification
* More advanced inpainting strategies
* Improved frequency-domain filtering
* Perceptual quality evaluation
* Larger and more diverse datasets
* Human-based evaluation
* Optional comparison with deep learning methods

These improvements can be considered in future versions without changing the core classical computer vision architecture.

---

## 21. Expected Deliverables

The completed project provides:

* Modular Python implementation
* Dataset loading and validation
* Degradation analysis
* Adaptive restoration pipeline
* Artifact removal
* Frequency-domain analysis
* Quantitative evaluation
* Real-image evaluation
* Streamlit application
* Automated test suite
* Dataset documentation
* Project README
* Project statement
* Evaluation outputs

---

## 22. Expected Outcome

The expected outcome is a functional and explainable classical computer vision system capable of:

1. Receiving an old or degraded photograph.
2. Measuring its major degradation characteristics.
3. Selecting suitable restoration operations.
4. Producing a restored image.
5. Showing the processing decisions made by the system.
6. Providing quantitative evaluation when ground truth is available.
7. Supporting qualitative evaluation when ground truth is unavailable.

The project demonstrates how multiple classical computer vision concepts can be integrated into a complete application rather than being treated as isolated algorithms.

---

## 23. Conclusion

The **Old Photo Restoration and Damage Analysis System** addresses photograph degradation through a modular classical computer vision pipeline.

The system combines degradation analysis, filtering, enhancement, morphology, inpainting, sharpening, Fourier analysis, and image-quality evaluation. The adaptive architecture allows restoration operations to be selected according to detected image characteristics.

The synthetic dataset provides 438 clean/degraded pairs for quantitative evaluation, while the 54 real damaged images provide a practical qualitative testing set. The final implementation successfully processes the complete real-image set and passes all 51 automated tests.

The project therefore provides an end-to-end demonstration of classical computer vision techniques applied to a meaningful image restoration problem.
