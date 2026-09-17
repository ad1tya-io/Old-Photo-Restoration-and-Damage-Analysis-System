# Dataset Documentation

## Folder Structure

The dataset is located in the `dataset/` directory and is organized into three main groups:

- `01_Clean_Candidates_GT/`: Contains clean reference photographs. No explicit degradations.
- `02_Damaged_Testing_Set/`: Contains real damaged photographs intended for testing the final pipeline.
- `03_Synthetic_Dataset/`: Contains paired images for development and evaluation.
  - `Train_GT_Clean/`: The ground truth (clean) source images.
  - `Train_Input_Degraded/`: The artificially degraded versions of the clean images.

## Role of Each Dataset Group

1. **Synthetic Dataset (`03_Synthetic_Dataset`)**: This is the primary dataset for algorithm development, experimentation, and quantitative evaluation. Because it contains perfect ground-truth pairs, we can mathematically evaluate restoration success.
2. **Real Damaged Testing Set (`02_Damaged_Testing_Set`)**: This is for qualitative/demo testing. It shows how well the algorithms generalize to real-world historical photos that do not have a ground truth.
3. **Clean Candidates (`01_Clean_Candidates_GT`)**: Used as reference distributions for color correction, histogram matching, or potentially for generating additional synthetic data in the future.

## Synthetic Pairing Strategy

Images in the synthetic dataset are paired using their source IDs extracted from their filenames. 

- A clean image is named `lrp_imgXXX.jpg`.
- Its corresponding degraded image(s) are named `lrp_imgXXX_(Degradation_Type).jpg`.

The `DatasetLoader` automatically scans these directories, extracts the common source ID (`lrp_imgXXX`), and groups them into `SyntheticImagePair` objects.

## Degradation Filename Convention

In the `Train_Input_Degraded` directory, the filenames encode the degradation type in parentheses. 
For example: `lrp_img100_(Noise_JPEG_Blur).jpg`.

The `DatasetLoader` parses this to extract:
- **Source ID**: `lrp_img100`
- **Degradation Type**: `Noise_JPEG_Blur`
- **Degradation Components**: `['Noise', 'JPEG', 'Blur']`

*Note: These labels are dataset-provided and do not necessarily represent objectively measured degradation severity. They indicate what type of artificial degradation was applied.*

## Synthetic Evaluation vs. Real Damaged Testing

**Synthetic Evaluation** uses the paired dataset to compare a restored image directly against its perfect ground truth pixel-by-pixel.

**Real Damaged Testing** involves running the restoration pipeline on a real historical photo and visually inspecting the results, as no perfect clean version exists.

## Quantitative Metrics (PSNR/SSIM)

- **Why PSNR/SSIM can be used for synthetic paired data:** Since we have the exact original clean image, we can mathematically calculate the Peak Signal-to-Noise Ratio (PSNR) and Structural Similarity Index (SSIM) between the restored output and the true original to measure how much information was recovered.
- **Why PSNR/SSIM cannot directly be used for the real damaged testing images:** These metrics require a reference (ground truth) image. Real historical photos do not have a "clean" reference, so these full-reference metrics cannot be calculated. (No-reference metrics like BRISQUE could theoretically be used instead).

## Assumptions

- The loader assumes that all images can be loaded into an RGB format. Grayscale and RGBA images are converted to RGB upon loading.
- The loader does not perform any automatic resizing. The actual restoration algorithms must handle dimension requirements or patches.
