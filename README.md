# Imagizer: Image Processing Desktop Application

Imagizer is a desktop application built using PyQt to perform various image processing algorithms. It allows users to manipulate standard images (grayscale and color) with functionalities including noise addition, filtering, edge detection, histogram analysis, normalization, equalization, thresholding, color to grayscale conversion, frequency domain filtering, and hybrid image generation.

## Table of Contents
1. [Installation](#installation)
2. [Usage](#usage)
3. [Features](#features)
4. [Requirements](#requirements)
5. [Contributors](#contributors)

## Installation
To install the project, clone the repository and install the dependencies:

```bash
# Clone the repository
git clone https://github.com/Zoz-HF/Image_Descriptor

# Navigate to the project directory
cd Image_Descriptor
```

## Usage
To run the application, use the following command:

```bash
python index.py
```

## Features

### 1. Additive Noise
   - Uniform Noise
   - Gaussian Noise
   - Salt and Pepper Noise 


### 2. Filtering
   - Average Filter
   - Gaussian Filter
   - Median Filter

   ![Noise and smoothing](assets/gifs/Noise_smooth.gif)

### 3. Edge Detection
   - Sobel Operator
   - Roberts Operator
   - Prewitt Operator
   - Laplace Operator
   - Canny Edge Detector

   ![Edge Detection](assets/gifs/edges.gif)

### 4. Histogram Analysis
   - Histogram Plotting
   - Distribution Curve Plotting


### 5. Equalization

### 6. Normalization

   ![Histogram Analysis](assets/gifs/Histograms.gif)


### 7. Thresholding
   - Local Thresholding
   - Global Thresholding

   ![Thresholding](assets/gifs/Thresholding.gif)

### 8. Color to Grayscale Conversion
   - Plotting R, G, and B histograms
   - Distribution Function Plotting

### 9. Frequency Domain Filtering
   - High Pass Filters
   - Low Pass Filters

### 10. Hybrid Images

   ![Hybrid Images](assets/gifs/Hybrid_image.gif)


## Requirements

- Python 3.x
- PyQt5
- OpenCV

## Contributors

- [Ziyad El Fayoumy](https://github.com/Zoz-HF)
- [Assem Hussein](https://github.com/RushingBlast)
- [Mohamed Sayed Diab](https://github.com/MohamedSayedDiab)
- [Abdel Rahman Shawky](https://github.com/AbdulrahmanGhitani)
- [Omar Shaban](https://github.com/omarshaban02)
