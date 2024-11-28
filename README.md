# Imagizer: Image Processing Desktop Application

Imagizer is a desktop application built using PyQt to perform various image processing algorithms. It allows users to manipulate standard images (grayscale and color) with functionalities including noise addition, filtering, edge detection, histogram analysis, normalization, equalization, thresholding, color to grayscale conversion, frequency domain filtering, and hybrid image generation.

## Table of Contents
1. [Installation](#installation)
2. [Usage](#usage)
3. [Features](#features)
4. [Dependencies](#dependencies)
5. [Contributors](#contributors)

## Installation
To install the project, clone the repository and install the requirements:

```bash
# Clone the repository
git clone https://github.com/Zoz-HF/Image_Descriptor
```
```bash
# Navigate to the project directory
cd Image_Descriptor
```
```bash
# Install the required packages:
pip install -r requirements.txt
```
```bash
# Run the application:
python main.py
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


## Dependencies
This project requires the following Python packages listed in the `requirements.txt` file:
- PyQt5
- opencv
- numpy
- scipy 
  
## Contributors <a name = "contributors"></a>
<table>
  <tr>
    <td align="center">
    <a href="https://github.com/AbdulrahmanGhitani" target="_black">
    <img src="https://avatars.githubusercontent.com/u/114954706?v=4" width="150px;" alt="Abdulrahman Shawky"/>
    <br />
    <sub><b>Abdulrahman Shawky</b></sub></a>
    </td>
  <td align="center">
    <a href="https://github.com/Ziyad-HF" target="_black">
    <img src="https://avatars.githubusercontent.com/u/99608059?v=4" width="150px;" alt="Ziyad El Fayoumy"/>
    <br />
    <sub><b>Ziyad El Fayoumy</b></sub></a>
    </td>
<td align="center">
    <a href="https://github.com/omarnasser0" target="_black">
    <img src="https://avatars.githubusercontent.com/u/100535160?v=4" width="150px;" alt="omarnasser0"/>
    <br />
    <sub><b>Omar Abdulnasser</b></sub></a>
    </td>
    <td align="center">
    <a href="https://github.com/MohamedSayedDiab" target="_black">
    <img src="https://avatars.githubusercontent.com/u/90231744?v=4" width="150px;" alt="Mohammed Sayed Diab"/>
    <br />
    <sub><b>Mohammed Sayed Diab</b></sub></a>
    </td>
     <td align="center">
    <a href="https://github.com/RushingBlast" target="_black">
    <img src="https://avatars.githubusercontent.com/u/96780345?v=4" width="150px;" alt="Assem Hussein"/>
    <br />
    <sub><b>Assem Hussein</b></sub></a>
    </td>
      </tr>
 </table>
