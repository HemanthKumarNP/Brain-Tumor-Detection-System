# Brain-Tumor-Detection-System

## Overview

This project is a brain tumor detection system that utilizes a Convolutional Neural Network (CNN) with EfficientNetB1 for classifying brain MRI images. The system can identify four types of brain conditions: glioma, meningioma, pituitary adenoma, and no tumor. The model achieved 99% accuracy on training data and 98% accuracy on testing data. Additionally, the project integrates Grad-CAM for visualizing regions of interest and the Gemini Pro Vision model for providing detailed insights, including tumor type, location, recommended treatments, and associated symptoms.

## Features

- **Accurate Classification**: Classifies glioma, meningioma, pituitary adenoma, and no tumor with high accuracy.
- **Visualization**: Utilizes Grad-CAM to generate heatmaps indicating areas of the MRI scan that the model focuses on.
- **Comprehensive Insights**: Integrates Gemini Pro Vision for detailed information on tumor type, location, recommended treatments, and symptoms.
- **User-Friendly Interface**: Frontend built with HTML, CSS, and JavaScript.
- **Backend Integration**: Backend developed with Flask, and the application containerized using Docker.
- **Award-Winning**: Won 3rd prize at the college project exhibition.
- **Publication**: [Published](https://www.irjmets.com/uploadedfiles/paper//issue_5_may_2024/56574/final/fin_irjmets1716062680.pdf) in the International Research Journal of Modernization in Engineering, Technology, and Science.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/HemanthKumarNP/Brain-Tumor-Detection-System.git
    cd Brain-Tumor-Detection-System
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Start the Flask server:
    ```bash
    python app.py
    ```

2. Build and run the Docker container:
    ```bash
    docker build -t brain-tumor-detection .
    docker run -p 5000:5000 brain-tumor-detection
    ```

3. Open `home.html` in your browser and upload an MRI image to get the classification, heatmap visualization, and detailed insights.

## Dataset

The dataset consists of MRI images categorized into four directories:
- `glioma`
- `meningioma`
- `pituitary`
- `no tumor`

## Model Architecture

The model uses EfficientNetB1 as the base model and includes the following layers:
- GlobalAveragePooling2D
- Dropout (0.5)
- Dense layer with softmax activation

The model is fine-tuned using transfer learning.

## Results

- **Accuracy**: 99% on training data, 98% on testing data.
- **Visualization**: Grad-CAM heatmaps to highlight regions of interest.
- **Insights**: Gemini Pro Vision provides detailed tumor type, location, recommended treatments, and symptoms.

## License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- **EfficientNetB1**: [EfficientNet](https://arxiv.org/abs/1905.11946) by Google AI
- **Grad-CAM**: [Grad-CAM: Visual Explanations from Deep Networks](https://arxiv.org/abs/1610.02391)
- **Gemini Pro Vision**: For providing comprehensive insights on tumor characteristics.
