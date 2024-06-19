# Vehicle Verification and Image Management App

## Overview

This project is a Streamlit-based application designed for managing and verifying vehicle images using machine learning. The app allows users to upload and manage vehicle images and verify if a new vehicle image matches any pre-approved vehicles based on a similarity threshold.

## Features

- **Image Manager:** Upload, view, and delete vehicle images.
- **Vehicle Verification:** Check if a new vehicle image matches any allowed vehicles using a machine learning model.
- **Threshold Adjustment:** Set the similarity threshold for verification via a slider.

## Approach

The project uses a pre-trained ResNet50 model to extract features from vehicle images. These features are compared using cosine similarity to determine how similar the uploaded vehicle image is to the pre-approved images.

### Steps:

1. **Image Preprocessing:** Images are resized, center-cropped, and normalized.
2. **Feature Extraction:** Features are extracted using the ResNet50 model.
3. **Similarity Calculation:** Cosine similarity is used to compare features between images.
4. **Threshold Check:** The similarity score is checked against a user-defined threshold to verify if the vehicle is allowed.

## Installation

To install the required packages, run:
```sh
pip install -r requirements.txt
```

## Usage

1. Run the Streamlit app:
   ```sh
   streamlit run app.py
   ```
2. Use the sidebar to upload and manage vehicle images.
3. Upload a new vehicle image in the main interface to verify if it is allowed based on the similarity threshold.

## Requirements

- Python 3.7+
- Streamlit
- Torch
- Torchvision
- Pillow
- Numpy
- Pandas
- Scikit-learn

## License

This project is licensed under the MIT License. See the LICENSE file for details.