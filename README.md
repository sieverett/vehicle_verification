# Vehicle Verification

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)

Vehicle image verification using ResNet50 feature extraction and cosine similarity.

[![Deploy To Azure](https://aka.ms/deploytoazurebutton)](https://portal.azure.com/#create/Microsoft.Template/uri/https://raw.githubusercontent.com/sieverett/vehicle_verification/main/azuredeploy.json)

---

## About

This application determines whether a given vehicle image matches any image in a set of pre-approved vehicles. It uses a pretrained ResNet50 convolutional neural network to extract feature vectors from images, then compares those vectors using cosine similarity against a configurable threshold.

The interface is built with Streamlit and includes an image management sidebar for uploading and removing reference images, a similarity threshold slider, and a detailed comparison table.

## How It Works

1. **Preprocessing** -- Input images are resized to 256px, center-cropped to 224px, and normalized to ImageNet statistics.
2. **Feature Extraction** -- A pretrained ResNet50 model (ImageNet weights) produces a 1000-dimensional feature vector for each image.
3. **Reference Database** -- Feature vectors for all approved vehicle images are serialized to a pickle file for fast lookup.
4. **Similarity Scoring** -- Cosine similarity is computed between the uploaded image's feature vector and every reference vector.
5. **Threshold Decision** -- If any similarity score meets or exceeds the user-defined threshold (default 0.8), the vehicle is marked as allowed.

## Getting Started

### Prerequisites

- Python 3.11+

### Installation

```bash
git clone https://github.com/sieverett/vehicle_verification.git
cd vehicle_verification
pip install -r requirements.txt
```

### Running the Application

```bash
cd app
streamlit run app.py
```

1. Use the sidebar **Image Manager** to upload reference vehicle images.
2. Adjust the **similarity threshold** in the sidebar Settings panel.
3. Upload a vehicle image in the main area to verify it against the reference set.

## Project Structure

```
vehicle_verification/
  app/
    app.py              # Streamlit application
    vehicles/           # Reference vehicle images (user-managed, gitignored)
  azuredeploy.json      # Azure App Service ARM template
  azuredeploy.parameters.json
  requirements.txt
```

## License

MIT. See [LICENSE](LICENSE) for details.
