# AI Image Classifier

A simple web app that uses a pre-trained MobileNetV2 model to classify images. Built with [Streamlit](https://streamlit.io/), [TensorFlow](https://www.tensorflow.org/), and [OpenCV](https://opencv.org/).

## Features

- Upload an image and get the top 3 predictions for its content.
- Fast and interactive UI powered by Streamlit.
- Uses MobileNetV2 trained on ImageNet for robust image recognition.

## Installation

1. **Clone the repository:**
   ```sh
   git clone https://github.com/yourusername/ai-image-classifier.git
   cd ai-image-classifier
   ```

2. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```

## Usage

Run the app with Streamlit:

```sh
streamlit run main.py
```

Open your browser and go to the URL shown in the terminal (usually http://localhost:8501).

## Project Structure

- `main.py` — Main application code.
- `requirements.txt` — List of required Python packages.

## Credits

- Model: [MobileNetV2](https://keras.io/api/applications/mobilenet/)
- UI: [Streamlit](https://streamlit.io/)
- Author: Bibek Subedi
