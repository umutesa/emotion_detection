# Emotion Detection from Photos

A web app to detect faces and their emotions from uploaded photos

## Features

- Upload a photo (JPG/PNG)
- Detects all faces in the image
- Identifies emotions for each face 
- Annotated image and emotion breakdown

## How to Run

### Install Requirements

It's recommended to use a virtual environment.

```bash
pip install -r requirements.txt
```

### Run the App

```bash
streamlit run app.py
```

#### Alteranative : Run on Virtual Environmnet
```
sudo apt install python3.10 python3.10-venv

# Create a virtual environment
python3.10 -m venv .venv
source .venv/bin/activate

# Reinstall packages
pip install --upgrade pip
pip install onnxruntime opencv-python-headless streamlit numpy pillow requests


```

The app will open in your browser at [http://localhost:8501](http://localhost:8501).

### 4. Usage

1. Upload a photo with visible faces.
2. Wait a few seconds for processing.
3. See bounding boxes and emotion labels on the image

