# Facial Emotion Detection

Detect faces and their emotions 

<img width="1920" height="1166" alt="image" src="https://github.com/user-attachments/assets/4d8281dd-e5c5-4a35-b09a-048b2055c107" />

# Tools Used

<p style="display: flex; justify-content: start; gap: 55px;">
  <img src="https://upload.wikimedia.org/wikipedia/commons/c/c3/Python-logo-notext.svg" alt="Python" height="100" width="120"/>
  <img src="https://upload.wikimedia.org/wikipedia/commons/3/32/OpenCV_Logo_with_text_svg_version.svg" alt="OpenCV" height="100" width="120"/>
  <img src="https://huggingface.co/front/assets/huggingface_logo.svg" alt="Hugging Face" height="100" width="120"/>
</p>

# About Project

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


```


### 4. Usage

1. Upload a photo with visible faces.
2. Wait a few seconds for processing.
3. See bounding boxes and emotion labels on the image

