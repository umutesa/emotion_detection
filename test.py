import urllib.request
import os
import onnxruntime as ort

def download_model():
    model_url = "https://github.com/onnx/models/raw/main/validated/vision/body_analysis/emotion_ferplus/model/emotion-ferplus-8.onnx"
    model_dir = "models"
    model_path = os.path.join(model_dir, "emotion-ferplus-8.onnx")
    
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    if not os.path.exists(model_path):
        st.info("Downloading emotion model...")
        urllib.request.urlretrieve(model_url, model_path)
        st.success("Model downloaded successfully!")
    
    return model_path

# Use it like this:
onnx_model_path = download_model()
session = ort.InferenceSession(onnx_model_path)