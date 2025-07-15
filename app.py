import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from PIL import Image, ImageDraw, ImageFont
import onnxruntime as ort
import os
import urllib.request

# Emotion labels
EMOTION_LABELS = ["neutral", "happy", "surprise", "sad", "angry", "disgust", "fear", "contempt"]

@st.cache_resource
def load_model():
    """Load the ONNX emotion model with error handling"""
    # Try multiple download sources
    model_urls = [
        "https://github.com/onnx/models/raw/main/validated/vision/body_analysis/emotion_ferplus/model/emotion-ferplus-8.onnx",
        "https://huggingface.co/webml/models/resolve/main/emotion-ferplus-8.onnx"
    ]
    model_dir = "models"
    model_path = os.path.join(model_dir, "emotion-ferplus.onnx")
    
    # Create models directory if it doesn't exist
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    # Download model if it doesn't exist
    if not os.path.exists(model_path):
        downloaded = False
        for i, model_url in enumerate(model_urls):
            try:
                st.info(f"Downloading emotion model from source {i+1}... This may take a moment.")
                urllib.request.urlretrieve(model_url, model_path)
                st.success("Model downloaded successfully!")
                downloaded = True
                break
            except Exception as e:
                st.warning(f"Failed to download from source {i+1}: {e}")
                if i == len(model_urls) - 1:  # Last attempt
                    st.error("Failed to download from all sources. Please download manually.")
                    st.error("Manual download command:")
                    st.code("wget https://github.com/onnx/models/raw/main/validated/vision/body_analysis/emotion_ferplus/model/emotion-ferplus-8.onnx -O models/emotion-ferplus.onnx")
                    st.stop()
        
        if not downloaded:
            st.stop()
    
    # Load the model
    try:
        session = ort.InferenceSession(model_path)
        return session
    except Exception as e:
        st.error(f"Failed to load ONNX model: {e}")
        st.stop()

def preprocess_face_multiple_methods(face_img):
    """Try multiple preprocessing methods for better emotion detection"""
    methods = []
    
    # Convert to grayscale if needed
    if len(face_img.shape) == 3:
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = face_img
    
    # Method 1: Original approach
    face1 = cv2.resize(gray, (64, 64)).astype(np.float32)
    face1 = (face1 - 128) / 128
    face1 = face1[np.newaxis, np.newaxis, :, :]
    methods.append(("Original", face1))
    
    # Method 2: Histogram equalization + different normalization
    face2 = cv2.equalizeHist(gray)
    face2 = cv2.resize(face2, (64, 64)).astype(np.float32)
    face2 = face2 / 255.0  # 0-1 normalization
    face2 = face2[np.newaxis, np.newaxis, :, :]
    methods.append(("Histogram Eq + 0-1", face2))
    
    # Method 3: CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    face3 = clahe.apply(gray)
    face3 = cv2.resize(face3, (64, 64)).astype(np.float32)
    face3 = (face3 - 127.5) / 127.5  # -1 to 1 normalization
    face3 = face3[np.newaxis, np.newaxis, :, :]
    methods.append(("CLAHE", face3))
    
    # Method 4: Standard normalization
    face4 = cv2.resize(gray, (64, 64)).astype(np.float32)
    face4 = (face4 - np.mean(face4)) / (np.std(face4) + 1e-7)
    face4 = face4[np.newaxis, np.newaxis, :, :]
    methods.append(("Z-score", face4))
    
    return methods

def predict_emotion_advanced(face_img, session):
    """Advanced emotion prediction with multiple methods and debugging"""
    try:
        methods = preprocess_face_multiple_methods(face_img)
        all_results = []
        
        for method_name, blob in methods:
            try:
                outputs = session.run(None, {"Input3": blob})
                probabilities = outputs[0][0]
                
                # Try different probability interpretations
                # Method 1: Direct probabilities
                if len(probabilities) == 8:  # Should be 8 emotions
                    probs1 = probabilities
                    if np.any(probs1 < 0):  # If negative values, apply softmax
                        probs1 = np.exp(probs1) / np.sum(np.exp(probs1))
                    else:
                        probs1 = probs1 / np.sum(probs1)  # Normalize
                    
                    emotion_idx = np.argmax(probs1)
                    confidence = probs1[emotion_idx]
                    
                    all_results.append({
                        'method': method_name,
                        'emotion': EMOTION_LABELS[emotion_idx],
                        'confidence': confidence,
                        'probabilities': probs1
                    })
                    
            except Exception as e:
                st.warning(f"Method {method_name} failed: {e}")
                continue
        
        if not all_results:
            return "unknown", 0.0, [], []
        
        # Find the result with highest non-neutral confidence
        best_non_neutral = None
        best_overall = max(all_results, key=lambda x: x['confidence'])
        
        for result in all_results:
            if result['emotion'] != 'neutral' and result['confidence'] > 0.3:
                if best_non_neutral is None or result['confidence'] > best_non_neutral['confidence']:
                    best_non_neutral = result
        
        # Use non-neutral if available and reasonably confident
        final_result = best_non_neutral if best_non_neutral else best_overall
        
        # Get top 3 emotions from best result
        probs = final_result['probabilities']
        top_3_indices = np.argsort(probs)[-3:][::-1]
        top_3_emotions = [(EMOTION_LABELS[i], probs[i]) for i in top_3_indices]
        
        return final_result['emotion'], final_result['confidence'], top_3_emotions, all_results
        
    except Exception as e:
        st.error(f"Error predicting emotion: {e}")
        return "unknown", 0.0, [], []

def draw_results(image, results):
    """Draw bounding boxes and emotion labels on image"""
    pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    
    try:
        font = ImageFont.truetype("arial.ttf", 16)
        font_small = ImageFont.truetype("arial.ttf", 12)
    except:
        font = ImageFont.load_default()
        font_small = ImageFont.load_default()
    
    # Color mapping for emotions
    emotion_colors = {
        "happy": "green",
        "sad": "blue",
        "angry": "red",
        "surprise": "orange",
        "fear": "purple",
        "disgust": "brown",
        "contempt": "darkred",
        "neutral": "gray"
    }
    
    for (x, y, w, h, emotion, confidence) in results:
        color = emotion_colors.get(emotion, "blue")
        
        # Draw bounding box
        draw.rectangle([x, y, x + w, y + h], outline=color, width=3)
        
        # Draw emotion label with confidence
        label = f"{emotion} ({confidence:.1%})"
        draw.text((x, y - 30), label, fill=color, font=font)
    
    return pil_img

def main():
    st.set_page_config(page_title="Face & Emotion Detector", layout="centered")
    st.title("Emotion Detector")
    
    # Load model
    session = load_model()    
    uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        try:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, 1)
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h_img, w_img, _ = rgb_img.shape
            st.image(rgb_img, caption="Uploaded Image", use_container_width=True)
            
            with st.spinner("Detecting faces and analyzing emotions..."):
                mp_face = mp.solutions.face_detection
                results = []
                
                with mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.5) as detector:
                    detection = detector.process(rgb_img)
                    
                    if detection.detections:
                        for det in detection.detections:
                            bbox = det.location_data.relative_bounding_box
                            x = int(bbox.xmin * w_img)
                            y = int(bbox.ymin * h_img)
                            w = int(bbox.width * w_img)
                            h = int(bbox.height * h_img)
                            
                            # Ensure coordinates are within image bounds
                            x = max(0, x)
                            y = max(0, y)
                            w = min(w, w_img - x)
                            h = min(h, h_img - y)
                            
                            # Add padding to face crop for better emotion detection
                            padding = 20
                            x_pad = max(0, x - padding)
                            y_pad = max(0, y - padding)
                            w_pad = min(w_img - x_pad, w + 2 * padding)
                            h_pad = min(h_img - y_pad, h + 2 * padding)
                            
                            face_crop = img[y_pad:y_pad+h_pad, x_pad:x_pad+w_pad]
                            if face_crop.size > 0:
                                emotion, confidence, top_3, all_methods = predict_emotion_advanced(face_crop, session)
                                results.append((x, y, w, h, emotion, confidence, top_3, all_methods))
                    else:
                        st.warning("No faces detected.")
            
            if results:
                st.success(f"Detected {len(results)} face(s).")
                
                # Draw results (only need first 6 elements for drawing)
                results_for_drawing = [(x, y, w, h, emotion, confidence) for x, y, w, h, emotion, confidence, top_3, all_methods in results]
                result_img = draw_results(img, results_for_drawing)
                st.image(result_img, caption="Detected Faces & Emotions", use_container_width=True)
                
                st.subheader("Detailed Emotion Analysis")
                
                # Create columns for better layout
                for idx, (_, _, _, _, emotion, confidence, top_3, all_methods) in enumerate(results):
                    with st.expander(f"Face {idx + 1}: {emotion.capitalize()} ({confidence:.1%})", expanded=True):
                        
                        # Show confidence bar
                        st.metric("Selected Emotion", emotion.capitalize(), f"{confidence:.1%}")
                        
                        # Show top 3 emotions
                        st.write("**Top 3 Emotions:**")
                        for i, (emo, prob) in enumerate(top_3):
                            st.write(f"{i+1}. {emo.capitalize()}: {prob:.1%}")
                        
                        # Show results from all methods
                        st.write("**All Detection Methods:**")
                        methods_df_data = []
                        for method_result in all_methods:
                            methods_df_data.append({
                                'Method': method_result['method'],
                                'Emotion': method_result['emotion'].capitalize(),
                                'Confidence': f"{method_result['confidence']:.1%}"
                            })
                        
                        if methods_df_data:
                            import pandas as pd
                            methods_df = pd.DataFrame(methods_df_data)
                            st.dataframe(methods_df, use_container_width=True)
                        
                        # Visualization of probabilities
                        st.write("**Probability Distribution:**")
                        if top_3:
                            import pandas as pd
                            df = pd.DataFrame({
                                'Emotion': [emo.capitalize() for emo, _ in top_3],
                                'Probability': [prob for _, prob in top_3]
                            })
                            st.bar_chart(df.set_index('Emotion'))
                        
                        # Debug information
                        with st.expander("Debug Information"):
                            st.write("**Raw probabilities from best method:**")
                            if all_methods:
                                best_method = max(all_methods, key=lambda x: x['confidence'])
                                for i, (label, prob) in enumerate(zip(EMOTION_LABELS, best_method['probabilities'])):
                                    st.write(f"{label}: {prob:.4f}")
                        
                        # Add emotion interpretation
                        if confidence > 0.7:
                            st.success("High confidence detection")
                        elif confidence > 0.4:
                            st.warning("Medium confidence detection")
                        else:
                            st.info("Low confidence detection - results may be uncertain")
            else:
                st.warning("Try a clearer image with visible faces.")
                
        except Exception as e:
            st.error(f"Error processing image: {e}")
    else:
        st.info("Upload an image to get started.")

if __name__ == "__main__":
    main()