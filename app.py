import streamlit as st
import cv2
from PIL import Image
from synesthesia_ai import analyze_emotion_text, poetic_line, caption_image, detect_objects, text_to_speech

st.set_page_config(page_title="AI Synesthesia for Accessibility", layout="wide")

st.title("🎨 AI Synesthesia – Accessible Art for the Visually Challenged")

mode = st.radio("Choose Mode:", ["📷 Camera (Real-time Object → Sound)", "🖼 Image Upload → Poem & Music", "✍️ Text/Food → Poem & Voice"])

# ---- CAMERA MODE ----
if mode == "📷 Camera (Real-time Object → Sound)":
    st.write("Show an object in front of your webcam, I'll describe and play its soundscape 🎶")
    camera_input = st.camera_input("Take a snapshot")
    if camera_input:
        img = Image.open(camera_input)
        frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        objects = detect_objects(frame)
        if objects:
            desc = f"I see a {objects[0]}."
            st.write(desc)
            audio_file = text_to_speech(desc)
            st.audio(audio_file, format="audio/wav")

# ---- IMAGE UPLOAD MODE ----
elif mode == "🖼 Image Upload → Poem & Music":
    uploaded = st.file_uploader("Upload an image", type=["jpg","png","jpeg"])
    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        st.image(img, caption="Uploaded Image")
        caption = caption_image(img)
        poem = poetic_line(caption)
        st.write(f"📜 Caption: {caption}")
        st.write(f"🎶 Poem: {poem}")
        audio_file = text_to_speech(poem)
        st.audio(audio_file, format="audio/wav")

# ---- TEXT MODE ----
elif mode == "✍️ Text/Food → Poem & Voice":
    text = st.text_area("Enter text (food, feelings, objects)...")
    if st.button("Generate"):
        if text:
            emotion = analyze_emotion_text(text)
            poem = poetic_line(text)
            st.subheader(f"🔎 Detected Emotion: {emotion}")
            st.write(f"📜 Poem: {poem}")
            audio_file = text_to_speech(poem)
            st.audio(audio_file, format="audio/wav")
