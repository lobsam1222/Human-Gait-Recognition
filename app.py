import streamlit as st
import os
import numpy as np
from utils.video_to_frames import extract_frames
from utils.extract_silhouette import get_silhouettes
from utils.feature_extractor import compute_gait_feature
from utils.matcher import match_gait
from utils.foreground_segmenter import segment_foreground_mediapipe

# Set page config
st.set_page_config(page_title="Gait Recognition System", layout="wide")
st.title("🧍‍♂️ Gait Recognition System")

# Select mode
mode = st.radio("Select Mode", ["Enroll New Person", "Recognize Person"])

if mode == "Enroll New Person":
    st.subheader("👤 Enrollment Mode")
    with st.form("enroll_form"):
        name = st.text_input("Enter your name")
        uploaded_file = st.file_uploader("Upload your walking video", type=["mp4", "avi"], key="enroll")
        submit = st.form_submit_button("Enroll")

    if submit and uploaded_file and name:
        video_path = f"data/uploads/{name}_{uploaded_file.name}"
        os.makedirs("data/uploads", exist_ok=True)
        with open(video_path, "wb") as f:
            f.write(uploaded_file.read())

        st.video(video_path)
        st.info("Processing video for enrollment...")

        # Extract frames
        frames = extract_frames(video_path, output_dir=f"data/frames/{name}")
        segmented_paths = segment_foreground_mediapipe(frames, output_dir=f"data/foreground/{name}")

        # Display segmented images
        st.subheader("Segmented Frames")
        cols = st.columns(5)
        for i, img_path in enumerate(segmented_paths[:10]):
            with cols[i % 5]:
                st.image(img_path, use_container_width=True)

        # Extract silhouettes and compute gait features
        silhouettes = get_silhouettes(frames, output_dir=f"data/silhouettes/{name}")
        gait_feature = compute_gait_feature(silhouettes)

        # Save feature to database
        db_path = "data/database.npy"
        if os.path.exists(db_path):
            database = np.load(db_path, allow_pickle=True).item()
        else:
            database = {}
        database[name] = gait_feature
        np.save(db_path, database)

        st.success(f"{name} successfully enrolled!")

elif mode == "Recognize Person":
    st.subheader("🔍 Recognition Mode")
    uploaded_file = st.file_uploader("Upload a short walking video for recognition", type=["mp4", "avi"], key="recognize")

    if uploaded_file:
        video_path = f"data/test/{uploaded_file.name}"
        os.makedirs("data/test", exist_ok=True)
        with open(video_path, "wb") as f:
            f.write(uploaded_file.read())

        st.video(video_path)
        st.info("Processing video for recognition...")

        # Extract frames
        frames = extract_frames(video_path, output_dir="data/frames/test")
        segmented_paths = segment_foreground_mediapipe(frames, output_dir="data/foreground/test")

        # Display segmented images
        st.subheader("Segmented Frames")
        cols = st.columns(5)
        for i, img_path in enumerate(segmented_paths[:10]):
            with cols[i % 5]:
                st.image(img_path, use_container_width=True)

        # Extract silhouettes and compute gait feature
        silhouettes = get_silhouettes(frames, output_dir="data/silhouettes/test")
        st.success(f"{len(silhouettes)} silhouettes extracted.")

        gait_feature = compute_gait_feature(silhouettes)

        # Match with database
        results = match_gait(gait_feature, db_path="data/database.npy")

        st.subheader("Recognition Results")
        st.table(results)

        if results:
            st.success(f"Person identified as: **{results[0]['name']}** with {results[0]['probability']:.2f}% similarity.")
