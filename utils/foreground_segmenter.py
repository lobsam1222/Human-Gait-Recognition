# utils/foreground_segmenter.py

import cv2
import numpy as np
import mediapipe as mp
import os

def segment_foreground_mediapipe(frame_paths, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    mp_selfie_segmentation = mp.solutions.selfie_segmentation
    segmented_images = []

    with mp_selfie_segmentation.SelfieSegmentation(model_selection=1) as selfie_seg:
        for path in frame_paths:
            image = cv2.imread(path)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = selfie_seg.process(image_rgb)
            
            condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.3
            bg_image = np.ones(image.shape, dtype=np.uint8) * 255  # white background
            output_image = np.where(condition, image, bg_image)
            
            out_path = os.path.join(output_dir, os.path.basename(path))
            cv2.imwrite(out_path, output_image)
            segmented_images.append(out_path)

    return segmented_images
