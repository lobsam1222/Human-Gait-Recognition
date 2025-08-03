import cv2
import os

def get_silhouettes(frame_paths, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    bg_subtractor = cv2.createBackgroundSubtractorMOG2()
    silhouettes = []

    for frame_path in frame_paths:
        frame = cv2.imread(frame_path)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        fg_mask = bg_subtractor.apply(gray)
        _, thresh = cv2.threshold(fg_mask, 250, 255, cv2.THRESH_BINARY)
        out_path = os.path.join(output_dir, os.path.basename(frame_path))
        cv2.imwrite(out_path, thresh)
        silhouettes.append(out_path)
    return silhouettes
