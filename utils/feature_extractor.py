import cv2
import numpy as np

def compute_gait_feature(silhouettes):
    imgs = [cv2.imread(path, 0) for path in silhouettes]
    imgs = [cv2.resize(img, (64, 128)) for img in imgs]  # normalize
    avg_img = np.mean(imgs, axis=0)
    return avg_img.flatten()
