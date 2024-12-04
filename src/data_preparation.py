import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import cv2
import kagglehub

# Download latest version
def download_images():
    path = kagglehub.dataset_download("soumikrakshit/div2k-high-resolution-images")

    print("Path to dataset files:", path)
    return path

# Data Preparation: Load images and create low-resolution versions
def load_images(path, img_size=(256, 256)):
    images = []
    for dirpath, _, filenames in os.walk(path):
        for file in filenames:
            if file.endswith(".png") or file.endswith(".jpg"):
                img_path = os.path.join(dirpath, file)
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, img_size)
                images.append(img)
    return np.array(images)

# Create low-res images
def create_low_res(hr_images, scale = 2):
  lr_images = []
  for img in hr_images:
    lr_img = cv2.resize(img, (img.shape[1] // scale, img.shape[0] // scale), interpolation=cv2.INTER_CUBIC)
    lr_img = cv2.resize(lr_img, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)
    lr_images.append(lr_img)
  return np.array(lr_images)


# Extract patches from images
def extract_patches(images, patch_size=64, stride=32):
    patches = []
    for img in images:
        for i in range(0, img.shape[0] - patch_size + 1, stride):
            for j in range(0, img.shape[1] - patch_size + 1, stride):
                patch = img[i:i + patch_size, j:j + patch_size]
                patches.append(patch)
    return np.array(patches)