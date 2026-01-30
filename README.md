# Face Parsing using U-Net Architecture

## 📌 Project Overview
This project implements a **U-Net** based deep learning model for semantic segmentation of facial attributes (Face Parsing). The model is designed to receive a facial image as input and generate pixel-wise masks for various facial components. This allows for precise separation and labeling of specific facial features.

## 🎯 Key Features
* **Architecture:** U-Net with **Sigmoid** activation function to handle multi-class segmentation tasks efficiently.
* **Input Resolution:** 512x512 pixels.
* **Model Complexity:** Approximately **32 Million parameters**.
* **Segmentation Classes:** The model segments the face into distinct regions, including:
    * Eyes & Eyebrows
    * Nose
    * Lips (Upper & Lower)
    * Hair
    * Ears
    * Facial Skin

## 📂 Dataset
The model was trained on the **CelebAMask-HQ** dataset.
* **Dataset Size:** ~5.5 GB
* **Content:** High-quality facial images with corresponding segmentation masks.

## ⚙️ Training Details
* **Epochs:** Trained for 5-10 epochs.
* **Optimization:** The model utilizes the U-Net structure to capture both local features (via the contracting path) and global context (via the expanding path), ensuring accurate boundary detection for facial parts.

## 🚀 Applications
This model has practical applications in various domains, including:
* **Face Editing & Retouching:** Automated makeup application and skin smoothing.
* **Face Generation:** Conditional GANs and avatar creation.
* **Medical Imaging:** The underlying U-Net architecture is also applicable to medical segmentation tasks (e.g., CT scans, Thyroid nodule detection).
