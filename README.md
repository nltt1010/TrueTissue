# AI-Powered Virtual Staining & Abnormality Detection

## 1. Project Overview
This project applies Artificial Intelligence (AI) in Healthcare, specifically focusing on **Virtual Staining**.
In clinical practice, the traditional chemical staining process of tissue slides using H&E (Hematoxylin & Eosin) is time-consuming, labor-intensive, and costly. This project provides a Deep Learning solution to directly transform uncolored, microscopic tissue images (Grayscale) into sharp, virtually H&E-stained images.
Furthermore, the system automatically analyzes, detects, and isolates abnormal cellular regions (Tumors) on the newly stained images. This serves as a powerful assistive tool, saving significant time for pathologists during diagnosis.

## 2. Execution Workflow
The system operates automatically through the following seamless pipeline:

**Upload Image (Grayscale)** -> **Preprocessing & Slicing (Grid Slicing for large images)** -> **Virtual Staining** -> **Abnormality Classification** -> **Heatmap Extraction (Grad-CAM)** -> **Pathology Bounding/Circling** -> **Sharpness Enhancement (Upscale 4x)** -> **Display & Evaluation (Web UI)**

## 3. Model Architecture & Differences
The system utilizes a coordinated ensemble of 3 primary models:

*   **Base Model:** Uses a GAN architecture with a basic UNet Generator. The execution process simply learns to map pixels from grayscale to colored images. However, the resulting images occasionally suffer from color bleeding, lack of sharpness in microscopic cellular structures, and noise.
*   **My Model (Improved Model):** Fine-tuned and optimized from the Base Model. This model applies advanced architectural improvements and techniques during training. In practical execution, My Model produces virtually stained images with highly realistic colors, extremely sharp boundaries between the nucleus and cytoplasm, and completely eliminates the color noise present in the Base Model.
*   **Classifier Model (ResNet-18):** After the image is virtually stained, it is passed through this model. ResNet-18 analyzes and calculates the probability of the image being Normal (Healthy) or Tumor (Abnormal). Simultaneously, using Grad-CAM techniques, the model back-traces to highlight suspicious areas via a heatmap and precisely circles the cancerous cell regions.

## 4. Best Training Results (Benchmarks)
Based on the detailed evaluation logs (Log Summary) throughout the training process of each model, below are the peak performance metrics achieved:

### 4.1. Base Model (Standard Staining Model)
The best result was recorded at **gen_5**:
*   **PSNR:** 30.7844
*   **SSIM:** 0.9738
*   **LPIPS:** 0.0408

*(Insert Image Here)*
<br><br><br><br>

### 4.2. My Model (Improved Staining Model)
The best result was recorded at **gen_8**, comprehensively outperforming the Base Model across all metrics:
*   **PSNR:** 33.1607 *(Significant noise reduction)*
*   **SSIM:** 0.9834 *(Structural accuracy approaching real images)*
*   **LPIPS:** 0.0292 *(Highly realistic visual perception)*

*(Insert Image Here)*
<br><br><br><br>

### 4.3. Classifier Model (Classification & Bounding)
The best testing result on a large dataset (35,517 test images) was recorded at **Epoch 7**:
*   **Accuracy:** 92.86%
*   **Sensitivity (Tumor Detection):** 94.20%
*   **Specificity (Normal Detection):** 91.24%
*   **F1-Score:** 93.52%
*   **ROC-AUC:** 0.9784

*(Insert Image Here)*
<br><br><br><br>
