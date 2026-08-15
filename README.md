# AI-Powered Virtual Staining & Abnormality Detection

## 1. Project Overview
This project applies Artificial Intelligence (AI) in Healthcare, specifically focusing on **Virtual Staining** and **Pathology Image Analysis**.

In clinical practice, the traditional chemical staining process of tissue slides using H&E (Hematoxylin & Eosin) is time-consuming, labor-intensive, and costly. This project provides a Deep Learning solution to directly transform uncolored microscopic tissue images (Grayscale) into sharp, high-fidelity, virtually H&E-stained images. 

Furthermore, the system automatically analyzes, detects, and isolates abnormal cellular regions (Tumors) on the generated images, serving as an assistive tool to accelerate diagnostic workflows for pathologists.

### Target Tissue Types
The model was trained and evaluated across **three distinct human tissue types**:
* **Lymph Node**
* **Prostate**
* **Digestive System**

### Training Environment & Constraints
To demonstrate efficiency and lightweight deployment capabilities, the entire training pipeline and experimental workflows were conducted directly on a standard resource-constrained environment:
* **Hardware:** Personal Laptop (CPU-only, No GPU)
* **RAM:** 8 GB

## 2. Execution Workflow
The system operates automatically through the following seamless pipeline:

**Upload Image (Grayscale)** -> **Preprocessing & Slicing (Grid Slicing for large images)** -> **Virtual Staining** -> **Abnormality Classification** -> **Heatmap Extraction (Grad-CAM)** -> **Pathology Bounding/Circling** -> **Sharpness Enhancement (Upscale 4x)** -> **Display & Evaluation (Web UI)**

## 3. Model Architecture & Differences
The system utilizes a coordinated ensemble of 3 primary models:

*   **Base Model:** Uses a GAN architecture with a basic UNet Generator. The execution process simply learns to map pixels from grayscale to colored images. However, the resulting images occasionally suffer from color bleeding, lack of sharpness in microscopic cellular structures, and noise.
*   **My Model (Improved Model):** Fine-tuned and optimized from the Base Model. This model applies advanced architectural improvements and techniques during training. In practical execution, My Model produces virtually stained images with highly realistic colors, extremely sharp boundaries between the nucleus and cytoplasm, and completely eliminates the color noise present in the Base Model.
*   **Classifier Model (ResNet-18):** After the image is virtually stained, it is passed through this model. ResNet-18 analyzes and calculates the probability of the image being Normal (Healthy) or Tumor (Abnormal). Simultaneously, using Grad-CAM techniques, the model back-traces to highlight suspicious areas via a heatmap and precisely circles the cancerous cell regions.

## 4. Training Results:
### 4.1. Base Model (Standard Staining Model)
The best result was recorded at **gen_5**:
*   **PSNR:** 30.7844
*   **SSIM:** 0.9738
*   **LPIPS:** 0.0408


<table>
  <tr>
    <td align="center">
      <h3>Normal</h3>
      <img src="https://github.com/user-attachments/assets/6e8f6be5-1a07-46c0-9c23-698206332b73" alt="Grid_basemodel_normal" width="100%" />
    </td>
    <td align="center">
      <h3>Tumor</h3>
      <img src="https://github.com/user-attachments/assets/690e48f9-0c6e-411c-9f3d-8701f4fbc2fa" alt="Grid_basemodel_tumor" width="100%" />
    </td>
  </tr>
</table>


### 4.2. My Model (Improved Staining Model)
The best result was recorded at **gen_8**:
*   **PSNR:** 33.1607
*   **SSIM:** 0.9834 
*   **LPIPS:** 0.0292


<table>
  <tr>
    <td align="center">
      <h3>Normal</h3>
      <img src="https://github.com/user-attachments/assets/6fc8dbce-e84a-4fe9-b9f7-d4cea756484a" alt="Grid_mymodel_normal" width="100%" />
    </td>
    <td align="center">
      <h3>Tumor</h3>
      <img src="https://github.com/user-attachments/assets/29c7a8b0-7bf8-475b-afcb-5b99b6d487eb" alt="Grid_mymodel_tumor" width="100%" />
    </td>
  </tr>
</table>


### 4.3. Classifier Model (Classification & Bounding)
The best testing result on a large dataset (35,517 test images) was recorded at **Epoch 7**:
*   **Accuracy:** 92.86%
*   **Sensitivity (Tumor Detection):** 94.20%
*   **Specificity (Normal Detection):** 91.24%
*   **F1-Score:** 93.52%
*   **ROC-AUC:** 0.9784

## 5. Interface
### 5.1 Flask
<table>
  <tr>
    <td align="center">
      <h3>Interface</h3>
      <img width="1890" height="889" alt="Ảnh chụp màn hình 2026-08-15 162428" src="https://github.com/user-attachments/assets/8f480bd4-6aec-4832-a96b-a512141ab160" />
    </td>
    <td align="center">
      <h3>Result</h3>
      <img width="1850" height="907" alt="image" src="https://github.com/user-attachments/assets/3274c530-c29f-4d46-bf57-9cfc47c61b19" />
    </td>
  </tr>
</table>

### 5.2 Streamlit
<table>
  <tr>
    <td align="center">
      <h3>Interface</h3>
      <img width="1883" height="795" alt="image" src="https://github.com/user-attachments/assets/9bd93f76-f963-4688-90cc-70c6ce96eeae" />
    </td>
    <td align="center">
      <h3>Result</h3>
      <img width="1815" height="861" alt="image" src="https://github.com/user-attachments/assets/e105c784-87f3-49d1-940d-139c217e5e20" />
    </td>
  </tr>
</table>

