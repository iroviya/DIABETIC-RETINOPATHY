# Diabetic Retinopathy Detection with Deep Learning
## Description
Revolutionizing Diabetic Retinopathy detection with a two-step deep learning model, ensuring early and accurate identification. Join us in the journey to prevent vision loss through innovative medical image analysis.

## Overview
This repository contains the implementation of a two-step deep learning model for the early detection and severity classification of diabetic retinopathy. The project leverages InceptionResNetV2 and DenseNet121 architectures to analyze retinal fundus images, providing an automated and efficient solution for healthcare.

## Key Features
- Two-step detection: First, identify the presence of diabetic retinopathy. Second, classify severity.
- Deep learning power: Utilizes InceptionResNetV2 and DenseNet121 for robust and accurate predictions.
- Dataset balancing: Addresses class imbalance challenges through strategic data augmentation.
## Dataset
The dataset used for this project is the [APTOS 2019 Blindness Detection dataset] (https://www.kaggle.com/c/aptos2019-blindness-detection) from Kaggle. It consists of retinal fundus images with labels for severity levels of diabetic retinopathy. To replicate and experiment with the project, download the dataset from the provided Kaggle link and place it in the `data/` directory.

## How to Use
1. Clone the repository: `git clone https://github.com/Mohithv13/DiabeticRetinopathyDetection.git`
2. Install dependencies: `pip install -r requirements.txt`
3. Run the application: `streamlit run app.py`

## Project Structure
- `Final_main/`: Contains pre-trained models for InceptionResNetV2 and DenseNet121.
- `streamlit run app.py/`: Main script to run the application and predict new images.
