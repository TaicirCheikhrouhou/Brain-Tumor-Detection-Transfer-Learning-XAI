# Brain Tumor Detection with Transfer Learning and XAI

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13+-orange)
![NumPy](https://img.shields.io/badge/NumPy-1.21+-blue)
![SHAP](https://img.shields.io/badge/SHAP-0.41+-purple)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-yellow)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green)

## Overview

This project demonstrates the application of transfer learning and fine-tuning techniques using MobileNetV2 for binary classification of brain MRI images to detect tumors. The notebook also explores Explainable Artificial Intelligence (XAI) methods including Grad-CAM, LIME, and SHAP to interpret model predictions.

## Dataset

The dataset used is the "Brain MRI Images for Brain Tumor Detection" from Kaggle (https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection). It contains MRI images categorized into two classes:

- **Yes**: Images with brain tumors
- **No**: Healthy brain images

The original dataset is imbalanced and augmented to balance the classes.

## Methodology

### Data Preparation

1. **Data Loading**: Images are loaded from the dataset directories.
2. **Data Augmentation**: Techniques like rotation, shift, zoom, brightness adjustment, and horizontal flip are applied to balance the dataset.
3. **Train-Validation-Test Split**: 70% training, 10% validation, 20% test.

### Model Architecture

- **Base Model**: MobileNetV2 pre-trained on ImageNet.
- **Transfer Learning**: Freeze all base layers, add custom classification head (GlobalAveragePooling2D, Dense 256, Dropout 0.5, Dense 1 with sigmoid).
- **Fine-Tuning**: Unfreeze the last 10 layers of the base model for further training.

### Training

- **Optimizer**: Adam with learning rates 1e-4 (TL) and 1e-5 (FT).
- **Loss**: Binary Crossentropy.
- **Metrics**: Accuracy, Precision, Recall, AUC.
- **Callbacks**: Early Stopping and Model Checkpointing.

### Explainable AI

- **Grad-CAM**: Visualizes the regions of the image that contribute most to the prediction.
- **LIME**: Provides local explanations by perturbing the image and observing changes in predictions.
- **SHAP**: Uses gradient-based explanations to show feature importance.

## Results

The models achieve high accuracy on the validation set. Transfer learning provides a good baseline, while fine-tuning improves performance by adapting the pre-trained features to the specific task.

## Usage

### Prerequisites

- Python 3.8+
- TensorFlow 2.x
- Required libraries: numpy, pandas, matplotlib, scikit-learn, opencv-python, lime, shap, etc.

Install dependencies:

```bash
pip install -r requirements.txt
```

### Running the Notebook

1. Download the dataset from Kaggle and place it in the appropriate directory.
2. Update the data paths in the notebook if necessary.
3. Run the cells sequentially.

### Key Variables to Configure

- `data_dir`: Path to the dataset directory.
- `save_dir`: Directory to save models and checkpoints.

## Project Structure

- `TransFerLearning+Finetuning+XAI.ipynb`: Main notebook.
- `requirements.txt`: Python dependencies.
- `data/`: Dataset directory (not included, download from Kaggle).

## Future Improvements

- Hyperparameter tuning.
- Deployment as a web application.

## License
This project is for educational purposes. 

## Author

**Taicir Cheikhrouhou**  
- GitHub: [TaicirCheikhrouhou](https://github.com/TaicirCheikhrouhou)
- LinkedIn: [https://www.linkedin.com/in/taicir-cheikhrouhou/](https://www.linkedin.com/in/cheikhrouhou-taicir/)
