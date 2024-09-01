# Classification-Model-with-CNN

This project aims to detect stop signs on the road using deep learning techniques. Implemented using PyTorch and CV Studio, this project includes steps from data processing to model training and evaluation.

## Table of Contents

1. [About the Project](#about-the-project)
2. [Requirements](#requirements)
3. [Installation](#installation)
4. [Data Preparation](#data-preparation)
5. [Model Architecture](#model-architecture)
6. [Training the Model](#training-the-model)
7. [Evaluating the Model](#evaluating-the-model)
8. [Reporting Results](#reporting-results)
9. [Saving and Deploying the Model](#saving-and-deploying-the-model)
10. [Usage](#usage)

## About the Project

This project involves training a deep learning model based on the ResNet-18 architecture to detect stop signs on the road. The project workflow includes:

- Processing image data (resizing, normalizing, and data augmentation).
- Training a model on a labeled dataset of images.
- Validating the model to ensure its generalization to new data.
- Saving the best-performing model and preparing it for deployment.

## Requirements

To run this project, you will need the following software and libraries:

- Python (>=3.6)
- PyTorch
- NumPy
- Matplotlib
- Pillow
- Other required libraries: skillsnetwork, torchvision, ipywidgets, tqdm

## Installation

To install the required libraries, follow these steps:

1. Ensure Python and pip are installed.
2. Install the necessary Python packages:

   ```bash
   pip install -r requirements.txt
## Data Preparation
A well-labeled dataset is crucial for the success of this project. You can download the dataset using CV Studio and perform the following operations:
- 1.Resize and normalize the images.
- 2.Perform data augmentation for the training dataset.
- 3.Normalize the test dataset.

## Model Architecture

The model used in this project is ResNet-18, a commonly used pre-trained image classification model in deep learning. In this project, the final layer is reconfigured to detect stop signs.

## Training the Model

- Preparing the training and validation datasets.
- Defining the training loop and training the model on the training data.
- Monitoring loss values and accuracy rates during training.

## Evaluating the Model

After training, the model is tested on the validation dataset to evaluate its performance. The best-performing model is saved for future use.

## Reporting Results

The training results and validation metrics are reported to CV Studio, including:
- Loss values and accuracy rates obtained during training.
- Details and training parameters of the best-performing model.

## Saving and Deploying the Model

The trained model is saved as model.pt and is ready for deployment for future use. Additionally, the model's performance and hyperparameters are reported to CV Studio.

## Usage

- Load the saved model file (model.pt).
- Use the model to make predictions on test data.
- Review the prediction results to evaluate the model's performance.

## Outputs

### Stop Signs:

![Ekran görüntüsü 2024-02-28 193654](https://github.com/user-attachments/assets/ae71467c-2d46-4607-95ce-711ae06fedb8)
![Ekran görüntüsü 2024-02-28 193738](https://github.com/user-attachments/assets/5cb51baa-0f72-4825-817f-e29204cc5e40)
![Ekran görüntüsü 2024-02-28 193807](https://github.com/user-attachments/assets/e5636541-ca5c-4237-945d-19900288dd57)

### Not Stop Signs:

![Ekran görüntüsü 2024-02-28 193943](https://github.com/user-attachments/assets/9a3ddddb-e690-4af4-acd3-640d8c1aaf5a)
![Ekran görüntüsü 2024-02-28 194004](https://github.com/user-attachments/assets/9870ed44-7022-46c6-a4a5-2f387ba2ff8d)
![Ekran görüntüsü 2024-02-28 214305](https://github.com/user-attachments/assets/100b009e-1ed8-483e-9724-094c6204787d)




