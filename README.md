# Cropeye Model Version 2.0 for Drone Precision Farming
! Model Summary
## Overview
This project implements a deep learning model, Cropeye Version 2.0, designed for drone precision farming. The model utilizes Convolutional Neural Networks (CNNs) to classify plant diseases, focusing primarily on potato diseases. It is trained on the Potato Leaf Disease dataset available on Kaggle.

## Dataset
The dataset used for training the model consists of images belonging to three classes:
- Potato___Early_blight
- Potato___Late_blight
- Potato___healthy

## Dependencies
- TensorFlow
- NumPy
- Matplotlib

## Functionality
- **Import Dependencies**: Import required libraries for model development.
- **Import Dataset**: Load the dataset using TensorFlow's `image_dataset_from_directory` API.
- **Visualize Images**: Visualize sample images from the dataset.
- **Split Dataset**: Split the dataset into training, validation, and test subsets.
- **Data Augmentation**: Apply data augmentation techniques to the training dataset.
- **Model Architecture**: Define the CNN model architecture using TensorFlow's Keras API.
- **Compile the Model**: Compile the model using the Adam optimizer and sparse categorical cross-entropy loss.
- **Train the Model**: Train the model on the training dataset and validate it on the validation dataset.
- **Evaluate the Model**: Evaluate the model's performance on the test dataset.
- **Plot Accuracy and Loss Curves**: Visualize the training and validation accuracy/loss curves.
- **Inference**: Make predictions on sample images using the trained model.
- **Save the Model**: Save the trained model for future use.

## Usage
1. Clone the repository.
2. Install the required dependencies using `pip install -r requirements.txt`.
3. Run the Jupyter Notebook or Python script to train the model and make predictions.

## Model Deployment
The trained model can be deployed in various ways, such as:
- Integration into drone systems for real-time disease detection.
- Deployment as a web service using frameworks like Flask or Django.

## Credits
- Dataset: https://www.kaggle.com/datasets/muhammadardiputra/potato-leaf-disease-dataset
