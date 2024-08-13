# Image Classifier

This project is part of Udacity's AI Programming with Python Nanodegree program. It involves developing an image classifier using PyTorch and converting it into a command-line application. The classifier is designed to recognize images and classify them into predefined categories.

## Project Overview

The image classifier uses a Convolutional Neural Network (CNN) built with PyTorch. The model is trained to classify images into various categories, and the application allows users to interact with the classifier through a command-line interface.

## Key Features

- **Image Classification**: Classify images into predefined categories using a trained CNN model.
- **Command-Line Interface**: A CLI that allows users to input an image and receive the classification result.
- **Model Training**: Train the model on a dataset of images.
- **Model Evaluation**: Evaluate the performance of the model on test data.

## Project Files

- `Image_Classifier_Project.ipynb`: Jupyter Notebook with code for developing and testing the image classifier.
- `predict.py`: Script for classifying images using the trained model.
- `train.py`: Script for training the model.
- `workspace-utils.py`: Contains utility functions for the project.
- `cat_to_name.json`: File mapping category labels to human-readable names.
- `.gitignore`: Specifies files and directories to be ignored by Git.
- `LICENSE`: License file for the project.
- `README.md`: This file.

## Getting Started

To get started with this project, follow these steps:

### 1. **Clone the Repository**

Clone the repository to your local machine:

`git clone https://github.com/your-username/image-classifier.git`

`cd image-classifier`

### 2. **Set Up the Environment**

Create a virtual environment and install the required dependencies:

`python -m venv venv`

`source venv/bin/activate +`

On Windows use `venv\Scripts\activate`

`pip install -r requirements.txt`

### 3. **Prepare the Data**

You need to have a dataset of images for training and testing the model. Ensure your dataset is organized and available in the appropriate directory. Update the paths in the configuration files if necessary.

### 4. **Train the Model**

To train the model, run the training script. This script will train the model on the provided dataset:

`python train.py --data_dir path_to_data --save_dir path_to_save_model`

--data_dir: Path to the directory containing the dataset.

--save_dir: Path to the directory where the trained model will be saved.

### 5. **Classify Images**

Once the model is trained, you can classify images using the CLI. Run the following command:

`python classify.py --image path_to_image --model path_to_model --top_k 5 --category_names path_to_category_names`

--image: Path to the image file to be classified.

--model: Path to the trained model.

--top_k: Number of top classes to return (default is 5).

--category_names: Path to a file mapping category labels to human-readable names.

### 6. **Evaluate the Model**

To evaluate the model's performance, you can use the evaluation script:

`python evaluate.py --data_dir path_to_data --model path_to_model`

--data_dir: Path to the directory containing the test dataset.

--model: Path to the trained model.

## Requirements
- Python 3.x
- PyTorch
- Numpy
- Matplotlib
- Pillow

## Acknowledgements
- Udacity's AI Programming with Python Nanodegree
- PyTorch Documentation
- ImageNet Dataset



