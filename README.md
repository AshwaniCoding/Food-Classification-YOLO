# Food Classification Using YOLOv8

## Introduction

The Food Classification YOLO project leverages the power of the YOLO (You Only Look Once) object detection algorithm to classify and detect various food items in images. This project aims to streamline the process of identifying food types, making it useful for applications in dietary tracking, food recognition apps, and automated kitchen systems.

YOLO's real-time detection capability allows for rapid and accurate classification of food items, even in complex environments. The model has been trained on a diverse dataset of food images to recognize and categorize different types of food, ensuring high accuracy and reliability.

This repository contains the code, model weights, and instructions needed to train, test, and deploy the Food Classification YOLO model.

## Table of Contents
- [Installation](#installation)
- [Dataset](#dataset)
- [Usage](#usage)
- [Features](#features)
- [Configuration](#configuration)
- [Acknowledgments](#acknowledgments)
- [Contact](#contact)

## Installation

To get started with this project, follow these steps:

1. Clone the repository:

        git clone https://github.com/yourusername/food-classification-yolo.git
        cd Food-Classification-YOLO


2. Install Dependencies:
   Make sure you have Python installed (preferably Python 3.8 or higher). Install the required dependencies by running:

        pip install ultralytics

    This will install the ultralytics library along with its dependencies, including YOLOv8.

3. Download or Prepare Your Dataset:
   Ensure you have your dataset ready, with images labeled according to the YOLO format. Modify data.yaml to match your dataset configuration.

## Dataset

 The dataset used in this project consists of images categorized into different types of Indian food items. The food items included are:

 - Apple Pie
 - Idli
 - Chapati
 - Jalebi
 - Butter Naan
 - Dal Makhani
 - Dhokla
 - Kadhai Paneer

 ### Project Folder Structure

  The dataset is organized into the following folder structure:
  ```
Food Classification YOLO/
├── YOLO/
│   ├── test/
│   │   ├── images/
│   │   │   ├── image1.jpg
│   │   │   ├── image2.jpg
│   │   │   └── ...
│   │   └── labels/
│   │       ├── image1.txt
│   │       ├── image2.txt
│   │       └── ...
│   ├── train/
│   │   ├── images/
│   │   │   ├── image1.jpg
│   │   │   ├── image2.jpg
│   │   │   └── ...
│   │   └── labels/
│   │       ├── image1.txt
│   │       ├── image2.txt
│   │       └── ...
│   └── valid/
│       ├── images/
│       │   ├── image1.jpg
│       │   ├── image2.jpg
│       │   └── ...
│       └── labels/
│           ├── image1.txt
│           ├── image2.txt
│           └── ...
├── data.yaml
├── yolo.ipynb
└── yolov8m.pt
```

 **Description**

- **Food Classification YOLO/**: The root folder of the project, containing all necessary files and folders for the YOLO model.
  
- **YOLO/**: Contains the dataset organized into different subsets:
  - **test/**: Includes images and corresponding label files used for testing the model.
    - **images/**: Contains food images for testing.
    - **labels/**: Contains label files for the test images.
  - **train/**: Includes images and label files used for training the model.
    - **images/**: Contains food images for training.
    - **labels/**: Contains label files for the training images.
  - **valid/**: Includes images and label files used for validation.
    - **images/**: Contains food images for validation.
    - **labels/**: Contains label files for the validation images.

    [**Download Dataset**](https://drive.google.com/drive/folders/1xXFvhGCkcrFYWya74-50FxtzEnnSeq2M?usp=drive_link)

- **data.yaml**: Configuration file for the dataset, defining the classes and paths for training, validation, and test data.

- **yolo.ipynb**: Jupyter Notebook containing the code for training and testing the YOLO model.

- **yolov8m.pt**: Pre-trained YOLOv8 model file used for initializing the model ([Download](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8m.pt)).

This structure ensures that all components are well-organized for efficient training, validation, and testing of the food classification model using YOLO.
  
        

## Usage

  To train the YOLO model for food classification, use the following command:

    from ultralytics import YOLO
    from IPython.display import display, Image

    # Train the YOLO model
    !yolo task=detect mode=train model=yolov8m.pt data=data.yaml epochs=25 imgsz=224 plots=True


  Once training is complete, you can use the trained model to predict and classify food items in images:

    # Predict using the trained model
    !yolo predict model="runs/detect/train/weights/best.pt" source='YOLO/test/images/ch40.jpg'


## Features

  **Efficient Training:** Utilizes YOLOv8 for fast and accurate object detection and classification.  
  **Customizable:** Easily adaptable to different datasets by modifying the configuration files.  
  **Visualization:** Includes plot generation during training to visualize model performance.  

## Configuration

  **Model:** You can change the model by altering the model parameter in the training command. For example, replace yolov8m.pt with yolov8s.pt for a smaller model.  
  **Data:** Ensure the data.yaml file is correctly configured for your dataset, specifying the paths to training and validation data.  
  **Epochs:** Adjust the epochs parameter to control the number of training iterations.  

## Acknowledgements

  This project leverages the [Ultralytics YOLO](https://ultralytics.com/) library, which provides an easy-to-use implementation of the YOLOv8 model. Special thanks to the open-source    community for providing tools and datasets that make projects like this possible.

## Contact

For any inquiries or further information, feel free to reach out:

**Name:** Ashwani Kumar Dwivedi  
**Email:** ashwanidwivedi7898@gmail.com  
**GitHub:** [AshwaniCoding](https://github.com/AshwaniCoding)  
