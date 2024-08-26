# Food Classification Using YOLOv8

## Introduction

The Food Classification YOLO project leverages the power of the YOLO (You Only Look Once) object detection algorithm to classify and detect various food items in images. This project aims to streamline the process of identifying food types, making it useful for applications in dietary tracking, food recognition apps, and automated kitchen systems.

YOLO's real-time detection capability allows for rapid and accurate classification of food items, even in complex environments. The model has been trained on a diverse dataset of food images to recognize and categorize different types of food, ensuring high accuracy and reliability.

This repository contains the code, model weights, and instructions needed to train, test, and deploy the Food Classification YOLO model.

## Table of Contents
- [Installation](#installation)
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

## Usage

  To train the YOLO model for food classification, use the following command:

    from ultralytics import YOLO
    from IPython.display import display, Image

    # Train the YOLO model
    !yolo task=detect mode=train model=yolov8m.pt data=data.yaml epochs=25 imgsz=224 plots=True


  Once training is complete, you can use the trained model to predict and classify food items in images:

    # Predict using the trained model
    !yolo predict model=runs/detect/train/weights/best.pt source='YOLO/test/images/ch40.jpg'


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
