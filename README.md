Plant Disease Detection using Deep Learning
Project Overview:
This project implements a deep learning-based Convolutional Neural Network (CNN) for plant disease classification using leaf images. It leverages transfer learning (VGG16) and image preprocessing techniques to detect plant diseases from the PlantVillage dataset. The model achieves high accuracy and can be used for real-world agricultural disease detection.
Tech Stack & Tools:
Programming Language: Python
ML/DL Frameworks: TensorFlow, Keras, OpenCV
Preprocessing: ImageDataGenerator (data augmentation), Label Encoding
Model: CNN (with VGG16-based Transfer Learning)
Evaluation Metrics: Accuracy, Precision, Recall, F1-score, Confusion Matrix 
Dataset: PlantVillage Dataset
Contains 39 plant species with diseased & healthy leaf images.
Images are resized to 256×256 and normalized using OpenCV.
 Features & Implementation:
 Implemented Convolutional Neural Network (CNN) using TensorFlow/Keras for classification.
 Applied data augmentation (rotation, zoom, horizontal flip, normalization) to improve model generalization.
 Used VGG16 transfer learning to enhance accuracy (~97%).
 Evaluated the model using accuracy, precision, recall, F1-score, and confusion matrix.
 Model Performance:
Accuracy: ~97%
Best model: Transfer learning using VGG16
Evaluation Metrics:
Precision
Recall
F1-score
Confusion Matrix
