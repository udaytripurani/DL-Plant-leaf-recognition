# DL-Plant-leaf-recognition  Deep Learning


## Introduction
This project explores deep learning techniques for classifying plant leaves into different types using a dataset of leaf images.

## Dataset Description
The dataset consists of images of plant leaves belonging to 14 different types of plants, with labels indicating healthy leaves and leaves with various types of spots. We have selected 3 types of leaves for this project.

## Data Preprocessing
### Background Removal
We applied techniques such as thresholding and edge detection to separate the leaf region from the background.
### Rescaling
The preprocessed images were resized to 224x224 pixels to maintain uniformity.
### Data Augmentation
Various transformations were applied, including rotation, flipping, scaling, translation, shearing, and adding noise to augment the dataset.

## Model Implementations
1. Binary Classification with Sequential Model
2. Binary Classification with Various Optimization Techniques
3. Comparison of Models
4. Multi-class Classification with Sequential Model
5. Binary Classification with Mini-batch Evaluations
6. Convolutional Neural Network (CNN)
7. CNN with Regularization
8. VGG
9. Recurrent Neural Network (RNN)
10. CNN + LSTM
11. Autoencoder
12. Denoise Autoencoder
13. Autoencoder + ResNet
14. Variational Autoencoder (VAE)

## Flask Application
We have provided a Flask application in the 'flask' folder to use all these models for input and output.

## Results and Discussion
Detailed results of each model's performance are provided in the project report.

## Future Work
Possible future improvements include exploring advanced architectures and incorporating more sophisticated regularization techniques.


