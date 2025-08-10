# Digit Recognizer – Train on MNIST, Predict from Any Image
This Python program trains a Convolutional Neural Network (CNN) on the MNIST handwritten digit dataset and allows you to predict digits from your own images.
It automatically preprocesses arbitrary images into MNIST-like format (28×28 grayscale, centered) so you can test it with photos, scans, or screenshots of handwritten numbers.

# Features-
1.Full MNIST Training – Uses TensorFlow/Keras to train a CNN from scratch or load a saved model (mnist_model.h5).
2.Image Preprocessing – Converts any input image into MNIST style:
 a.Converts to grayscale
 b.Inverts if background is light
 c.Crops the digit area
 d.Resizes to 28×28 pixels and centers it
 e.Normalizes pixel values to the range [-1, 1]

3.Data Augmentation – Adds random rotations, shifts, shears, and zooms to improve model generalization.
4.Tkinter File Picker – Easily select one or more images from your computer.
5.Side-by-Side Prediction View – Displays original and preprocessed image with predicted label.

# File Stucture-
digit_recognizer.py   # Main program
mnist_model.h5        # Trained model (created after first run)
README.md             # This file

# Requirements -
Install dependencies using:
pip install tensorflow numpy opencv-python pillow matplotlib

# How to Run -
1.Run the script:
python digit_recognizer.py
2.Select your images from the file dialog window-
Supported formats: .png, .jpg, .jpeg, .bmp
3.iew the prediction – The program will show:
Original image
MNIST-style processed image
Predicted digit in console and image title

# Model Architecture-
1.Input Layer: 28×28×1 grayscale
2.Conv2D: 32 filters, 3×3, ReLU, MaxPooling, Dropout
3.Conv2D: 64 filters, 3×3, ReLU, MaxPooling, Dropout
4.Flatten
5.Dense: 128 neurons, ReLU, Dropout
6.Dense: 10 neurons (softmax output)

# Training Details-
1.Dataset: MNIST (60,000 train / 10,000 test images)
2.Optimizer: Adam
3.Loss Function: Sparse Categorical Crossentropy
4.Epochs: 8
5.Batch Size: 128
6.Augmentation: Rotation ±15°, shifts, shear, zoom
