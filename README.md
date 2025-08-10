🖋 Digit Recognizer – Train on MNIST, Predict from Any Image
This Python program trains a Convolutional Neural Network (CNN) on the MNIST handwritten digit dataset and allows you to predict digits from your own images.

It automatically preprocesses arbitrary images into MNIST-like format (28×28 grayscale, centered) so you can test it with photos, scans, or screenshots of handwritten numbers.

📌 Features
Full MNIST Training – Uses TensorFlow/Keras to train a CNN from scratch or load a saved model (mnist_model.h5).

Image Preprocessing – Converts any input image into MNIST style:

Converts to grayscale

Inverts if background is light

Crops the digit area

Resizes to 28×28 pixels and centers it

Normalizes pixel values to the range [-1, 1]

Data Augmentation – Adds random rotations, shifts, shears, and zooms to improve model generalization.

Tkinter File Picker – Easily select one or more images from your computer.

Side-by-Side Prediction View – Displays original and preprocessed image with predicted label.

📂 File Structure
bash
Copy
Edit
digit_recognizer.py   # Main program
mnist_model.h5        # Trained model (created after first run)
README.md             # This file
🔧 Requirements
Install dependencies using:

bash
Copy
Edit
pip install tensorflow numpy opencv-python pillow matplotlib
🚀 How to Run
Clone the repository:

bash
Copy
Edit
git clone https://github.com/yourusername/digit-recognizer.git
cd digit-recognizer
Run the script:

bash
Copy
Edit
python digit_recognizer.py
Select your images from the file dialog window.
Supported formats: .png, .jpg, .jpeg, .bmp

View the prediction – The program will show:

Original image

MNIST-style processed image

Predicted digit in console and image title

🧠 Model Architecture
Input Layer: 28×28×1 grayscale

Conv2D: 32 filters, 3×3, ReLU, MaxPooling, Dropout

Conv2D: 64 filters, 3×3, ReLU, MaxPooling, Dropout

Flatten

Dense: 128 neurons, ReLU, Dropout

Dense: 10 neurons (softmax output)

📊 Training Details
Dataset: MNIST (60,000 train / 10,000 test images)

Optimizer: Adam

Loss Function: Sparse Categorical Crossentropy

Epochs: 8

Batch Size: 128

Augmentation: Rotation ±15°, shifts, shear, zoom
