# Digit Recognizer - Train on MNIST, Predict from Any Image
# This script trains (or loads) a CNN model on MNIST digits and
# lets you pick your own image(s) from the file dialog to predict.
# It also preprocesses any arbitrary image into "MNIST style"
# (white digit, black background, centered, 28×28 px, normalized)
# so the CNN can correctly interpret it.

#Imports-----
import os                # For file system operations (checking if model file exists, saving/loading)
import cv2               # OpenCV for image processing (thresholding, resizing, finding bounding boxes)
import numpy as np       # NumPy for fast array handling and numerical operations
import tensorflow as tf  # TensorFlow/Keras for building, training, and using neural networks
from PIL import Image    # Pillow for opening and converting images between formats
import matplotlib.pyplot as plt  # Matplotlib for displaying images and prediction results

# Keras-specific modules for building and training CNN
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Config ---
MODEL_FILE = "mnist_model.h5"  # Name of the file to save/load the trained CNN model

# Helper: Normalize image like MNIST dataset ---
# MNIST pixels are normalized from uint8 [0–255] to [-1, 1].
# This helps the neural network converge faster during training and improves accuracy.
normalize = lambda x: (x.astype("float32") / 255.0 - 0.5) / 0.5


# Preprocessing function - Convert ANY image into MNIST format

def preprocess(img_path):
    """
    Load an image from disk, clean it up, center it,
    resize it to 28×28 pixels, and normalize to [-1, 1].
    This ensures the input matches the format used during training.
    """
    # 1. Open image and convert to grayscale ('L' mode = 8-bit pixels, 0=black, 255=white)
    img = Image.open(img_path).convert("L")

    # 2. Convert Pillow image to NumPy array for OpenCV operations
    arr = np.array(img)

    # 3. If background is lighter than foreground, invert it
    #    MNIST format expects: digit in white, background in black.
    if arr.mean() > 127:
        arr = 255 - arr

    # 4. Apply binary threshold:
    #    Pixels > 10 → white (255), else black (0). Removes noise.
    _, thresh = cv2.threshold(arr, 10, 255, cv2.THRESH_BINARY)

    # 5. Find coordinates of all white pixels (digit region)
    coords = cv2.findNonZero(thresh)

    # 6. Get smallest bounding rectangle that encloses the digit
    x, y, w, h = cv2.boundingRect(coords)

    # 7. Crop to this bounding box to isolate the digit
    digit = arr[y:y+h, x:x+w]

    # 8. Resize the digit to 20×20 (MNIST digits are roughly that size inside a 28×28 image)
    digit = cv2.resize(digit, (20, 20), interpolation=cv2.INTER_AREA)

    # 9. Create an empty 28×28 black canvas
    canvas = np.zeros((28, 28), dtype=np.uint8)

    # 10. Compute offsets so that the digit is centered in the canvas
    x_off = (28 - digit.shape[1]) // 2
    y_off = (28 - digit.shape[0]) // 2

    # 11. Place the digit into the center of the canvas
    canvas[y_off:y_off+digit.shape[0], x_off:x_off+digit.shape[1]] = digit

    # 12. Normalize the image to [-1, 1] and add channel dimension (needed for CNN)
    return normalize(canvas).reshape(28, 28, 1)

# Model building function - Small CNN for digit classification

def build_model():
    """
    Build a lightweight CNN that works well for MNIST digit recognition.
    CNN learns spatial features automatically, making it ideal for images.
    """
    return models.Sequential([
        layers.Input((28, 28, 1)),  # Input layer (28×28 grayscale image)

        # First convolutional block:
        # Conv → extract low-level features like edges
        # MaxPooling → reduce size and keep important info
        layers.Conv2D(32, 3, activation='relu', padding='same'),
        layers.MaxPooling2D(),
        layers.Dropout(0.25),  # Randomly drop 25% neurons to reduce overfitting

        # Second convolutional block:
        layers.Conv2D(64, 3, activation='relu', padding='same'),
        layers.MaxPooling2D(),
        layers.Dropout(0.25),

        # Flatten → convert 2D feature maps to 1D vector
        layers.Flatten(),

        # Dense layer → fully connected network for combining features
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),  # More dropout here for regularization

        # Output layer: 10 neurons for digits 0–9 (softmax = probability distribution)
        layers.Dense(10, activation='softmax')
    ])

# Training function - MNIST dataset + augmentation

def train_model():
    """
    Load MNIST data, normalize it, apply slight random transformations (augmentation)
    to make the model more robust, and train the CNN.
    Saves the best model version to disk.
    """
    # 1. Load MNIST dataset (already split into train/test sets)
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # 2. Normalize and add channel dimension for CNN (shape: N×28×28×1)
    x_train = normalize(x_train[..., None])
    x_test = normalize(x_test[..., None])

    # 3. Data augmentation: makes the model generalize better to varied handwriting
    aug = ImageDataGenerator(
        rotation_range=15,      # Rotate randomly ±15 degrees
        width_shift_range=0.1,  # Shift horizontally up to ±10%
        height_shift_range=0.1, # Shift vertically up to ±10%
        shear_range=10,         # Slant the image
        zoom_range=0.1          # Random zoom in/out up to ±10%
    )

    # 4. Build and compile the model
    model = build_model()
    model.compile(
        optimizer='adam',  # Adam optimizer is efficient for this type of task
        loss='sparse_categorical_crossentropy',  # Works with integer labels
        metrics=['accuracy']
    )

    # 5. Train with augmented data
    model.fit(
        aug.flow(x_train, y_train, batch_size=128),  # Feed batches with augmentation
        validation_data=(x_test, y_test),            # Monitor test accuracy during training
        epochs=8,                                    # 8 passes over the dataset
        callbacks=[
            callbacks.ModelCheckpoint(MODEL_FILE, save_best_only=True)  # Save best version
        ]
    )

    return model

# Prediction function - Process image → Model → Show result

def predict(img_path, model):
    """
    Given an image path and a trained model:
    1. Preprocess it into MNIST format
    2. Get prediction probabilities
    3. Display both original and processed images with the result
    """
    # 1. Preprocess and add batch dimension for CNN (shape: 1×28×28×1)
    x = np.expand_dims(preprocess(img_path), 0)

    # 2. Predict class probabilities, take the class with highest probability
    pred = np.argmax(model.predict(x, verbose=0))

    # 3. Plot original grayscale image
    plt.subplot(1, 2, 1)
    plt.imshow(Image.open(img_path).convert("L"), cmap='gray')
    plt.axis('off')
    plt.title("Original")

    # 4. Plot processed (MNIST-style) image with prediction label
    plt.subplot(1, 2, 2)
    plt.imshow(x[0].squeeze(), cmap='gray')
    plt.axis('off')
    plt.title(f"Pred: {pred}")

    # 5. Show the side-by-side comparison
    plt.show()

    return pred


# Main script - Load existing model OR train a new one

if __name__ == "__main__":
    # Suppress TensorFlow info/warning messages for cleaner output
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    # Load trained model if file exists; else train from scratch
    if os.path.exists(MODEL_FILE):
        model = tf.keras.models.load_model(MODEL_FILE)
    else:
        model = train_model()

    # File picker dialog (Tkinter) to select image(s) for prediction
    from tkinter import Tk, filedialog
    Tk().withdraw()  # Hide the Tkinter main window
    files = filedialog.askopenfilenames(
        title="Select images to recognize",
        filetypes=[("Images", "*.png;*.jpg;*.jpeg;*.bmp")]
    )

    # Predict and display results for each selected file
    for f in files:
        print(f, "→", predict(f, model))
