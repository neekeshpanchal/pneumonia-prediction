import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np

# Load the pre-trained model from the saved file
model = load_model('pneumonia_detection_model.h5')

def open_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        img = Image.open(file_path)

        # Preprocess the image for prediction
        processed_img = preprocess_image(img)

        # Convert the processed image to a batch with a single sample
        batched_img = processed_img.reshape((1,) + processed_img.shape)

        # Use the model to predict the class probabilities (1: Pneumonia, 0: Normal)
        prediction = model.predict(batched_img)

        # Get the predicted class (0 or 1)
        predicted_class = int(prediction[0][0])

        # Define the class labels
        class_labels = {0: "Normal", 1: "Pneumonia"}

        # Display the result
        result_label.config(text="Prediction: " + class_labels[predicted_class])

        # Resize the image and display it on the GUI
        img = img.resize((300, 300))
        img = ImageTk.PhotoImage(img)
        panel.configure(image=img)
        panel.image = img

def preprocess_image(image):
    # Preprocess the image before passing it to the model for prediction
    # You need to resize the image, convert it to an array, and normalize the pixel values
    # Resize the image to the required input size of the model
    image = image.resize((150, 150))

    # Convert the PIL image to a NumPy array
    image_array = img_to_array(image)

    # Handle images with 4 channels (e.g., RGBA)
    if image_array.shape[-1] == 4:
        image_array = image_array[:, :, :3]  # Discard the alpha channel

    # Ensure the image has 3 channels (RGB)
    if image_array.shape[-1] == 1:
        image_array = np.repeat(image_array, 3, axis=-1)
    elif image_array.shape[-1] != 3:
        raise ValueError("Image has an unexpected number of channels. Expected 1 or 3 channels, found {} channels.".format(image_array.shape[-1]))

    # Preprocess the image for the model
    image_array = preprocess_input(image_array)

    # Return the preprocessed image
    return image_array

# Create GUI window
window = tk.Tk()
window.title("Pneumonia Detection")

btn = tk.Button(window, text="Upload X-ray Image", command=open_image)
btn.pack(padx=10, pady=5)

panel = tk.Label(window)
panel.pack(padx=10, pady=5)

result_label = tk.Label(window, text="", font=("Helvetica", 16))
result_label.pack(padx=10, pady=5)

window.mainloop()
