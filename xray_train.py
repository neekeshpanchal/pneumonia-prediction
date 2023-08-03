import os
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input


data_dir = r'C:\Users\Neeku\Downloads\Pneumonia Prediction\images'
train_dir = os.path.join(data_dir, 'train')
test_dir = os.path.join(data_dir, 'test')
val_dir = os.path.join(data_dir, 'val')

# Image preprocessing and data augmentation
batch_size = 32
image_size = (150, 150)

train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(train_dir,
                                                    target_size=image_size,
                                                    batch_size=batch_size,
                                                    class_mode='binary')

test_generator = test_datagen.flow_from_directory(test_dir,
                                                  target_size=image_size,
                                                  batch_size=batch_size,
                                                  class_mode='binary')

val_generator = test_datagen.flow_from_directory(val_dir,
                                                 target_size=image_size,
                                                 batch_size=batch_size,
                                                 class_mode='binary')

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary()

epochs = 10
history = model.fit(train_generator,
                    steps_per_epoch=len(train_generator),
                    epochs=epochs,
                    validation_data=val_generator,
                    validation_steps=len(val_generator))
                    
model = load_model('pneumonia_detection_model.h5')
model.save('pneumonia_detection_model_new.h5')

def open_image():
    file_path = filedialog.askopenfilename()
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
