import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
from keras.preprocessing import image
import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from custommodel import build_custom_model


# Function to preprocess an input image
def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array


# Function to make a prediction
def predict_fruit(image_path):
    img_array = preprocess_image(image_path)
    prediction = loaded_model.predict(img_array)
    predicted_class_index = np.argmax(prediction)
    predicted_class = classes[predicted_class_index]
    confidence = prediction[0][predicted_class_index]
    return predicted_class, confidence


# Function to handle the button click event
def browse_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])
    if file_path:
        # Perform object detection on the selected image
        predicted_class, confidence = predict_fruit(file_path)

        # Display the result
        img = Image.open(file_path)
        img = img.resize((128, 128), Image.ANTIALIAS)
        img = ImageTk.PhotoImage(img)

        result_label.config(text=f"Predicted: {predicted_class} ({confidence:.2f})")
        image_label.config(image=img)
        image_label.image = img


# Build and train the custom model
test_dir, batch_size = build_custom_model()

# Load the trained model
loaded_model = tf.keras.models.load_model("fruit_model.keras")

# Define the classes
classes = ['apple', 'orange', 'banana', 'grape', 'watermelon']

# Create the main window
root = tk.Tk()
root.title("Fruit Object Detection")

# Create GUI components
browse_button = tk.Button(root, text="Browse Image", command=browse_image)
browse_button.pack(pady=10)

result_label = tk.Label(root, text="")
result_label.pack(pady=10)

image_label = tk.Label(root)
image_label.pack()

# Start the Tkinter event loop
root.mainloop()
