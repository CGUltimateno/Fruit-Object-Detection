# run.py
import tensorflow as tf
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from custommodel import build_custom_model

# Build and train the custom model
test_dir, batch_size = build_custom_model()

# Load the trained model
loaded_model = tf.keras.models.load_model("model_custom.h5")

# Define the classes
classes = ['apple', 'orange', 'banana', 'grape', 'watermelon']


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


# Load test data
test_datagen = ImageDataGenerator(rescale=1. / 255)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(128, 128),
    batch_size=batch_size,
    class_mode='categorical',
    classes=classes,
    shuffle=False
)

# Predict on the test data
y_pred = loaded_model.predict(test_generator)
y_true = test_generator.classes

# Get class labels
class_labels = list(test_generator.class_indices.keys())

# Generate confusion matrix
cm = confusion_matrix(y_true, np.argmax(y_pred, axis=1))

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# testing
image_path = "test.jpeg"
predicted_class, confidence = predict_fruit(image_path)

# Display the result
plt.imshow(image.load_img(image_path, target_size=(128, 128)))
plt.title(f"Predicted: {predicted_class} ({confidence:.2f})")
plt.show()
