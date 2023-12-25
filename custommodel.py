from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os


def build_custom_model():
    # path to dataset
    dataset_path = "dataset"

    # Define the classes (fruits) you want to detect
    classes = ['apple', 'orange', 'banana', 'grape', 'watermelon']

    # Set up the project directories
    train_dir = os.path.join(dataset_path, 'train')
    test_dir = os.path.join(dataset_path, 'test')
    # Create an ImageDataGenerator for data augmentation
    batch_size = 32
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(128, 128),  # Adjust the target size as needed
        batch_size=batch_size,
        class_mode='categorical',
        classes=classes
    )

    # Build a custom neural network model
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(len(classes), activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    epochs = 10
    history = model.fit(train_generator, epochs=epochs)

    # Save the trained model
    model.save("model_custom.h5")

    return test_dir, batch_size
