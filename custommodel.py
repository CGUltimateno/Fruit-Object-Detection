from keras import Sequential
from keras.callbacks import EarlyStopping
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Flatten, Dense, Dropout, BatchNormalization


def create_generators(train_directory, validation_directory, test_directory, selected_categories):
    # Define the data augmentation parameters for training
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )

    # Define only rescaling for validation and testing (no data augmentation)
    valid_test_datagen = ImageDataGenerator(rescale=1. / 255)

    # Create generators for training, validation, and testing data
    train_generator = train_datagen.flow_from_directory(
        train_directory,
        target_size=(128, 128),
        batch_size=32,
        class_mode='categorical',
        classes=selected_categories
    )

    validation_generator = valid_test_datagen.flow_from_directory(
        validation_directory,
        target_size=(128, 128),
        batch_size=32,
        class_mode='categorical',
        classes=selected_categories
    )

    test_generator = valid_test_datagen.flow_from_directory(
        test_directory,
        target_size=(128, 128),
        batch_size=32,
        class_mode='categorical',
        classes=selected_categories
    )

    return train_generator, validation_generator, test_generator


def create_model(selected_categories):
    # Build a custom neural network model
    model = Sequential([
        Flatten(input_shape=(128, 128, 3)),
        BatchNormalization(),
        Dense(550, activation='relu'),
        Dropout(0.5),
        Dense(300, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(150, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(150, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(len(selected_categories), activation='softmax')
    ])

    return model


def compile_and_train_model(model, train_generator, validation_generator):
    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Define callbacks
    early_stopping_callback = EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)

    # Train the model
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // 32,
        epochs=10,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // 32,
    )

    return model


def build_custom_model():
    # Define the main directories
    train_directory = 'dataset/train'
    validation_directory = 'dataset/valid'
    test_directory = 'dataset/test'

    # Specify the desired categories
    selected_categories = ['apple', 'orange', 'banana', 'grape', 'watermelon']

    # Create generators
    train_generator, validation_generator, test_generator = create_generators(
        train_directory, validation_directory, test_directory, selected_categories)

    # Create model
    model = create_model(selected_categories)

    # Compile and train model
    model = compile_and_train_model(model, train_generator, validation_generator)

    # Save the trained model
    model.save("fruit_model.keras")

    return train_directory, 32
