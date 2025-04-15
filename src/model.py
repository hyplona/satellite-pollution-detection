import tensorflow as tf
from tensorflow.keras import layers, models

def build_cnn_model(input_shape=(128, 128, 3)):
    """
    Build the Convolutional Neural Network (CNN) model for pollution detection
    from satellite images.

    Args:
        input_shape: The shape of the input images (height, width, channels).
                      Default is (128, 128, 3) for RGB images.

    Returns:
        A compiled CNN model.
    """

    # Initialize the Sequential model
    model = models.Sequential()

    # Add the first convolutional block
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))

    # Add the second convolutional block
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    # Add the third convolutional block
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    # Flatten the 3D output to 1D for the fully connected layers
    model.add(layers.Flatten())

    # Fully connected layer
    model.add(layers.Dense(128, activation='relu'))

    # Output layer with a single neuron for binary classification
    model.add(layers.Dense(1, activation='sigmoid'))  # Sigmoid for binary classification (polluted vs clean)

    # Compile the model
    model.compile(optimizer='adam', 
                  loss='binary_crossentropy', 
                  metrics=['accuracy'])

    return model

def summarize_model(model):
    """
    Prints the summary of the model architecture.

    Args:
        model: The CNN model to be summarized.
    """
    model.summary()

def save_model(model, filepath='pollution_detection_model.h5'):
    """
    Save the trained model to a file.

    Args:
        model: The trained CNN model.
        filepath: The path where the model will be saved.
    """
    model.save(filepath)
    print(f"Model saved to {filepath}")

