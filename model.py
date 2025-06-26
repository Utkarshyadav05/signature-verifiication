import os
import numpy as np
import keras
from keras import layers, Model, optimizers
from keras import callbacks 
import cv2

EarlyStopping = callbacks.EarlyStopping
ModelCheckpoint = callbacks.ModelCheckpoint

# Step 1: Register the euclidean_distance function
@keras.saving.register_keras_serializable()
def euclidean_distance(vectors):
    """
    Compute the Euclidean distance between two vectors.
    """
    x, y = vectors
    return keras.ops.sqrt(keras.ops.sum(keras.ops.square(x - y), axis=1, keepdims=True))

# Step 2: Generate Synthetic Data
def generate_synthetic_data(num_samples=1000, image_shape=(128, 128, 1)):
    """
    Generate synthetic data for training.
    Each sample consists of a pair of images and a label (1 for genuine, 0 for forged).
    """
    data = []
    labels = []

    for _ in range(num_samples):
        # Generate a random "base" image (genuine signature)
        base_image = np.random.rand(*image_shape) * 255
        base_image = base_image.astype(np.uint8)

        # Generate a genuine pair (slightly modified base image)
        genuine_image = base_image + np.random.normal(0, 10, image_shape)
        genuine_image = np.clip(genuine_image, 0, 255).astype(np.uint8)
        data.append([base_image, genuine_image])
        labels.append(1)  # Genuine pair

        # Generate a forged pair (completely random image)
        forged_image = np.random.rand(*image_shape) * 255
        forged_image = forged_image.astype(np.uint8)
        data.append([base_image, forged_image])
        labels.append(0)  # Forged pair

    # Convert to numpy arrays
    data = np.array(data, dtype=np.float32) / 255.0  # Normalize to [0, 1]
    labels = np.array(labels, dtype=np.float32)

    return data, labels

# Step 3: Build the Siamese Network
def build_siamese_model(input_shape):
    """
    Build the base model for feature extraction.
    """
    input_layer = layers.Input(shape=input_shape)
    
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(input_layer)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    
    x = layers.Flatten()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    
    model = Model(input_layer, x)
    return model

# Step 4: Build the Comparison Model
def build_comparison_model(input_shape):
    """
    Build the Siamese Network for signature verification.
    """
    base_model = build_siamese_model(input_shape)
    
    input_a = layers.Input(shape=input_shape)
    input_b = layers.Input(shape=input_shape)
    
    encoded_a = base_model(input_a)
    encoded_b = base_model(input_b)
    
    distance = layers.Lambda(euclidean_distance)([encoded_a, encoded_b])
    
    output = layers.Dense(1, activation='sigmoid')(distance)
    
    model = Model(inputs=[input_a, input_b], outputs=output)
    model.compile(loss='binary_crossentropy', optimizer=optimizers.Adam(learning_rate=0.0001), metrics=['accuracy'])
    
    return model

# Step 5: Train the Model
def train_model(model, data, labels, batch_size=32, epochs=20):
    """
    Train the Siamese Network.
    """
    # Split data into training and validation sets
    split = int(0.8 * len(data))
    train_data, val_data = data[:split], data[split:]
    train_labels, val_labels = labels[:split], labels[split:]

    # Callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ModelCheckpoint("model/signature_model_final.keras", monitor='val_loss', save_best_only=True)
    ]

    # Train the model
    history = model.fit(
        [train_data[:, 0], train_data[:, 1]], train_labels,
        validation_data=([val_data[:, 0], val_data[:, 1]], val_labels),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=callbacks
    )
    return history

# Step 6: Main Function
def main():
    # Define image shape
    image_shape = (128, 128, 1)
    
    # Generate synthetic data
    data, labels = generate_synthetic_data(num_samples=1000, image_shape=image_shape)
    
    # Build the model
    model = build_comparison_model(image_shape)
    
    # Train the model
    train_model(model, data, labels)
    
    # Save the final model
    model.save("model/signature_model_final.keras")
    print("Model saved as 'model/signature_model_final.keras'.")

# Run the program
if __name__ == "__main__":
    main()