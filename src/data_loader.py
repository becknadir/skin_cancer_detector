import os
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def load_dataset(data_dir, img_size=(224, 224), batch_size=32, validation_split=0.2):
    """
    Load and preprocess the skin cancer dataset
    
    Parameters:
    -----------
    data_dir : str
        Path to the data directory
    img_size : tuple
        Target size for the images
    batch_size : int
        Batch size for training
    validation_split : float
        Fraction of data to use for validation
        
    Returns:
    --------
    train_generator, validation_generator : tuple
        Data generators for training and validation
    """
    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=validation_split
    )
    
    # Only rescaling for validation
    validation_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=validation_split
    )
    
    # Load training data
    train_generator = train_datagen.flow_from_directory(
        os.path.join(data_dir, 'train'),
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training'
    )
    
    # Load validation data
    validation_generator = validation_datagen.flow_from_directory(
        os.path.join(data_dir, 'train'),
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'
    )
    
    return train_generator, validation_generator

def load_test_data(data_dir, img_size=(224, 224), batch_size=32):
    """
    Load test dataset
    
    Parameters:
    -----------
    data_dir : str
        Path to the test data directory
    img_size : tuple
        Target size for the images
    batch_size : int
        Batch size for testing
        
    Returns:
    --------
    test_generator : ImageDataGenerator
        Data generator for test data
    """
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    test_generator = test_datagen.flow_from_directory(
        os.path.join(data_dir, 'test'),
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )
    
    return test_generator

def visualize_samples(data_generator, num_samples=5):
    """
    Visualize sample images from the dataset
    
    Parameters:
    -----------
    data_generator : ImageDataGenerator
        Data generator to draw samples from
    num_samples : int
        Number of samples to visualize
    """
    # Get a batch of images
    images, labels = next(data_generator)
    
    # Get class labels
    class_indices = data_generator.class_indices
    class_names = {v: k for k, v in class_indices.items()}
    
    # Plot images
    plt.figure(figsize=(15, 5))
    for i in range(min(num_samples, len(images))):
        plt.subplot(1, num_samples, i + 1)
        plt.imshow(images[i])
        label_idx = np.argmax(labels[i])
        plt.title(class_names[label_idx])
        plt.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Example usage
    data_dir = "../data"
    train_gen, val_gen = load_dataset(data_dir)
    print(f"Found {train_gen.samples} training images")
    print(f"Found {val_gen.samples} validation images")
    
    # Visualize some samples
    visualize_samples(train_gen) 