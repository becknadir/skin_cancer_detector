import os
import argparse
import pickle
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard
import datetime

# Import from local modules
from data_loader import load_dataset
from model import create_cnn_model, create_transfer_learning_model, get_callbacks, fine_tune_model

def plot_training_history(history, save_path=None):
    """
    Plot training and validation metrics
    
    Parameters:
    -----------
    history : tf.keras.callbacks.History
        Training history
    save_path : str, optional
        Path to save the plot
    """
    plt.figure(figsize=(12, 4))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    plt.show()

def save_history(history, save_path):
    """
    Save training history to a file
    
    Parameters:
    -----------
    history : tf.keras.callbacks.History
        Training history
    save_path : str
        Path to save the history
    """
    with open(save_path, 'wb') as f:
        pickle.dump(history.history, f)
    print(f"History saved to {save_path}")

def main():
    parser = argparse.ArgumentParser(description='Train a skin cancer detection model')
    parser.add_argument('--data_dir', type=str, default='../data',
                        help='Directory containing the dataset')
    parser.add_argument('--model_dir', type=str, default='../models',
                        help='Directory to save model files')
    parser.add_argument('--model_type', type=str, default='transfer',
                        choices=['cnn', 'transfer'],
                        help='Type of model to train (basic CNN or transfer learning)')
    parser.add_argument('--base_model', type=str, default='efficientnet',
                        choices=['vgg16', 'resnet50', 'efficientnet'],
                        help='Base model for transfer learning')
    parser.add_argument('--img_size', type=int, default=224,
                        help='Image size (width and height)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of training epochs')
    parser.add_argument('--fine_tune', action='store_true',
                        help='Whether to fine-tune the model after initial training')
    parser.add_argument('--fine_tune_epochs', type=int, default=10,
                        help='Number of fine-tuning epochs')
    
    args = parser.parse_args()
    
    # Create model directory if it doesn't exist
    os.makedirs(args.model_dir, exist_ok=True)
    
    # Set up input shape and load data
    input_shape = (args.img_size, args.img_size, 3)
    train_generator, validation_generator = load_dataset(
        args.data_dir,
        img_size=(args.img_size, args.img_size),
        batch_size=args.batch_size
    )
    
    # Get number of classes from the generator
    num_classes = len(train_generator.class_indices)
    print(f"Number of classes: {num_classes}")
    print(f"Class indices: {train_generator.class_indices}")
    
    # Create model based on specified type
    if args.model_type == 'cnn':
        model = create_cnn_model(input_shape, num_classes)
        model_name = 'cnn_model'
    else:  # Transfer learning model
        model = create_transfer_learning_model(input_shape, num_classes, args.base_model)
        model_name = f"{args.base_model}_model"
    
    # Define checkpoint path and callbacks
    checkpoint_path = os.path.join(args.model_dir, f"{model_name}.h5")
    callbacks = get_callbacks(checkpoint_path)
    
    # Add TensorBoard callback
    log_dir = os.path.join(args.model_dir, "logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
    callbacks.append(tensorboard_callback)
    
    # Print model summary
    model.summary()
    
    # Train the model
    print(f"Training {model_name}...")
    history = model.fit(
        train_generator,
        epochs=args.epochs,
        validation_data=validation_generator,
        callbacks=callbacks
    )
    
    # Fine-tune the model if specified
    if args.fine_tune and args.model_type == 'transfer':
        print(f"Fine-tuning {model_name}...")
        model = fine_tune_model(model, num_layers_to_unfreeze=10)
        
        # Train with fine-tuning
        fine_tune_history = model.fit(
            train_generator,
            epochs=args.fine_tune_epochs,
            validation_data=validation_generator,
            callbacks=callbacks
        )
        
        # Combine histories
        for key in fine_tune_history.history:
            history.history[key].extend(fine_tune_history.history[key])
    
    # Save the final model
    final_model_path = os.path.join(args.model_dir, f"{model_name}_final.h5")
    model.save(final_model_path)
    print(f"Final model saved to {final_model_path}")
    
    # Save and plot training history
    history_path = os.path.join(args.model_dir, f"{model_name}_history.pkl")
    save_history(history, history_path)
    
    plot_path = os.path.join(args.model_dir, f"{model_name}_training_plot.png")
    plot_training_history(history, save_path=plot_path)
    
    print("Training completed successfully!")

if __name__ == "__main__":
    main() 