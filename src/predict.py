import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import cv2

def load_and_preprocess_image(img_path, target_size=(224, 224)):
    """
    Load and preprocess an image for prediction
    
    Parameters:
    -----------
    img_path : str
        Path to the image file
    target_size : tuple
        Target size for resizing the image
        
    Returns:
    --------
    img_array : numpy.ndarray
        Preprocessed image array
    """
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize
    
    return img_array

def predict_image(model, img_path, class_names, target_size=(224, 224)):
    """
    Make a prediction for a single image
    
    Parameters:
    -----------
    model : tf.keras.Model
        Trained model
    img_path : str
        Path to the image file
    class_names : list
        List of class names
    target_size : tuple
        Target size for resizing the image
        
    Returns:
    --------
    prediction : dict
        Dictionary containing prediction results
    """
    # Load and preprocess image
    img_array = load_and_preprocess_image(img_path, target_size)
    
    # Make prediction
    predictions = model.predict(img_array)
    predicted_class_idx = np.argmax(predictions[0])
    predicted_class = class_names[predicted_class_idx]
    confidence = predictions[0][predicted_class_idx]
    
    # Return results
    return {
        'class': predicted_class,
        'confidence': float(confidence),
        'probabilities': {class_names[i]: float(predictions[0][i]) for i in range(len(class_names))}
    }

def visualize_prediction(img_path, prediction, target_size=(224, 224)):
    """
    Visualize an image with its prediction
    
    Parameters:
    -----------
    img_path : str
        Path to the image file
    prediction : dict
        Prediction results
    target_size : tuple
        Target size for displaying the image
    """
    # Load image
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)
    
    # Create figure
    plt.figure(figsize=(10, 6))
    
    # Plot image
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title(f"Prediction: {prediction['class']}")
    plt.axis('off')
    
    # Plot probabilities
    plt.subplot(1, 2, 2)
    classes = list(prediction['probabilities'].keys())
    probs = list(prediction['probabilities'].values())
    y_pos = np.arange(len(classes))
    
    plt.barh(y_pos, probs, align='center')
    plt.yticks(y_pos, classes)
    plt.xlabel('Probability')
    plt.title('Class Probabilities')
    
    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Make predictions on skin cancer images')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the trained model file')
    parser.add_argument('--image', type=str, required=True,
                        help='Path to the image file to predict')
    parser.add_argument('--class_names', type=str, nargs='+',
                        default=['benign', 'malignant'],
                        help='List of class names (in order of model output)')
    parser.add_argument('--visualize', action='store_true',
                        help='Whether to visualize the prediction')
    
    args = parser.parse_args()
    
    # Check if the model file exists
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model file not found: {args.model_path}")
    
    # Check if the image file exists
    if not os.path.exists(args.image):
        raise FileNotFoundError(f"Image file not found: {args.image}")
    
    # Load the model
    model = load_model(args.model_path)
    print(f"Model loaded from {args.model_path}")
    
    # Make prediction
    prediction = predict_image(model, args.image, args.class_names)
    
    # Print results
    print("\nPrediction Results:")
    print(f"Class: {prediction['class']}")
    print(f"Confidence: {prediction['confidence']:.4f}")
    print("\nClass Probabilities:")
    for class_name, prob in prediction['probabilities'].items():
        print(f"  {class_name}: {prob:.4f}")
    
    # Visualize if specified
    if args.visualize:
        visualize_prediction(args.image, prediction)

if __name__ == "__main__":
    main() 