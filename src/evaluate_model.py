import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

from data_loader import load_test_data

def plot_confusion_matrix(cm, class_names, save_path=None):
    """
    Plot confusion matrix
    
    Parameters:
    -----------
    cm : numpy.ndarray
        Confusion matrix
    class_names : list
        List of class names
    save_path : str, optional
        Path to save the plot
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    
    if save_path:
        plt.savefig(save_path)
    
    plt.show()

def plot_roc_curve(y_true, y_pred, class_names, save_path=None):
    """
    Plot ROC curve for multi-class classification
    
    Parameters:
    -----------
    y_true : numpy.ndarray
        True labels (one-hot encoded)
    y_pred : numpy.ndarray
        Predicted probabilities
    class_names : list
        List of class names
    save_path : str, optional
        Path to save the plot
    """
    n_classes = len(class_names)
    
    # Compute ROC curve and ROC area for each class
    fpr = {}
    tpr = {}
    roc_auc = {}
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Plot all ROC curves
    plt.figure(figsize=(10, 8))
    
    colors = plt.cm.get_cmap('tab10', n_classes)
    
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f'{class_names[i]} (AUC = {roc_auc[i]:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    
    if save_path:
        plt.savefig(save_path)
    
    plt.show()

def evaluate_model(model, test_generator, class_names, output_dir=None):
    """
    Evaluate model performance
    
    Parameters:
    -----------
    model : tf.keras.Model
        Trained model
    test_generator : ImageDataGenerator
        Test data generator
    class_names : list
        List of class names
    output_dir : str, optional
        Directory to save evaluation results
    
    Returns:
    --------
    metrics : dict
        Dictionary containing evaluation metrics
    """
    # Get true labels and predictions
    steps = test_generator.samples // test_generator.batch_size + 1
    y_pred_probs = model.predict(test_generator, steps=steps)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    # Get true labels
    true_classes = test_generator.classes[:len(y_pred)]
    y_true_one_hot = np.zeros((len(true_classes), len(class_names)))
    for i, label in enumerate(true_classes):
        y_true_one_hot[i, label] = 1
    
    # Compute confusion matrix
    cm = confusion_matrix(true_classes, y_pred)
    
    # Compute classification report
    report = classification_report(true_classes, y_pred, target_names=class_names, output_dict=True)
    
    # Create outputs
    metrics = {
        'accuracy': report['accuracy'],
        'weighted_avg_precision': report['weighted avg']['precision'],
        'weighted_avg_recall': report['weighted avg']['recall'],
        'weighted_avg_f1': report['weighted avg']['f1-score'],
        'confusion_matrix': cm.tolist(),
        'classification_report': report
    }
    
    # Save and show plots if output directory is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        # Save confusion matrix
        cm_path = os.path.join(output_dir, 'confusion_matrix.png')
        plot_confusion_matrix(cm, class_names, save_path=cm_path)
        
        # Save ROC curve
        roc_path = os.path.join(output_dir, 'roc_curve.png')
        plot_roc_curve(y_true_one_hot, y_pred_probs, class_names, save_path=roc_path)
        
        # Save metrics to CSV
        metrics_df = pd.DataFrame({
            'Metric': list(metrics.keys())[:4],  # Only the scalar metrics
            'Value': list(metrics.values())[:4]
        })
        metrics_df.to_csv(os.path.join(output_dir, 'metrics.csv'), index=False)
        
        # Save detailed report
        report_df = pd.DataFrame(report).transpose()
        report_df.to_csv(os.path.join(output_dir, 'classification_report.csv'))
    else:
        # Just display the plots
        plot_confusion_matrix(cm, class_names)
        plot_roc_curve(y_true_one_hot, y_pred_probs, class_names)
    
    return metrics

def main():
    parser = argparse.ArgumentParser(description='Evaluate a skin cancer detection model')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the trained model file')
    parser.add_argument('--data_dir', type=str, default='../data',
                        help='Directory containing the test dataset')
    parser.add_argument('--output_dir', type=str, default='../models/evaluation',
                        help='Directory to save evaluation results')
    parser.add_argument('--class_names', type=str, nargs='+',
                        help='List of class names (in order of model output)')
    parser.add_argument('--img_size', type=int, default=224,
                        help='Image size (width and height)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for evaluation')
    
    args = parser.parse_args()
    
    # Check if the model file exists
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model file not found: {args.model_path}")
    
    # Load the model
    model = load_model(args.model_path)
    print(f"Model loaded from {args.model_path}")
    
    # Load test data
    test_generator = load_test_data(
        args.data_dir,
        img_size=(args.img_size, args.img_size),
        batch_size=args.batch_size
    )
    
    # If class names are not provided, get them from the test generator
    if args.class_names is None:
        class_indices = test_generator.class_indices
        class_names = [k for k, v in sorted(class_indices.items(), key=lambda item: item[1])]
    else:
        class_names = args.class_names
    
    print(f"Evaluating model on {test_generator.samples} test images")
    
    # Evaluate model
    metrics = evaluate_model(model, test_generator, class_names, args.output_dir)
    
    # Print summary of results
    print("\nEvaluation Results:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Weighted Avg Precision: {metrics['weighted_avg_precision']:.4f}")
    print(f"Weighted Avg Recall: {metrics['weighted_avg_recall']:.4f}")
    print(f"Weighted Avg F1 Score: {metrics['weighted_avg_f1']:.4f}")
    
    if args.output_dir:
        print(f"\nDetailed results saved to {args.output_dir}")

if __name__ == "__main__":
    main() 