#!/usr/bin/env python3
"""
Skin Cancer Detection - Main Script

This script serves as the main entry point for the skin cancer detection project.
It provides a command-line interface to train, evaluate, and use the model.
"""

import os
import argparse
import sys

def print_header():
    """Print header information"""
    print("\n" + "="*80)
    print(" "*30 + "SKIN CANCER DETECTION")
    print("="*80 + "\n")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Skin Cancer Detection')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train the model')
    train_parser.add_argument('--data_dir', type=str, default='../data',
                         help='Directory containing the dataset')
    train_parser.add_argument('--model_dir', type=str, default='../models',
                         help='Directory to save model files')
    train_parser.add_argument('--model_type', type=str, default='transfer',
                         choices=['cnn', 'transfer'],
                         help='Type of model to train (basic CNN or transfer learning)')
    train_parser.add_argument('--base_model', type=str, default='efficientnet',
                         choices=['vgg16', 'resnet50', 'efficientnet'],
                         help='Base model for transfer learning')
    train_parser.add_argument('--epochs', type=int, default=20,
                         help='Number of training epochs')
    
    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate the model')
    eval_parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the trained model file')
    eval_parser.add_argument('--data_dir', type=str, default='../data',
                        help='Directory containing the test dataset')
    
    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Make predictions on an image')
    predict_parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the trained model file')
    predict_parser.add_argument('--image', type=str, required=True,
                        help='Path to the image file to predict')
    predict_parser.add_argument('--visualize', action='store_true',
                        help='Whether to visualize the prediction')
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return
    
    print_header()
    
    if args.command == 'train':
        from train_model import main as train_main
        sys.argv = [sys.argv[0]] + [
            '--data_dir', args.data_dir,
            '--model_dir', args.model_dir,
            '--model_type', args.model_type,
            '--base_model', args.base_model,
            '--epochs', str(args.epochs)
        ]
        train_main()
    
    elif args.command == 'evaluate':
        from evaluate_model import main as eval_main
        sys.argv = [sys.argv[0]] + [
            '--model_path', args.model_path,
            '--data_dir', args.data_dir
        ]
        eval_main()
    
    elif args.command == 'predict':
        from predict import main as predict_main
        cmd = [sys.argv[0], '--model_path', args.model_path, '--image', args.image]
        if args.visualize:
            cmd.append('--visualize')
        sys.argv = cmd
        predict_main()

if __name__ == "__main__":
    main() 