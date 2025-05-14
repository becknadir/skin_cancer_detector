# Skin Cancer Detection

This repository contains a machine learning model for detecting skin cancer from images.

## Project Overview

This model is designed to classify skin lesion images into different categories of skin conditions, with a focus on identifying malignant versus benign lesions.

## Dataset

The model requires a large image dataset (approximately 5GB) for training. Due to the size limitations, the training data is not included in this repository. 

**Note:** The training dataset will be added in the future once a suitable hosting solution is found.

### Expected Dataset Structure

```
data/
    train/
        benign/
            img1.jpg
            img2.jpg
            ...
        malignant/
            img1.jpg
            img2.jpg
            ...
    validation/
        benign/
            img1.jpg
            img2.jpg
            ...
        malignant/
            img1.jpg
            img2.jpg
            ...
    test/
        benign/
            img1.jpg
            img2.jpg
            ...
        malignant/
            img1.jpg
            img2.jpg
            ...
```

## Installation

1. Clone this repository:
```
git clone https://github.com/yourusername/skin-cancer-detection.git
cd skin-cancer-detection
```

2. Install the required dependencies:
```
pip install -r requirements.txt
```

## Usage

1. First, prepare your dataset according to the structure above.
2. Run the training script:
```
python src/train_model.py
```
3. For predictions on new images:
```
python src/predict.py --image path/to/your/image.jpg
```

## Model Architecture

This project uses a convolutional neural network (CNN) built with TensorFlow/Keras for image classification.

## License

[MIT License](LICENSE)

## Acknowledgments

- Dataset source will be credited once added
- Any other acknowledgments or references will be added here 