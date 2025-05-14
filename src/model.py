import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Input
from tensorflow.keras.applications import VGG16, ResNet50, EfficientNetB0
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

def create_cnn_model(input_shape, num_classes):
    """
    Create a basic CNN model for skin cancer detection
    
    Parameters:
    -----------
    input_shape : tuple
        Shape of input images (height, width, channels)
    num_classes : int
        Number of output classes
        
    Returns:
    --------
    model : tf.keras.Model
        Compiled Keras model
    """
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dropout(0.5),
        Dense(512, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def create_transfer_learning_model(input_shape, num_classes, base_model='vgg16'):
    """
    Create a transfer learning model for skin cancer detection
    
    Parameters:
    -----------
    input_shape : tuple
        Shape of input images (height, width, channels)
    num_classes : int
        Number of output classes
    base_model : str
        Name of the pre-trained model to use ('vgg16', 'resnet50', or 'efficientnet')
        
    Returns:
    --------
    model : tf.keras.Model
        Compiled Keras model
    """
    # Create the base pre-trained model
    if base_model == 'vgg16':
        base = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    elif base_model == 'resnet50':
        base = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    elif base_model == 'efficientnet':
        base = EfficientNetB0(weights='imagenet', include_top=False, input_shape=input_shape)
    else:
        raise ValueError("base_model must be one of 'vgg16', 'resnet50', or 'efficientnet'")
    
    # Freeze the base model
    base.trainable = False
    
    # Add custom classification layers
    inputs = Input(shape=input_shape)
    x = base(inputs, training=False)
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs, outputs)
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def get_callbacks(checkpoint_path, patience=5):
    """
    Get callbacks for model training
    
    Parameters:
    -----------
    checkpoint_path : str
        Path to save model checkpoints
    patience : int
        Number of epochs with no improvement to wait before early stopping
        
    Returns:
    --------
    callbacks : list
        List of Keras callbacks
    """
    callbacks = [
        ModelCheckpoint(
            filepath=checkpoint_path,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max'
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=3,
            min_lr=1e-6
        )
    ]
    
    return callbacks

def fine_tune_model(model, num_layers_to_unfreeze):
    """
    Fine-tune a pre-trained model
    
    Parameters:
    -----------
    model : tf.keras.Model
        Model to fine-tune
    num_layers_to_unfreeze : int
        Number of layers to unfreeze from the end of the base model
        
    Returns:
    --------
    model : tf.keras.Model
        Fine-tuned model
    """
    # Unfreeze the specified number of layers from the end
    if hasattr(model.layers[0], 'layers'):
        base_model = model.layers[0]
        for layer in base_model.layers[:-num_layers_to_unfreeze]:
            layer.trainable = False
        for layer in base_model.layers[-num_layers_to_unfreeze:]:
            layer.trainable = True
    
    # Recompile the model with a lower learning rate
    model.compile(
        optimizer=Adam(learning_rate=1e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model 