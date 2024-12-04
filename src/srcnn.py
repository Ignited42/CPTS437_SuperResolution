import tensorflow as tf
from tensorflow.keras import layers, models, optimizers

def build_srcnn():
    input_layer = layers.Input(shape=(None, None, 3))

    # Initial large kernel for broader feature extraction
    x = layers.Conv2D(64, (9, 9), activation='relu', padding='same', kernel_initializer='he_normal')(input_layer)
    x = layers.Conv2D(64, (5, 5), activation='relu', padding='same', kernel_initializer='he_normal')(x)

    # Deeper layers with varied filter numbers for more detailed features
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(128, (3, 3), activation='sigmoid', padding='same', kernel_initializer='he_normal')(x)
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(256, (3, 3), activation='sigmoid', padding='same', kernel_initializer='he_normal')(x)

    # Adding more depth with skip connections
    skip = x
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(x)
    x = layers.Conv2D(256, (3, 3), activation='sigmoid', padding='same', kernel_initializer='he_normal')(x)
    x = layers.Add()([x, skip])  # Residual connection

    # Final convolution layers to reconstruct the output
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(x)
    x = layers.Conv2D(64, (3, 3), activation='sigmoid', padding='same', kernel_initializer='he_normal')(x)
    output_layer = layers.Conv2D(3, (5, 5), activation='sigmoid', padding='same')(x)

    return models.Model(inputs=input_layer, outputs=output_layer)
