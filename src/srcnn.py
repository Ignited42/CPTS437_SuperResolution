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

# Build model
srcnn = build_srcnn()
optimizer = optimizers.Adam(learning_rate=1e-6)

# Loss function: Mean Squared Error (simpler than perceptual loss)
srcnn.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mean_squared_error'])

# Training parameters
batch_size = 128
steps_per_epoch = len(lr_patches) // batch_size

# Add early stopping and model checkpoint for training stability
lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=3, min_lr=1e-8)

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3, restore_best_weights=True)
checkpoint = tf.keras.callbacks.ModelCheckpoint('srcnn_checkpoint.weights.h5', monitor='loss', save_best_only=True, save_weights_only=True)

# Training the model
history = srcnn.fit(
    tf.data.Dataset.from_tensor_slices((lr_patches, hr_patches)).batch(batch_size).repeat(),
    epochs=500,
    steps_per_epoch=steps_per_epoch,
    validation_data=(lr_patches[:batch_size], hr_patches[:batch_size]),
    callbacks=[early_stopping, checkpoint, lr_scheduler]
)

# Get the final validation loss
final_val_loss = history.history['val_mean_squared_error'][-1]
srcnn.load_weights('srcnn_checkpoint.weights.h5')
# Save the entire model once training is complete
srcnn.save('srcnn_final_model.h5')

# Send a notification with the final validation loss
send_ifttt_notification("training_complete", f"Your SRCNN model has finished training. Final validation loss: {final_val_loss}")

