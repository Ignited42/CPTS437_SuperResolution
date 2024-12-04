import os
import numpy as np
import matplotlib.pyplot as plt
import build_cnn as sr
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import data_preparation as datprep
import plot_result as plot

# Download images and return image location
path = datprep.download_images()

# Verify if the path exists and list the files
if os.path.exists(path):
    print("Path exists. Listing files in the directory:")
    for dirpath, _, filenames in os.walk(path):
        for file in filenames[:5]:  # List the first 5 files only
            print(os.path.join(dirpath, file))
else:
    print("Path does not exist. Please check the path.")

# Attempt to load a single image from the path
try:
    sample_images = datprep.load_images(path, img_size=(256, 256))
    if len(sample_images) > 0:
        print("Successfully loaded images.")
        plt.imshow(sample_images[0])
        plt.title("Sample Loaded Image")
        plt.show()
    else:
        print("No images were found in the specified path.")
except Exception as e:
    print(f"An error occurred while loading images: {e}")

#Load the high res images
hr_images = datprep.load_images(path)

#create low res images
lr_images = datprep.create_low_res(hr_images)

# Extract patches from high-resolution and low-resolution images
hr_patches = datprep.extract_patches(hr_images)
lr_patches = datprep.extract_patches(lr_images)
# plt.imshow(hr_patches[0])
# plt.show()

# ASK EDD ON THIS PART
if np.max(lr_images) > 1:
    hr_images = hr_images / 255.0
    lr_images = lr_images / 255.0

# Normalize the patches to [0, 1] range
lr_patches = lr_patches / 255.0
hr_patches = hr_patches / 255.0


# Build model
srcnn = sr.build_srcnn()
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
# send_ifttt_notification("training_complete", f"Your SRCNN model has finished training. Final validation loss: {final_val_loss}")

'''
=============================================================
'''

# Test the model on multiple sample patches
sample_lr = lr_patches[10:15]  # Selecting 5 low-resolution patches
sample_hr = hr_patches[10:15]  # Selecting the corresponding 5 high-resolution patches

# Predict the super-resolved images
sample_sr = srcnn.predict(sample_lr)

# Clip or normalize the output images to [0, 1] range
sample_sr = np.clip(sample_sr, 0, 1).astype(np.float32)

# Plot predicted pixels histogram
plot.plot_pred_pixels(sample_sr)

# Call the function with the predicted samples
plot.plot_samples(sample_lr, sample_hr, sample_sr)

# Show full samples
for i in range(9):
  plot.samplefull(lr_images, hr_images, i, srcnn)