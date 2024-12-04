import matplotlib.pyplot as plt
import numpy as np

# Evaluate the model and plot multiple samples
def plot_samples(lr_images, hr_images, sr_images):
    num_samples = len(lr_images)
    for i in range(num_samples):
        lr = lr_images[i]
        hr = hr_images[i]
        sr = sr_images[i]

        # Ensure the output image is properly scaled to [0, 1] range if needed
        sr = np.clip(sr, 0, 1)

        # Plot the LR, HR, and SR images side by side
        plt.figure(figsize=(20, 10))
        plt.subplot(1, 3, 1)
        plt.title("Low Resolution", fontsize=16)
        plt.imshow(lr)
        plt.axis('off')  # Remove axes for better visualization

        plt.subplot(1, 3, 2)
        plt.title("High Resolution", fontsize=16)
        plt.imshow(hr)
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.title("Super Resolved", fontsize=16)
        plt.imshow(sr)
        plt.axis('off')

        plt.tight_layout()  # Adjust layout to prevent overlapping
        plt.show()

def plot_pred_pixels(sample_sr):
    # Plot the histogram of predicted pixel values
    plt.figure(figsize=(8, 4))
    plt.hist(sample_sr.flatten(), bins=50, color='blue', alpha=0.7)
    plt.title('Histogram of Predicted Pixel Values')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.show()

def samplefull(lr_images, hr_images, index, srcnn):
  # Test the model on a single full-size image
  sample_lr_full = lr_images[index]
  sample_hr_full = hr_images[index]
  sample_sr_full = srcnn.predict(np.expand_dims(sample_lr_full, axis=0))[0]

  # Ensure the output image is properly scaled to [0, 1] range
  sample_sr_full = np.clip(sample_sr_full, 0, 1).astype(np.float32)

  # Plot the result for the full-size image
  plt.figure(figsize=(60, 20))  # Adjust the figure size if 60x20 is too large
  plt.subplot(1, 3, 1)
  plt.title("Low Resolution (Full Image)", fontsize=16)
  plt.imshow(sample_lr_full)
  plt.axis('off')  # Remove axes for better visualization

  plt.subplot(1, 3, 2)
  plt.title("High Resolution (Full Image)", fontsize=16)
  plt.imshow(sample_hr_full)
  plt.axis('off')

  plt.subplot(1, 3, 3)
  plt.title("Super Resolved (Full Image)", fontsize=16)
  plt.imshow(sample_sr_full)
  plt.axis('off')

  plt.tight_layout()  # Adjust layout to prevent overlapping titles
  plt.show()