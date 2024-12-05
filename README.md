# CPTS437_SuperResolution

The purpose of the project is to upscale images with acceptable quality. Most of the development is done through Google Colab, where it's eventually imported here as a Python project for both storage and documentation purposes.

## Project Specifications
Libraries Used:
* `tensorflow`
* `matplotlib`
* `kagglehub`

<br>

Dataset Used:
[DIV2K](https://www.kaggle.com/datasets/soumikrakshit/div2k-high-resolution-images)

Original dataset: https://data.vision.ee.ethz.ch/cvl/DIV2K/

## Prerequisites
It is highly recommended **NOT** to run this on your local machine as the size of the dataset is around 3.8 GB! Instead, consider running the given Notebook on Google Colab.

While not strictly necessary, make sure you have a Kaggle key, and by extension, a Kaggle account, to download the dataset from `kagglehub`. Documentation regarding such process can be found on [their repository](https://github.com/Kaggle/kagglehub).

## Recommended Usage

  
To Train Model:

* Download SuperResolutionNotebook.ipynb and open them in a Colab environment.
* It is recommended to use an upgraded version of Colab, as this model was trained on an A100 GPU.
* For the base version of Colab, reduce the batch_size to a manageable level for the GPU being used to avoid memory issues. **(IMPORTANT)**

To Test Model:

* Download .h5 file from bin folder.
* Download the SRCNNUsage notebook and open it in a Colab environment.
* Place .h5 file in reachable location for the Colab environment and step through the notebook.
