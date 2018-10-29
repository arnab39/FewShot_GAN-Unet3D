# Few-shot 3D Multi-modal Medical Image Segmentation using Generative Adversarial Learning
This repository contain the tensorflow implementation of the model we proposed in our paper of the same name: Few-shot 3D Multi-modal Medical Image Segmentation using Generative Adversarial Learning, submitted in Medical Image Analysis, October 2018.

## Requirements

- The code has been written in Python (3.5.2) and Tensorflow (1.7.0)
- Make sure to install all the libraries given in requirement.txt (You can do so by the following command)
```
pip install -r requirement.txt
```

## Proposed model architecture
The following shows the model architecture of the proposed model.
<br>
<img src="https://github.com/arnab39/FewShot_GAN-Unet3D/blob/master/images/Diagram.jpg" />
<br>

## Some results from our paper

* Visual comparison of the segmentation by each model, for two test subjects of the iSEG-2017 dataset, when training with different numbers of labeled examples.
<p float="left">
  <img src="https://github.com/arnab39/FewShot_GAN-Unet3D/blob/master/images/Subject9.jpg" width="420" />
  <img src="https://github.com/arnab39/FewShot_GAN-Unet3D/blob/master/images/Subject10.jpg" width="420" /> 
</p>

* Segmentation of Subject 10 of the iSEG-2017 dataset predicted by different GAN-based models, when trained with 2 labeled images. The red box highlights a region in the ground truth where all these models give noticeable differences.
<br>
<img src="https://github.com/arnab39/FewShot_GAN-Unet3D/blob/master/images/ganwar_mod.jpg" />
<br>
