# Few-shot 3D Medical Image Segmentation using Generative Adversarial Learning
This repository contains the tensorflow and pytorch implementation of the model we proposed in our paper of the same name: [Few-shot 3D Medical Image Segmentation using Generative Adversarial Learning](https://arxiv.org/abs/1810.12241)

The code is available in both tensorflow and pytorch. To run the project kindly refer to the individual readme file.

### Dataset
[iSEG 2017](http://iseg2017.web.unc.edu/) dataset was chosen to substantiate our proposed method.
It contains the 3D multi-modal brain MRI data of 10 labeled training subjects and 13 unlabeled testing subjects.
We split the 10 labeled training data into training, validation and testing images for both the models.(Eg- 2,1 and 7)
Rest of the 13 unlabeled testing images are only used for training the GAN based model.

[MRBrainS13](https://mrbrains13.isi.uu.nl/) dataset was also used to test the robustness of our proposed model.
It contains 3 modalities (T1-weighted, T1-weighted inversion recovery and FLAIR).
Original Dataset consists of 5 labeled training and 6 unlabeled testing subjects.
We split the 5 labeled training data into 1,1,3 for training, validation and testing. The 6 testing subjects are used as unlabeled data for the GAN based models.

## Proposed model architecture
The following shows the model architecture of the proposed model. (Read our paper for further details)

<br>
<img src="https://github.com/arnab39/FewShot_GAN-Unet3D/blob/master/tensorflow/images/Diagram.jpg" width="800"/>
<br>

## Some results from our paper

* Visual comparison of the segmentation by each model, for two test subjects of the iSEG-2017 dataset, when training with different numbers of labeled examples.
<p float="left">
  <img src="https://github.com/arnab39/FewShot_GAN-Unet3D/blob/master/tensorflow/images/train1.png" width="800" />
  <img src="https://github.com/arnab39/FewShot_GAN-Unet3D/blob/master/tensorflow/images/train2.png" width="800" />
</p>

* Segmentation of Subject 10 of the iSEG-2017 dataset predicted by different GAN-based models, when trained with 2 labeled images. The red box highlights a region in the ground truth where all these models give noticeable differences.
<br>
<img src="https://github.com/arnab39/FewShot_GAN-Unet3D/blob/master/tensorflow/images/comparison.png" width="820"/>
<br>
* More such results can be found in the paper.

## Contact
You can mail us at: sanu.arnab@gmail.com, aakash.garg80@gmail.com
If you use this code for your research, please consider citing the original paper:

- Arnab Kumar Mondal, Jose Dolz, Christian Desrosiers. [Few-shot 3D Multi-modal Medical Image Segmentation using Generative Adversarial Learning](https://arxiv.org/abs/1810.12241).
