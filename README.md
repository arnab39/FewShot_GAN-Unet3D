# Few-shot 3D Multi-modal Medical Image Segmentation using Generative Adversarial Learning
This repository contains the tensorflow implementation of the model we proposed in our paper of the same name: [Few-shot 3D Multi-modal Medical Image Segmentation using Generative Adversarial Learning](https://arxiv.org/abs/1810.12241), submitted in Medical Image Analysis, October 2018.

## Requirements

- The code has been written in Python (3.5.2) and Tensorflow (1.7.0)
- Make sure to install all the libraries given in requirement.txt (You can do so by the following command)
```
pip install -r requirement.txt
```
- Install [ANTs N4BiasFieldCorrection](https://github.com/ANTsX/ANTs/releases) and add the location of the ANTs binaries to the PATH environmental variable.

## Dataset
[iSEG 2017](http://iseg2017.web.unc.edu/) dataset was chosen to substantiate our proposed method. 
It contains the 3D multi-modal brain MRI data of 10 labeled training subjects and 13 unlabeled testing subjects.
We split the 10 labeled training data into training, validation and testing images for both the models.(Eg- 2,1 and 7)
Rest of the 13 unlabeled testing images are only used for training the GAN based model. 

## How to use the code?
* Download the iSEG-2017 data and place it in data folder. (Visit [this](http://iseg2017.web.unc.edu/download/) link to download the data. You need to register for the challenge.)
* To perform image wise normalization and correction( run preprocess_static once more by changing dataset argument from "labeled" to "unlabeled"):
```
$ cd preprocess
$ python
>>> from preprocess import preprocess_static
>>> preprocess_static("../data/iSEG", "../data/iSEG_preprocessed",dataset="labeled")
```
* The preprocessed images will be stored in iSEG_preprocessed folder
* If you fail to install ANTs N4BiasFieldCorrection you can also skip the above preprocessing step and continue working with the original dataset. (Just change the data_directory flag to the original data directory while running the models)
* You can run standard 3D U-Net & our proposed model(both Feature matching GAN and bad GAN) with this code and compare their performance.

### How to run 3D U-Net?
```
$ cd ../unet3D
```
* Configure the flags according to your experiment.
* To run training
```
$ python main_unet.py --training
```
* This will train your model and save the best checkpoint according to your validation performance. 
* You can also resume training from saved checkpoint by setting the load_chkpt flag.
* You can run the testing to predict segmented output which will be saved in your result folder as ".nii.gz" files.
* To run testing
```
$ python main_unet.py --testing
```
* This version of code only compute dice coefficient to evaluate the testing performance.( Once the output segmented images are created you can use them to compute any other evaluation metric)
* Note that the U-Net used here is modified according to the U-Net used in proposed model.(To stabilise the GAN training)
* To use the original U-Net you need to change the replace network_dis with network(both networks are provided) in build_model function of class U-Net(in model_unet.py). 
 
### How to run GAN based 3D U-Net?
```
$ cd ../proposed_model
```
* Configure the flags according to your experiment.
* To run training
```
$ python main.py --training
```
* By default it trains Feature Matching GAN based model. To train the bad GAN based model
```
$ python main.py --training --badGAN
```
* To run testing
```
$ python main.py --testing
``` 
* Once you run both the model you can compare the difference between their perfomance when 1 or 2 training images are available. 

## Proposed model architecture
The following shows the model architecture of the proposed model. (Read our paper for further details)

<br>
<img src="https://github.com/arnab39/FewShot_GAN-Unet3D/blob/master/images/Diagram.jpg" width="800"/>
<br>

## Some results from our paper

* Visual comparison of the segmentation by each model, for two test subjects of the iSEG-2017 dataset, when training with different numbers of labeled examples.
<p float="left">
  <img src="https://github.com/arnab39/FewShot_GAN-Unet3D/blob/master/images/Subject9.jpg" width="420" />
  <img src="https://github.com/arnab39/FewShot_GAN-Unet3D/blob/master/images/Subject10.jpg" width="420" /> 
</p>

* Segmentation of Subject 10 of the iSEG-2017 dataset predicted by different GAN-based models, when trained with 2 labeled images. The red box highlights a region in the ground truth where all these models give noticeable differences.
<br>
<img src="https://github.com/arnab39/FewShot_GAN-Unet3D/blob/master/images/ganwar_mod.jpg" width="820"/>
<br>
* More such results can be found in the paper.

## Contact
You can mail me at: sanu.arnab@gmail.com  
If you use this code for your research, please consider citing the original paper:

- Arnab Kumar Mondal, Jose Dolz, Christian Desrosiers. [Few-shot 3D Multi-modal Medical Image Segmentation using Generative Adversarial Learning](https://arxiv.org/abs/1810.12241), submitted in Medical Image Analysis, October 2018.

