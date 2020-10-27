
## How to use the Tensorflow code?
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
* You can run standard 3D U-Net & our proposed model(Feature matching GAN, weighted Feature Matching GAN and bad GAN) with this code and compare their performance.

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
* By default it trains Feature Matching GAN based model. To train the weighted Feature Matching GAN based model
```
$ python main.py --training --use_weighted_fm
```
* To train the bad GAN based model
```
$ python main.py --training --badGAN
```
* To run testing
```
$ python main.py --testing
```
* Once you run both the model you can compare the difference between their perfomance when 1 or 2 training images are available.
