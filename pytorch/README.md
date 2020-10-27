[Note that results with PyTorch implementation may vary slightly from the paper.]
## How to use the PyTorch code?
* Download the iSEG-2017 data and place it in data folder. (Visit [this](http://iseg2017.web.unc.edu/download/) link to download the data. You need to register for the challenge.)
* Do the preprocessing as mentioned in the tensorflow readme. (optional)
* To run the code for your own dataset, change utils/preprocess.py according to the dataset. Originally it is adapted for iSEG-2017 dataset.

### How to run 3D U-Net?
* Configure the unet.json file inside config directory according to your experiment.
* To run training
```
$ python main.py configs/unet.json
```
* This will train your model and save the best checkpoint according to your validation performance inside checkpoint_dir mentioned in the json file.
* You can also resume training from saved checkpoint by setting the load_chkpt as True and running the same command as in Step 2.
* To run testing, configure the phase as "testing" in json file and run the command mentioned in Step 2.
* This version of code only compute dice coefficient to evaluate the testing performance.
* Note that the U-Net used here is modified according to the U-Net used in proposed model.(To stabilise the GAN training)

### How to run GAN based 3D U-Net?
* Configure the fmgan.json or badgan.json file inside config directory according to your experiment.
* To run training (example: Feature Matching GAN)
```
$ python main.py configs/fmgan.json
```
* This will train your model and save the best checkpoint according to your validation performance inside checkpoint_dir mentioned in the json file.
* You can also resume training from saved checkpoint by setting the load_chkpt as True and running the same command as in Step 2.
* To run testing, configure the phase as "testing" in json file and run the command mentioned in Step 2.
* This version of code only compute dice coefficient to evaluate the testing performance.

This PyTorch code is built over the template from this [repo](https://github.com/moemen95/Pytorch-Project-Template), for more details about the structure, kindly refer to it.
