import os
import glob
import nibabel as nib
import numpy as np
import SimpleITK as sitk
import shutil
from nipype.interfaces.ants import N4BiasFieldCorrection
from sklearn.feature_extraction.image import extract_patches as sk_extract_patches
from sklearn.utils import shuffle
import tensorflow as tf
import scipy.misc
#import pdb

F = tf.app.flags.FLAGS


seed = 7
np.random.seed(seed)

all_modalities={'T1','T2'}


def get_filename(set_name, case_idx, input_name, loc):
    pattern = '{0}/{1}/{3}/subject-{2}-{3}.nii'
    return pattern.format(loc, set_name, case_idx, input_name)

def get_set_name(case_idx):
    return 'Training' if case_idx < 11 else 'Testing'

def read_data(case_idx, input_name, loc):
    set_name = get_set_name(case_idx)

    image_path = get_filename(set_name, case_idx, input_name, loc)
    print(image_path)

    return nib.load(image_path)

def read_vol(case_idx, input_name, dir):
    image_data = read_data(case_idx, input_name, dir)
    return image_data.get_data()

def correct_bias(in_file, out_file):
    correct = N4BiasFieldCorrection()
    correct.inputs.input_image = in_file
    correct.inputs.output_image = out_file
    done = correct.run()
    return done.outputs.output_image

def normalise(case_idx, input_name, in_dir, out_dir,copy=False):
	set_name = get_set_name(case_idx)
	image_in_path = get_filename(set_name, case_idx, input_name, in_dir)
	image_out_path = get_filename(set_name, case_idx, input_name, out_dir)
	if copy:
		shutil.copy(image_in_path, image_out_path)
	else:
		correct_bias(image_in_path, image_out_path)
	print(image_in_path + " done.")



"""
To extract patches from a 3D image
"""
def extract_patches(volume, patch_shape, extraction_step,datype='float32'):
  patch_h, patch_w, patch_d = patch_shape[0], patch_shape[1], patch_shape[2]
  stride_h, stride_w, stride_d = extraction_step[0], extraction_step[1], extraction_step[2]
  img_h, img_w, img_d = volume.shape[0],volume.shape[1],volume.shape[2]
  N_patches_h = (img_h-patch_h)//stride_h+1
  N_patches_w = (img_w-patch_w)//stride_w+1
  N_patches_d = (img_d-patch_d)//stride_d+1
  N_patches_img = N_patches_h * N_patches_w * N_patches_d
  raw_patch_martrix = np.zeros((N_patches_img,patch_h,patch_w,patch_d),dtype=datype)
  k=0

  #iterator over all the patches
  for h in range((img_h-patch_h)//stride_h+1):
    for w in range((img_w-patch_w)//stride_w+1):
      for d in range((img_d-patch_d)//stride_d+1):
        raw_patch_martrix[k]=volume[h*stride_h:(h*stride_h)+patch_h,\
                            			w*stride_w:(w*stride_w)+patch_w,\
                              				d*stride_d:(d*stride_d)+patch_d]
        k+=1
  assert(k==N_patches_img)
  return raw_patch_martrix

"""
To extract labeled patches from array of 3D labeled images
"""
def get_patches_lab(T1_vols, T2_vols, label_vols, extraction_step,
                    patch_shape,validating,testing,num_images_training):
    patch_shape_1d=patch_shape[0]
    # Extract patches from input volumes and ground truth
    x = np.zeros((0, patch_shape_1d, patch_shape_1d, patch_shape_1d, 2),dtype="float32")
    y = np.zeros((0, patch_shape_1d, patch_shape_1d, patch_shape_1d),dtype="uint8")
    for idx in range(len(T1_vols)) :
        y_length = len(y)
        if testing:
            print(("Extracting Patches from Image %2d ....")%(num_images_training+idx+2))
        elif validating:
            print(("Extracting Patches from Image %2d ....")%(num_images_training+idx+1))
        else:
            print(("Extracting Patches from Image %2d ....")%(1+idx))
        label_patches = extract_patches(label_vols[idx], patch_shape, extraction_step,
        														datype="uint8")

        # Select only those who are important for processing
        if testing or validating:
            valid_idxs = np.where(np.sum(label_patches, axis=(1, 2, 3)) != -1)
        else:
            valid_idxs = np.where(np.count_nonzero(label_patches, axis=(1, 2, 3)) > 6000)

        # Filtering extracted patches
        label_patches = label_patches[valid_idxs]

        x = np.vstack((x, np.zeros((len(label_patches), patch_shape_1d, 
                                                patch_shape_1d, patch_shape_1d, 2),dtype="float32")))
        y = np.vstack((y, np.zeros((len(label_patches), patch_shape_1d, 
                                                patch_shape_1d, patch_shape_1d),dtype="uint8")))

        y[y_length:, :, :, :] = label_patches

        # Sampling strategy: reject samples which labels are mostly 0 and have less than 6000 nonzero elements
        T1_train = extract_patches(T1_vols[idx], patch_shape, extraction_step,datype="float32")
        x[y_length:, :, :, :, 0] = T1_train[valid_idxs]
        
        # Sampling strategy: reject samples which labels are mostly 0 and have less than 6000 nonzero elements
        T2_train = extract_patches(T2_vols[idx], patch_shape, extraction_step,datype="float32")
        x[y_length:, :, :, :, 1] = T2_train[valid_idxs]
    return x, y

"""
To preprocess the labeled training data
"""
def preprocess_dynamic_lab(dir,num_classes, extraction_step,patch_shape,num_images_training=2,
                                validating=False,testing=False,num_images_testing=7):
    if testing:
        print("Testing")
        r1=num_images_training+2
        r2=num_images_training+num_images_testing+2
        c=num_images_training+1
        T1_vols = np.empty((num_images_testing, 144, 192, 256),dtype="float32")
        T2_vols = np.empty((num_images_testing, 144, 192, 256),dtype="float32")
        label_vols = np.empty((num_images_testing, 144, 192, 256),dtype="uint8")
    elif validating:
        print("Validating")
        r1=num_images_training+1
        r2=num_images_training+2
        c=num_images_training
        T1_vols = np.empty((1, 144, 192, 256),dtype="float32")
        T2_vols = np.empty((1, 144, 192, 256),dtype="float32")
        label_vols = np.empty((1, 144, 192, 256),dtype="uint8")
    else:
        print("Training")
        r1=1
        r2=num_images_training+1
        c=0
        T1_vols = np.empty((num_images_training, 144, 192, 256),dtype="float32")
        T2_vols = np.empty((num_images_training, 144, 192, 256),dtype="float32")
        label_vols = np.empty((num_images_training, 144, 192, 256),dtype="uint8")
    for case_idx in range(r1, r2) :
        print(case_idx)
        T1_vols[(case_idx-c-1), :, :, :] = read_vol(case_idx, 'T1', dir)
        T2_vols[(case_idx-c-1), :, :, :] = read_vol(case_idx, 'T2', dir)
        label_vols[(case_idx-c-1), :, :, :] = read_vol(case_idx, 'label', dir)
    T1_mean = T1_vols.mean()
    T1_std = T1_vols.std()
    T1_vols = (T1_vols - T1_mean) / T1_std
    T2_mean = T2_vols.mean()
    T2_std = T2_vols.std()
    T2_vols = (T2_vols - T2_mean) / T2_std

    for i in range(T1_vols.shape[0]):
        T1_vols[i] = ((T1_vols[i] - np.min(T1_vols[i])) / 
                                    (np.max(T1_vols[i])-np.min(T1_vols[i])))*255
    for i in range(T2_vols.shape[0]):
        T2_vols[i] = ((T2_vols[i] - np.min(T2_vols[i])) / 
                                    (np.max(T2_vols[i])-np.min(T2_vols[i])))*255    
    T1_vols = T1_vols/127.5 -1.
    T2_vols = T2_vols/127.5 -1.
    x,y=get_patches_lab(T1_vols,T2_vols,label_vols,extraction_step,patch_shape,validating=validating,
                                testing=testing,num_images_training=num_images_training)
    print("Total Extracted Labelled Patches Shape:",x.shape,y.shape)
    if testing:
        return x, label_vols
    elif validating:
        return x, y, label_vols
    else:
        return x, y


"""
To extract labeled patches from array of 3D ulabeled images
"""
def get_patches_unlab(T1_vols, T2_vols, extraction_step,patch_shape,dir):
    patch_shape_1d=patch_shape[0]
    # Extract patches from input volumes and ground truth
    label_ref= np.empty((1, 144, 192, 256),dtype="uint8")
    x = np.zeros((0, patch_shape_1d, patch_shape_1d, patch_shape_1d, 2))
    label_ref = read_vol(1, 'label', dir)
    for idx in range(len(T1_vols)) :

        x_length = len(x)
        print(("Processing the Image %2d ....")%(idx+11))
        label_patches = extract_patches(label_ref, patch_shape, extraction_step)

        # Select only those who are important for processing
        # Sampling strategy: reject samples which labels are mostly 0 and have less than 6000 nonzero elements
        valid_idxs = np.where(np.count_nonzero(label_patches, axis=(1, 2, 3)) > 6000)

        label_patches = label_patches[valid_idxs]
        x = np.vstack((x, np.zeros((len(label_patches), patch_shape_1d, 
                                            patch_shape_1d, patch_shape_1d, 2))))

        T1_train = extract_patches(T1_vols[idx], patch_shape, extraction_step,datype="float32")
        x[x_length:, :, :, :, 0] = T1_train[valid_idxs]
        
        T2_train = extract_patches(T2_vols[idx], patch_shape, extraction_step,datype="float32")
        x[x_length:, :, :, :, 1] = T2_train[valid_idxs]
    return x

"""
To preprocess the unlabeled training data
"""
def preprocess_dynamic_unlab( dir,extraction_step,patch_shape,num_images_training_unlab):
    T1_vols = np.empty((num_images_training_unlab, 144, 192, 256),dtype="float32")
    T2_vols = np.empty((num_images_training_unlab, 144, 192, 256),dtype="float32")
    for case_idx in range(11, 11+num_images_training_unlab) :
        T1_vols[(case_idx - 11), :, :, :] = read_vol(case_idx, 'T1', dir)
        T2_vols[(case_idx - 11), :, :, :] = read_vol(case_idx, 'T2', dir)
        #print(read_vol(case_idx, 'T2', dir).shape)
    T1_mean = T1_vols.mean()
    T1_std = T1_vols.std()
    T1_vols = (T1_vols - T1_mean) / T1_std
    T2_mean = T2_vols.mean()
    T2_std = T2_vols.std()
    T2_vols = (T2_vols - T2_mean) / T2_std
    for i in range(T1_vols.shape[0]):
        T1_vols[i] = ((T1_vols[i] - np.min(T1_vols[i])) / 
                                        (np.max(T1_vols[i])-np.min(T1_vols[i])))*255
    for i in range(T2_vols.shape[0]):
        T2_vols[i] = ((T2_vols[i] - np.min(T2_vols[i])) / 
                                        (np.max(T2_vols[i])-np.min(T2_vols[i])))*255  
    T1_vols = T1_vols/127.5 -1.
    T2_vols = T2_vols/127.5 -1.
    x=get_patches_unlab(T1_vols, T2_vols, extraction_step, patch_shape,dir)
    print("Total Extracted Unlabelled Patches Shape:",x.shape)
    return x


def preprocess_static( org_dir, prepro_dir, dataset="labeled", overwrite=False):
    if not os.path.exists(prepro_dir):
        os.makedirs(prepro_dir)
    for subject_folder in glob.glob(os.path.join(org_dir, "*", "*")):
        if os.path.isdir(subject_folder):
            subject = os.path.basename(subject_folder)
            new_subject_folder = os.path.join(prepro_dir, 
                os.path.basename(os.path.dirname(subject_folder)),subject)
            if not os.path.exists(new_subject_folder) or overwrite:
                if not os.path.exists(new_subject_folder):
                    os.makedirs(new_subject_folder)
    if(dataset=="labeled"):
        for case_idx in range(1, 11) :
            normalise(case_idx, 'T1',org_dir,prepro_dir)
            normalise(case_idx, 'T2',org_dir,prepro_dir)
            normalise(case_idx, 'label',org_dir,prepro_dir,
                           copy=True)
    else:
        for case_idx in range(11, 24) :
            normalise(case_idx, 'T1',org_dir,prepro_dir)
            normalise(case_idx, 'T2',org_dir,prepro_dir)
            
"""
dataset class for preparing training data of basic U-Net
"""
class dataset(object):
  def __init__(self,num_classes, extraction_step, number_images_training, batch_size, patch_shape,data_directory):
    # Extract labelled and unlabelled patches
    self.batch_size=batch_size
    self.data_lab, self.label = preprocess_dynamic_lab(
                                        data_directory,num_classes,extraction_step,
                                        patch_shape,number_images_training)

    self.data_lab, self.label = shuffle(self.data_lab, 
                                                    self.label, random_state=0)
    print("Data_shape:",self.data_lab.shape)
    print("Data lab max and min:",np.max(self.data_lab),np.min(self.data_lab))
    print("Label unique:",np.unique(self.label))

  def batch_train(self):
    self.num_batches = len(self.data_lab) // self.batch_size
    for i in range(self.num_batches):
      yield self.data_lab[i*self.batch_size:(i+1)*self.batch_size],\
             self.label[i*self.batch_size:(i+1)*self.batch_size]


"""
dataset_badGAN class for preparing data of our model
"""
class dataset_badGAN(object):
  def __init__(self,num_classes, extraction_step, number_images_training, batch_size, 
                    patch_shape, number_unlab_images_training,data_directory):
    # Extract labelled and unlabelled patches,
    self.batch_size=batch_size
    self.data_lab, self.label = preprocess_dynamic_lab(
                                data_directory,num_classes,extraction_step,
                                        patch_shape,number_images_training)

    self.data_lab, self.label = shuffle(self.data_lab, self.label, random_state=0)
    self.data_unlab = preprocess_dynamic_unlab(data_directory,extraction_step,
                                                patch_shape, number_unlab_images_training)
    self.data_unlab = shuffle(self.data_unlab, random_state=0)

    # If training, repeat labelled data to make its size equal to unlabelled data
    factor = len(self.data_unlab) // len(self.data_lab)
    print("Factor for labeled images:",factor)
    rem = len(self.data_unlab)%len(self.data_lab)
    temp = self.data_lab[:rem]
    self.data_lab = np.concatenate((np.repeat(self.data_lab, factor, axis=0), temp), axis=0)
    temp = self.label[:rem]
    self.label = np.concatenate((np.repeat(self.label, factor, axis=0), temp), axis=0)
    assert(self.data_lab.shape == self.data_unlab.shape)
    print("Data_shape:",self.data_lab.shape,self.data_unlab.shape)
    print("Data lab max and min:",np.max(self.data_lab),np.min(self.data_lab))
    print("Data unlab max and min:",np.max(self.data_unlab),np.min(self.data_unlab))
    print("Label unique:",np.unique(self.label))

  def batch_train(self):
    self.num_batches = len(self.data_lab) // self.batch_size
    for i in range(self.num_batches):
      yield self.data_lab[i*self.batch_size:(i+1)*self.batch_size],\
             self.data_unlab[i*self.batch_size:(i+1)*self.batch_size],\
                self.label[i*self.batch_size:(i+1)*self.batch_size]


#preprocess_static( actual_data_directory, preprocesses_data_directory, overwrite=True)