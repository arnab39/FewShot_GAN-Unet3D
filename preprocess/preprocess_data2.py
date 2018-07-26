import os
import glob
import nibabel as nib
import numpy as np
import SimpleITK as sitk
import shutil
from nipype.interfaces.ants import N4BiasFieldCorrection
from sklearn.utils import shuffle
import tensorflow as tf
import scipy.misc
import pdb

preprocesses_data_directory='/home/AP84830/Current_work_3DGANISEG/data/IBSR_Shakeri'
class_mapper = {0:0, 2:0, 3:0, 4:0, 5:0, 7:0, 8:0, 10:1, 11:2, 12:3, 13:4,
                14:0,15:0,16:5,17:6,18:7,24:0,26:0, 28:0,29:0,30:0,41:0, 42:0,
                43:0, 44:0, 46:0, 47:0, 48:8, 49:8, 50:9, 51:10, 52:11, 53:12,
                54:13, 58:0,60:0,61:0,62:0,72:0}

def get_filename(case_idx, input_name, loc):
    if input_name=='orig_nii':
        if(case_idx<10):
            pattern = '{0}/{1}/IBSR_0{2}_ana_strip.nii'
        else:
            pattern = '{0}/{1}/IBSR_{2}_ana_strip.nii'
    else:
        if(case_idx<10):
            pattern = '{0}/{1}/IBSR_0{2}_seg_ana.nii'
        else:
            pattern = '{0}/{1}/IBSR_{2}_seg_ana.nii'
    return pattern.format(loc, input_name, case_idx)

def get_set_name(case_idx):
    return 'Training' if case_idx < 11 else 'Testing'

def read_data(case_idx, input_name, loc):
    image_path = get_filename(case_idx, input_name, loc)
    print(image_path)

    return nib.load(image_path)

def read_vol(case_idx, input_name, dir):
    image_data = read_data(case_idx, input_name, dir)
    return image_data.get_data()

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


def get_patches_lab(image_vols, label_vols, extraction_step,
                    patch_shape,validating,testing,num_images_training):
    patch_shape_1d=patch_shape[0]
    # Extract patches from input volumes and ground truth
    x = np.zeros((0, patch_shape_1d, patch_shape_1d, patch_shape_1d,1),dtype="float32")
    y = np.zeros((0, patch_shape_1d, patch_shape_1d, patch_shape_1d),dtype="uint8")
    for idx in range(len(image_vols)) :
        y_length = len(y)
        if testing:
            print(("Extracting Patches from Image %2d ....")%(num_images_training+idx+2))
        elif validating:
            print(("Extracting Patches from Image %2d ....")%(num_images_training+idx+1))
        else:
            print(("Extracting Patches from Image %2d ....")%(1+idx))
        label_patches = extract_patches(label_vols[idx], patch_shape, extraction_step,datype="uint8")
        print(len(label_patches))                                                       
        # Select only those who are important for processing
        if testing or validating:
            valid_idxs = np.where(np.sum(label_patches, axis=(1, 2, 3)) != -1)
        else:
            valid_idxs = np.where(np.count_nonzero(label_patches, axis=(1, 2, 3)) > 6000)
            print("training")
        # Filtering extracted patches
        label_patches = label_patches[valid_idxs]


        x = np.vstack((x, np.zeros((len(label_patches), patch_shape_1d, 
                                                patch_shape_1d, patch_shape_1d, 1),dtype="float32")))
        y = np.vstack((y, np.zeros((len(label_patches), patch_shape_1d, 
                                                patch_shape_1d, patch_shape_1d),dtype="uint8")))

        y[y_length:, :, :, :] = label_patches

        # Sampling strategy: reject samples which labels are only zeros
        image_train = extract_patches(image_vols[idx], patch_shape, extraction_step,
                                                                datype="float32")
        x[y_length:, :, :, :, 0] = image_train[valid_idxs]

    print(x.shape,y.shape)
        
    return x, y

def get_patches_unlab(image_vols, extraction_step,patch_shape):
    patch_shape_1d=patch_shape[0]
    # Extract patches from input volumes and ground truth
    label_ref= np.zeros((1, 158, 123, 147),dtype="uint8")
    x = np.zeros((0, patch_shape_1d, patch_shape_1d, patch_shape_1d, 1))
    label_ref[:,:,:,1:146] = read_vol(1, 'Seg', preprocesses_data_directory)
    for class_idx in class_mapper:
        label_ref[label_ref == class_idx] = class_mapper[class_idx]
    for idx in range(len(image_vols)) :

        x_length = len(x)
        print(("Processing the Image %2d ....")%(idx+11))
        label_patches = extract_patches(label_ref[0], patch_shape, extraction_step)

        # Select only those who are important for processing
        valid_idxs = np.where(np.count_nonzero(label_patches, axis=(1, 2, 3)) > 6000)
        label_patches = label_patches[valid_idxs]
        x = np.vstack((x, np.zeros((len(label_patches), patch_shape_1d, 
                                            patch_shape_1d, patch_shape_1d, 1))))

        # Sampling strategy: reject samples which labels are only zeros
        image_train = extract_patches(image_vols[idx], patch_shape, extraction_step, datype="float32")
        x[x_length:, :, :, :, 0] = image_train[valid_idxs]
        
    return x

def preprocess_dynamic_lab(dir, extraction_step,patch_shape,num_images_training=2,
                                validating=False,testing=False,num_images_testing=7):
    if testing:
        print("Testing")
        r1=num_images_training+2
        r2=num_images_training+num_images_testing+2
        c=num_images_training+1
        image_vols = np.zeros((num_images_testing, 160, 127, 145),dtype="float32")
        label_vols = np.zeros((num_images_testing, 160, 127, 145),dtype="uint8")
    elif validating:
        print("Validating")
        r1=num_images_training+1
        r2=num_images_training+2
        c=num_images_training
        image_vols = np.zeros((1, 160, 127, 145),dtype="float32")
        label_vols = np.zeros((1, 160, 127, 145),dtype="uint8")
    else:
        print("Training")
        r1=1
        r2=num_images_training+1
        c=0
        image_vols = np.zeros((num_images_training, 160, 127, 145),dtype="float32")
        label_vols = np.zeros((num_images_training, 160, 127, 145),dtype="uint8")
    for case_idx in range(r1, r2) :
        print(case_idx)
        image_vols[(case_idx-c-1), 1:159, 2:125, :] = read_vol(case_idx, 'orig_nii', dir)
        label_vols[(case_idx-c-1), 1:159, 2:125, :] = read_vol(case_idx, 'Seg', dir)
    image_mean = image_vols.mean()
    image_std = image_vols.std()
    image_vols = (image_vols - image_mean) / image_std

    for i in range(image_vols.shape[0]):
        image_vols[i] = ((image_vols[i] - np.min(image_vols[i])) / 
                                    (np.max(image_vols[i])-np.min(image_vols[i])))*255
    
    image_vols = image_vols/127.5 -1.
    print(image_vols.shape)
    print(np.unique(label_vols))
    unique, counts = np.unique(label_vols, return_counts=True)
    a=dict(zip(unique, counts))
    print(a)

    for class_idx in class_mapper:
        label_vols[label_vols == class_idx] = class_mapper[class_idx]

    unique, counts = np.unique(label_vols, return_counts=True)
    a=dict(zip(unique, counts))
    print(a)
    x,y=get_patches_lab(image_vols,label_vols,extraction_step,patch_shape,validating=validating,
                                testing=testing,num_images_training=num_images_training)
    print("Total Extracted Labeled Patches Shape:",x.shape,y.shape)
    if testing:
        return x, label_vols
    elif validating:
        return x, y, label_vols
    else:
        return x, y

def preprocess_dynamic_unlab( dir,extraction_step,patch_shape,num_images_training_unlab):
    image_vols = np.zeros((num_images_training_unlab, 160, 127, 145),dtype="float32")
    for case_idx in range(11, 11+num_images_training_unlab) :
        image_vols[(case_idx - 11),1:159, 2:125, :] = read_vol(case_idx, 'orig_nii', dir)
    image_mean = image_vols.mean()
    image_std = image_vols.std()
    image_vols = (image_vols - image_mean) / image_std
    for i in range(image_vols.shape[0]):
        image_vols[i] = ((image_vols[i] - np.min(image_vols[i])) / 
                                        (np.max(image_vols[i])-np.min(image_vols[i])))*255
    image_vols = image_vols/127.5 -1.
    np.unique(image_vols)
    x=get_patches_unlab(image_vols, extraction_step, patch_shape)
    print("Total Extracted Unlabeled Patches Shape:",x.shape)
    return x

class dataset(object):
  def __init__(self, extraction_step, number_images_training, batch_size, patch_shape):
    # Extract labelled and unlabelled patches
    self.batch_size=batch_size
    self.data_lab, self.label = preprocess_dynamic_lab(
                                    preprocesses_data_directory,extraction_step,
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



class dataset_badGAN(object):
  def __init__(self, extraction_step, number_images_training, batch_size, 
                    patch_shape, number_unlab_images_training):
    # Extract labelled and unlabelled patches
    self.batch_size=batch_size
    self.data_lab, self.label = preprocess_dynamic_lab(
                                preprocesses_data_directory,extraction_step,
                                        patch_shape,number_images_training)

    self.data_lab, self.label = shuffle(self.data_lab, self.label, random_state=0)
    self.data_unlab = preprocess_dynamic_unlab(preprocesses_data_directory,extraction_step,
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
'''
x,y=preprocess_dynamic_lab(preprocesses_data_directory,(8,9,9),(64,64,64),2)
unique, counts = np.unique(y, return_counts=True)
a=dict(zip(unique, counts))
print(a)
'''

