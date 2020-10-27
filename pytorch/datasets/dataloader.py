"""
CelebA Dataloader implementation, used in DCGAN
"""
import numpy as np

import imageio

import torch
import torchvision.transforms as v_transforms
import torchvision.utils as v_utils
import torchvision.datasets as v_datasets

from torch.utils.data import DataLoader, TensorDataset, Dataset
from utils.preprocess import *

# For UNET based supervised method
class Supervised_Dataloader(Dataset):
    def __init__(self, config, phase):
        assert phase in ['training', 'validating', 'testing']
        self.phase = phase
        if phase == 'training':
            self.patches, self.label = preprocess_dynamic_lab(config.data_directory, config.seed, config.num_classes,\
                                                                config.extraction_step, config.patch_shape,\
                                                                config.number_images_training)
            print("Label unique:",np.unique(self.label))
        if phase == 'validating':
            self.patches, self.label, self.whole_vol = preprocess_dynamic_lab(config.data_directory, config.seed, config.num_classes,\
                                                                config.extraction_step, config.patch_shape,\
                                                                config.number_images_training, validating=True)
            print("Label unique:",np.unique(self.label))

        if phase == 'testing':
            self.patches, self.whole_vol = preprocess_dynamic_lab(config.data_directory, config.seed, config.num_classes,\
                                                                config.extraction_step, config.patch_shape,\
                                                                config.number_images_training, testing=True)

        # self.data_lab, self.label = shuffle(self.data_lab, self.label, random_state=0)
        print("Data_shape:",self.patches.shape)
        print("Data lab max and min:",np.max(self.patches),np.min(self.patches))

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, index):
        if self.phase == 'training':
            return self.patches[index], self.label[index]
        if self.phase == 'validating':
            return self.patches[index], self.label[index], self.whole_vol
        if self.phase == 'testing':
            return self.patches[index], self.whole_vol


class Supervised_Dataset:
    def __init__(self, config, phase):
        self.config = config

        self.dataset = Supervised_Dataloader(config, phase)
        if phase == 'training':
            shuffle = True
        else:
            shuffle = False
        self.loader = DataLoader(self.dataset,
                                 batch_size=config.batch_size,
                                 shuffle=shuffle,
                                 num_workers=config.data_loader_workers,
                                 pin_memory=config.pin_memory)

        self.num_iterations = len(self.loader)

    def finalize(self):
        pass

# For GAN based few-shot method
class FewShot_Dataloader(Dataset):
    def __init__(self, config, phase):

        assert phase in ['training', 'validating', 'testing']
        self.phase = phase
        if phase == 'training':
            self.patches, self.label = preprocess_dynamic_lab(config.data_directory, config.seed, config.num_classes,\
                                                                config.extraction_step, config.patch_shape,\
                                                                config.number_images_training)
            print("Label unique:",np.unique(self.label))
            self.patches_unlab = preprocess_dynamic_unlab(config.data_directory, config.extraction_step,
                                                        config.patch_shape, config.number_unlab_images_training)
            self.patches_unlab = shuffle(self.patches_unlab, random_state=0)
            factor = len(self.patches_unlab) // len(self.patches)
            print("Factor for labeled images:",factor)
            rem = len(self.patches_unlab)%len(self.patches)
            temp = self.patches[:rem]
            self.patches = np.concatenate((np.repeat(self.patches, factor, axis=0), temp), axis=0)
            temp = self.label[:rem]
            self.label = np.concatenate((np.repeat(self.label, factor, axis=0), temp), axis=0)
            assert(self.patches.shape == self.patches_unlab.shape)
            print("Data_shape:",self.patches.shape,self.patches_unlab.shape)
            print("Data lab max and min:",np.max(self.patches),np.min(self.patches))
            print("Data unlab max and min:",np.max(self.patches_unlab),np.min(self.patches_unlab))
            print("Label unique:",np.unique(self.label))

        if phase == 'validating':
            self.patches, self.label, self.whole_vol = preprocess_dynamic_lab(config.data_directory, config.seed, config.num_classes,\
                                                                config.extraction_step, config.patch_shape,\
                                                                config.number_images_training, validating=True)
            print("Label unique:",np.unique(self.label))

        if phase == 'testing':
            self.patches, self.whole_vol = preprocess_dynamic_lab(config.data_directory, config.seed, config.num_classes,\
                                                                config.extraction_step, config.patch_shape,\
                                                                config.number_images_training, testing=True)



    def __len__(self):
        return len(self.patches)

    def __getitem__(self, index):
        if self.phase == 'training':
            return self.patches[index], self.patches_unlab[index], self.label[index]
        if self.phase == 'validating':
            return self.patches[index], self.label[index], self.whole_vol
        if self.phase == 'testing':
            return self.patches[index], self.whole_vol

class FewShot_Dataset:
    def __init__(self, config, phase):
        self.config = config

        self.dataset = FewShot_Dataloader(config, phase)
        if phase == 'training':
            shuffle = True
        else:
            shuffle = False
        self.loader = DataLoader(self.dataset,
                                 batch_size=config.batch_size,
                                 shuffle=shuffle,
                                 num_workers=config.data_loader_workers,
                                 pin_memory=config.pin_memory)

        self.num_iterations = len(self.loader)

    def finalize(self):
        pass
