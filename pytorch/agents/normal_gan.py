import numpy as np
import os

from tqdm import tqdm
import shutil
import random

import torch
from torch import nn
from torch.backends import cudnn
from torch.autograd import Variable
import torchvision.utils as vutils
from sklearn.metrics import f1_score
import torch.nn.functional as F

from agents.base import BaseAgent
from graphs.models.generator import Generator
from graphs.models.discriminator import Discriminator
from datasets.dataloader import FewShot_Dataset

from tensorboardX import SummaryWriter
from utils.metrics import AverageMeter, AverageMeterList
from utils.misc import print_cuda_statistics
from utils.recompose import recompose3D_overlap

cudnn.benchmark = True

class NormalGAN_Model(BaseAgent):
    def __init__(self, config):
        super().__init__(config)

        self.generator = Generator(self.config)
        self.discriminator = Discriminator(self.config) # Segmenation Network
        if self.config.phase == 'testing':
            self.testloader = FewShot_Dataset(self.config, "testing")
        else:
            self.trainloader = FewShot_Dataset(self.config, "training")
            self.valloader = FewShot_Dataset(self.config, "validating")

        # optimizer
        self.g_optim = torch.optim.Adam(self.generator.parameters(), lr=self.config.learning_rate_G, betas=(self.config.beta1G, self.config.beta2G))
        self.d_optim = torch.optim.Adam(self.discriminator.parameters(), lr=self.config.learning_rate_D, betas=(self.config.beta1D, self.config.beta2D))
        # counter initialization
        self.current_epoch = 0
        self.best_validation_dice = 0
        self.current_iteration = 0

        self.is_cuda = torch.cuda.is_available()
        if self.is_cuda and not self.config.cuda:
            self.logger.info("WARNING: You have a CUDA device, so you should probably enable CUDA")

        self.cuda = self.is_cuda & self.config.cuda

        if self.cuda:
            self.generator = self.generator.cuda()
            self.discriminator = self.discriminator.cuda()

        class_weights = torch.tensor([[0.33, 1.5, 0.83, 1.33]])
        if self.cuda:
            class_weights = torch.FloatTensor(class_weights).cuda()
        self.criterion = nn.CrossEntropyLoss(class_weights)

        # set the manual seed for torch
        if not self.config.seed:
            self.manual_seed = random.randint(1, 10000)
        else:
            self.manual_seed = self.config.seed
        self.logger.info ("seed: %d" , self.manual_seed)
        random.seed(self.manual_seed)
        if self.cuda:
            self.device = torch.device("cuda")
            torch.cuda.set_device(self.config.gpu_device)
            torch.cuda.manual_seed_all(self.manual_seed)
            self.logger.info("Program will run on *****GPU-CUDA***** ")
            print_cuda_statistics()
        else:
            self.device = torch.device("cpu")
            torch.manual_seed(self.manual_seed)
            self.logger.info("Program will run on *****CPU***** ")

        if(self.config.load_chkpt == True):
            self.load_checkpoint(self.config.phase)

    def load_checkpoint(self, phase):
        try:
            if phase == 'training':
                filename = os.path.join(self.config.checkpoint_dir, 'checkpoint.pth.tar')
            elif phase == 'testing':
                filename = os.path.join(self.config.checkpoint_dir, 'model_best.pth.tar')
            self.logger.info("Loading checkpoint '{}'".format(filename))
            checkpoint = torch.load(filename)

            self.current_epoch = checkpoint['epoch']
            self.generator.load_state_dict(checkpoint['generator'])
            self.discriminator.load_state_dict(checkpoint['discriminator'])
            self.manual_seed = checkpoint['manual_seed']

            self.logger.info("Checkpoint loaded successfully from '{}' at (epoch {})\n"
                  .format(self.config.checkpoint_dir, checkpoint['epoch']))

        except OSError as e:
            self.logger.info("No checkpoint exists from '{}'. Skipping...".format(self.config.checkpoint_dir))
            self.logger.info("**First time to train**")

    def save_checkpoint(self, is_best=False):
        file_name="checkpoint.pth.tar"
        state = {
            'epoch': self.current_epoch,
            'generator': self.generator.state_dict(),
            'discriminator': self.discriminator.state_dict(),
            'manual_seed': self.manual_seed
        }
        torch.save(state, os.path.join(self.config.checkpoint_dir , file_name))
        if is_best:
            print("SAVING BEST CHECKPOINT !!!\n")
            shutil.copyfile(self.config.checkpoint_dir + file_name,
                            self.config.checkpoint_dir + 'model_best.pth.tar')

    def run(self):
        try:
            if self.config.phase == 'training':
                self.train()
            if self.config.phase == 'testing':
                self.load_checkpoint(self.config.phase)
                self.test()
        except KeyboardInterrupt:
            self.logger.info("You have entered CTRL+C.. Wait to finalize")

    def train(self):
        for epoch in range(self.current_epoch, self.config.epochs):
            self.current_epoch = epoch
            self.current_iteration = 0
            self.train_one_epoch()
            self.save_checkpoint()
            if(self.current_epoch % self.config.validation_every_epoch == 0):
                self.validate()

    def train_one_epoch(self):
        # initialize tqdm batch
        tqdm_batch = tqdm(self.trainloader.loader, total=self.trainloader.num_iterations, desc="epoch-{}-".format(self.current_epoch))

        self.generator.train()
        self.discriminator.train()
        epoch_loss_gen = AverageMeter()
        epoch_loss_dis = AverageMeter()
        epoch_loss_ce = AverageMeter()
        epoch_loss_unlab = AverageMeter()
        epoch_loss_fake = AverageMeter()

        for curr_it, (patches_lab, patches_unlab, labels) in enumerate(tqdm_batch):
            #y = torch.full((self.batch_size,), self.real_label)
            if self.cuda:
                patches_lab = patches_lab.cuda()
                patches_unlab = patches_unlab.cuda()
                labels = labels.cuda()

            patches_lab = Variable(patches_lab)
            patches_unlab = Variable(patches_unlab.float())
            labels = Variable(labels).long()

            noise_vector = torch.tensor(np.random.uniform(-1, 1, [self.config.batch_size, self.config.noise_dim])).float()
            if self.cuda:
                noise_vector = noise_vector.cuda()
            patches_fake = self.generator(noise_vector)

            ## Discriminator
            # Supervised loss
            lab_output, lab_output_sofmax = self.discriminator(patches_lab)
            lab_loss = self.criterion(lab_output, labels)

            unlab_output, unlab_output_softmax = self.discriminator(patches_unlab)
            fake_output, fake_output_softmax = self.discriminator(patches_fake.detach())

            # Unlabeled Loss and Fake loss
            unlab_lsp = torch.logsumexp(unlab_output, dim=1)
            fake_lsp = torch.logsumexp(fake_output, dim=1)
            unlab_loss = - 0.5 * torch.mean(unlab_lsp) + 0.5 * torch.mean(F.softplus(unlab_lsp, 1))
            fake_loss = 0.5 * torch.mean(F.softplus(fake_lsp,1))
            discriminator_loss = lab_loss + unlab_loss + fake_loss

            self.d_optim.zero_grad()
            discriminator_loss.backward()
            self.d_optim.step()

            ## Generator
            fake_output, fake_output_softmax = self.discriminator(patches_fake)
            fake_lsp = torch.logsumexp(fake_output, dim=1)
            generator_loss = - 0.5 * torch.mean(fake_lsp) + 0.5 * torch.mean(F.softplus(fake_lsp, 1))

            self.g_optim.zero_grad()
            self.d_optim.zero_grad()
            generator_loss.backward()
            self.g_optim.step()

            epoch_loss_gen.update(generator_loss.item())
            epoch_loss_dis.update(discriminator_loss.item())
            epoch_loss_ce.update(lab_loss.item())
            epoch_loss_unlab.update(unlab_loss.item())
            epoch_loss_fake.update(fake_loss.item())
            self.current_iteration += 1

            print("Epoch: {0}, Iteration: {1}/{2}, Gen loss: {3}, Dis loss: {4} :: CE loss {5}, Unlab loss: {6}, Fake loss: {7}".format(
                                self.current_epoch, self.current_iteration,\
                                self.trainloader.num_iterations, generator_loss.item(), discriminator_loss.item(),\
                                lab_loss.item(), unlab_loss.item(), fake_loss.item()))

        tqdm_batch.close()

        self.logger.info("Training at epoch-" + str(self.current_epoch) + " | " +\
         " Generator loss: " + str(epoch_loss_gen.val) +\
          " Discriminator loss: " + str(epoch_loss_dis.val) +\
           " CE loss: " + str(epoch_loss_ce.val) + " Unlab loss: " + str(epoch_loss_unlab.val) + " Fake loss: " + str(epoch_loss_fake.val))

    def validate(self):
        self.discriminator.eval()

        prediction_image = torch.zeros([self.valloader.dataset.label.shape[0], self.config.patch_shape[0],\
                                        self.config.patch_shape[1], self.config.patch_shape[2]])
        whole_vol = self.valloader.dataset.whole_vol
        for batch_number, (patches, label, _) in enumerate(self.valloader.loader):
            patches = patches.cuda()
            _, batch_prediction_softmax = self.discriminator(patches)
            batch_prediction = torch.argmax(batch_prediction_softmax, dim=1).cpu()
            prediction_image[batch_number*self.config.batch_size:(batch_number+1)*self.config.batch_size,:,:,:] = batch_prediction

            print("Validating.. [{0}/{1}]".format(batch_number, self.valloader.num_iterations))

        vol_shape_x, vol_shape_y, vol_shape_z = self.config.volume_shape
        prediction_image = prediction_image.numpy()
        val_image_pred = recompose3D_overlap(prediction_image, vol_shape_x, vol_shape_y, vol_shape_z, self.config.extraction_step[0],
                                                    self.config.extraction_step[1],self.config.extraction_step[2])
        val_image_pred = val_image_pred.astype('uint8')
        pred2d=np.reshape(val_image_pred,(val_image_pred.shape[0]*vol_shape_x*vol_shape_y*vol_shape_z))
        lab2d=np.reshape(whole_vol,(whole_vol.shape[0]*vol_shape_x*vol_shape_y*vol_shape_z))

        classes = list(range(0, self.config.num_classes))
        F1_score = f1_score(lab2d, pred2d, classes, average=None)
        print("Validation Dice Coefficient.... ")
        print("Background:",F1_score[0])
        print("CSF:",F1_score[1])
        print("GM:",F1_score[2])
        print("WM:",F1_score[3])

        current_validation_dice = F1_score[2] + F1_score[3]
        if(self.best_validation_dice < current_validation_dice):
            self.best_validation_dice = current_validation_dice
            self.save_checkpoint(is_best = True)

    def test(self):
        self.discriminator.eval()

        prediction_image = torch.zeros([self.testloader.dataset.patches.shape[0], self.config.patch_shape[0],\
                                        self.config.patch_shape[1], self.config.patch_shape[2]])
        whole_vol = self.testloader.dataset.whole_vol
        for batch_number, (patches, _) in enumerate(self.testloader.loader):
            patches = patches.cuda()
            _, batch_prediction_softmax = self.discriminator(patches)
            batch_prediction = torch.argmax(batch_prediction_softmax, dim=1).cpu()
            prediction_image[batch_number*self.config.batch_size:(batch_number+1)*self.config.batch_size,:,:,:] = batch_prediction

            print("Testing.. [{0}/{1}]".format(batch_number, self.testloader.num_iterations))

        vol_shape_x, vol_shape_y, vol_shape_z = self.config.volume_shape
        prediction_image = prediction_image.numpy()
        test_image_pred = recompose3D_overlap(prediction_image, vol_shape_x, vol_shape_y, vol_shape_z, self.config.extraction_step[0],
                                                    self.config.extraction_step[1],self.config.extraction_step[2])
        test_image_pred = test_image_pred.astype('uint8')
        pred2d=np.reshape(test_image_pred,(test_image_pred.shape[0]*vol_shape_x*vol_shape_y*vol_shape_z))
        lab2d=np.reshape(whole_vol,(whole_vol.shape[0]*vol_shape_x*vol_shape_y*vol_shape_z))

        classes = list(range(0, self.config.num_classes))
        F1_score = f1_score(lab2d, pred2d, classes, average=None)
        print("Test Dice Coefficient.... ")
        print("Background:",F1_score[0])
        print("CSF:",F1_score[1])
        print("GM:",F1_score[2])
        print("WM:",F1_score[3])


    def finalize(self):
        """
        Finalize all the operations of the 2 Main classes of the process the operator and the data loader
        :return:
        """
        self.logger.info("Please wait while finalizing the operation.. Thank you")
        self.save_checkpoint()
        # self.summary_writer.export_scalars_to_json("{}all_scalars.json".format(self.config.summary_dir))
        # self.summary_writer.close()
        self.testloader.finalize()
