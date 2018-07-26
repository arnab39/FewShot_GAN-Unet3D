from __future__ import division
import os
import pickle 
import tensorflow as tf
import numpy as np
from six.moves import xrange
from sklearn.metrics import f1_score

import sys
sys.path.insert(0, '../preprocess/')
sys.path.insert(0, '../lib/')

from operations import *
from utils import *
import preprocess as pre
import preprocess_data2 as pre2


F = tf.app.flags.FLAGS


"""
Model class
"""
class model(object):
  def __init__(self, sess, patch_shape, extraction_step):
    self.sess = sess
    self.patch_shape = patch_shape
    self.extraction_step = extraction_step
    self.g_bns = [batch_norm(name='g_bn{}'.format(i,)) for i in range(4)]
    self.e_bns = [batch_norm(name='e_bn{}'.format(i,)) for i in range(3)]

  """
  Network model
  Parameters:
  * image - input image for the network
  * reuse - boolean variable to reuse weights
  Returns: logits 
  """
  def discriminator(self, patch, phase, reuse=False):
    with tf.variable_scope('D') as scope:
      if reuse:
        scope.reuse_variables()

      h0 = lrelu(conv3d_WN(patch, 32, name='d_h0_conv'))
      h1 = lrelu(conv3d_WN(h0, 32, name='d_h1_conv'))
      p1 = avg_pool3D(h1)

      h2 = lrelu(conv3d_WN(p1, 64, name='d_h2_conv'))
      h3 = lrelu(conv3d_WN(h2, 64, name='d_h3_conv'))
      p3 = avg_pool3D(h3)

      h4 = lrelu(conv3d_WN(p3, 128, name='d_h4_conv'))
      h5 = lrelu(conv3d_WN(h4, 128, name='d_h5_conv'))
      p5 = avg_pool3D(h5)

      h6 = lrelu(conv3d_WN(p5, 256, name='d_h6_conv'))
      h7 = lrelu(conv3d_WN(h6, 256, name='d_h7_conv'))

      up1 = deconv3d_WN(h7,256,name='d_up1_deconv')
      up1 = tf.concat([h5,up1],4)
      h8 = lrelu(conv3d_WN(up1, 128, name='d_h8_conv'))
      h9 = lrelu(conv3d_WN(h8, 128, name='d_h9_conv'))
      
      up2 = deconv3d_WN(h9,128,name='d_up2_deconv')
      up2 = tf.concat([h3,up2],4)
      h10 = lrelu(conv3d_WN(up2, 64, name='d_h10_conv'))
      h11 = lrelu(conv3d_WN(h10, 64, name='d_h11_conv'))

      up3 = deconv3d_WN(h11,64,name='d_up3_deconv')
      up3 = tf.concat([h1,up3],4)
      h12 = lrelu(conv3d_WN(up3, 32, name='d_h12_conv'))
      h13 = lrelu(conv3d_WN(h12, 32, name='d_h13_conv'))

      h14 = conv3d_WN(h13, F.num_classes,name='d_h14_conv')

      return h14,tf.nn.softmax(h14),h6

  def generator(self, z, phase):
    with tf.variable_scope('G') as scope:
      sh1, sh2, sh3, sh4 = int(self.patch_shape[0]/16), int(self.patch_shape[0]/8),\
                           int(self.patch_shape[0]/4), int(self.patch_shape[0]/2)

      h0 = linear(z, sh1*sh1*sh1*512,'g_h0_lin')
      h0 = tf.reshape(h0, [F.batch_size, sh1, sh1, sh1, 512])
      h0 = relu(self.g_bns[0](h0,phase))

      h1 = relu(self.g_bns[1](deconv3d(h0, [F.batch_size,sh2,sh2,sh2,256], 
                                                          name='g_h1_deconv'),phase))

      h2 = relu(self.g_bns[2](deconv3d(h1, [F.batch_size,sh3,sh3,sh3,128], 
                                                          name='g_h2_deconv'),phase))   

      h3 = relu(self.g_bns[3](deconv3d(h2, [F.batch_size,sh4,sh4,sh4,64], 
                                                          name='g_h3_deconv'),phase))

      h4 = deconv3d_WN(h3, F.num_mod, name='g_h4_deconv')

      return tf.nn.tanh(h4)

  def generator_WN(self, z, phase):
    with tf.variable_scope('G') as scope:
      sh1 = int(self.patch_shape[0]/16)

      h0 = linear_WN(z, sh1*sh1*sh1*512,'g_h0_lin')
      h0 = tf.reshape(h0, [F.batch_size, sh1, sh1, sh1, 512])
      h0 = relu(h0)

      h1 = relu(deconv3d_WN(h0, 256, name='g_h1_deconv'))

      h2 = relu(deconv3d_WN(h0, 128, name='g_h2_deconv'))   

      h3 = relu(deconv3d_WN(h0,  64, name='g_h3_deconv'))

      h4 = deconv3d_WN(h3, F.num_mod, name='g_h4_deconv')

      return tf.nn.tanh(h4)

  def encoder(self, patch, phase):
    with tf.variable_scope('E') as scope:
      h0 = relu(self.e_bns[0](conv3d(patch, 128, 5,5,5, 2,2,2, name='e_h0_conv'),phase))
      h1 = relu(self.e_bns[1](conv3d(h0, 256, 5,5,5, 2,2,2, name='e_h1_conv'),phase))
      h2 = relu(self.e_bns[2](conv3d(h1, 512, 5,5,5, 2,2,2, name='e_h2_conv'),phase))

      h2 = tf.reshape(h2, [h2.shape[0],h2.shape[1]*h2.shape[2]*h2.shape[3]*h2.shape[4]])
      h3 = linear_WN(h2, F.noise_dim*2,'e_h3_lin')
      
      h3 = tf.split(h3,2,1)
      return h3

  def encoder_WN(self, patch, phase):
    with tf.variable_scope('E') as scope:
      h0 = relu(conv3d_WN(patch, 128, [5,5,5], [2,2,2], name='e_h0_conv'))
      h1 = relu(conv3d_WN(h0, 256, [5,5,5], [2,2,2], name='e_h1_conv'))
      h2 = relu(conv3d_WN(h1, 512, [5,5,5], [2,2,2], name='e_h2_conv'))

      h2 = tf.reshape(h2, [h2.shape[0],h2.shape[1]*h2.shape[2]*h2.shape[3]*h2.shape[4]])
      h3 = linear_WN(h2, F.noise_dim*2,'e_h3_lin')
      
      h3 = tf.split(h3,2,1)
      return h3

  """
  Defines the UNET model and losses
  """
  def build_model(self):
    self.patches_lab = tf.placeholder(tf.float32, [F.batch_size, self.patch_shape[0], 
                                self.patch_shape[1], self.patch_shape[2], F.num_mod], name='real_images_l')
    self.patches_unlab = tf.placeholder(tf.float32, [F.batch_size, self.patch_shape[0], 
                                self.patch_shape[1], self.patch_shape[2], F.num_mod], name='real_images_unl')

    self.z_gen = tf.placeholder(tf.float32, [None, F.noise_dim], name='noise')
    self.labels = tf.placeholder(tf.uint8, [F.batch_size, self.patch_shape[0], self.patch_shape[1],
                                                         self.patch_shape[2]], name='image_labels')
    self.labels_1hot = tf.one_hot(self.labels, depth=F.num_classes)
    self.phase = tf.placeholder(tf.bool)

    # To generate samples from noise
    self.patches_fake = self.generator(self.z_gen, self.phase)

    # Mean and standard deviation for variational inference loss
    if F.badGAN:
      self.mu, self.log_sigma = self.encoder(self.patches_fake, self.phase)

    # Forward pass through network
    self.D_logits_lab, self.D_probdist, _= self.discriminator(self.patches_lab, self.phase, reuse=False)
    self.D_logits_unlab, _, self.features_unlab\
                       = self.discriminator(self.patches_unlab, self.phase, reuse=True)
    self.D_logits_fake, _, self.features_fake\
                       = self.discriminator(self.patches_fake, self.phase, reuse=True)


    #Validation Output
    self.Val_output = tf.argmax(self.D_probdist, axis=-1)

    # Weighted cross entropy loss
    if F.data2:
      class_weights = tf.constant([[0.06, 0.7, 0.9, 0.6, 1.1,0.4,1,2,0.5,0.5,0.4,0.7,1.2,2]])
      weights = tf.reduce_sum(class_weights * self.labels_1hot, axis=-1)
      unweighted_losses = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.D_logits_lab, labels=self.labels_1hot)
      weighted_losses = unweighted_losses * weights
      self.d_loss_lab = tf.reduce_mean(weighted_losses)
    else:
      class_weights = tf.constant([[0.33, 1.5, 0.83, 1.33]])
      weights = tf.reduce_sum(class_weights * self.labels_1hot, axis=-1)
      unweighted_losses = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.D_logits_lab, labels=self.labels_1hot)
      weighted_losses = unweighted_losses * weights
      self.d_loss_lab = tf.reduce_mean(weighted_losses)

    # Unlabelled/GAN loss
    self.unl_lsexp = tf.reduce_logsumexp(self.D_logits_unlab,-1)
    self.fake_lsexp = tf.reduce_logsumexp(self.D_logits_fake,-1)

    self.true_loss = - 0.5 * tf.reduce_mean(self.unl_lsexp) + 0.5 * tf.reduce_mean(tf.nn.softplus(self.unl_lsexp))
    self.fake_loss = 0.5 * tf.reduce_mean(tf.nn.softplus(self.fake_lsexp))
    self.d_loss_unlab = self.true_loss + self.fake_loss

    self.d_loss = self.d_loss_lab + self.d_loss_unlab


    self.g_loss_fm = tf.reduce_mean(tf.abs(tf.reduce_mean(self.features_unlab,0) \
                                                  - tf.reduce_mean(self.features_fake,0)))

    
    # Generator Loss via variational inference
    if F.badGAN:
      self.vi_loss = gaussian_nll(self.mu, self.log_sigma, self.z_gen)

    # Total Generator Loss
    if F.badGAN:
      self.g_loss = self.g_loss_fm + F.vi_weight * self.vi_loss
    else:
      self.g_loss = self.g_loss_fm

    t_vars = tf.trainable_variables()
    
    self.d_vars = [var for var in t_vars if 'd_' in var.name]
    self.g_vars = [var for var in t_vars if 'g_' in var.name]
    if F.badGAN:
      self.e_vars = [var for var in t_vars if 'e_' in var.name]

    self.saver = tf.train.Saver()


  """
  Train function
  Defines learning rates and optimizers.
  Performs Network update and saves the losses
  """
  def train(self):
    if F.data2:
      data = pre2.dataset_badGAN(extraction_step=self.extraction_step, 
              number_images_training=F.number_train_images,batch_size=F.batch_size, 
              patch_shape=self.patch_shape,number_unlab_images_training=F.number_train_unlab_images)
    else:
      data = pre.dataset_badGAN(num_classes=F.num_classes,extraction_step=self.extraction_step, 
              number_images_training=F.number_train_images,batch_size=F.batch_size, 
              patch_shape=self.patch_shape,number_unlab_images_training=F.number_train_unlab_images)


    d_optim = tf.train.AdamOptimizer(F.learning_rate_D, beta1=F.beta1D)\
                  .minimize(self.d_loss,var_list=self.d_vars)
    g_optim = tf.train.AdamOptimizer(F.learning_rate_G, beta1=F.beta1G)\
                  .minimize(self.g_loss,var_list=self.g_vars)
    if F.badGAN:
      e_optim = tf.train.AdamOptimizer(F.learning_rate_E, beta1=F.beta1E)\
                  .minimize(self.g_loss,var_list=self.e_vars)

    tf.global_variables_initializer().run()

    # Load checkpoints if required
    if F.load_chkpt:
      try:
        load_model(F.checkpoint_dir, self.sess, self.saver)
        print("\n [*] Checkpoint loaded succesfully!")
      except:
        print("\n [!] Checkpoint loading failed!")
    else:
      print("\n [*] Checkpoint load not required.")

    if F.data2:
      patches_val, labels_val_patch, labels_val = pre2.preprocess_dynamic_lab(F.preprocesses_data2_directory,
                                    self.extraction_step,self.patch_shape,
                                    F.number_train_images,validating=F.training,
                                    testing=F.testing,num_images_testing=F.number_test_images)
    else:
      patches_val, labels_val_patch, labels_val = pre.preprocess_dynamic_lab(F.preprocesses_data_directory,
                                    F.num_classes,self.extraction_step,self.patch_shape,
                                    F.number_train_images,validating=F.training,
                                    testing=F.testing,num_images_testing=F.number_test_images)
    
    predictions_val = np.zeros((patches_val.shape[0],self.patch_shape[0],self.patch_shape[1],
                                                              self.patch_shape[2]),dtype="uint8")
    max_par=0.0
    max_loss=100
    for epoch in xrange(int(F.epoch)):
      idx = 0
      batch_iter_train = data.batch_train()
      total_val_loss=0
      total_train_loss_CE=0
      total_train_loss_UL=0
      total_train_loss_FK=0
      total_gen_FMloss =0

      for patches_lab, patches_unlab, labels in batch_iter_train:
        # Network update
        sample_z_gen = np.random.uniform(-1, 1, [F.batch_size, F.noise_dim]).astype(np.float32)

        _ = self.sess.run(d_optim,feed_dict={self.patches_lab:patches_lab,self.patches_unlab:patches_unlab,
                                    self.z_gen:sample_z_gen,self.labels:labels, self.phase: True})  

        if F.badGAN:
          _,_ = self.sess.run([e_optim,g_optim],feed_dict={self.patches_unlab:patches_unlab, self.z_gen:sample_z_gen,
                                                                  self.z_gen:sample_z_gen,self.phase: True})
        else:
          _ = self.sess.run(g_optim,feed_dict={self.patches_unlab:patches_unlab, self.z_gen:sample_z_gen,
                                                                  self.z_gen:sample_z_gen,self.phase: True})
        feed_dict = {self.patches_lab:patches_lab,self.patches_unlab:patches_unlab,
                                    self.z_gen:sample_z_gen,self.labels:labels, self.phase: True} 

        # Evaluate loss for plotting/printing purposes   
        d_loss_lab = self.d_loss_lab.eval(feed_dict)
        d_loss_unlab_true = self.true_loss.eval(feed_dict)
        d_loss_unlab_fake = self.fake_loss.eval(feed_dict)
        g_loss_fm = self.g_loss_fm.eval(feed_dict)
        if F.badGAN:
          vi_loss = self.vi_loss.eval(feed_dict)

        total_train_loss_CE=total_train_loss_CE+d_loss_lab
        total_train_loss_UL=total_train_loss_UL+d_loss_unlab_true
        total_train_loss_FK=total_train_loss_FK+d_loss_unlab_fake
        total_gen_FMloss=total_gen_FMloss+g_loss_fm

        idx += 1
        if F.badGAN:
          print(("Epoch:[%2d] [%4d/%4d] Labeled loss:%.2e Unlabeled loss true:%.2e Unlabeled loss fake:%.2e Generator FM loss:%.8f Generator VI loss:%.8f\n")%
                      (epoch, idx,data.num_batches,d_loss_lab,d_loss_unlab_true,d_loss_unlab_fake,g_loss_fm,vi_loss))
        else:
          print(("Epoch:[%2d] [%4d/%4d] Labeled loss:%.2e Unlabeled loss true:%.2e Unlabeled loss fake:%.2e Generator loss:%.8f \n")%
                      (epoch, idx,data.num_batches,d_loss_lab,d_loss_unlab_true,d_loss_unlab_fake,g_loss_fm))
      
      # Save model
      save_model(F.checkpoint_dir, self.sess, self.saver)
      avg_train_loss_CE=total_train_loss_CE/(idx*1.0)
      avg_train_loss_UL=total_train_loss_UL/(idx*1.0)
      avg_train_loss_FK=total_train_loss_FK/(idx*1.0)
      avg_gen_FMloss=total_gen_FMloss/(idx*1.0)
      
      print('\n\n')

      total_batches = int(patches_val.shape[0]/F.batch_size)
      print("Total number of Patches: ",patches_val.shape[0])
      print("Total number of Batches: ",total_batches)

      for batch in range(total_batches):
        patches_feed = patches_val[batch*F.batch_size:(batch+1)*F.batch_size,:,:,:,:]
        labels_feed = labels_val_patch[batch*F.batch_size:(batch+1)*F.batch_size,:,:,:]
        feed_dict={self.patches_lab:patches_feed,
                                              self.labels:labels_feed, self.phase:False}
        preds = self.Val_output.eval(feed_dict)
        val_loss = self.d_loss_lab.eval(feed_dict)

        predictions_val[batch*F.batch_size:(batch+1)*F.batch_size,:,:,:]=preds
        print(("Validated Patch:[%8d/%8d]")%(batch,total_batches))
        total_val_loss=total_val_loss+val_loss

      avg_val_loss=total_val_loss/(total_batches*1.0)

      if(max_loss>avg_val_loss):
        max_loss=avg_val_loss
        save_model(F.best_checkpoint_dir_val, self.sess, self.saver)
        print("Best checkpoint updated from validation loss.")


      print("All validation patches Predicted")

      print("Shape of predictions_val, min and max:",predictions_val.shape,np.min(predictions_val),
                                                                        np.max(predictions_val))

      if F.data2:
        val_image_pred = recompose3D_overlap(predictions_val,160, 127, 145, 8, 9, 9)
      else:
        val_image_pred = recompose3D_overlap(predictions_val,144, 192, 256, 8, 8, 8)
      val_image_pred = val_image_pred.astype('uint8')

      print("Shape of Predicted Output Groundtruth Images:",val_image_pred.shape,
                                                np.unique(val_image_pred),
                                                np.unique(labels_val),
                                                np.mean(val_image_pred),np.mean(labels_val))

      if F.data2:
        pred2d=np.reshape(val_image_pred,(val_image_pred.shape[0]*160*127*145))
        lab2d=np.reshape(labels_val,(labels_val.shape[0]*160*127*145))
        F1_score = f1_score(lab2d, pred2d,[0,1,2,3,4,5,6,7,8,9,10,11,12,13]
                                                          ,average=None)
        print("Validation Dice Coefficient.... ")
        print("Background:",F1_score[0])

        print("Left-Thalamus:",F1_score[1])
        print("Left-Caudate:",F1_score[2])
        print("Left-Putamen:",F1_score[3])
        print("Left-Pallidum :",F1_score[4])
        print("Left-Hippocampus:",F1_score[6])
        print("Left-Amygdala:",F1_score[7])

        print("Right-Thalamus:",F1_score[8])
        print("Right-Caudate:",F1_score[9])
        print("Right-Putamen:",F1_score[10])
        print("Right-Pallidum:",F1_score[11])
        print("Right-Hippocampus:",F1_score[12])
        print("Right-Amygdala:",F1_score[13])

        print("Brain-Stem:",F1_score[5])
      else:
        pred2d=np.reshape(val_image_pred,(val_image_pred.shape[0]*144*192*256))
        lab2d=np.reshape(labels_val,(labels_val.shape[0]*144*192*256))
        F1_score = f1_score(lab2d, pred2d,[0,1,2,3],average=None)
        print("Validation Dice Coefficient.... ")
        print("Background:",F1_score[0])
        print("CSF:",F1_score[1])
        print("GM:",F1_score[2])
        print("WM:",F1_score[3])

        if(max_par<(F1_score[2]+F1_score[3])):
          max_par=(F1_score[2]+F1_score[3])
          save_model(F.best_checkpoint_dir, self.sess, self.saver)
          print("Best checkpoint got updated from validation results.")
        


      print("Average Validation Loss:",avg_val_loss)
      with open('Val_loss_GAN.txt', 'a') as f:
        f.write('%.2e \n' % avg_val_loss)
      with open('Train_loss_CE.txt', 'a') as f:
        f.write('%.2e \n' % avg_train_loss_CE)
      with open('Train_loss_UL.txt', 'a') as f:
        f.write('%.2e \n' % avg_train_loss_UL)
      with open('Train_loss_FK.txt', 'a') as f:
        f.write('%.2e \n' % avg_train_loss_FK)
      with open('Train_loss_FM.txt', 'a') as f:
        f.write('%.2e \n' % avg_gen_FMloss)
    return