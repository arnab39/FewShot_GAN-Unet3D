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
from preprocess import * 


F = tf.app.flags.FLAGS


"""
Model class
"""
class UNET(object):
  def __init__(self, sess, patch_shape, extraction_step):
    self.sess = sess
    self.patch_shape = patch_shape
    self.extraction_step = extraction_step
    self.d_bns = [batch_norm(name='u_bn{}'.format(i,)) for i in range(14)]


  def network_dis(self, patch, reuse=False):
    """
    Parameters:
    * patch - input image for the network
    * reuse - boolean variable to reuse weights
    Returns: 
    * logits
    * softmax of logits
    * features extracted from encoding path 
    """
    with tf.variable_scope('U') as scope:
      if reuse:
        scope.reuse_variables()

      h0 = lrelu(conv3d_WN(patch, 32, name='u_h0_conv'))
      h1 = lrelu(conv3d_WN(h0, 32, name='u_h1_conv'))
      p1 = avg_pool3D(h1)

      h2 = lrelu(conv3d_WN(p1, 64, name='u_h2_conv'))
      h3 = lrelu(conv3d_WN(h2, 64, name='u_h3_conv'))
      p3 = avg_pool3D(h3)

      h4 = lrelu(conv3d_WN(p3, 128, name='u_h4_conv'))
      h5 = lrelu(conv3d_WN(h4, 128, name='u_h5_conv'))
      p5 = avg_pool3D(h5)

      h6 = lrelu(conv3d_WN(p5, 256, name='u_h6_conv'))
      h7 = lrelu(conv3d_WN(h6, 256, name='u_h7_conv'))

      up1 = deconv3d_WN(h7,256,name='u_up1_deconv')
      up1 = tf.concat([h5,up1],4)
      h8 = lrelu(conv3d_WN(up1, 128, name='u_h8_conv'))
      h9 = lrelu(conv3d_WN(h8, 128, name='u_h9_conv'))
      
      up2 = deconv3d_WN(h9,128,name='u_up2_deconv')
      up2 = tf.concat([h3,up2],4)
      h10 = lrelu(conv3d_WN(up2, 64, name='u_h10_conv'))
      h11 = lrelu(conv3d_WN(h10, 64, name='u_h11_conv'))

      up3 = deconv3d_WN(h11,64,name='u_up3_deconv')
      up3 = tf.concat([h1,up3],4)
      h12 = lrelu(conv3d_WN(up3, 32, name='u_h12_conv'))
      h13 = lrelu(conv3d_WN(h12, 32, name='u_h13_conv'))

      h14 = conv3d_WN(h13, F.num_classes,name='u_h14_conv')

      return h14,tf.nn.softmax(h14)

  """
  Network model
  Parameters:
  * image - input image for the network
  * reuse - boolean variable to reuse weights
  Returns: logits 
  """
  def network(self, patch, phase,pshape, reuse=False):
    with tf.variable_scope('U') as scope:
      if reuse:
        scope.reuse_variables()

      sh1, sh2, sh3 = int(pshape[0]/4),\
                           int(pshape[0]/2), int(pshape[0])

      h0 = relu(self.d_bns[0](conv3d(patch, 32, name='u_h0_conv'),phase))
      h1 = relu(self.d_bns[1](conv3d(h0, 32, name='u_h1_conv'),phase))
      p1 = max_pool3D(h1)

      h2 = relu(self.d_bns[2](conv3d(p1, 64, name='u_h2_conv'),phase))
      h3 = relu(self.d_bns[3](conv3d(h2, 64, name='u_h3_conv'),phase))
      p3 = max_pool3D(h3)

      h4 = relu(self.d_bns[4](conv3d(p3, 128, name='u_h4_conv'),phase))
      h5 = relu(self.d_bns[5](conv3d(h4, 128, name='u_h5_conv'),phase))
      p5 = max_pool3D(h5)

      h6 = relu(self.d_bns[6](conv3d(p5, 256, name='u_h6_conv'),phase))
      h7 = relu(self.d_bns[7](conv3d(h6, 256, name='u_h7_conv'),phase))

      up1 = deconv3d(h7,[F.batch_size,sh1,sh1,sh1,256],name='d_up1_deconv')
      up1 = tf.concat([h5,up1],4)
      h8 = relu(self.d_bns[8](conv3d(up1, 128, name='u_h8_conv'),phase))
      h9 = relu(self.d_bns[9](conv3d(h8, 128, name='u_h9_conv'),phase))
      
      up2 = deconv3d(h9,[F.batch_size,sh2,sh2,sh2,128],name='d_up2_deconv')
      up2 = tf.concat([h3,up2],4)
      h10 = relu(self.d_bns[10](conv3d(up2, 64, name='u_h10_conv'),phase))
      h11 = relu(self.d_bns[11](conv3d(h10, 64, name='u_h11_conv'),phase))

      up3 = deconv3d(h11,[F.batch_size,sh3,sh3,sh3,64],name='d_up3_deconv')
      up3 = tf.concat([h1,up3],4)
      h12 = relu(self.d_bns[12](conv3d(up3, 32, name='u_h12_conv'),phase))
      h13 = relu(self.d_bns[13](conv3d(h12, 32, name='u_h13_conv'),phase))

      h14 = conv3d(h13, F.num_classes, name='u_h14_conv')

      return h14,tf.nn.softmax(h14)


  """
  Defines the UNET model and losses
  """
  def build_model(self):
    self.patches_labeled = tf.placeholder(tf.float32, [F.batch_size, self.patch_shape[0], 
                                self.patch_shape[1], self.patch_shape[2], F.num_mod], name='real_images_l')

    self.labels = tf.placeholder(tf.uint8, [F.batch_size, self.patch_shape[0], self.patch_shape[1],
                                                         self.patch_shape[2]], name='image_labels')
    self.labels_1hot = tf.one_hot(self.labels, depth=F.num_classes)
    self.phase = tf.placeholder(tf.bool)

    # Forward pass through network
    # To use original 3D U-Net use ***network*** function and don't forget to change the testing file
    #self._logits_labeled, self._probdist = self.network(self.patches_labeled, self.phase, self.patch_shape, reuse=False)
    self._logits_labeled, self._probdist = self.network_dis(self.patches_labeled, reuse=False)

    #Validation Output
    self.Val_output = tf.argmax(self._probdist, axis=-1)

    # Weighted ross entropy loss
    # Weights of different class are: Background- 0.33, CSF- 1.5, GM- 0.83, WM- 1.33
    class_weights = tf.constant([[0.33, 1.5, 0.83, 1.33]])
    weights = tf.reduce_sum(class_weights * self.labels_1hot, axis=-1)
    unweighted_losses = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self._logits_labeled, labels=self.labels_1hot)
    weighted_losses = unweighted_losses * weights
    self.u_loss = tf.reduce_mean(weighted_losses)

    #define the trainable variables
    t_vars = tf.trainable_variables()
    self.u_vars = [var for var in t_vars if 'u_' in var.name]

    self.saver = tf.train.Saver()


  """
  Train function
  Defines learning rates and optimizers.
  Performs Network update and saves the losses
  """
  def train(self):
    data = dataset(num_classes=F.num_classes,extraction_step=self.extraction_step, number_images_training=
                    F.number_train_images,batch_size=F.batch_size, patch_shape=self.patch_shape,data_directory=F.data_directory)
    
    global_step = tf.placeholder(tf.int32, [], name="global_step_epochs")


    # Optimizer operation
    _optim = tf.train.AdamOptimizer(F.learning_rate_, beta1=F.beta1).minimize(self.u_loss,
                                                                       var_list=self.u_vars)

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


    patches_val, labels_val_patch, labels_val = preprocess_dynamic_lab(F.data_directory,
                                    F.num_classes,self.extraction_step,self.patch_shape,
                                    F.number_train_images,validating=F.training,
                                    testing=F.testing,num_images_testing=F.number_test_images)
    
    predictions_val = np.zeros((patches_val.shape[0],self.patch_shape[0],self.patch_shape[1],
                                                        self.patch_shape[2]),dtype='uint8')
    max_par=0.0
    max_loss=100
    for epoch in xrange(int(F.epoch)):
      idx = 0
      batch_iter_train = data.batch_train()
      total_val_loss=0
      total_train_loss=0

      for patches_lab, labels in batch_iter_train:
        # Network update
        feed_dict = {self.patches_labeled:patches_lab,self.labels:labels,
                                        self.phase:True, global_step: epoch}
        _optim.run(feed_dict)

        # Evaluate loss for plotting/printing purposes
        feed_dict = {self.patches_labeled:patches_lab,self.labels:labels,
                                        self.phase:True, global_step: epoch}    
        u_loss = self.u_loss.eval(feed_dict)
        total_train_loss=total_train_loss+u_loss

        idx += 1
        print(("Epoch:[%2d] [%4d/%4d] Loss:%.2e \n")%(epoch, idx,data.num_batches,u_loss))


      # Save model
      save_model(F.checkpoint_dir, self.sess, self.saver)


      # Validation runs every third epoch
      if epoch%3==0:
        avg_train_loss=total_train_loss/(idx*1.0)
        print('\n\n')

        total_batches = int(patches_val.shape[0]/F.batch_size)
        print("Total number of Patches: ",patches_val.shape[0])
        print("Total number of Batches: ",total_batches)

        for batch in range(total_batches):
          patches_feed = patches_val[batch*F.batch_size:(batch+1)*F.batch_size,:,:,:,:]
          labels_feed = labels_val_patch[batch*F.batch_size:(batch+1)*F.batch_size,:,:,:]
          feed_dict={self.patches_labeled:patches_feed,
                                                self.labels:labels_feed, self.phase:False}
          preds = self.Val_output.eval(feed_dict)
          val_loss = self.u_loss.eval(feed_dict)

          predictions_val[batch*F.batch_size:(batch+1)*F.batch_size,:,:,:]=preds
          print(("Validated Patch:[%8d/%8d]")%(batch,total_batches))
          total_val_loss=total_val_loss+val_loss

        avg_val_loss=total_val_loss/(total_batches*1.0)

        print("All validation patches Predicted")


        print("Shape of predictions_val, min and max:",predictions_val.shape,np.min(predictions_val),
                                                                          np.max(predictions_val))


        val_image_pred = recompose3D_overlap(predictions_val,144, 192, 256, self.extraction_step[0],
                                                        self.extraction_step[1],self.extraction_step[2])
        val_image_pred = val_image_pred.astype('uint8')

        print("Shape of Predicted Output Groundtruth Images:",val_image_pred.shape,
                                                  np.unique(val_image_pred),
                                                  np.unique(labels_val),
                                                  np.mean(val_image_pred),np.mean(labels_val))

        
        pred2d=np.reshape(val_image_pred,(val_image_pred.shape[0]*144*192*256))
        lab2d=np.reshape(labels_val,(labels_val.shape[0]*144*192*256))
        F1_score = f1_score(lab2d, pred2d,[0,1,2,3],average=None)
        print("Validation Dice Coefficient.... ")
        print("Background:",F1_score[0])
        print("CSF:",F1_score[1])
        print("GM:",F1_score[2])
        print("WM:",F1_score[3])

        # To save the best model based on validation
        if(max_par<(F1_score[2]+F1_score[3])):
          max_par=(F1_score[2]+F1_score[3])
          save_model(F.best_checkpoint_dir, self.sess, self.saver)
          print("Best checkpoint got updated from validation results.")

        # To save losses for plotting 
        '''
        print("Average Validation Loss:",avg_val_loss)
        print("Average Training Loss",avg_train_loss)
        with open('Val_loss.txt', 'a') as f:
          f.write('%.2e \n' % avg_val_loss)
        with open('Train_loss.txt', 'a') as f:
          f.write('%.2e \n' % avg_train_loss)
        '''

    return