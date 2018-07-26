from __future__ import division
import os
import pickle 
import tensorflow as tf
import numpy as np
from sklearn.metrics import f1_score
import nibabel as nib

import sys
sys.path.insert(0, '../preprocess/')
sys.path.insert(0, '../lib/')

from operations import *
from utils import *
from preprocess import *


F = tf.app.flags.FLAGS

g_bns = [batch_norm(name='g_bn{}'.format(i,)) for i in range(4)]
d_bns = [batch_norm(name='u_bn{}'.format(i,)) for i in range(14)]

cur_dir='/home/AP84830/Current_work_3DGANISEG/Experiments/'

def Unet3D( patch, phase, reuse=False):
    with tf.variable_scope('U') as scope:
      if reuse:
        scope.reuse_variables()

      h0 = lrelu(d_bns[0](conv3d(patch, 32, name='u_h0_conv'),phase))
      h1 = lrelu(d_bns[1](conv3d(h0, 32, name='u_h1_conv'),phase))
      p1 = avg_pool3D(h1)

      h2 = lrelu(d_bns[2](conv3d(p1, 64, name='u_h2_conv'),phase))
      h3 = lrelu(d_bns[3](conv3d(h2, 64, name='u_h3_conv'),phase))
      p3 = avg_pool3D(h3)

      h4 = lrelu(d_bns[4](conv3d(p3, 128, name='u_h4_conv'),phase))
      h5 = lrelu(d_bns[5](conv3d(h4, 128, name='u_h5_conv'),phase))
      p5 = avg_pool3D(h5)

      h6 = lrelu(d_bns[6](conv3d(p5, 256, name='u_h6_conv'),phase))
      h7 = lrelu(d_bns[7](conv3d(h6, 256, name='u_h7_conv'),phase))

      up1 = deconv3d(h7,[F.batch_size,8,8,8,256],name='d_up1_deconv')
      up1 = tf.concat([h5,up1],4)
      h8 = lrelu(d_bns[8](conv3d(up1, 128, name='u_h8_conv'),phase))
      h9 = lrelu(d_bns[9](conv3d(h8, 128, name='u_h9_conv'),phase))
      
      up2 = deconv3d(h9,[F.batch_size,16,16,16,128],name='d_up2_deconv')
      up2 = tf.concat([h3,up2],4)
      h10 = lrelu(d_bns[10](conv3d(up2, 64, name='u_h10_conv'),phase))
      h11 = lrelu(d_bns[11](conv3d(h10, 64, name='u_h11_conv'),phase))

      up3 = deconv3d(h11,[F.batch_size,32,32,32,64],name='d_up3_deconv')
      up3 = tf.concat([h1,up3],4)
      h12 = lrelu(d_bns[12](conv3d(up3, 32, name='u_h12_conv'),phase))
      h13 = lrelu(d_bns[13](conv3d(h12, 32, name='u_h13_conv'),phase))

      h14 = conv3d(h13, F.num_classes, name='u_h14_conv')

      return h14,tf.nn.softmax(h14),h6

def discriminator(patch, reuse=False):
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

def generator(z, phase, patch_shape):
    with tf.variable_scope('G') as scope:
      sh1, sh2, sh3, sh4 = int(patch_shape[0]/16), int(patch_shape[0]/8),\
                           int(patch_shape[0]/4), int(patch_shape[0]/2)

      h0 = linear(z, sh1*sh1*sh1*512,'g_h0_lin')
      h0 = tf.reshape(h0, [F.batch_size, sh1, sh1, sh1, 512])
      h0 = relu(g_bns[0](h0,phase))

      h1 = relu(g_bns[1](deconv3d(h0, [F.batch_size,sh2,sh2,sh2,256], 
                                                          name='g_h1_deconv'),phase))

      h2 = relu(g_bns[2](deconv3d(h1, [F.batch_size,sh3,sh3,sh3,128], 
                                                          name='g_h2_deconv'),phase))   

      h3 = relu(g_bns[3](deconv3d(h2, [F.batch_size,sh4,sh4,sh4,64], 
                                                          name='g_h3_deconv'),phase))

      h4 = deconv3d_WN(h3, F.num_mod, name='g_h4_deconv')

      return tf.nn.tanh(h4)

def save_image(direc,i1,i2,i3,i4):
  print(direc)
  img = nib.Nifti1Image(i1, None)
  nib.save(img, os.path.join(direc,'probmapfake.nii.gz'))
  img = nib.Nifti1Image(i2, None)
  nib.save(img, os.path.join(direc,'probmapunlab.nii.gz'))
  img = nib.Nifti1Image(i3, None)
  nib.save(img, os.path.join(direc,'fakeimg.nii.gz'))
  img = nib.Nifti1Image(i4, None)
  nib.save(img, os.path.join(direc,'unlabimg.nii.gz'))


def exp_test_unet(patch_shape,extraction_step,gen_patches):
  
  with tf.Graph().as_default():
    patches_lab = tf.placeholder(tf.float32, [F.batch_size, patch_shape[0], 
                                patch_shape[1], patch_shape[2], F.num_mod], 
                                name='real_images_l')
    patches_unlab = tf.placeholder(tf.float32, [F.batch_size, patch_shape[0], 
                                patch_shape[1], patch_shape[2], F.num_mod], 
                                name='real_images_unl')
    patches_fake = tf.placeholder(tf.float32, [F.batch_size, patch_shape[0], 
                                patch_shape[1], patch_shape[2], F.num_mod], 
                                name='fake_images')

    labels = tf.placeholder(tf.uint8, [F.batch_size, patch_shape[0], patch_shape[1],
                                                         patch_shape[2]], name='image_labels')
    labels_1hot = tf.one_hot(labels, depth=F.num_classes)
    phase = tf.placeholder(tf.bool)

    # To generate samples from noise


    # Forward pass through network

    D_logits_lab, D_probdist, _= Unet3D(patches_lab,phase, reuse=False)
    D_logits_unlab, _, features_unlab\
                     = Unet3D(patches_unlab,phase, reuse=True)
    D_logits_fake, _, features_fake\
                     = Unet3D(patches_fake,phase, reuse=True)

    # Output
    Output = tf.argmax(D_probdist, axis=-1)

    # Weighted cross entropy loss

    class_weights = tf.constant([[2.0, 9.0, 5.0, 7.0]])
    weights = tf.reduce_sum(class_weights * labels_1hot, axis=-1)
    unweighted_losses = tf.nn.softmax_cross_entropy_with_logits_v2(logits=D_logits_lab, labels=labels_1hot)
    weighted_losses = unweighted_losses * weights
    d_loss_lab = tf.reduce_mean(weighted_losses)

    # Unlabelled/GAN loss
    unl_lsexp = tf.reduce_logsumexp(D_logits_unlab,-1)
    fake_lsexp = tf.reduce_logsumexp(D_logits_fake,-1)

    fake_probdist_fake=tf.exp(-1.0*tf.nn.softplus(fake_lsexp))
    fake_probdist_unlab=tf.exp(-1.0*tf.nn.softplus(unl_lsexp))

    true_loss = - 0.5 * tf.reduce_mean(unl_lsexp) + 0.5 * tf.reduce_mean(tf.nn.softplus(unl_lsexp))
    fake_loss = 0.5 * tf.reduce_mean(tf.nn.softplus(fake_lsexp))
    d_loss_unlab = true_loss + fake_loss

    d_loss = d_loss_lab + d_loss_unlab


    saver = tf.train.Saver()
    with tf.Session() as sess:

      print("For the Unet3D model losses:")
      print("################################")
      try:
        load_model(F.checkpoint_unet, sess, saver)
        print(" Checkpoint loaded succesfully!....\n")
      except:
        print(" [!] Checkpoint loading failed!....\n")
        return
      patches_lab_exp, labels_exp = preprocess_dynamic_lab(F.preprocesses_data_directory,
                                    F.num_classes,extraction_step,patch_shape,
                                    num_images_training=1)
      patches_unlab_exp=preprocess_dynamic_unlab(F.preprocesses_data_directory,extraction_step,
                                    patch_shape,num_images_training_unlab=1)

      total_batches = int(patches_lab_exp.shape[0]/F.batch_size)

      print("Total number of Batches: ",total_batches)
      total_lab_loss,total_unlab_loss,total_fake_loss=0.0,0.0,0.0
      sample_z_gen = np.random.uniform(-1, 1, [F.batch_size, F.noise_dim]).astype(np.float32)

      for batch in range(total_batches):
        lab_patches_feed = patches_lab_exp[batch*F.batch_size:(batch+1)*F.batch_size,:,:,:,:]
        unlab_patches_feed = patches_unlab_exp[batch*F.batch_size:(batch+1)*F.batch_size,:,:,:,:]
        fake_patches_feed = gen_patches[batch*F.batch_size:(batch+1)*F.batch_size,:,:,:,:]
        labels_feed = labels_exp[batch*F.batch_size:(batch+1)*F.batch_size,:,:,:]

        lab_loss, unlab_loss, fake_loss_exp = sess.run([d_loss_lab,true_loss,fake_loss], 
                                        feed_dict={patches_lab:lab_patches_feed, labels:labels_feed, 
                                        patches_unlab:unlab_patches_feed, patches_fake:fake_patches_feed, phase:False})
        if batch==(total_batches/2):
          fake_probdist,unlab_probdist= sess.run([fake_probdist_fake,fake_probdist_unlab], 
                feed_dict={ patches_unlab:unlab_patches_feed, patches_fake:fake_patches_feed, phase:False})
          print(np.max(fake_probdist),np.min(fake_probdist),np.mean(fake_probdist),
                    np.max(unlab_probdist),np.min(unlab_probdist),np.mean(unlab_probdist))
          save_image(cur_dir+F.results_dir+'GMT1/',fake_probdist[0],unlab_probdist[0],fake_patches_feed[0,:,:,:,0],unlab_patches_feed[0,:,:,:,0])
          save_image(cur_dir+F.results_dir+'WMT2/',fake_probdist[0],unlab_probdist[0],fake_patches_feed[0,:,:,:,1],unlab_patches_feed[0,:,:,:,1])

        total_lab_loss=total_lab_loss+lab_loss
        total_unlab_loss=total_unlab_loss+unlab_loss
        total_fake_loss=total_fake_loss+fake_loss_exp
        print(("Processed_batch:[%8d/%8d]")%(batch,total_batches))

      print("Avg CE loss:",total_lab_loss/total_batches)
      print("Avg unlab loss:",total_unlab_loss/total_batches)
      print("Avg fake loss:",total_fake_loss/total_batches)

  return




def exp_test_unetGAN(patch_shape,extraction_step):
  
  with tf.Graph().as_default():
    patches_lab = tf.placeholder(tf.float32, [F.batch_size, patch_shape[0], 
                                patch_shape[1], patch_shape[2], F.num_mod], 
                                name='real_images_l')
    patches_unlab = tf.placeholder(tf.float32, [F.batch_size, patch_shape[0], 
                                patch_shape[1], patch_shape[2], F.num_mod], 
                                name='real_images_unl')

    z_gen = tf.placeholder(tf.float32, [F.batch_size, F.noise_dim], name='noise')
    labels = tf.placeholder(tf.uint8, [F.batch_size, patch_shape[0], patch_shape[1],
                                                         patch_shape[2]], name='image_labels')
    print(labels.shape)
    labels_1hot = tf.one_hot(labels, depth=F.num_classes)
    phase = tf.placeholder(tf.bool)

    # To generate samples from noise
    patches_fake = generator(z_gen, phase, patch_shape)


    # Forward pass through network

    D_logits_lab, D_probdist, _= discriminator(patches_lab, reuse=False)
    D_logits_unlab,_, features_unlab\
                     = discriminator(patches_unlab, reuse=True)
    D_logits_fake,_, features_fake\
                     = discriminator(patches_fake, reuse=True)

    # Output
    Output = tf.argmax(D_probdist, axis=-1)

    # Weighted cross entropy loss

    class_weights = tf.constant([[2.0, 9.0, 5.0, 7.0]])
    weights = tf.reduce_sum(class_weights * labels_1hot, axis=-1)
    unweighted_losses = tf.nn.softmax_cross_entropy_with_logits_v2(logits=D_logits_lab, labels=labels_1hot)
    weighted_losses = unweighted_losses * weights
    d_loss_lab = tf.reduce_mean(weighted_losses)

    # Unlabelled/GAN loss
    unl_lsexp = tf.reduce_logsumexp(D_logits_unlab,-1)
    fake_lsexp = tf.reduce_logsumexp(D_logits_fake,-1)

    fake_probdist_fake=tf.exp(-1.0*tf.nn.softplus(fake_lsexp))
    fake_probdist_unlab=tf.exp(-1.0*tf.nn.softplus(unl_lsexp))

    true_loss = - 0.5 * tf.reduce_mean(unl_lsexp) + 0.5 * tf.reduce_mean(tf.nn.softplus(unl_lsexp))
    fake_loss = 0.5 * tf.reduce_mean(tf.nn.softplus(fake_lsexp))
    d_loss_unlab = true_loss + fake_loss

    d_loss = d_loss_lab + d_loss_unlab


    saver = tf.train.Saver()
    with tf.Session() as sess:

      print("For the Unet3D GAN model losses:")
      print("################################")
      try:
        load_model(F.checkpoint_unetGAN, sess, saver)
        print(" Checkpoint loaded succesfully!....\n")
      except:
        print(" [!] Checkpoint loading failed!....\n")
        return
      patches_lab_exp, labels_exp = preprocess_dynamic_lab(F.preprocesses_data_directory,
                                    F.num_classes,extraction_step,patch_shape,
                                    num_images_training=1)
      patches_unlab_exp=preprocess_dynamic_unlab(F.preprocesses_data_directory,extraction_step,
                                    patch_shape,num_images_training_unlab=1)

      total_batches = int(patches_lab_exp.shape[0]/F.batch_size)
      total_fake_patches = np.empty(patches_lab_exp.shape)

      print("Total number of Batches: ",total_batches)
      total_lab_loss,total_unlab_loss,total_fake_loss=0.0,0.0,0.0
      
      for batch in range(total_batches):
        lab_patches_feed = patches_lab_exp[batch*F.batch_size:(batch+1)*F.batch_size,:,:,:,:]
        unlab_patches_feed = patches_unlab_exp[batch*F.batch_size:(batch+1)*F.batch_size,:,:,:,:]
        sample_z_gen = np.random.uniform(-1, 1, [F.batch_size, F.noise_dim]).astype(np.float32)
        labels_feed = labels_exp[batch*F.batch_size:(batch+1)*F.batch_size,:,:,:]

        lab_loss, unlab_loss, fake_loss_exp = sess.run([d_loss_lab,true_loss,fake_loss], 
                                        feed_dict={patches_lab:lab_patches_feed, labels:labels_feed, 
                                        patches_unlab:unlab_patches_feed, z_gen:sample_z_gen, phase:False})
        fake_patch_sample= sess.run(patches_fake, feed_dict={ z_gen:sample_z_gen, phase:False})

        if batch==(total_batches/2):
          fake_probdist,unlab_probdist= sess.run([fake_probdist_fake,fake_probdist_unlab], 
                feed_dict={ patches_unlab:unlab_patches_feed, z_gen:sample_z_gen, phase:False})
          print(np.max(fake_probdist),np.min(fake_probdist),np.mean(fake_probdist),
                    np.max(unlab_probdist),np.min(unlab_probdist),np.mean(unlab_probdist))
          #save_image(cur_dir+F.results_dir+'GMT1/',fake_probdist[0],unlab_probdist[0],fake_patch_sample[0,:,:,:,0],unlab_patches_feed[0,:,:,:,0])
          #save_image(cur_dir+F.results_dir+'WMT2/',fake_probdist[0],unlab_probdist[0],fake_patch_sample[0,:,:,:,1],unlab_patches_feed[0,:,:,:,1])
        total_fake_patches[batch*F.batch_size:(batch+1)*F.batch_size,:,:,:,:]=fake_patch_sample
        total_lab_loss=total_lab_loss+lab_loss
        total_unlab_loss=total_unlab_loss+unlab_loss
        total_fake_loss=total_fake_loss+fake_loss_exp
        print(("Processed_batch:[%8d/%8d]")%(batch,total_batches))

      print("Avg CE loss:",total_lab_loss/total_batches)
      print("Avg unlab loss:",total_unlab_loss/total_batches)
      print("Avg fake loss:",total_fake_loss/total_batches)
      exp_test_unet(patch_shape,extraction_step,gen_patches=total_fake_patches)


  return







