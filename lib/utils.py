import os
import numpy as np
import tensorflow as tf
#import pdb

F = tf.app.flags.FLAGS


"""
Save tensorflow model
Parameters:
* checkpoint_dir - name of the directory where model is to be saved
* sess - current tensorflow session
* saver - tensorflow saver
"""
def save_model(checkpoint_dir, sess, saver):
  model_name = "model.ckpt"
  if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
  saver.save(sess, os.path.join(checkpoint_dir, model_name))


"""
Load tensorflow model
Parameters:
* checkpoint_dir - name of the directory where model is to be loaded from
* sess - current tensorflow session
* saver - tensorflow saver
Returns: True if the model loaded successfully, else False
"""
def load_model(checkpoint_dir, sess, saver):
  print(" [*] Reading checkpoints...")
  ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
  if ckpt and ckpt.model_checkpoint_path:
    ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
    saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))
    return True
  else:
    return False

"""
To recompose an array of 3D images from patches
"""
def recompose3D_overlap(preds, img_h, img_w, img_d, stride_h, stride_w, stride_d):
  patch_h = preds.shape[1]
  patch_w = preds.shape[2]
  patch_d = preds.shape[3]
  N_patches_h = (img_h-patch_h)//stride_h+1
  N_patches_w = (img_w-patch_w)//stride_w+1
  N_patches_d = (img_d-patch_d)//stride_d+1
  N_patches_img = N_patches_h * N_patches_w * N_patches_d
  print("N_patches_h: " ,N_patches_h)
  print("N_patches_w: " ,N_patches_w)
  print("N_patches_d: " ,N_patches_d)
  print("N_patches_img: ",N_patches_img)
  assert(preds.shape[0]%N_patches_img==0)
  N_full_imgs = preds.shape[0]//N_patches_img
  print("According to the dimension inserted, there are " \
          +str(N_full_imgs) +" full images (of " +str(img_h)+"x" +str(img_w)+"x" +str(img_d) +" each)")
  # itialize to zero mega array with sum of Probabilities
  raw_pred_martrix = np.zeros((N_full_imgs,img_h,img_w,img_d)) 
  raw_sum = np.zeros((N_full_imgs,img_h,img_w,img_d))
  final_matrix = np.zeros((N_full_imgs,img_h,img_w,img_d),dtype='uint16')

  k = 0 
  # iterator over all the patches
  for i in range(N_full_imgs):
    for h in range((img_h-patch_h)//stride_h+1):
      for w in range((img_w-patch_w)//stride_w+1):
        for d in range((img_d-patch_d)//stride_d+1):
          raw_pred_martrix[i,h*stride_h:(h*stride_h)+patch_h,\
                                w*stride_w:(w*stride_w)+patch_w,\
                                  d*stride_d:(d*stride_d)+patch_d]+=preds[k]
          raw_sum[i,h*stride_h:(h*stride_h)+patch_h,\
                          w*stride_w:(w*stride_w)+patch_w,\
                            d*stride_d:(d*stride_d)+patch_d]+=1.0
          k+=1
  assert(k==preds.shape[0])
  #To check for non zero sum matrix
  assert(np.min(raw_sum)>=1.0)
  final_matrix = np.around(raw_pred_martrix/raw_sum)
  return final_matrix