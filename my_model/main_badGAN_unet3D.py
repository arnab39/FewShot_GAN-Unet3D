import os
import pprint
import tensorflow as tf

from model_badGAN_unet3D import model
from badGAN_unet3Dtesting import *


# Define flags
flags = tf.app.flags
flags.DEFINE_integer("epoch", 100000, "Number of training epochs (default: 100000)")
flags.DEFINE_float("learning_rate_D", 0.0001, "Learning rate of Adam optimizer for Discriminator (default: 0.0001)")
flags.DEFINE_float("learning_rate_G", 0.0001, "Learning rate of Adam optimizer for Generator (default: 0.0001)")
flags.DEFINE_float("learning_rate_E", 0.0001, "Learning rate of Adam optimizer for Encoder (default: 0.0001)")
flags.DEFINE_float("beta1D", 0.5, "Momentum term of Adam optimizer for Discriminator (default: 0.5)")
flags.DEFINE_float("beta1G", 0.5, "Momentum term of Adam optimizer for Generator (default: 0.5)")
flags.DEFINE_float("beta1E", 0.5, "Momentum term of Adam optimizer for Encoder (default: 0.5)")

flags.DEFINE_float("gpu_frac", 0.95, "Gpu fraction")

flags.DEFINE_integer("number_train_images", 2, "No. of images for training")
flags.DEFINE_integer("number_train_unlab_images", 6, "No. of images for testing")
flags.DEFINE_integer("number_test_images", 7, "No. of images for testing")

flags.DEFINE_string("actual_data_directory", "../data/iSEG", "Directory name containing the actual dataset")
flags.DEFINE_string("preprocesses_data_directory", "../data/iSEG_preprocessed", "Directory name containing the preprocessed dataset")
flags.DEFINE_string("preprocesses_data2_directory", "../data/IBSR_Shakeri", "Directory name containing the preprocessed dataset 2")
flags.DEFINE_string("checkpoint_dir", "checkpoint/current", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("best_checkpoint_dir", "checkpoint/best", "Directory name to save the best checkpoints [checkpoint]")
flags.DEFINE_string("best_checkpoint_dir_val", "checkpoint/best_val", "Directory name to save the best checkpoints [checkpoint]")
flags.DEFINE_string("results_dir", "results/", "Directory name to save the results [results]")

flags.DEFINE_boolean("load_chkpt", True, "True for loading saved checkpoint")
flags.DEFINE_boolean("training", False, "True for Training ")
flags.DEFINE_boolean("testing", False, "True for Testing ")
flags.DEFINE_boolean("badGAN", False, "True for badGAN ")
flags.DEFINE_boolean("data2", True, "True for the second dataset ")


flags.DEFINE_integer("batch_size", 10, "The size of batch images [10]")

flags.DEFINE_integer("num_mod", 1, "Number of channels in input image")
flags.DEFINE_integer("num_classes", 14, "Number of output classes")
flags.DEFINE_integer("noise_dim", 200, "Dimension of noise vector")
flags.DEFINE_float("vi_weight", 0.01, "Weight of variational inference loss")

FLAGS = flags.FLAGS

def main(_):
  # Create required directories
  if not os.path.exists(FLAGS.checkpoint_dir):
    os.makedirs(FLAGS.checkpoint_dir)

  if not os.path.exists(FLAGS.best_checkpoint_dir):
    os.makedirs(FLAGS.best_checkpoint_dir)

  gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=FLAGS.gpu_frac)

  if F.data2:
    patch_shape=(64,64,64)
    extraction_step=(8,9,9)
    testing_extraction_shape=(8,9,9)
  else:
    patch_shape=(32,32,32)
    extraction_step=(8,8,8)
    testing_extraction_shape=(8,8,8)
  

  if FLAGS.training:
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
      network = model(sess,patch_shape,extraction_step)
      network.build_model()
      network.train()
  if FLAGS.testing:
      test(patch_shape,testing_extraction_shape)


if __name__ == '__main__':
  tf.app.run()