import os
import pprint
import tensorflow as tf

from model_unet import UNET
from testing_unet import *


# Define flags
flags = tf.app.flags
flags.DEFINE_integer("epoch", 100000, "Number of training epochs (default: 100000)")
flags.DEFINE_float("learning_rate_", 0.0001, "Learning rate of Adam optimizer for Discriminator (default: 0.0001)")
flags.DEFINE_float("beta1", 0.9, "Momentum term of Adam optimizer for Discriminator (default: 0.5)")

flags.DEFINE_float("gpu_frac", 0.95, "Gpu fraction")


flags.DEFINE_integer("number_train_images", 2, "No. of images for training")
flags.DEFINE_integer("number_test_images", 1, "No. of images for testing")

flags.DEFINE_string("data_directory", "../data/iSEG_preprocessed", "Directory name containing the preprocessed dataset")
flags.DEFINE_string("checkpoint_dir", "checkpoint/current", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("best_checkpoint_dir", "checkpoint/best", "Directory name to save the best checkpoints [checkpoint]")
flags.DEFINE_string("results_dir", "results/", "Directory name to save the results [results]")

flags.DEFINE_boolean("load_chkpt", False, "True for loading saved checkpoint")
flags.DEFINE_boolean("training", False, "True for Training ")
flags.DEFINE_boolean("testing", False, "True for Testing ")


flags.DEFINE_integer("batch_size", 30, "The size of batch images(30 for data1 and 20 for data2")

flags.DEFINE_integer("num_mod", 2, "Number of channels in input image(2 for data1 and 1 for data2)")
flags.DEFINE_integer("num_classes", 4, "Number of output classes(4 for data1 and 15 for data2)")

FLAGS = flags.FLAGS

def main(_):
  # Create required directories
  if not os.path.exists(FLAGS.checkpoint_dir):
    os.makedirs(FLAGS.checkpoint_dir)

  if not os.path.exists(FLAGS.results_dir):
    os.makedirs(FLAGS.results_dir)

  if not os.path.exists(FLAGS.best_checkpoint_dir):
    os.makedirs(FLAGS.best_checkpoint_dir)

  # To configure the GPU fraction
  gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=FLAGS.gpu_frac)

  # Parameters of extracted training and testing patches
  patch_shape=(32,32,32)
  extraction_step=(8,8,8)
  testing_extraction_shape=(8,8,8)

  if FLAGS.training:
    # For training the network
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
      network = UNET(sess,patch_shape,extraction_step)
      network.build_model()
      network.train()
  if FLAGS.testing:
      # For testing the network
      test(patch_shape,testing_extraction_shape)


if __name__ == '__main__':
  tf.app.run()