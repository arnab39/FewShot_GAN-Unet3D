import os
import pprint
import tensorflow as tf

import exp as ep


# Define flags
flags = tf.app.flags

flags.DEFINE_float("gpu_frac", 0.95, "Gpu fraction")

flags.DEFINE_string("actual_data_directory", "../data/iSEG", "Directory name containing the actual dataset")
flags.DEFINE_string("preprocesses_data_directory", "../data/iSEG_preprocessed", "Directory name containing the preprocessed dataset")
flags.DEFINE_string("checkpoint_unet", "checkpoint_unet/current", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("checkpoint_unetGAN", "checkpoint_unetGAN/current", "Directory name to save the best checkpoints [checkpoint]")
flags.DEFINE_string("results_dir", "results/", "Directory name to save the results [results]")

flags.DEFINE_integer("batch_size", 10, "The size of batch images [64]")
flags.DEFINE_integer("patch_size", 32, "The size of image patches [32]")

flags.DEFINE_integer("num_mod", 2, "Number of channels in input image")
flags.DEFINE_integer("num_classes", 4, "Number of output classes")
flags.DEFINE_integer("noise_dim", 200, "Dimension of noise vector")



FLAGS = flags.FLAGS

def main(_):

  gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=FLAGS.gpu_frac)
  

  patch_shape=(32,32,32)
  extraction_step=(8,8,8)
  testing_extraction_shape=(8,8,8)

  #ep.exp_test_unet(patch_shape,testing_extraction_shape)
  ep.exp_test_unetGAN(patch_shape,testing_extraction_shape)


if __name__ == '__main__':
  tf.app.run()