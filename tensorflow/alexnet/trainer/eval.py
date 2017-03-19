#!/usr/bin/env python
# Copyright 2016 Google Inc. All Rights Reserved.
# Modified by dkoes.

"""A script evaluating a saved session on new TEST data.

  # Using a model from the local filesystem:
  python eval.py --model_dir=output/${JOB_NAME}/model TEST

"""

### NEED AN export FILE IN THE MODEL FOLDER

import scipy.io
import argparse
import collections
import json
import os
import time
import subprocess
import numpy as np
import sys
from scipy import misc
#from google.cloud.ml import session_bundle
from tensorflow.contrib.session_bundle import constants
from tensorflow.contrib.session_bundle import manifest_pb2
from tensorflow.contrib.session_bundle import session_bundle
from tensorflow.python.lib.io import file_io
from sklearn import metrics
import tensorflow as tf
import tensorflow.contrib.slim as slim
import pandas as pd

# Basic model parameters as external flags.
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('max_steps', 20001, 'Number of steps to run trainer.')
flags.DEFINE_integer('batch_size', 24, 'Batch size.')
flags.DEFINE_integer('im_size', 512, 'Batch size.')
#flags.DEFINE_string('train_data_dir', '/home/lun5/ADH/non_masked_images_512', 'Directory containing training data')
flags.DEFINE_string('train_data_dir', '/home/lun5/ADH/masked_images_512', 'Directory containing training data')
flags.DEFINE_string('train_test_lists_dir','/home/lun5/ADH/train_test_lists', 'Directory containing list of train/val/test')
#flags.DEFINE_string('model_dir', '/home/lun5/ADH/tensorflow/alexnet/non_masked_images/model', 'Directory to put the model into.')
#flags.DEFINE_string('test_dir', '/home/lun5/ADH/tensorflow/alexnet/non_masked_images/', 'Directory to store the test results')
flags.DEFINE_string('model_dir', '/home/lun5/ADH/tensorflow/alexnet/masked_im/model', 'Directory to put the model into.')
flags.DEFINE_string('test_dir', '/home/lun5/ADH/tensorflow/alexnet/masked_im/', 'Directory to store the test results')
#flags.DEFINE_string('train_output_dir', '/home/lun5/ADH/tensorflow/alexnet/non_masked_images/data', 'Directory containing output data')
#flags.DEFINE_string('model_dir', '/home/lun5/ADH/tensorflow/alexnet/masked_images/model', 'Directory to put the model into.')
#flags.DEFINE_string('train_output_dir', '/home/lun5/ADH/tensorflow/alexnet/masked_images/data', 'Directory containing output data')
#flags.DEFINE_string('train_output_dir', '/home/lun5/tissue-db/tensorflow/alexnet/data', 'Directory containing output data')
# Feel free to add additional flags to assist in setting hyper parameters
flags.DEFINE_float('L2reg',0.0005 , 'L2 regularization for weights')
flags.DEFINE_float('Adam_lr',1e-4, 'Adam optimizer learning rate')
flags.DEFINE_float('Adam_beta1',.9 , 'Adam optimizer beta1')
flags.DEFINE_float('Adam_beta2',.999 , 'Adam optimizer beta2')
flags.DEFINE_float('Adam_eps',1 , 'Adam optimizer epsilon')

labelmap = {'ADH':1,
 'Flat Epithelial':1,
 'Columnar':0,
 'Normal Duct':0
           }

n_classes = len(set(labelmap.values()))

def parse_args():
  """Parses arguments specified on the command-line."""

  argparser = argparse.ArgumentParser('Evaluate predictions.')

  argparser.add_argument(
      '--test_fname',
      dest='test_fname',
      help=('File containing list of TEST images'))

  argparser.add_argument(
      '--model_dir',
      dest='model_dir',
      help=('The path to the model where the tensorflow meta graph '
            'proto and checkpoint files are saved.'))


  return argparser.parse_args()

def read_im_list(fname):
    """
    Read <train_data_dir>/TRAIN which containing paths and labels in
    the format label, channel1 file, channel2 file, channel3 
    Returns:
        List with all filenames in file image_list_file
    """
    image_list_file = FLAGS.train_test_lists_dir + '/' + fname
    f = pd.read_csv(image_list_file)
    filenames = list(f['Image'])
    labels = np.asarray(f['Label'])
    n_classes = len(np.unique(labels))
    labels = [[int(labelmap[l] == 0), int(labelmap[l] == 1)] for l in labels]
    filenames = [os.path.join(FLAGS.train_data_dir,ff) for ff in filenames]
    #onehot = [onehot[i] for i in xrange(onehot.shape[0])]
    return zip( labels,filenames)

def read_test_list(testdir):
    """
    Read <train_data_dir>/TEST which containing paths and labels in
    the format label, channel1 file, channel2 file, channel3 
    Returns:
        List with all filenames in file image_list_file
    """
    image_list_file = testdir + '/TEST_05'
    f = file_io.FileIO(image_list_file, 'r') #this can read files from the cloud
    filenames = []
    labels = []
    n_classes = len(labelmap)
    for line in f:
        files, label = line.rstrip().split(' ')
        #convert labels into onehot encoding
        onehot = np.zeros(n_classes)
        onehot[labelmap[label]] = 1.0
        labels.append(onehot)
        #create absolute paths for image files
        filenames.append(testdir + files)
    
    return zip( labels,filenames)

# def network(inputs):
#     '''Define the network'''
#     with slim.arg_scope([slim.conv2d, slim.fully_connected],
#                       activation_fn=tf.nn.relu,
#                       weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
#                       weights_regularizer=slim.l2_regularizer(0.0005)):
#         net = tf.reshape(inputs,[-1,FLAGS.im_size ,FLAGS.im_size,3])
#         net = slim.conv2d(net, 32, [3,3], scope='conv1')
#         net = slim.max_pool2d(net, [4,4], scope = 'conv1')
#         net = slim.conv2d(net,128,[3,3], scope = 'conv2')
#         net = slim.max_pool2d(net,[4,4], scope = 'pool2')
#         net = slim.flatten(net)
#         net = slim.fully_connected(net,64, scope = 'fc')
#         net = slim.fully_connected(net, len(labelmap), activation_fn = None, scope = 'output')
#     return net


if __name__ == '__main__':   
  #args = parse_args()    
  # Create a session for running Ops on the Graph.
  session, _ = session_bundle.load_session_bundle_from_path(
    FLAGS.model_dir + '/' + '12000/')
  print session
  
  # get the mappings between aliases and tensor names
  # for both inputs and outputs
  input_alias_map = json.loads(session.graph.get_collection('inputs')[0])
  output_alias_map = json.loads(session.graph.get_collection('outputs')[0])
  aliases, tensor_names = zip(*output_alias_map.items())
  np.random.seed(45) #shuffle the same way each time for consistency
  #examples = read_test_list(args.test_dir)
  examples = read_im_list('tf_test.csv')
  np.random.shuffle(examples)
  start_time = time.time() 
  y_true = []
  y_pred = []
  count = 0
  
  for (label, files) in examples:
      channels = misc.imread(files)
      image = misc.imresize(channels,[FLAGS.im_size, FLAGS.im_size])
      #print image.shape
      feed_dict = {input_alias_map['image']: [image]}
      predict, scores = session.run(fetches=[output_alias_map['prediction'],
                                             output_alias_map['scores']],
                                    feed_dict=feed_dict)
      y_true.append(np.argmax(label))
      y_pred.append(predict[0])
      
      count += 1
      if (count%100== 0) and (count > 0):
          accuracy = metrics.accuracy_score(y_true,y_pred)
          #f1macro = metrics.f1_score(y_true,y_pred,average='macro')
          #f1micro = metrics.f1_score(y_true,y_pred,average='micro')
          #print('Example %d: accuracy=%g, f1macro=%g, f1micro=%g \n' %(
          #  count, accuracy, f1macro, f1micro))
          eval_results = [(y_true[j],y_pred[j]) for j in xrange(len(y_true))]
        
          f1 = metrics.f1_score(y_true, y_pred)
          print('Accuracy: %g, F1: %g\n' %(accuracy, f1))
          print('Confusion matrix is')
          print(metrics.confusion_matrix(y_true, y_pred))
          with open(os.path.join(FLAGS.test_dir,'eval_results.txt'), 'w') as fp:
                fp.write('\n'.join('%s %s' % x for x in eval_results))  
  
  duration = time.time()-start_time
  accuracy = metrics.accuracy_score(y_true,y_pred)
  #f1macro = metrics.f1_score(y_true,y_pred,average='macro')
  #f1micro = metrics.f1_score(y_true,y_pred,average='micro')
  f1 = metrics.f1_score(y_true, y_pred)
  f1_weighted = metrics.f1_score(y_true, y_pred)
  print('Test Accuracy: %g, F1: %g, F1_weighted: %g\n' %(accuracy, f1, f1_weighted))
  print('Confusion matrix is')
  print(metrics.confusion_matrix(y_true, y_pred))
  print "PredictTime",duration


