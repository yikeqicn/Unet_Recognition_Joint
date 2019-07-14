from glob import glob
import numpy as np
import sys
import tensorflow as tf
from os.path import join
from Densenet4htr import Densenet4htr
import utils


class DecoderType:
  BestPath = 0
  BeamSearch = 1
  WordBeamSearch = 2


class Model:
  # model constants
  # batchSize = 50 #qyk
  # imgSize = (128, 32)
  # imgSize = (192, 48) #qyk
  maxTextLen = 32  # qyk?
  
  def __init__(self, args, charList, decoderType=DecoderType.BestPath, mustRestore=False):
    "init model: add CNN, RNN and CTC and initialize TF"
    self.charList = charList
    self.decoderType = decoderType
    self.mustRestore = mustRestore
    # self.FilePaths = FilePaths
    self.batchsize = args.batchsize
    self.lrInit = args.lrInit
    self.args = args
    
    with tf.name_scope('graph_segmentation'):
  
      self.loss_segmentation, output = YIKE_FUNCTION_HERE()

    
    with tf.name_scope('graph_recognition'):
        
        # Input
        # self.inputImgs = tf.placeholder(tf.float32, shape=(None, args.imgsize[0], args.imgsize[1]))  # self.batchsize yike
        self.inputImgs = output
        self.inputImgs = tf.placeholder(blash)
        
        # CNN
        if args.nondensenet:
          cnnOut4d = self.setupCNN(self.inputImgs)
        else:  # use densenet by default
          cnnOut4d = self.setupCNNdensenet(self.inputImgs, args)
        
        # RNN
        rnnOut3d = self.setupRNN(cnnOut4d)
        
        # CTC
        (self.ctcloss, self.decoder) = self.setupCTC(rnnOut3d)
        
        # Explicit regularizers
        self.loss_recognition = self.ctcloss + args.wdec * self.setupWdec(args)
      
    # combine losses
    self.loss = (1 - beta) * self.loss_recognition + beta * self.loss_segmentation
    
    # optimizer for NN parameters
    self.batchesTrained = args.batchesTrained
    self.learningRate = tf.placeholder(tf.float32, shape=[])
    
    # optimize only the segnet variables
    var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "graph_segmentation")
    self.optimizer = tf.train.RMSPropOptimizer(self.learningRate).minimize(self.loss, var_list=var_list)
    
    # initialize TF
    (self.sess, self.saver) = self.setupTF()
    # when training recognition model, you might want to try something like this
    train_op = self.sess.run([self.train_op, self.loss], feed_dict={self.inputImgs: your_numpy_image_data})

    # after training, use saver to save only the recognition variables
    var_list = tf.global_variables('graph_segmentation')
    saver = tf.train.Saver(var_list=var_list, max_to_keep=1)
    saver.save(self.sess)
      
    
