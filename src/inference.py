from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import tempfile
import os, pickle
import tensorflow as tf
from train_detector import cnn_model_struct 
import config
from scipy.optimize import differential_evolution
import numpy as np
import tqdm
import matplotlib.pyplot as plt

class Infer:
    def __init__(self, config):
	self.cfg = config
	self.target = pickle.load(open('../data/angle_ndt_base_simulations_1.pickle','rb'))[0][0].reshape((-1,))
	self.inp = tf.placeholder(tf.float32, self.cfg.test_param_dims)
	self.initialized = False
	with tf.device('/gpu:0'):
	    with tf.variable_scope("model") as scope:
		self.model = cnn_model_struct()
		self.model.build(self.inp, self.cfg.test_param_dims[1:], self.cfg.output_hist_dims[1:], train_mode=False)
	    self.gpuconfig = tf.ConfigProto()
	    self.gpuconfig.gpu_options.allow_growth = True
	    self.gpuconfig.allow_soft_placement = True
	    self.saver = tf.train.Saver()

    def __getitem__(self, item):
	return getattr(self, item)

    def __contains__(self, item):
	return hasattr(self, item)

    def klDivergence(self, x, y):
	return np.sum(x * np.log(1e-30 + x/y))

    def objectivefn(self, params):
	if self.initialized == False:
	    self.sess = tf.Session(config=self.gpuconfig)
	    ckpts = '/media/data_cifs/lakshmi/projectABC/models/cnn-v0/mlp_cnn_angle_1404250.ckpt-1404250'
	    self.saver.restore(self.sess, ckpts)
	    self.initialized = True
	pred_hist = self.sess.run(self.model.output, feed_dict={self.inp:params.reshape(1,1,5,1)})
	return self.klDivergence(pred_hist, self.target)

def model_inference():
    cfg = config.Config()
    inference_class = Infer(config=cfg)
    bounds = [(-2,2), (-2,2), (-2,2), (-2,2), (-2,2)]
    simulated_data = pickle.load(open('../data/angle_ndt_base_simulations_1.pickle','rb'))
    nsamples = simulated_data[0].shape[0]
    #nsamples = 10

    recovered_params = []
    for idx in tqdm.tqdm(np.arange(nsamples)):
	inference_class.target = simulated_data[0][idx].reshape((-1,))
	output = differential_evolution(inference_class.objectivefn,bounds)
	recovered_params.append(output.x)
    #import ipdb; ipdb.set_trace()

    RP = np.array(recovered_params)
    # make the plots
    for k in range(5):
        plt.figure(); plt.scatter(simulated_data[1][:nsamples,k], RP[:nsamples,k]); plt.xlabel('groundtruth parameter value'); plt.ylabel('recovered parameter value'); plt.title('angle_ndt param #{}'.format(k))
    plt.show(); 

if __name__ == '__main__':
    model_inference()
