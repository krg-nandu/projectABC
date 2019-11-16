import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import tqdm, pickle, os
from mpl_toolkits.mplot3d import Axes3D
from reverse_model import cnn_reverse_model
from train_detector import cnn_model_struct
import config
import tensorflow as tf

class ImportanceSampler:
    def __getitem__(self, item):
	return getattr(self, item)

    def __contains__(self, item):
	return hasattr(self, item)

    """
    We need to initialize two models here:
    Model1: dataset -> params (this will give us an initial proposal)
    Model2: params -> dataset (allows us to evaluate likelihood of subsequent IS iterations)
    """
    def __init__(self, config, max_iters=100, tol=1e-7, nsamples=1e5):	

	self.max_iters = max_iters
	self.tol = tol
	self.N = nsamples

	self.cfg = config
	self.target = []

	# placeholder for forward model
	self.forward_model_inpdims = [10000] + self.cfg.param_dims[1:]
	self.forward_input = tf.placeholder(tf.float32, self.forward_model_inpdims)
	self.forward_initialized = False

	with tf.device('/gpu:0'):
	    with tf.variable_scope("model", reuse=tf.AUTO_REUSE) as scope:
		# build the forward model
		self.forward_model = cnn_model_struct()
		self.forward_model.build(self.forward_input, self.cfg.param_dims[1:], self.cfg.output_hist_dims[1:], train_mode=False, verbose=False)

	    self.gpuconfig = tf.ConfigProto()
	    self.gpuconfig.gpu_options.allow_growth = True
	    self.gpuconfig.allow_soft_placement = True
	    self.saver = tf.train.Saver()
	
	self.forward_sess = tf.Session(config=self.gpuconfig)
	ckpts = tf.train.latest_checkpoint(self.cfg.model_output)
	self.saver.restore(self.forward_sess, ckpts)

	# placeholder for inverse model
	self.inv_model_inpdims = [1] + self.cfg.output_hist_dims[1:]
	self.inv_input = tf.placeholder(tf.float32, self.inv_model_inpdims)
	
	with tf.device('/gpu:1'):
	    with tf.variable_scope("reversemodel", reuse=tf.AUTO_REUSE) as scope:
		# build the inverse model
		self.inv_model = cnn_reverse_model()
		self.inv_model.build(self.inv_input, self.cfg.output_hist_dims[1:], self.cfg.param_dims[1:], train_mode=False, verbose=False)

	    self.gpuconfig1 = tf.ConfigProto()
	    self.gpuconfig1.gpu_options.allow_growth = True
	    self.gpuconfig1.allow_soft_placement = True
	    self.saver1 = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='reversemodel'))

	self.inv_sess = tf.Session(config=self.gpuconfig1)
	ckpts = tf.train.latest_checkpoint(os.path.join(self.cfg.base_dir, 
							'models',
							'rev_'+self.cfg.model_name+'_'+self.cfg.model_suffix))
	self.saver1.restore(self.inv_sess, ckpts)

    def initializeMoG(self, mu_initial, std_initial, n_components=3, mu_perturbation=(-2., 2.), spread=10.):
        
	n_dims = mu_initial.shape[0]
	# set the inital component weights
	alpha_p = np.random.randint(low=1, high=n_components-1, size=n_components).astype(np.float32)
	alpha_p[0] = n_components
	alpha_p = alpha_p / alpha_p.sum()

	# initialize the mu and sigma
	mu_p = np.zeros((n_components, n_dims))
	std_p = np.zeros((n_components, n_dims, n_dims))

	# set the first component right on the point estimate
	mu_p[0] = mu_initial
	std_p[0] = np.diag(std_initial)

	# initialize the other components by perturbing them a little bit around the point estimate
	for c in range(1, n_components):
            mu_p[c] = mu_p[0] + np.random.uniform(low=mu_perturbation[0], high=mu_perturbation[1], size=n_dims)*std_initial
            std_p[c] = spread * std_p[0]
        
	# set the members
        self.n_components = n_components
	self.alpha_p = alpha_p
	self.mu_p = mu_p
	self.std_p = std_p

    def klDivergence(self, x, y, eps1=1e-7, eps2=1e-30):
	return np.sum(x * np.log(eps2 + x/(y+eps1)))

    def likelihood(self, x, y, eps=1e-7):
	return np.sum(-np.log(x+eps)*y)

    '''
    feed forward through the inverse model to get a point estimate of the parameters that could've generated a given dataset
    '''
    def getPointEstimate(self, dataset):
	params = self.inv_sess.run(self.inv_model.output, feed_dict={self.inv_input: dataset.reshape(self.inv_model_inpdims)})
	nparams = np.prod(self.cfg.param_dims[1:])
	means, stds = params[0][:nparams], params[0][nparams:]
	return means, stds

    def getLikelihoodFromProposals(self, params, target):

	params = np.expand_dims(np.expand_dims(params,-1),1)
	import time;
	t = time.time()
	pred_data = self.forward_sess.run(self.forward_model.output, feed_dict={self.forward_input:params})
	print (time.time() - t)
	import ipdb; ipdb.set_trace()
	return self.likelihood(pred_hist, self.target)

    '''
    def eval_likelihood(x, mu_l, std_l, alpha_l):
        n_components = alpha_l.shape[0]
        n_dims = mu_l.shape[1]
        target = np.zeros((x.shape[0],))
        for a in range(n_components):
            target += alpha_l[a] * stats.multivariate_normal.pdf(x, mean=mu_l[a], cov=std_l[a])
        return target

    def eval_proposal_by_component(x, mu_d, std_d):
        target = stats.multivariate_normal.pdf(x, mean=mu_d, cov=std_d)
        return target
    '''
    def generateFromProposal(self, mu_p, std_p, alpha_p, n):
        component_indices = np.random.choice(alpha_p.shape[0], # number of components
		                         p = alpha_p,          # component probabilities
		     			 size = n,
		     			 replace = True)
        _, unique_counts = np.unique(component_indices, return_counts=True)
        samples = np.array([])
        for c in range(alpha_p.shape[0]):
            if c >= unique_counts.shape[0]:
	        continue
 	    cur_samps = np.random.multivariate_normal(size = unique_counts[c], mean = mu_p[c], cov = std_p[c])
	    if c == 0:
	        samples = cur_samps
	    else:
                samples = np.concatenate([samples, cur_samps], axis=0)
        return samples

def main():
    # let's choose the dataset for which we'll try to get posteriors
    my_data = pickle.load(open('../data/ddm/parameter_recovery/ddm_param_recovery_data_n_3000.pickle', 'rb'))
    data, params = my_data[0][0], my_data[1][0]

    # load in the configurations
    cfg = config.Config()

    # initialize the importance sampler
    i_sampler = ImportanceSampler(cfg, max_iters=100, tol=1e-7, nsamples=10000)

    # get an initial point estimate
    mu_initial, std_initial = i_sampler.getPointEstimate(data)
    
    # Initializing the mixture
    i_sampler.initializeMoG(mu_initial, std_initial, n_components=3, mu_perturbation=(-2., 2.), spread=10.)

    # convergence metric
    norm_perplexity, cur_iter = 0., 0.

    while (cur_iter < max_iters):
	print(alpha_p)

	# sample parameters from the proposal distribution
	X = i_sampler.generateFromProposal(mu_p, std_p, alpha_p, N)

	# evaluate the likelihood of observering these parameters
	target = i_sampler.getLikelihoodFromProposals(X, data)
	import ipdb; ipdb.set_trace()	
	rho = np.zeros((n_components, N))
        # get rhos
	for c in range(n_components):
	    rho[c] = alpha_p[c] * eval_proposal_by_component(X, mu_p[c], std_p[c])
	rho_sum = np.sum(rho, axis = 0)
	rho = rho / rho_sum
	w = np.exp(np.log(target) - np.log(rho_sum))
	w = w / np.sum(w)

	entropy = -1*np.sum(w * np.log(w))
	norm_perplexity_cur = np.exp(entropy)/N
	if (norm_perplexity - norm_perplexity_cur)**2 < tol:
	    break
	norm_perplexity = norm_perplexity_cur

        # update proposal model parameters; in our case it is the alpha (s), mu (s) and std (s)
	for c in range(n_components):
	    tmp = w * rho[c]
	    alpha_p[c] = np.sum(tmp)
	    mu_p[c] = np.sum(X.transpose() * tmp, axis = 1) / alpha_p[c]
	    cov_mats = np.array([np.outer(x,x) for x in (X - mu_p[c])])
	    std_p[c] =  np.sum(cov_mats * tmp[:,np.newaxis, np.newaxis], axis = 0) / alpha_p[c]

	cur_iter += 1
    return X, w, X_true

if __name__ == '__main__':
    X, w, X_true = main()

    '''
    post_idx = np.random.choice(w.shape[0], p = w, replace = True, size = 100000)
    posterior_samples = X[post_idx, :]

    fig = plt.figure()
    #plt.hist2d(X_true[:,0], X_true[:,1], bins = 100, alpha = 0.2)
    plt.hist(posterior_samples[:, 0], bins = 100, alpha = 0.2)
    plt.hist(X_true[:, 0], bins = 100, alpha = 0.2)
    plt.figure()
    plt.hist(posterior_samples[:, 1], bins = 100, alpha = 0.2)
    plt.hist(X_true[:, 1], bins = 100, alpha = 0.2)
    plt.show()
    '''
