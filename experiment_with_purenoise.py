import os
from time import time
import numpy as np
# from CNNScorer import NoIOCNNScorer
from ExperimentBase import ExperimentBase
from Optimizer import CMAES, Genetic, CholeskyCMAES
#import utils
from utils import add_neuron_subdir, add_trial_subdir
import matplotlib.pylab as plt
from utils import generator

np.set_printoptions(precision=4, suppress=True)
code_length = 4096

homedir = r"D:\Generator_DB_Windows" #os.path.expanduser('~')
initcodedir = os.path.join(homedir, 'stimuli/texture006')
natstimdir = os.path.join(homedir, 'stimuli/natimages-guapoCh9')
exp_dir = os.path.join(homedir, 'data/purenoise')

population_size = 40
n_natural_stimuli = 40
random_seed = 0

# Parameters for a genetic algorithm
mutation_rate = 0.25
mutation_size = 0.75
kT_multiplier = 2
n_conserve = 10
parental_skew = 0.75

class Null_Experiment(ExperimentBase):
    def __init__(self, recorddir, logdir, max_steps, init_sigma=5, init_code=None, Aupdate_freq=None,
                 optimizer_name='cmaes', optim_params={}, random_seed=None, saveimg=True):
        '''
        optimizer_name: can be set as 'cmaes' or 'genetic'
        optim_param: is input parameters into 'genetic' algorithm. And 'cmaes' only uses 'init_sigma' for parameter.
        noise_scheme=None, noise_param=None, noise_rand_seed=0, are directly forward into Scorer
        '''
        super(Null_Experiment, self).__init__(logdir, random_seed)
        self.logdir = logdir
        self.max_steps = max_steps
        self.saveimg = saveimg
        # initialize optimizer and scorer
        if optimizer_name == 'cmaes':
            optimizer = CMAES(recorddir=recorddir, space_dimen=code_length, init_sigma=init_sigma)
        elif optimizer_name == 'cholcmaes':
            optimizer = CholeskyCMAES(recorddir=recorddir, space_dimen=code_length, init_sigma=init_sigma, init_code=init_code,
                                      Aupdate_freq=Aupdate_freq, optim_params=optim_params)
        elif optimizer_name == "genetic":
            optimizer = Genetic(population_size=population_size, random_seed=random_seed, recorddir=recorddir, **optim_params)
        # Variant Optimizer can be added here !
        else:
            raise NotImplementedError
        optimizer.load_init_population(initcodedir, size=population_size)
        self.attach_optimizer(optimizer)

    def run(self):
        '''
        Running each iteration in the order:

        `self.order.score()`
        `self.optimizer.save_current_codes()
        self.optimizer.save_current_genealogy()
        self.scorer.save_current_scores()
        t2 = time()
        # use results to update optimizer
        self.optimizer.step(synscores)`

        Note:
        The only methods that a `scorer` module should get are
            `scorer.score(stimuli, stimuli_ids)`,
            `scorer.save_current_scores()`
            `scorer.load_classifier()`
        So to implement a `scorer` class just make these 3 functions. that's enough

        '''
        self.istep = 0
        self.codes_all = []
        self.img_ids = []
        self.generations = []
        while self._istep < self.max_steps:
            print('\n>>> step %d' % self.istep)
            t0 = time()
            # score images
            # synscores = self.scorer.score(self.optimizer.current_images, self.optimizer.current_image_ids)
            synscores = np.random.randn(len(self.optimizer._curr_sample_ids))
            t1 = time()
            t2 = time()
            # use results to update optimizer
            self.optimizer.step(synscores, no_image=True)
            t3 = time()
            self.img_ids.extend(self.optimizer._curr_sample_ids)
            self.codes_all.append(self.optimizer._curr_samples)
            self.generations.extend([self.istep] * len(self.optimizer._curr_sample_ids))
            # TODO: may add optimizer status saving function
            # summarize scores & delays
            print('synthetic img scores: mean {}, all {}'.format(np.nanmean(synscores), synscores))
            print(('step %d time: total %.2fs | ' +
                   'wait for results %.2fs  write records %.2fs  optimizer update %.2fs')
                  % (self.istep, t3 - t0, t1 - t0, t2 - t1, t3 - t2))

            self.logger.flush()
            self.istep += 1
        self.codes_all = np.concatenate(tuple(experiment.codes_all), axis=0)
        self.generations = np.array(self.generations)
        np.savez(os.path.join(self.logdir, "codes_all.npz"), codes_all=self.codes_all,
                 generations=self.generations,
                 image_ids=self.img_ids)
        self.visualize_image_evolution(exp_title_str="pure_noise_evolution")
        # self.save_experiment()

    def visualize_image_evolution(self, save=True, exp_title_str='', col_n=10, savedir=''):
        '''
        # CurDataDir:  "/home/poncelab/Documents/data/with_CNN/caffe-net_fc6_0001/backup/"
        # block_num: the number of block to visualize 20
        # title_cmap: define the colormap to do the code, plt.cm.viridis
        # col_n: number of column in a plot 6
        # FIXED: on Oct. 7th support new name format, and align the score are image correctly
        '''

        image_num = len(self.img_ids)
        gen_num = self.generations.max() + 1
        row_n = np.ceil(gen_num / col_n)
        figW = 12
        figH = figW / col_n * row_n + 1
        fig = plt.figure(figsize=[figW, figH])
        for geni in range(gen_num):
            code_tmp = self.codes_all[self.generations == geni, :].mean(axis=0)
            img_tmp = generator.visualize(code_tmp)
            plt.subplot(row_n, col_n, geni + 1)
            plt.imshow(img_tmp)
            plt.axis('off')
            plt.title("%d" % (geni))

        plt.suptitle(exp_title_str, fontsize=16)
        plt.tight_layout(h_pad=0.1, w_pad=0, rect=(0, 0, 0.95, 0.9))
        if save:
            if savedir == '':
                savedir = self.logdir
            plt.savefig(os.path.join(savedir, exp_title_str))
        # plt.show()
        return fig
# %%
this_exp_dir = exp_dir
optim_params = {} # Original case
import logging
for trial_i in range(100):
    random_seed = int(time())
    trial_title = 'choleskycma_sgm3_uf10_trial%d' % (trial_i)
    trialdir = add_trial_subdir(this_exp_dir, trial_title)
    experiment = Null_Experiment(recorddir=trialdir, logdir=trialdir, max_steps=100,
                                optimizer_name='cholcmaes', init_sigma=3, Aupdate_freq=10,
                                optim_params=optim_params, random_seed=random_seed, saveimg=False)
    experiment.run()

# utils.codes_summary(trialdir, True)
# utils.gen_visualize_image_score_each_block(trialdir, block_num=198, exp_title_str=trial_title)