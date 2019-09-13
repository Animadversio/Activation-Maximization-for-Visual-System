import os
from time import time
import numpy as np
from CNNScorer import NoIOCNNScorer
from ExperimentBase import ExperimentBase
from Optimizer import CMAES, Genetic, CholeskyCMAES
import utils
from utils import add_neuron_subdir, add_trial_subdir

target_neuron = ('caffe-net', 'fc6', 1)
np.set_printoptions(precision=4, suppress=True)
code_length = 4096

homedir = r"D:\Generator_DB_Windows" #os.path.expanduser('~')
initcodedir = os.path.join(homedir, 'stimuli/texture006')
natstimdir = os.path.join(homedir, 'stimuli/natimages-guapoCh9')
exp_dir = os.path.join(homedir, 'data/with_CNN')

population_size = 36
n_natural_stimuli = 40
random_seed = 0

# Parameters for a genetic algorithm
mutation_rate = 0.25
mutation_size = 0.75
kT_multiplier = 2
n_conserve = 10
parental_skew = 0.75

class CNNExperiment_Simplify(ExperimentBase):
    def __init__(self, recorddir, logdir, max_steps, init_sigma=5, init_code=None, Aupdate_freq=None, optimizer_name='cmaes', optim_params={},
                 target_neuron=target_neuron, noise_scheme=None, noise_param=None, noise_rand_seed=0,
                 random_seed=None, saveimg=True, record_pattern=False):
        '''
        optimizer_name: can be set as 'cmaes' or 'genetic'
        optim_param: is input parameters into 'genetic' algorithm. And 'cmaes' only uses 'init_sigma' for parameter.
        noise_scheme=None, noise_param=None, noise_rand_seed=0, are directly forward into Scorer
        '''
        super(CNNExperiment_Simplify, self).__init__(logdir, random_seed)
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
        optimizer.save_init_population()
        scorer = NoIOCNNScorer(target_neuron=target_neuron, writedir=recorddir, record_pattern=record_pattern,
                               noise_scheme=noise_scheme, noise_param=noise_param, noise_rand_seed=noise_rand_seed, )
        self.attach_optimizer(optimizer)  # TODO: Maybe we can attach more than one optimizer here?
        self.attach_scorer(scorer)

        # load & backup_images natural stimuli
        self.attach_natural_stimuli(natstimdir, n_natural_stimuli)
        self.save_natural_stimuli(recorddir)

    def _set_fpath(self):         # the file at self._fpath will be copied as a backup_images;
        self._fpath = __file__    # this function makes sure self._fpath points to *this* file

    def load_nets(self):
        self.optimizer.load_generator()
        self.scorer.load_classifier()

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
        self.load_nets()    # nets are not loaded in __init__; this enables exp to be run with multiprocessing
        self.istep = 0
        while self._istep < self.max_steps:
            print('\n>>> step %d' % self.istep)
            t0 = time()

            if self.istep == 0:
                # score images
                natscores = self.scorer.score(self.natural_stimuli, self.natural_stimuli_ids)
                t1 = time()
                # backup scores
                self.scorer.save_current_scores()
                t2 = time()
                # summarize scores & delays
                print('natural image scores: mean {}, all {}'.format(np.nanmean(natscores), natscores))
                print('step %d time: total %.2fs | wait for results %.2fs  write records %.2fs'
                      % (self.istep, t2 - t0, t1 - t0, t2 - t1))
                # self.scorer.score(self.optimizer._init_population)
            else:
                # score images
                synscores = self.scorer.score(self.optimizer.current_images, self.optimizer.current_image_ids)
                t1 = time()
                # before update, backup_images codes (optimizer) and scores (scorer)
                self.optimizer.save_current_codes()
                if self.saveimg:
                    self.optimizer.save_current_state()  # current state save code and image
                else:
                    self.optimizer.save_current_codes()
                # self.optimizer.save_current_genealogy()
                self.scorer.save_current_scores()
                t2 = time()
                # use results to update optimizer
                self.optimizer.step(synscores)
                t3 = time()
                # TODO: may add optimizer status saving function
                # summarize scores & delays
                print('synthetic img scores: mean {}, all {}'.format(np.nanmean(synscores), synscores))
                print(('step %d time: total %.2fs | ' +
                       'wait for results %.2fs  write records %.2fs  optimizer update %.2fs')
                      % (self.istep, t3 - t0, t1 - t0, t2 - t1, t3 - t2))

            self.logger.flush()
            self.istep += 1



if __name__ == '__main__':
    # from CNNScorer import NoIOCNNScorer
    # code = np.random.randn(1, 4096)
    # img = utils.generator.visualize(code)
    # target_neuron = ('caffe-net', 'fc8', 1)
    # scorer = NoIOCNNScorer(target_neuron, exp_dir, record_pattern=['conv5', 'fc6', 'fc7', 'fc8'])
    # scorer.load_classifier()
    # score, pattern_dict = scorer.test_score(img)
    neuron = ('caffe-net', 'fc8', 1)
    this_exp_dir = add_neuron_subdir(neuron, exp_dir)
    optim_params = {}  # Original case
    for i in range(5, 10):
        random_seed = int(time())
        trial_title = 'choleskycma_norm_trial%d' % (i)
        trialdir = add_trial_subdir(this_exp_dir, trial_title)
        experiment = CNNExperiment_Simplify(recorddir=trialdir, logdir=trialdir, target_neuron=neuron, max_steps=200,
                                            optimizer_name='cholcmaes', init_sigma=3, Aupdate_freq=10,
                                            optim_params=optim_params,
                                            random_seed=random_seed, saveimg=False,
                                            record_pattern=['conv5', 'fc6', 'fc7', 'fc8'], )
        experiment.run()
        utils.codes_summary(trialdir, True)
        utils.scores_imgname_summary(trialdir, True)
        utils.visualize_score_trajectory(trialdir, save=True, title_str=trial_title)