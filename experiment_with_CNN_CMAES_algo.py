import os
from time import time
import numpy as np
from CNNScorer import NoIOCNNScorer
from ExperimentBase import ExperimentBase
from Optimizer import CMAES, Genetic

target_neuron = ('caffe-net', 'fc6', 1)
np.set_printoptions(precision=4, suppress=True)
code_length = 4096

homedir = os.path.expanduser('~')
initcodedir = os.path.join(homedir, 'Documents/stimuli/texture006')
natstimdir = os.path.join(homedir, 'Documents/stimuli/natimages-guapoCh9')
exp_dir = os.path.join(homedir, 'Documents/data/with_CNN')

population_size = 36
n_natural_stimuli = 40
random_seed = 0

population_size = 36
mutation_rate = 0.25
mutation_size = 0.75
kT_multiplier = 2
n_conserve = 10
parental_skew = 0.75

class CNNExperiment_Simplify(ExperimentBase):
    def __init__(self, recorddir, logdir, max_steps, init_sigma=5, optimizer_name='cmaes', optim_params={},
                 target_neuron=target_neuron, random_seed=None, saveimg=True, record_pattern=False):
        super(CNNExperiment_Simplify, self).__init__(logdir, random_seed)
        self.max_steps = max_steps
        self.saveimg = saveimg
        # initialize optimizer and scorer
        if optimizer_name == 'cmaes':
            optimizer = CMAES(recorddir=recorddir, space_dimen=code_length, init_sigma=init_sigma)
        elif optimizer_name == "genetic":
            optimizer = Genetic(population_size=population_size, random_seed=random_seed, recorddir=recorddir, **optim_params)
        else:
            raise NotImplementedError
        optimizer.load_init_population(initcodedir, size=population_size)
        optimizer.save_init_population()
        scorer = NoIOCNNScorer(target_neuron=target_neuron, writedir=recorddir, record_pattern=record_pattern)
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
        So to realize a `scorer` class just make these 3 functions. that's enough

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


def add_neuron_subdir(neuron, exp_dir):
    if len(neuron) == 5:
        subdir = '%s_%s_%04d_%d,%d' % (neuron[0], neuron[1].replace('/', '_'), neuron[2], neuron[3], neuron[4])
    else:
        subdir = '%s_%s_%04d' % (neuron[0], neuron[1].replace('/', '_'), neuron[2])
    this_exp_dir = os.path.join(exp_dir, subdir)
    for dir_ in (this_exp_dir,):
        if not os.path.isdir(dir_):
            os.mkdir(dir_)
    return this_exp_dir


def add_trial_subdir(neuron_dir, trial_title):
    trialdir = os.path.join(neuron_dir, trial_title)
    if not os.path.isdir(trialdir):
        os.mkdir(trialdir)
    return trialdir

import utils
if __name__ == '__main__':
    # random_seed = int(time())
    # neuron = ('caffe-net', 'fc6', 5)
    # this_exp_dir = add_neuron_subdir(neuron, exp_dir)
    #
    # trial_title ='trial_cma5_noeig_sgm20'
    # trialdir = add_trial_subdir(this_exp_dir, trial_title)
    # experiment = CNNExperiment_Simplify(recorddir=trialdir, logdir=trialdir, target_neuron=neuron, max_steps=300,
    #                                     init_sigma=20, random_seed=random_seed)
    # experiment.run()
    # utils.visualize_score_trajectory(trialdir, save=True, title_str=trial_title)

    # neuron = ('caffe-net', 'fc6', 30)
    # this_exp_dir = add_neuron_subdir(neuron, exp_dir)
    # for i in range(1, 5):
    #     random_seed = int(time())
    #     trial_title = 'cma_trial%d_noeig_sgm5' % i
    #     trialdir = add_trial_subdir(this_exp_dir, trial_title)
    #     experiment = CNNExperiment_Simplify(recorddir=trialdir, logdir=trialdir, target_neuron=neuron, max_steps=300,
    #                                         init_sigma=5, random_seed=random_seed)
    #     experiment.run()
    #     utils.visualize_score_trajectory(trialdir, save=True, title_str=trial_title)
    #
    # for i in range(5, 10):
    #     random_seed = int(time())
    #     trial_title = 'cma_trial%d_noeig_sgm10' % i
    #     trialdir = add_trial_subdir(this_exp_dir, trial_title)
    #     experiment = CNNExperiment_Simplify(recorddir=trialdir, logdir=trialdir, target_neuron=neuron, max_steps=300,
    #                                         init_sigma=10, random_seed=random_seed)
    #     experiment.run()
    #     utils.visualize_score_trajectory(trialdir, save=True, title_str=trial_title)
    #
    # neuron = ('caffe-net', 'fc6', 200)
    # this_exp_dir = add_neuron_subdir(neuron, exp_dir)
    # for i in range(5):
    #     random_seed = int(time())
    #     trial_title = 'cma_trial%d_noeig_sgm5' % i
    #     trialdir = add_trial_subdir(this_exp_dir, trial_title)
    #     experiment = CNNExperiment_Simplify(recorddir=trialdir, logdir=trialdir, target_neuron=neuron, max_steps=300,
    #                                         init_sigma=5, random_seed=random_seed)
    #     experiment.run()
    #     utils.visualize_score_trajectory(trialdir, save=True, title_str=trial_title)
    #
    # for i in range(5, 10):
    #     random_seed = int(time())
    #     trial_title = 'cma_trial%d_noeig_sgm10' % i
    #     trialdir = add_trial_subdir(this_exp_dir, trial_title)
    #     experiment = CNNExperiment_Simplify(recorddir=trialdir, logdir=trialdir, target_neuron=neuron, max_steps=300,
    #                                         init_sigma=10, random_seed=random_seed)
    #     experiment.run()
    #     utils.visualize_score_trajectory(trialdir, save=True, title_str=trial_title)
    #
    # neuron = ('caffe-net', 'fc7', 10)
    # this_exp_dir = add_neuron_subdir(neuron, exp_dir)
    # for i in range(5):
    #     random_seed = int(time())
    #     trial_title = 'cma_trial%d_noeig_sgm5' % i
    #     trialdir = add_trial_subdir(this_exp_dir, trial_title)
    #     experiment = CNNExperiment_Simplify(recorddir=trialdir, logdir=trialdir, target_neuron=neuron, max_steps=300,
    #                                         init_sigma=5, random_seed=random_seed)
    #     experiment.run()
    #     utils.visualize_score_trajectory(trialdir, save=True, title_str=trial_title)
    #
    # for i in range(5, 10):
    #     random_seed = int(time())
    #     trial_title = 'cma_trial%d_noeig_sgm10' % i
    #     trialdir = add_trial_subdir(this_exp_dir, trial_title)
    #     experiment = CNNExperiment_Simplify(recorddir=trialdir, logdir=trialdir, target_neuron=neuron, max_steps=300,
    #                                         init_sigma=10, random_seed=random_seed)
    #     experiment.run()
    #     utils.visualize_score_trajectory(trialdir, save=True, title_str=trial_title)
    #####   Genetic Trial
    optim_params = {'mutation_rate': mutation_rate, 'mutation_size': mutation_size, 'n_conserve': n_conserve,
                    'kT_multiplier': kT_multiplier, 'parental_skew': parental_skew, }

    neuron = ('caffe-net', 'fc6', 5)
    this_exp_dir = add_neuron_subdir(neuron, exp_dir)
    random_seed = int(time())
    trial_title = 'genetic_trial%d' % 0
    trialdir = add_trial_subdir(this_exp_dir, trial_title)
    experiment = CNNExperiment_Simplify(recorddir=trialdir, logdir=trialdir, target_neuron=neuron, max_steps=300,
                                        optimizer_name='genetic', optim_params=optim_params, init_sigma=5,
                                        random_seed=random_seed)
    experiment.run()
    utils.visualize_score_trajectory(trialdir, save=True, title_str=trial_title)
    for i in range(1,5):
        random_seed = int(time())
        trial_title = 'genetic_trial%d' % i
        trialdir = add_trial_subdir(this_exp_dir, trial_title)
        experiment = CNNExperiment_Simplify(recorddir=trialdir, logdir=trialdir, target_neuron=neuron, max_steps=300,
                                            optimizer_name='genetic', optim_params=optim_params, init_sigma=5,
                                            random_seed=random_seed, saveimg=False)
        experiment.run()
        utils.visualize_score_trajectory(trialdir, save=True, title_str=trial_title)

    neuron = ('caffe-net', 'fc6', 10)
    this_exp_dir = add_neuron_subdir(neuron, exp_dir)
    random_seed = int(time())
    trial_title = 'genetic_trial%d' % 0
    trialdir = add_trial_subdir(this_exp_dir, trial_title)
    experiment = CNNExperiment_Simplify(recorddir=trialdir, logdir=trialdir, target_neuron=neuron, max_steps=300,
                                        optimizer_name='genetic', optim_params=optim_params, init_sigma=5,
                                        random_seed=random_seed)
    experiment.run()
    utils.visualize_score_trajectory(trialdir, save=True, title_str=trial_title)
    for i in range(1,5):
        random_seed = int(time())
        trial_title = 'genetic_trial%d' % i
        trialdir = add_trial_subdir(this_exp_dir, trial_title)
        experiment = CNNExperiment_Simplify(recorddir=trialdir, logdir=trialdir, target_neuron=neuron, max_steps=300,
                                            optimizer_name='genetic', optim_params=optim_params, init_sigma=5,
                                            random_seed=random_seed, saveimg=False)
        experiment.run()
        utils.visualize_score_trajectory(trialdir, save=True, title_str=trial_title)

    neuron = ('caffe-net', 'fc6', 100)
    this_exp_dir = add_neuron_subdir(neuron, exp_dir)
    random_seed = int(time())
    trial_title = 'genetic_trial%d' % 0
    trialdir = add_trial_subdir(this_exp_dir, trial_title)
    experiment = CNNExperiment_Simplify(recorddir=trialdir, logdir=trialdir, target_neuron=neuron, max_steps=300,
                                        optimizer_name='genetic', optim_params=optim_params, init_sigma=5,
                                        random_seed=random_seed)
    experiment.run()
    utils.visualize_score_trajectory(trialdir, save=True, title_str=trial_title)
    for i in range(1,5):
        random_seed = int(time())
        trial_title = 'genetic_trial%d' % i
        trialdir = add_trial_subdir(this_exp_dir, trial_title)
        experiment = CNNExperiment_Simplify(recorddir=trialdir, logdir=trialdir, target_neuron=neuron, max_steps=300,
                                            optimizer_name='genetic', optim_params=optim_params, init_sigma=5,
                                            random_seed=random_seed, saveimg=False)
        experiment.run()
        utils.visualize_score_trajectory(trialdir, save=True, title_str=trial_title)

    neuron = ('caffe-net', 'fc7', 1)
    this_exp_dir = add_neuron_subdir(neuron, exp_dir)
    random_seed = int(time())
    trial_title = 'genetic_trial%d' % 0
    trialdir = add_trial_subdir(this_exp_dir, trial_title)
    experiment = CNNExperiment_Simplify(recorddir=trialdir, logdir=trialdir, target_neuron=neuron, max_steps=300,
                                        optimizer_name='genetic', optim_params=optim_params, init_sigma=5,
                                        random_seed=random_seed)
    experiment.run()
    utils.visualize_score_trajectory(trialdir, save=True, title_str=trial_title)
    for i in range(1,5):
        random_seed = int(time())
        trial_title = 'genetic_trial%d' % i
        trialdir = add_trial_subdir(this_exp_dir, trial_title)
        experiment = CNNExperiment_Simplify(recorddir=trialdir, logdir=trialdir, target_neuron=neuron, max_steps=300,
                                            optimizer_name='genetic', optim_params=optim_params, init_sigma=5,
                                            random_seed=random_seed, saveimg=False)
        experiment.run()
        utils.visualize_score_trajectory(trialdir, save=True, title_str=trial_title)

    neuron = ('caffe-net', 'fc7', 10)
    random_seed = int(time())
    trial_title = 'genetic_trial%d' % 0
    trialdir = add_trial_subdir(this_exp_dir, trial_title)
    experiment = CNNExperiment_Simplify(recorddir=trialdir, logdir=trialdir, target_neuron=neuron, max_steps=300,
                                        optimizer_name='genetic', optim_params=optim_params, init_sigma=5,
                                        random_seed=random_seed)
    experiment.run()
    utils.visualize_score_trajectory(trialdir, save=True, title_str=trial_title)
    this_exp_dir = add_neuron_subdir(neuron, exp_dir)
    for i in range(1,5):
        random_seed = int(time())
        trial_title = 'genetic_trial%d' % i
        trialdir = add_trial_subdir(this_exp_dir, trial_title)
        experiment = CNNExperiment_Simplify(recorddir=trialdir, logdir=trialdir, target_neuron=neuron, max_steps=300,
                                            optimizer_name='genetic', optim_params=optim_params, init_sigma=5,
                                            random_seed=random_seed, saveimg=False)
        experiment.run()
        utils.visualize_score_trajectory(trialdir, save=True, title_str=trial_title)