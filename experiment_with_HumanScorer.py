import os
from time import time
import matplotlib.pyplot as plt
import numpy as np
from Scorer import WithIOHumanScorer
from ExperimentBase import ExperimentBase
from Optimizer import Genetic

# Note : This Experiment is mostly for fun~ Try to rate the images according to subjective values and evolve images according to them

np.set_printoptions(precision=4, suppress=True)

# set file directories for I/O
homedir = os.path.expanduser('~')
initcodedir = os.path.join(homedir, 'Documents/stimuli/texture006')
natstimdir = os.path.join(homedir, 'Documents/stimuli/natimages-guapoCh9')
blockwritedir = os.path.join(homedir, 'Documents/data/with_humanscorer')
recorddir = os.path.join(blockwritedir, 'backup')

# set parameters
#   for scorer, which does image I/O
image_size = 83
# target_neuron = ('caffe-net', 'fc8', 1)
#   for optimizer
population_size = 36
mutation_rate = 0.25
mutation_size = 0.75
kT_multiplier = 2
n_conserve = 10
parental_skew = 0.75
#   for overall
max_steps = 300
random_seed = 0
n_natural_stimuli = 40

# auto assign directory

# natstimdir = os.path.join(natstimdir, subdir)
recorddir = os.path.join(blockwritedir, 'backup')

# make dir if needed
for dir_ in (blockwritedir, recorddir):
    if not os.path.isdir(dir_):
        os.mkdir(dir_)

# check existence of file and direcrtory
for dir_ in (initcodedir, natstimdir, blockwritedir, recorddir):
    if not os.path.isdir(dir_):
        raise OSError('directory not found: %s' % dir_)


class WithHumanExperiment(ExperimentBase):
    def __init__(self, logdir, random_seed=None):
        super(WithHumanExperiment, self).__init__(logdir, random_seed)

        # initialize optimizer and scorer
        optimizer = Genetic(population_size=population_size, mutation_rate=mutation_rate, mutation_size=mutation_size,
                            n_conserve=n_conserve, kT_multiplier=kT_multiplier, parental_skew=parental_skew,
                            random_seed=random_seed, recorddir=recorddir)
        optimizer.load_init_population(initcodedir, size=population_size)
        optimizer.save_init_population()
        scorer = WithIOHumanScorer(writedir=blockwritedir, backupdir=recorddir,
                                 image_size=image_size, random_seed=random_seed)
        self.attach_optimizer(optimizer)  # TODO: Maybe we can attach more than one optimizer here?
        self.attach_scorer(scorer)

        # load & backup_images natural stimuli
        self.attach_natural_stimuli(natstimdir, n_natural_stimuli)
        self.save_natural_stimuli(recorddir)

    def _set_fpath(self):         # the file at self._fpath will be copied as a backup_images;
        self._fpath = __file__    # this function makes sure self._fpath points to *this* file

    def load_nets(self):
        self.optimizer.load_generator()

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

        '''
        self.load_nets()    # nets are not loaded in __init__; this enables exp to be run with multiprocessing
        self.istep = 0
        while self._istep < max_steps:
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

            else:
                # score images
                synscores = self.scorer.score(self.optimizer.current_images, self.optimizer.current_image_ids)
                t1 = time()
                # before update, backup_images codes (optimizer) and scores (scorer)
                self.optimizer.save_current_codes()
                self.optimizer.save_current_genealogy()
                self.scorer.save_current_scores()
                t2 = time()
                # use results to update optimizer
                self.optimizer.step(synscores)
                t3 = time()
                # summarize scores & delays
                print('synthetic img scores: mean {}, all {}'.format(np.nanmean(synscores), synscores))
                print(('step %d time: total %.2fs | ' +
                       'wait for results %.2fs  write records %.2fs  optimizer update %.2fs')
                      % (self.istep, t3 - t0, t1 - t0, t2 - t1, t3 - t2))

            self.logger.flush()
            self.istep += 1


if __name__ == '__main__':
    experiment = WithHumanExperiment(logdir=recorddir, random_seed=random_seed)
    experiment.run()
