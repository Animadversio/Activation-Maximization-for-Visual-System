from ExperimentBase import ExperimentBase
from Optimizer import Genetic
from Scorer import ShuffledEPhysScorer
import numpy as np
import os
from time import time

np.set_printoptions(precision=4, suppress=True)

# set file directories for I/O
homedir = os.path.expanduser('~')
projectdir = os.path.join(homedir, 'Documents/playtest/shuffled_control/ringo_site10-180405')
initcodedir = os.path.join(homedir,
                           'Documents/playtest/180405/StimuliShared_gen1_george_180405_site7/backup/init_population')
natstimdir = os.path.join(homedir,
                          'Documents/playtest/180405/StimuliShared_gen1_george_180405_site7/backup/natural_stimuli')
blockwritedir = projectdir
matdir = projectdir
recorddir = os.path.join(projectdir, 'backup')
if not os.path.isdir(recorddir):
    os.mkdir(recorddir)

# check existence of file and direcrtory
for dir_ in (initcodedir, natstimdir, blockwritedir, matdir, recorddir):
    if not os.path.isdir(dir_):
        raise OSError('directory not found: %s' % dir_)

# set parameters
#   for scorer, which also handles image I/O
image_size = 83
block_size = 80
ichannel = None
reps = 1
#   for optimizer
population_size = 40
mutation_rate = 0.25
mutation_size = 0.75
kT_multiplier = 2
n_conserve = 10
parental_skew = 0.75
#   for immigration
do_immigration = False
immi_size = 10
#   for overall
random_seed = 0
n_natural_stimuli = 40


class ShuffledControlExperiment(ExperimentBase):
    def __init__(self, logdir, random_seed=None):
        super(ShuffledControlExperiment, self).__init__(logdir, random_seed)

        # initialize optimizer and scorer
        optimizer = Genetic(population_size=population_size, mutation_rate=mutation_rate, mutation_size=mutation_size,
                            n_conserve=n_conserve, kT_multiplier=kT_multiplier, parental_skew=parental_skew,
                            random_seed=random_seed, recorddir=recorddir)
        optimizer.load_init_population(initcodedir, size=population_size)
        optimizer.save_init_population()
        scorer = ShuffledEPhysScorer(writedir=blockwritedir, backupdir=recorddir, respdir=matdir,
                                     block_size=block_size, image_size=image_size, channel=ichannel,
                                     random_seed=random_seed, reps=reps, shuffle_first_n=40)
        self.attach_optimizer(optimizer)
        self.attach_scorer(scorer)

        # load & backup_images natural stimuli
        self.attach_natural_stimuli(natstimdir, n_natural_stimuli)
        # catalogue = np.load(natstim_catalogue_fpath)
        # sorted_toshow_args = catalogue['ringo_sortargs']    # high to low scores given by monkey
        # nat_toshow_args = np.concatenate((sorted_toshow_args[:15],       # first 15
        #                                   sorted_toshow_args[46:61],     # middle 15
        #                                   sorted_toshow_args[-10:],))    # last 10
        # self.attach_natural_stimuli_old_ver(natstimdir, natstim_catalogue_fpath, toshow_args=nat_toshow_args)

    def _set_fpath(self):         # the file at self._fpath will be copied as a backup_images;
        self._fpath = __file__    # this function makes sure self._fpath points to *this* file

    def run(self):
        self.load_nets()    # nets are not loaded in __init__; this enables exp to be run with multiprocessing
        self.istep = 0
        while True:
            print('\n>>> step %d' % self.istep)
            t00 = time()

            # before scoring, backup_images codes (optimizer)
            self.optimizer.save_current_codes()
            self.optimizer.save_current_genealogy()
            t01 = time()

            # get scores of images:
            #    1) combine synthesized & natural images
            #    2) write images to disk for evaluation; also, copy them to backup_images
            #    3) wait for & read results
            syn_nimgs = self.optimizer.nsamples
            combined_scores = self.scorer.score(self.optimizer.current_images + self.natural_stimuli,
                                                self.optimizer.current_image_ids + self.natural_stimuli_ids)
            synscores = combined_scores[:syn_nimgs]
            natscores = combined_scores[syn_nimgs:]
            t1 = time()

            # after scoring, backup_images scores (scorer)
            self.scorer.save_current_scores()
            t2 = time()

            # use results to update optimizer
            if len(synscores.shape) > 1:    # if scores for list of channels returned, pool channels
                synscores = np.mean(synscores, axis=-1)
                natscores = np.mean(natscores, axis=-1)    # not used by optimizer, but used in printout
            self.optimizer.step(synscores)
            t3 = time()

            # summarize scores & delays, & save log
            print('synthetic img scores: mean {}, all {}'.format(np.nanmean(synscores), synscores))
            print('natural image scores: mean {}, all {}'.format(np.nanmean(natscores), natscores))
            print(('step %d time: total %.2fs | ' +
                   'wait for results %.2fs  optimizer update %.2fs  write records %.2fs')
                  % (self.istep, t3 - t00, t1 - t01, t3 - t2, t2 - t1 + t01 - t00))
            self.logger.flush()

            # criterion for doing immigration
            if do_immigration and self.istep % 10 == 1:
                print('introducing immigrants')
                self.optimizer.add_immigrants(initcodedir, immi_size)

            self.istep += 1


if __name__ == '__main__':
    experiment = ShuffledControlExperiment(logdir=recorddir, random_seed=random_seed)
    experiment.run()
