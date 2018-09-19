from ExperimentBase import ExperimentBase
from Optimizer import Genetic
from Scorer import EPhysScorer
from utils import set_dynamic_parameters_by_file, write_dynamic_parameters_to_file
import numpy as np
import os
from time import time

# Main experiment script
np.set_printoptions(precision=4, suppress=True)

# set file directories for I/O
homedir = os.path.expanduser('~')
initcodedir = os.path.join(homedir, 'Documents/stimuli/texture006')
natstimdir = os.path.join(homedir, 'Documents/stimuli/natimages-guapoCh9')

# projectdir = '/Volumes/Neurobio/LivingstoneLab/Stimuli/2018-ProjectGenerate'
# initcodedir = os.path.join(projectdir, 'restart')
projectdir = os.path.join('/home/poncelab/shared')
blockwritedir = projectdir
matdir = projectdir
recorddir = os.path.join(projectdir, 'backup')
dynparam_fpath = 'dynparam.txt'
if not os.path.isdir(recorddir):
    os.mkdir(recorddir)

# check existence of file and direcrtory
for dir_ in (initcodedir, natstimdir, blockwritedir, matdir, recorddir):
    if not os.path.isdir(dir_):
        raise OSError('directory not found: %s' % dir_)

# set parameters
#   for scorer, which also handles image I/O
image_size = 83             # size (height/width) of images
ichannel = None             # index or list of indices for channels to select (0-based indexing); None means average all
reps = 1                    # times each image is shown
#   for genetic optimizer
population_size = 30        # size of population each generation
mutation_rate = 0.25        # fraction of code elements to mutate(on average); range 0 - 1
mutation_size = 0.75        # magnitude of mutation (on average); meaningful range 0 - ~1.5
kT_multiplier = 2           # selective pressure, with lower being more selective; range 0 - inf
n_conserve = 10             # number of best images to keep untouched per generation; range 0 - populationsize
parental_skew = 0.75        # how much one parent (of 2) contributes to each progeny; meaningful range 0.5 - 1
#   for immigration
do_immigration = False
immi_size = 10
#   for overall
random_seed = 0             # seed for all random generators used in the experiment (to ensure reproducibility)
n_natural_stimuli = None    # number of natural stimuli to show; None means default to however many is in natstimdir


class OnlineWithImmigrationExperiment(ExperimentBase):
    def __init__(self, logdir, random_seed=None):
        super(OnlineWithImmigrationExperiment, self).__init__(logdir, random_seed)

        # load & backup_images natural stimuli
        self.attach_natural_stimuli(natstimdir, n_natural_stimuli)
        self.save_natural_stimuli(recorddir)
        block_size = len(self.natural_stimuli) + population_size    # automatically computed after n_natstim is known

        # initialize optimizer and scorer
        optimizer = Genetic(population_size=population_size, mutation_rate=mutation_rate, mutation_size=mutation_size,
                            n_conserve=n_conserve, kT_multiplier=kT_multiplier, parental_skew=parental_skew,
                            random_seed=random_seed, recorddir=recorddir)
        optimizer.load_init_population(initcodedir, size=population_size)
        optimizer.save_init_population()
        scorer = EPhysScorer(writedir=blockwritedir, backupdir=recorddir, respdir=matdir,
                             block_size=block_size, image_size=image_size, channel=ichannel,
                             random_seed=random_seed, reps=reps)
        self.attach_optimizer(optimizer)
        self.attach_scorer(scorer)

        # initialized dynparam I/O file
        write_dynamic_parameters_to_file(dynparam_fpath, self.dynamic_parameters)

    def _set_fpath(self):         # the file at self._fpath will be copied as a backup_images;
        self._fpath = __file__    # this function makes sure self._fpath points to *this* file

    def run(self):
        self.load_nets()    # nets are not loaded in __init__; this enables exp to be run with multiprocessing
        self.istep = 0
        while True:
            print('\n>>> step %d' % self.istep)
            t00 = time()

            # load dynamic parameters from file, check them, apply update, then re-write file
            set_dynamic_parameters_by_file(dynparam_fpath, self.dynamic_parameters)
            self.optimizer.update_dynamic_parameters()
            write_dynamic_parameters_to_file(dynparam_fpath, self.dynamic_parameters)
            t01 = time()

            # before scoring, backup_images codes (optimizer)
            self.optimizer.save_current_codes()
            self.optimizer.save_current_genealogy()
            t02 = time()

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
                   'wait for results %.2fs  optimizer update %.2fs  write records %.2fs  update dynparams %.2fs')
                  % (self.istep, t3 - t00, t1 - t02, t3 - t2, t02 - t01 + t2 - t1, t01 - t00))
            self.logger.flush()

            # criterion for doing immigration
            if do_immigration and self.istep % 10 == 1:
                print('introducing immigrants')
                self.optimizer.add_immigrants(initcodedir, immi_size)

            self.istep += 1


if __name__ == '__main__':
    experiment = OnlineWithImmigrationExperiment(logdir=recorddir, random_seed=random_seed)
    experiment.run()
