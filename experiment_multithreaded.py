from ExperimentBase import ExperimentBase
from Optimizer import Genetic
from Scorer import EPhysScorer
import numpy as np
import os
from time import time

# Main experiment script
np.set_printoptions(precision=4, suppress=True)

# set file directories for I/O
homedir = os.path.expanduser('~')
initcodedir = os.path.join(homedir, 'Documents/static_data/Kreiman Lab/stimuli/Screening-3-sets')
natstimdir = os.path.join(homedir, 'Documents/static_data/Kreiman Lab/stimuli/Screening-3-sets')
# projectdir = '/Volumes/Neurobio/LivingstoneLab/Stimuli/2018-ProjectGenerate'
# initcodedir = os.path.join(projectdir, 'restart')
projectdir = os.path.join(homedir, 'Documents/playtest/test')
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
image_size = 83
block_size = 80
ichannel = None   # None for averaging across all channels (per thread)
reps = 1
#   for optimizer
population_size = 40
mutation_rate = 0.25
mutation_size = 0.75
kT_multiplier = 2
n_conserve = 10
parental_skew = 0.75
#   for overall
random_seed = 10
n_natural_stimuli = 40
#   for multithreading
nchannels_per_thread = 16
nthreads = 2

# re-format ichannel when multithreading
if nthreads > 1:
    if ichannel is None:
        ichannel = ...    # ellipsis means select all
    ichannel_new = []
    for thread in range(nthreads):
        ichannel_new.append((thread * nchannels_per_thread + np.arange(nchannels_per_thread))[ichannel])
    ichannel = ichannel_new
print('MultithreadedExperiment: listenning on the following channel(s) for each thread')
for thread in range(nthreads):
    print('\tthread %d: %s' % (thread, str(ichannel[thread])))


class MultithreadedExperiment(ExperimentBase):
    def __init__(self, logdir, random_seed=None):
        super(MultithreadedExperiment, self).__init__(logdir, random_seed)

        # initialize optimizer and scorer
        for thread in range(nthreads):
            if nthreads == 1:
                thread = None
            optimizer = Genetic(population_size=population_size, mutation_rate=mutation_rate, mutation_size=mutation_size,
                                n_conserve=n_conserve, kT_multiplier=kT_multiplier, parental_skew=parental_skew,
                                random_seed=random_seed, thread=thread, recorddir=recorddir)
            optimizer.load_init_population(initcodedir, size=population_size)
            optimizer.save_init_population()
            self.attach_optimizer(optimizer)
        scorer = EPhysScorer(writedir=blockwritedir, backupdir=recorddir, respdir=matdir,
                             block_size=block_size, image_size=image_size, channel=ichannel,
                             random_seed=random_seed, reps=reps)
        self.attach_scorer(scorer)

        # load & backup_images natural stimuli
        self.attach_natural_stimuli(natstimdir, n_natural_stimuli)
        self.save_natural_stimuli(recorddir)

    def _set_fpath(self):         # the file at self._fpath will be copied as a backup_images;
        self._fpath = __file__    # this function makes sure self._fpath points to *this* file

    def run(self):
        self.load_nets()    # nets are not loaded in __init__; this enables exp to be run with multiprocessing
        self.istep = 0
        while True:
            print('\n>>> step %d' % self.istep)
            t00 = time()

            # before scoring, backup_images codes (optimizer)
            for optimizer in self.optimizers:
                optimizer.save_current_codes()
                optimizer.save_current_genealogy()
            t01 = time()

            # get scores of images:
            #    1) combine synthesized & natural images
            #    2) write images to disk for evaluation; also, copy them to backup_images
            #    3) wait for & read results
            syn_nimgs = 0
            syn_sections = [0]
            syn_images = []
            syn_image_ids = []
            for optimizer in self.optimizers:
                syn_nimgs += optimizer.nsamples
                syn_sections.append(syn_nimgs)
                syn_images += optimizer.current_images
                syn_image_ids += optimizer.current_image_ids
            combined_scores = self.scorer.score(syn_images + self.natural_stimuli,
                                                syn_image_ids + self.natural_stimuli_ids)
            t1 = time()

            # after scoring, backup_images scores (scorer)
            self.scorer.save_current_scores()
            t2 = time()

            # use results to update optimizer
            threads_synscores = []
            threads_natscores = []
            for i, optimizer in enumerate( self.optimizers):
                thread_synscores = combined_scores[syn_sections[i]:syn_sections[i+1], i]
                thread_natscores = combined_scores[syn_nimgs:, i]
                if len(thread_synscores.shape) > 1:    # if scores for list of channels returned, pool channels
                    thread_synscores = np.mean(thread_synscores, axis=-1)
                    thread_natscores = np.mean(thread_natscores, axis=-1)    # unused by optimizer but used in printout
                threads_synscores.append(thread_synscores)
                threads_natscores.append(thread_natscores)
                optimizer.step(thread_synscores)       # update optimizer
            t3 = time()

            # summarize scores & delays, & save log
            for thread in range(nthreads):
                if not nthreads == 1:
                    print('thread %d: ' % thread)
                print('synthetic img scores: mean {}, all {}'.
                      format(np.nanmean(threads_synscores[thread]), threads_synscores[thread]))
                print('natural image scores: mean {}, all {}'.
                      format(np.nanmean(threads_natscores[thread]), threads_natscores[thread]))
            print(('step %d time: total %.2fs | ' +
                   'wait for results %.2fs  optimizer update %.2fs  write records %.2fs')
                  % (self.istep, t3 - t00, t1 - t01, t3 - t2, t2 - t1 + t01 - t00))
            self.logger.flush()

            self.istep += 1


if __name__ == '__main__':
    experiment = MultithreadedExperiment(logdir=recorddir, random_seed=random_seed)
    print('initialized')
    experiment.run()
