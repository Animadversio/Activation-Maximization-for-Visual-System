from Generator import Generator
from DynamicParameter import DynamicParameter
import utils
import numpy as np
import os
from shutil import copyfile,rmtree


class Optimizer:
    def __init__(self, recorddir, random_seed=None, thread=None):
        if thread is not None:
            assert isinstance(thread, int), 'thread must be an integer'
        assert os.path.isdir(recorddir)
        assert isinstance(random_seed, int) or random_seed is None, 'random_seed must be an integer or None'

        self._generator = None
        self._istep = 0
        self._dynparams = {}

        self._curr_samples = None       # array of codes
        self._curr_images = None        # list of image arrays
        self._curr_sample_idc = None    # range object
        self._curr_sample_ids = None    # list
        self._next_sample_idx = 0       # scalar

        self._best_code = None
        self._best_score = None

        self._thread = thread
        if thread is not None:
            recorddir = os.path.join(recorddir, 'thread%02d' % self._thread)
            if not os.path.isdir(recorddir):
                os.mkdir(recorddir)
        self._recorddir = recorddir

        self._random_generator = np.random.RandomState()
        if random_seed is not None:
            if self._thread is not None:
                print('random seed set to %d for optimizer (thread %d)' % (random_seed, self._thread))
            else:
                print('random seed set to %d for optimizer' % random_seed)
            self._random_seed = random_seed
            self._random_generator = np.random.RandomState(seed=self._random_seed)

    def load_generator(self):
        self._generator = Generator()
        self._prepare_images()

    def _prepare_images(self):
        if self._generator is None:
            raise RuntimeError('generator not loaded. please run optimizer.load_generator() first')

        curr_images = []
        for sample in self._curr_samples:
            im_arr = self._generator.visualize(sample)
            curr_images.append(im_arr)
        self._curr_images = curr_images

    def step(self, scores):
        '''Take in score for each sample and generate a next generation of samples.'''
        raise NotImplementedError

    def save_current_state(self, image_size=None):
        utils.write_images(self._curr_images, self._curr_sample_ids, self._recorddir, image_size)
        utils.write_codes(self._curr_samples, self._curr_sample_ids, self._recorddir)

    def save_current_codes(self):
        utils.write_codes(self._curr_samples, self._curr_sample_ids, self._recorddir)

    @property
    def current_images(self):
        if self._curr_images is None:
            raise RuntimeError('Current images have not been initialized. Is generator loaded?')
        return self._curr_images

    @property
    def current_images_copy(self):
        return list(np.array(self._curr_images).copy())

    @property
    def current_image_ids(self):
        return self._curr_sample_ids

    @property
    def curr_image_idc(self):
        return self._curr_sample_idc

    @property
    def nsamples(self):
        return len(self._curr_samples)

    @property
    def dynamic_parameters(self):
        return self._dynparams


def mutate(population, genealogy, mutation_size, mutation_rate, random_generator):
    do_mutate = random_generator.random_sample(population.shape) < mutation_rate
    population_new = population.copy()
    population_new[do_mutate] += random_generator.normal(loc=0, scale=mutation_size, size=np.sum(do_mutate))
    genealogy_new = ['%s+mut' % gen for gen in genealogy]
    return population_new, genealogy_new


def mate(population, genealogy, fitness, new_size, random_generator, skew=0.5):
    """
    fitness > 0
    """
    # clean data
    assert len(population) == len(genealogy)
    assert len(population) == len(fitness)
    if np.max(fitness) == 0:
        fitness[np.argmax(fitness)] = 0.001
    if np.min(fitness) <= 0:
        fitness[fitness <= 0] = np.min(fitness[fitness > 0])

    fitness_bins = np.cumsum(fitness)
    fitness_bins /= fitness_bins[-1]
    parent1s = np.digitize(random_generator.random_sample(new_size), fitness_bins)
    parent2s = np.digitize(random_generator.random_sample(new_size), fitness_bins)
    new_samples = np.empty((new_size, population.shape[1]))
    new_genealogy = []
    for i in range(new_size):
        parentage = random_generator.random_sample(population.shape[1]) < skew
        new_samples[i, parentage] = population[parent1s[i]][parentage]
        new_samples[i, ~parentage] = population[parent2s[i]][~parentage]
        new_genealogy.append('%s+%s' % (genealogy[parent1s[i]], genealogy[parent2s[i]]))
    return new_samples, new_genealogy


class Genetic(Optimizer):
    def __init__(self, population_size, mutation_rate, mutation_size, kT_multiplier, recorddir,
                 parental_skew=0.5, n_conserve=0, random_seed=None, thread=None):
        super(Genetic, self).__init__(recorddir, random_seed, thread)

        # various parameters
        self._popsize = int(population_size)
        self._mut_rate = float(mutation_rate)
        self._mut_size = float(mutation_size)
        self._kT_mul = float(kT_multiplier)
        self._kT = None    # deprecated; will be overwritten
        self._n_conserve = int(n_conserve)
        assert(self._n_conserve < self._popsize)
        self._parental_skew = float(parental_skew)

        # initialize dynamic parameters & their types
        self._dynparams['mutation_rate'] = \
            DynamicParameter('d', self._mut_rate, 'probability that each gene will mutate at each step')
        self._dynparams['mutation_size'] = \
            DynamicParameter('d', self._mut_size, 'stdev of the stochastic size of mutation')
        self._dynparams['kT_multiplier'] = \
            DynamicParameter('d', self._kT_mul, 'used to calculate kT; kT = kT_multiplier * stdev of scores')
        self._dynparams['n_conserve'] = \
            DynamicParameter('i', self._n_conserve, 'number of best individuals kept unmutated in each step')
        self._dynparams['parental_skew'] = \
            DynamicParameter('d', self._parental_skew, 'amount inherited from one parent; 1 means no recombination')
        self._dynparams['population_size'] = \
            DynamicParameter('i', self._popsize, 'size of population')

        # initialize samples & indices
        self._init_population = self._random_generator.normal(loc=0, scale=1, size=(self._popsize, 4096))
        self._init_population_dir = None
        self._init_population_fns = None
        self._curr_samples = self._init_population.copy()    # curr_samples is current population of codes
        self._genealogy = ['standard_normal'] * self._popsize
        self._curr_sample_idc = range(self._popsize)
        self._next_sample_idx = self._popsize
        if self._thread is None:
            self._curr_sample_ids = ['gen%03d_%06d' % (self._istep, idx) for idx in self._curr_sample_idc]
        else:
            self._curr_sample_ids = ['thread%02d_gen%03d_%06d' %
                                     (self._thread, self._istep, idx) for idx in self._curr_sample_idc]

        # # reset random seed to ignore any calls during init
        # if random_seed is not None:
        #     random_generator.seed(random_seed)

    def load_init_population(self, initcodedir, size):
        # make sure we are at the beginning of experiment
        assert self._istep == 0, 'initialization only allowed at the beginning'
        # make sure size <= population size
        assert size <= self._popsize, 'size %d too big for population of size %d' % (size, self._popsize)
        # load codes
        init_population, genealogy = utils.load_codes2(initcodedir, size)
        # fill the rest of population if size==len(codes) < population size
        if len(init_population) < self._popsize:
            remainder_size = self._popsize - len(init_population)
            remainder_pop, remainder_genealogy = mate(
                init_population, genealogy,    # self._curr_sample_ids[:size],
                np.ones(len(init_population)), remainder_size,
                self._random_generator, self._parental_skew
            )
            remainder_pop, remainder_genealogy = mutate(
                remainder_pop, remainder_genealogy, self._mut_size, self._mut_rate, self._random_generator
            )
            init_population = np.concatenate((init_population, remainder_pop))
            genealogy = genealogy + remainder_genealogy
        # apply
        self._init_population = init_population
        self._init_population_dir = initcodedir
        self._init_population_fns = genealogy    # a list of '*.npy' file names
        self._curr_samples = self._init_population.copy()
        self._genealogy = ['[init]%s' % g for g in genealogy]
        # no update for idc, idx, ids because popsize unchanged
        try:
            self._prepare_images()
        except RuntimeError:    # this means generator not loaded; on load, images will be prepared
            pass

    def save_init_population(self):
        '''Record experimental parameter: initial population
        in the directory "[:recorddir]/init_population"
        '''
        assert (self._init_population_fns is not None) and (self._init_population_dir is not None),\
            'please load init population first by calling load_init_population();' + \
            'if init is not loaded from file, it can be found in experiment backup_images folder after experiment runs'
        recorddir = os.path.join(self._recorddir, 'init_population')
        try:
            os.mkdir(recorddir)
        except OSError as e:
            if e.errno == 17:
                # ADDED Sep.17, To let user delete the directory if existing during the system running.
                chs = input("Dir %s exist input y to delete the dir and write on it, n to exit" % recorddir)
                if chs is 'y':
                    print("Directory %s all removed." % recorddir)
                    rmtree(recorddir)
                    os.mkdir(recorddir)
                else:
                    raise OSError('trying to save init population but directory already exists: %s' % recorddir)
            else:
                raise
        for fn in self._init_population_fns:
            copyfile(os.path.join(self._init_population_dir, fn), os.path.join(recorddir, fn))

    def step(self, scores):
        # clean variables
        assert len(scores) == len(self._curr_samples), \
            'number of scores (%d) != population size (%d)' % (len(scores), len(self._curr_samples))
        new_size = self._popsize    # this may != len(curr_samples) if it has been dynamically updated
        new_samples = np.empty((new_size, self._curr_samples.shape[1]))
        # instead of chaining the genealogy, alias it at every step
        curr_genealogy = np.array(self._curr_sample_ids, dtype=str)
        new_genealogy = [''] * new_size    # np array not used because str len will be limited by len at init

        # deal with nan scores:
        nan_mask = np.isnan(scores)
        n_nans = int(np.sum(nan_mask))
        valid_mask = ~nan_mask
        n_valid = int(np.sum(valid_mask))
        if n_nans > 0:
            print('optimizer: missing %d scores for samples %s' % (n_nans, str(np.array(self._curr_sample_idc)[nan_mask])))
            if n_nans > new_size:
                print('Warning: n_nans > new population_size because population_size has just been changed AND ' +
                      'too many images failed to score. This will lead to arbitrary loss of some nan score images.')
            if n_nans > new_size - self._n_conserve:
                print('Warning: n_nans > new population_size - self._n_conserve. ' +
                      'IFF population_size has just been changed, ' +
                      'this will lead to aribitrary loss of some/all nan score images.')
            # carry over images with no scores
            thres_n_nans = min(n_nans, new_size)
            new_samples[-thres_n_nans:] = self._curr_samples[nan_mask][-thres_n_nans:]
            new_genealogy[-thres_n_nans:] = curr_genealogy[nan_mask][-thres_n_nans:]

        # if some images have scores
        if n_valid > 0:
            valid_scores = scores[valid_mask]
            self._kT = max((np.std(valid_scores) * self._kT_mul, 1e-8))    # prevents underflow kT = 0
            print('kT: %f' % self._kT)
            sort_order = np.argsort(valid_scores)[::-1]    # sort from high to low
            valid_scores = valid_scores[sort_order]
            # Note: if new_size is smalled than n_valid, low ranking images will be lost
            thres_n_valid = min(n_valid, new_size)
            new_samples[:thres_n_valid] = self._curr_samples[valid_mask][sort_order][:thres_n_valid]
            new_genealogy[:thres_n_valid] = curr_genealogy[valid_mask][sort_order][:thres_n_valid]

            # if need to generate new samples
            if n_nans < new_size - self._n_conserve:
                fitness = np.exp((valid_scores - valid_scores[0]) / self._kT)
                # skips first n_conserve samples
                n_mate = new_size - self._n_conserve - n_nans
                new_samples[self._n_conserve:thres_n_valid], new_genealogy[self._n_conserve:thres_n_valid] = \
                    mate(
                        new_samples[:thres_n_valid], new_genealogy[:thres_n_valid],
                        fitness, n_mate, self._random_generator, self._parental_skew
                    )
                new_samples[self._n_conserve:thres_n_valid], new_genealogy[self._n_conserve:thres_n_valid] = \
                    mutate(
                        new_samples[self._n_conserve:thres_n_valid], new_genealogy[self._n_conserve:thres_n_valid],
                        self._mut_size, self._mut_rate, self._random_generator
                    )

            # if any score turned out to be best
            if self._best_score is None or self._best_score < valid_scores[0]:
                self._best_score = valid_scores[0]
                self._best_code = new_samples[0].copy()

        self._istep += 1
        self._curr_samples = new_samples
        self._genealogy = new_genealogy
        self._curr_sample_idc = range(self._next_sample_idx, self._next_sample_idx + new_size)
        self._next_sample_idx += new_size
        if self._thread is None:
            self._curr_sample_ids = ['gen%03d_%06d' % (self._istep, idx) for idx in self._curr_sample_idc]
        else:
            self._curr_sample_ids = ['thread%02d_gen%03d_%06d' %
                                     (self._thread, self._istep, idx) for idx in self._curr_sample_idc]
        self._prepare_images()

    # def step_with_immigration(self, scores, immigrants, immigrant_scores):
    #     assert len(immigrants.shape) == 2, 'population is not batch sized (dim != 2)'
    #     self._curr_samples = np.concatenate((self._curr_samples, immigrants))
    #     scores = np.concatenate((scores, immigrant_scores))
    #     self.step(scores)

    def add_immigrants(self, codedir, size, ignore_conserve=False):
        if not ignore_conserve:
            assert size <= len(self._curr_samples) - self._n_conserve,\
                'size of immigrantion should be <= size of unconserved population because ignore_conserve is False'
        else:
            assert size < len(self._curr_samples), 'size of immigrantion should be < size of population'
            if size > len(self._curr_samples) - self._n_conserve:
                print('Warning: some conserved codes are being overwritten')

        immigrants, immigrant_codefns = utils.load_codes2(codedir, size)
        n_immi = len(immigrants)
        n_conserve = len(self._curr_samples) - n_immi
        self._curr_samples = np.concatenate((self._curr_samples[:n_conserve], immigrants))
        self._genealogy = self._genealogy[:n_conserve] + ['[immi]%s' % fn for fn in immigrant_codefns]
        next_sample_idx = self._curr_sample_idc[n_conserve] + n_immi
        self._curr_sample_idc = range(self._curr_sample_idc[0], next_sample_idx)
        self._next_sample_idx = next_sample_idx
        if self._thread is None:
            new_sample_ids = ['gen%03d_%06d' % (self._istep, idx) for idx in self._curr_sample_idc[n_conserve:]]
        else:
            new_sample_ids = ['thread%02d_gen%03d_%06d' %
                              (self._thread, self._istep, idx) for idx in self._curr_sample_idc[n_conserve:]]
        self._curr_sample_ids = self._curr_sample_ids[:n_conserve] + new_sample_ids
        self._prepare_images()

    def update_dynamic_parameters(self):
        if self._dynparams['mutation_rate'].value != self._mut_rate:
            self._mut_rate = self._dynparams['mutation_rate'].value
            print('updated mutation_rate to %f at step %d' % (self._mut_rate, self._istep))
        if self._dynparams['mutation_size'].value != self._mut_size:
            self._mut_size = self._dynparams['mutation_size'].value
            print('updated mutation_size to %f at step %d' % (self._mut_size, self._istep))
        if self._dynparams['kT_multiplier'].value != self._kT_mul:
            self._kT_mul = self._dynparams['kT_multiplier'].value
            print('updated kT_multiplier to %.2f at step %d' % (self._kT_mul, self._istep))
        if self._dynparams['parental_skew'].value != self._parental_skew:
            self._parental_skew = self._dynparams['parental_skew'].value
            print('updated parental_skew to %.2f at step %d' % (self._parental_skew, self._istep))
        if self._dynparams['population_size'].value != self._popsize or \
                self._dynparams['n_conserve'].value != self._n_conserve:
            n_conserve = self._dynparams['n_conserve'].value
            popsize = self._dynparams['population_size'].value
            if popsize < n_conserve:            # both newest
                if popsize == self._popsize:    # if popsize hasn't changed
                    self._dynparams['n_conserve'].set_value(self._n_conserve)
                    print('rejected n_conserve update: new n_conserve > old population_size')
                else:                           # popsize has changed
                    self._dynparams['population_size'].set_value(self._popsize)
                    print('rejected population_size update: new population_size < new/old n_conserve')
                    if n_conserve <= self._popsize:
                        self._n_conserve = n_conserve
                        print('updated n_conserve to %d at step %d' % (self._n_conserve, self._istep))
                    else:
                        self._dynparams['n_conserve'].set_value(self._n_conserve)
                        print('rejected n_conserve update: new n_conserve > old population_size')
            else:
                if self._popsize != popsize:
                    self._popsize = popsize
                    print('updated population_size to %d at step %d' % (self._popsize, self._istep))
                if self._n_conserve != n_conserve:
                    self._n_conserve = n_conserve
                    print('updated n_conserve to %d at step %d' % (self._n_conserve, self._istep))

    def save_current_genealogy(self):
        savefpath = os.path.join(self._recorddir, 'genealogy_gen%03d.npz' % self._istep)
        save_kwargs = {'image_ids': np.array(self._curr_sample_ids, dtype=str),
                       'genealogy': np.array(self._genealogy, dtype=str)}
        utils.savez(savefpath, save_kwargs)

    @property
    def generation(self):
        '''Return current step number'''
        return self._istep

# TODO: Finish the CMAES optimizer
class CMAES(Optimizer):
    def __init__(self, population_size, mutation_rate, mutation_size, kT_multiplier, recorddir,
                 parental_skew=0.5, n_conserve=0, random_seed=None, thread=None):
        super(CMAES, self).__init__(recorddir, random_seed, thread)

#
# class FDGD(Optimizer):
#     def __init__(self, nsamples, mutation_size, learning_rate, antithetic=True, init_code=None):
#         self._nsamples = int(nsamples)
#         self._mut_size = float(mutation_size)
#         self._lr = float(learning_rate)
#         self._antithetic = antithetic
#         self.parameters = {'mutation_size': mutation_size, 'learning_rate': learning_rate,
#                            'nsamples': nsamples, 'antithetic': antithetic}
#
#         if init_code is not None:
#             self._init_code = init_code.copy().reshape(4096)
#         else:
#             # self.init_code = np.random.normal(loc=0, scale=1, size=(4096))
#             self._init_code = np.zeros(shape=(4096,))
#         self._curr = self._init_code.copy()
#         self._best_code = self._curr.copy()
#         self._best_score = None
#
#         self._pos_isteps = None
#         self._norm_isteps = None
#
#         self._prepare_next_samples()
#
#     def _prepare_next_samples(self):
#         self._pos_isteps = np.random.normal(loc=0, scale=self._mut_size, size=(self._nsamples, len(self._curr)))
#         self._norm_isteps = np.linalg.norm(self._pos_isteps, axis=1)
#
#         pos_samples = self._pos_isteps + self._curr
#         if self._antithetic:
#             neg_samples = -self._pos_isteps + self._curr
#             self._curr_samples = np.concatenate((pos_samples, neg_samples))
#         else:
#             self._curr_samples = np.concatenate((pos_samples, (self._curr.copy(),)))
#
#         self._curr_sample_idc = range(self._next_sample_idx, self._next_sample_idx + len(self._curr_samples))
#         self._next_sample_idx += len(self._curr_samples)
#
#         curr_images = []
#         for sample in self._curr_samples:
#             im_arr = self._generator.visualize(sample)
#             curr_images.append(im_arr)
#         self._curr_images = curr_images
#
#     def step(self, scores):
#         """
#         Use scores for current samples to update samples
#         :param scores: array or list of scalar scores, one for each current sample, in order
#         :param write:
#             if True, immediately writes after samples are prepared.
#             if False, user need to call .write_images(path)
#         :return: None
#         """
#         scores = np.array(scores)
#         assert len(scores) == len(self._curr_samples),\
#             'number of scores (%d) and number of samples (%d) are different' % (len(scores), len(self._curr_samples))
#
#         pos_scores = scores[:self._nsamples]
#         if self._antithetic:
#             neg_scores = scores[self._nsamples:]
#             dscore = (pos_scores - neg_scores) / 2.
#         else:
#             dscore = pos_scores - scores[-1]
#
#         grad = np.mean(dscore.reshape(-1, 1) * self._pos_isteps * (self._norm_isteps ** -2).reshape(-1, 1), axis=0)
#         self._curr += self._lr * grad
#
#         score_argmax = np.argsort(scores)[-1]
#         if self._best_score is None or self._best_score < scores[score_argmax]:
#             self._best_score = scores[score_argmax]
#             self._best_code = self._curr_samples[score_argmax]
#
#         self._prepare_next_samples()
