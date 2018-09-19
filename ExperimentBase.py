from Logger import Tee
from utils import read_image, sort_nicely
from shutil import copyfile,rmtree
import numpy as np
import os


class ExperimentBase:
    def __init__(self, logdir, random_seed=None):
        self._scorer = None
        self._optimizers = None
        self._dynparams = {}
        self._istep = -1    # step we are currently at; will count from 0 once experiment starts

        # only used if natural stimuli are attached
        self._natstimuli = None            # a list of images: (w, h, c) arrays, uint8
        self._natstimids = None            # a list of strs
        self._natstimfns = None            # a list of strs
        self._natstimdir = None            # dir of natstim
        # self._natstimnames = None          # a list of strs
        self._natstim_cumuscores = None    # an array, float
        self._natstim_nscores = None       # an array, int
        self._natstim_scores = None        # an array, float

        # logging utilities
        self._fpath = None
        self._logger = Tee(os.path.join(logdir, 'log.txt'))    # helps print to stdout & save a copy to logfpath
        self._copy_self(os.path.join(logdir, 'setting.py'))

        # apply random seed
        if random_seed is None:
            print('Note: random seed not set')
        else:
            print('random seed set to %d for experiment' % random_seed)
            np.random.seed(seed=random_seed)
        self._randomseed = random_seed

    def load_nets(self):
        if self._optimizers is None:
            raise RuntimeError('cannot load nets before optimizers are attached')
        for optimizer in self._optimizers:
            optimizer.load_generator()

    def run(self):
        raise NotImplementedError

    def attach_optimizer(self, optimizer):
        if self._optimizers is None:
            self._optimizers = [optimizer]
            self._update_dynparams(optimizer.dynamic_parameters)
        else:
            self._optimizers.append(optimizer)
            self._dynparams = {}    # dynparams not supported when multiple optimizers are loaded

    def attach_scorer(self, scorer):
        self._scorer = scorer
        # self._update_dynparams(scorer.dynamic_parameters)

    def attach_natural_stimuli(self, natstimdir, size=None, natstim_catalogue_fpath=None, shuffle=False):
        natstimfns = sort_nicely([fn for fn in os.listdir(natstimdir) if
                                  '.bmp' in fn or '.jpg' in fn or '.png' in fn or '.jepg' in fn or '.JPEG' in fn])
        natstimnames = [fn[:fn.rfind('.')] for fn in natstimfns]
        nstimuli = len(natstimnames)
        if nstimuli == 0:
            raise Exception('no images found in natual stimulus directory %s' % natstimdir)
        if size is None:
            size = nstimuli
        else:
            size = int(size)

        # choose images to load (if nstimuli != size)
        toshow_args = np.arange(nstimuli)
        if nstimuli < size:
            if shuffle:
                print('note: number of natural images (%d) < requested (%d); sampling with replacement'
                      % (nstimuli, size))
                toshow_args = np.random.choice(toshow_args, size=size, replace=True)
            else:
                print('note: number of natural images (%d) < requested (%d); repeating images'
                      % (nstimuli, size))
                toshow_args = np.repeat(toshow_args, int(np.ceil(size / float(nstimuli))))[:size]
        elif nstimuli > size:
            if shuffle:
                print('note: number of natural images (%d) > requested (%d); sampling (no replacement)'
                      % (nstimuli, size))
                toshow_args = np.random.choice(toshow_args, size=size, replace=False)
            else:
                print('note: number of natural images (%d) > requested (%d); taking first %d images'
                      % (nstimuli, size, size))
                toshow_args = toshow_args[:size]
        natstimnames = [natstimnames[arg] for arg in toshow_args]
        natstimfns = [natstimfns[arg] for arg in toshow_args]

        # load images
        natstimuli = []
        for natstimfn in natstimfns:
            natstimuli.append(read_image(os.path.join(natstimdir, natstimfn)))

        # make ids
        natstimids = natstimnames[:]
        #   strip off initial [], if any
        for i, id_ in enumerate(natstimids):
            if id_[0] == '[' and ']' in id_:
                natstimids[i] = id_[id_.find(']') + 1:]
        #   try to map filename (no extension) to short id, if catalogue given
        if natstim_catalogue_fpath is not None:
            catalogue = np.load(natstim_catalogue_fpath)
            name_2_id = {name: id_ for name, id_ in zip(catalogue['stimnames'], catalogue['stimids'])}
            for i, name in enumerate(natstimids):
                try:
                    natstimids[i] = name_2_id[name]
                except KeyError:
                    natstimids[i] = name
        #   resolve nonunique ids, if any
        if nstimuli < size:
            for i, id_ in enumerate(natstimids):
                icopy = 1
                new_id = id_
                while new_id in natstimids[:i]:
                    new_id = '%s_copy%02d' % (id_, icopy)
                natstimids[i] = new_id

        print('showing the following %d natural stimuli loaded from %s:' % (size, natstimdir))
        print(natstimids)

        # save results
        self._natstimuli = natstimuli
        self._natstimids = natstimids
        self._natstimfns = natstimfns
        self._natstimdir = natstimdir
        # self._natstimnames = natstimnames
        self._natstim_cumuscores = np.zeros(nstimuli, dtype='float')
        self._natstim_nscores = np.zeros(nstimuli, dtype='int')
        self._natstim_scores = np.full(nstimuli, np.nan, dtype='float')

    def save_natural_stimuli(self, recorddir):
        assert (self._natstimfns is not None) and (self._natstimdir is not None),\
            'please load natural stimuli first by calling attach_natural_stimuli()'
        recorddir = os.path.join(recorddir, 'natural_stimuli')
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
                    raise OSError('trying to save natural stimuli but directory already exists: %s' % recorddir)
            else:
                raise
        for fn in self._natstimfns:
            copyfile(os.path.join(self._natstimdir, fn), os.path.join(recorddir, fn))

    def attach_natural_stimuli_old_ver(self, natstimdir, natstim_catalogue_fpath, size=None, toshow_args=None):
        """
        loads natural stimuli
        :param natstimdir: dir containing natural stimiuli images
        :param natstim_catalogue_fpath: path to catalogue (.npz file) containing stimnames and stimids
        :param size: ignored if toshow_args is set
        :param toshow_args: args to select which ones to show
        :return: None
        """
        if size is None and toshow_args is None:
            raise ValueError('size must be set if toshow_args is not set')
        
        catalogue = np.load(natstim_catalogue_fpath)
        natstimnames = catalogue['stimnames']    # full filename minus extension
        natstimids = catalogue['stimids']        # short readable unique name
        natstimids = np.array(['nat_%s' % id_ for id_ in natstimids], dtype=str)
        # natstimcats = catalogue['stimcats']      # catagory
        # id_2_idc = {natstimid: i for i, natstimid in enumerate(natstimids)}
        # natstim_presentations = np.zeros(natstimids.shape, dtype='int')
        natstimuli = []
        for natstimname in natstimnames:
            natstimuli.append(read_image(os.path.join(natstimdir, natstimname + '.bmp')))
        natstimuli = np.array(natstimuli)

        if toshow_args is None:
            args = np.arange(len(natstimids))
            if size <= len(natstimuli):
                toshow_args = np.random.choice(args, size=size, replace=False)
            else:
                toshow_args = np.random.choice(args, size=size, replace=True)
        
        natstimuli = natstimuli[toshow_args]
        natstimids = natstimids[toshow_args]
        # natstimnames = natstimnames[toshow_args]
        print('showing the following %d natural stimuli:loaded from %s:' % (len(toshow_args), natstimdir))
        print(natstimids)

        self._natstimuli = list(natstimuli)
        self._natstimids = list(natstimids)
        # self._natstimnames = list(natstimnames)
        self._natstim_cumuscores = np.zeros(len(natstimids), dtype='float')
        self._natstim_nscores = np.zeros(len(natstimids), dtype='int')
        self._natstim_scores = np.full(len(natstimids), np.nan, dtype='float')

    def _update_dynparams(self, dynparams):
        for name in dynparams.keys():
            if name in self._dynparams.keys():
                print('handle to dynamic parameter %s is being overwritten' % name)
            self._dynparams[name] = dynparams[name]

    def _copy_self(self, expfpath):
        self._set_fpath()
        copyfile(self._fpath, expfpath)

    def _set_fpath(self):
        self._fpath = __file__

    @property
    def istep(self):
        return self._istep

    @istep.setter
    def istep(self, istep):
        self._istep = max(0, int(istep))

    @property
    def optimizer(self):
        if self._optimizers is None:
            return None
        elif len(self._optimizers) == 1:
            return self._optimizers[0]
        else:
            raise RuntimeError('more than 1 (%d) optimizers have been loaded; asking for "optimizer" is ambiguous'
                               % len(self._optimizers))
    @property
    def optimizers(self):
        return self._optimizers

    @property
    def scorer(self):
        return self._scorer

    @property
    def natural_stimuli_scores(self):
        return self._natstim_scores

    @property
    def natural_stimuli(self):
        return self._natstimuli

    @property
    def natural_stimuli_ids(self):
        return self._natstimids

    @property
    def dynamic_parameters(self):
        return self._dynparams

    @property
    def logger(self):
        return self._logger
