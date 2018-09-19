import os
import shutil
from time import time, sleep

import numpy as np

import utils


class Scorer:
    def __init__(self, backupdir, **kwargs):
        """
        :param backupdir: directory to save scores & other backup_images
        """
        assert os.path.isdir(backupdir), 'invalid backup_images directory: %s' % backupdir

        self._curr_images = None
        self._curr_imgids = None
        self._curr_scores = None
        self._backupdir = backupdir

    def score(self, images, image_ids):
        """
        :param images: a list of numpy arrays of dimensions (h, w, c) containing an image (uint8) to be scored
        :param image_ids: int or str as unique identifier of each image
        :param name_prefix: str of prefix to use before uid in filename
        :return: a score for an image array
        """
        raise NotImplementedError


class BlockWriter:
    def __init__(self, writedir, backupdir, block_size=None,
                 reps=1, image_size=None, random_seed=None, cleanupdir=None):
        self._writedir = writedir
        self._backupdir = backupdir
        self._cleanupdir = cleanupdir
        self._images = None
        self._imgids = None
        self._nimgs = None
        self._imsize = None
        self._curr_block_imgfns = []
        self._imgfn_2_imgid = {}
        self._imgid_2_local_idx = None
        self._remaining_times_toshow = None
        self._iblock = -1
        self._iloop = -1
        self._blocksize = None
        self._reps = None
        self._random_generator = np.random.RandomState()

        self.set_block_size(block_size)
        self.set_reps(reps)
        self.set_image_size(image_size)
        if random_seed is not None:
            self.set_random_seed(random_seed)

    def set_block_size(self, block_size):
        assert isinstance(block_size, int) or block_size is None, 'block_size must be an integer or None'
        self._blocksize = block_size

    def set_reps(self, reps):
        assert isinstance(reps, int), 'reps must be an integer'
        self._reps = reps

    def set_image_size(self, image_size):
        assert isinstance(image_size, int) or image_size is None, 'block_size must be an integer or None'
        self._imsize = image_size

    def set_random_seed(self, random_seed):
        assert isinstance(random_seed, int) or random_seed is None, 'random_seed must be an integer or None'
        if random_seed is not None:
            self._random_generator = np.random.RandomState(seed=random_seed)
            print('random seed set to %d for %s' % (random_seed, self.__class__.__name__))

    def cleanup(self):
        for image_fn in [fn for fn in os.listdir(self._writedir) if '.bmp' in fn]:
            try:
                if self._cleanupdir is None:
                    os.remove(os.path.join(self._writedir, image_fn))
                else:
                    shutil.move(os.path.join(self._writedir, image_fn), os.path.join(self._cleanupdir, image_fn))
            except OSError:
                print('failed to clean up %s' % image_fn)

    def show_images(self, images, imgids):
        nimgs = len(images)
        assert len(imgids) == nimgs
        if self._blocksize is not None:
            assert nimgs >= self._blocksize, 'not enough images for block'

        self._images = images
        self._imgids = imgids
        self._nimgs = nimgs
        self._remaining_times_toshow = np.full(len(images), self._reps, dtype=int)
        self._imgid_2_local_idx = {imgid: i for i, imgid in enumerate(imgids)}
        self._iloop = -1

    def write_block(self):
        assert self._images is not None, 'no images loaded'
        if self._blocksize is not None:
            blocksize = self._blocksize
        else:
            blocksize = len(self._images)

        self._iblock += 1
        self._iloop += 1

        view = self._random_generator.permutation(self._nimgs)
        prioritized_view = np.argsort(self._remaining_times_toshow[view])[::-1][:blocksize]
        block_images = self._images[view[prioritized_view]]
        block_imgids = self._imgids[view[prioritized_view]]
        block_ids = ['block%03d_%02d' % (self._iblock, i) for i in range(blocksize)]
        block_imgfns = ['%s_%s.bmp' % (blockid, imgid) for blockid, imgid in zip(block_ids, block_imgids)]
        imgfn_2_imgid = {name: imgid for name, imgid in zip(block_imgfns, block_imgids)}
        utils.write_images(block_images, block_imgfns, self._writedir, self._imsize)

        self._curr_block_imgfns = block_imgfns
        self._imgfn_2_imgid = imgfn_2_imgid
        self._remaining_times_toshow[view[prioritized_view]] -= 1
        return imgfn_2_imgid

    def backup_images(self):
        for imgfn in self._curr_block_imgfns:
            try:
                shutil.copyfile(os.path.join(self._writedir, imgfn), os.path.join(self._backupdir, imgfn))
            except OSError:
                print('failed to backup_images %s' % imgfn)

    def show_again(self, imgids):
        for imgid in imgids:
            try:
                local_idx = self._imgid_2_local_idx[imgid]
                self._remaining_times_toshow[local_idx] += 1
            except KeyError:
                print('BlockWriter: warning: cannot show image %s again; image is not registered' % imgid)

    @property
    def iblock(self):
        return self._iblock

    @property
    def iloop(self):
        return self._iloop

    @property
    def done(self):
        return np.all(self._remaining_times_toshow <= 0)

    @property
    def block_size(self):
        return self._blocksize


class WithIOScorer(Scorer):
    def __init__(self, writedir, backupdir, image_size=None, random_seed=None, **kwargs):
        super(WithIOScorer, self).__init__(backupdir)

        assert os.path.isdir(writedir), 'invalid write directory: %s' % writedir
        self._writedir = writedir
        self._score_shape = tuple()    # use an empty tuple to indicate shape of score is a scalar (not even a 1d array)
        self._curr_nimgs = None
        self._curr_listscores = None
        self._curr_cumuscores = None
        self._curr_nscores = None
        self._curr_scores_mat = None
        self._curr_imgfn_2_imgid = None
        self._istep = -1

        self._blockwriter = BlockWriter(self._writedir, self._backupdir)
        self._blockwriter.set_image_size(image_size)
        self._blockwriter.set_random_seed(random_seed)
        self._random_seed = random_seed

        self._require_response = False
        self._verbose = False

    def score(self, images, image_ids):
        nimgs = len(images)
        assert len(image_ids) == nimgs
        if self._blockwriter.block_size is not None:
            assert nimgs >= self._blockwriter.block_size, 'too few images for block'
        for imgid in image_ids:
            if not isinstance(imgid, str):
                raise ValueError('datatype of image_id %s not understood' % str(type(image_ids[0])))

        self._istep += 1

        self._curr_imgids = np.array(image_ids, dtype=str)
        self._curr_images = np.array(images)
        self._curr_nimgs = nimgs
        blockwriter = self._blockwriter
        blockwriter.show_images(self._curr_images, self._curr_imgids)
        self._curr_listscores = [[] for _ in range(nimgs)]
        self._curr_cumuscores = np.zeros((nimgs, *self._score_shape), dtype='float')
        self._curr_nscores = np.zeros(nimgs, dtype='int')
        while not blockwriter.done:
            t0 = time()

            self._curr_imgfn_2_imgid = blockwriter.write_block()
            t1 = time()

            blockwriter.backup_images()
            t2 = time()

            scores, scores_local_idx, novel_imgfns = self._get_scores()
            if self._score_shape == (0,) and len(scores) > 0:    # if score_shape is the inital placeholder
                self._score_shape = scores[0].shape
                self._curr_cumuscores = np.zeros((nimgs, *self._score_shape), dtype='float')
            for score, idx in zip(scores, scores_local_idx):
                self._curr_listscores[idx].append(score)
                self._curr_cumuscores[idx] += score
                self._curr_nscores[idx] += 1
            if self._require_response:
                unscored_imgids = set(self._curr_imgfn_2_imgid.values()) - set(self._curr_imgids[scores_local_idx])
                blockwriter.show_again(unscored_imgids)
            t3 = time()

            blockwriter.cleanup()
            t4 = time()

            # report delays
            if self._verbose:
                print(('block %03d time: total %.2fs | ' +
                       'write images %.2fs  backup_images images %.2fs  ' +
                       'wait for results %.2fs  clean up images %.2fs  (loop %d)') %
                      (blockwriter.iblock, t4 - t0, t1 - t0, t2 - t1, t3 - t2, t4 - t3, blockwriter.iloop))
                if len(novel_imgfns) > 0:
                    print('novel images:  {}'.format(sorted(novel_imgfns)))

        # consolidate & save data before returning
        # calculate average score
        scores = np.empty(self._curr_cumuscores.shape)
        valid_mask = self._curr_nscores != 0
        scores[~valid_mask] = np.nan
        if np.sum(valid_mask) > 0:    # if any valid scores
            if len(self._curr_cumuscores.shape) == 2:
                # if multiple channels, need to reshape nscores for correct array broadcasting
                scores[valid_mask] = self._curr_cumuscores[valid_mask] / self._curr_nscores[:, np.newaxis][valid_mask]
            else:
                scores[valid_mask] = self._curr_cumuscores[valid_mask] / self._curr_nscores[valid_mask]
        # make matrix of all individual scores
        scores_mat = np.full((*scores.shape, max(self._curr_nscores)), np.nan)
        for i in range(len(self._curr_imgids)):
            if self._curr_nscores[i] > 0:
                scores_mat[i, ..., :self._curr_nscores[i]] = np.array(self._curr_listscores[i]).T
        # record scores
        self._curr_scores = scores
        self._curr_scores_mat = scores_mat
        return scores    # shape of (nimgs, [nchannels,])

    def _get_scores(self):
        raise NotImplementedError

    def save_current_scores(self):
        savefpath = os.path.join(self._backupdir, 'scores_end_block%03d.npz' % self._blockwriter.iblock)
        save_kwargs = {'image_ids': self._curr_imgids, 'scores': self._curr_scores,
                       'scores_mat': self._curr_scores_mat, 'nscores': self._curr_nscores}
        print('saving scores to %s' % savefpath)
        utils.save_scores(savefpath, save_kwargs)

        
class EPhysScorer(WithIOScorer):
    def __init__(self, writedir, backupdir, block_size, channel=None, reps=1, image_size=None, random_seed=None,
                 respdir=None, require_response=False, verbose=True, match_imgfn_policy='strict'):
        super(EPhysScorer, self).__init__(writedir, backupdir, image_size, random_seed)
        self._blockwriter.set_reps(reps)
        self._blockwriter.set_block_size(block_size)

        if channel is None:
            self._channel = channel                # self._score_shape defaults to empty tuple()
        elif isinstance(channel, int):
            self._channel = channel
        elif channel is ...:
            self._channel = channel
            self._score_shape = (0,)               # placeholder; will be overwritten later
        elif hasattr(channel, '__len__'):
            channel_new = []
            for c in channel:
                if hasattr(c, '__len__'):
                    channel_new.append(np.array(c, dtype=int))
                else:
                    channel_new.append(int(c))
            self._channel = channel_new
            self._score_shape = (len(channel),)    # each score is a vector of dimension len(channel)
        else:
            raise ValueError('channel must be one of None, ... (Ellipsis), int, or a list/array of ints')

        if respdir is None:
            self._respdir = writedir
        else:
            assert os.path.isdir(respdir), 'invalid response directory: %s' % respdir
            self._respdir = respdir

        assert isinstance(require_response, bool), 'require_response must be True or False'
        assert isinstance(verbose, bool), 'verbose must be True or False'
        assert match_imgfn_policy in self._supported_match_imgfn_policies,\
            'match_imgfn_policy %s not supported; must be one of %s'\
            % (match_imgfn_policy, str(self._supported_match_imgfn_policies))
        self._match_imgfn_policy = match_imgfn_policy
        self._verbose = verbose
        self._require_response = require_response

    @property
    def _supported_match_imgfn_policies(self):
        return 'strict', 'loose'

    def _match_imgfn_2_imgid(self, result_imgfn):
        if self._match_imgfn_policy == 'strict':
            return self._curr_imgfn_2_imgid[result_imgfn]
        elif self._match_imgfn_policy == 'loose':
            try:
                return self._curr_imgfn_2_imgid[result_imgfn]
            except KeyError:
                imgid = result_imgfn[:result_imgfn.rfind('.')]
                if imgid.find('block') == 0:
                    imgid = imgid[12:]
                return imgid

    def _get_scores(self):
        imgid_2_local_idx = {imgid: i for i, imgid in enumerate(self._curr_imgids)}
        t0 = time()

        # wait for matf
        matfn = 'block%03d.mat' % self._blockwriter.iblock
        matfpath = os.path.join(self._respdir, matfn)
        print('waiting for %s' % matfn)
        while not os.path.isfile(matfpath):
            sleep(0.001)
        sleep(0.5)    # ensures mat file finish writing
        t1 = time()

        # load .mat file results
        result_imgfns, scores = utils.load_block_mat(matfpath)
        #    select the channel(s) to use
        if self._channel is None:
            scores = np.mean(scores, axis=-1)
        elif hasattr(self._channel, '__len__'):
            scores_new = np.empty((scores.shape[0], len(self._channel)))
            for ic, c in enumerate(self._channel):
                if hasattr(c, '__len__'):
                    scores_new[:, ic] = np.mean(scores[:, c], axis=-1)
                else:
                    scores_new[:, ic] = scores[:, c]
            scores = scores_new
        else:
            scores = scores[:, self._channel]

        t2 = time()
        print('read from %s: stimulusID %s  tEvokedResp %s' % (matfn, str(result_imgfns.shape), str(scores.shape)))

        # organize results
        organized_scores = []
        scores_local_idx = []
        novel_imgfns = []
        for result_imgfn, score in zip(result_imgfns, scores):
            try:
                imgid = self._match_imgfn_2_imgid(result_imgfn)
                local_idx = imgid_2_local_idx[imgid]
                organized_scores.append(score)
                scores_local_idx.append(local_idx)
            except KeyError:
                novel_imgfns.append(result_imgfn)
        t3 = time()

        print('wait for .mat file %.2fs  load .mat file %.2fs  organize results %.2fs' %
              (t1 - t0, t2 - t1, t3 - t2))
        return organized_scores, scores_local_idx, novel_imgfns


class WithIODummyScorer(WithIOScorer):
    def __init__(self, writedir, backupdir):
        super(WithIODummyScorer, self).__init__(writedir, backupdir)

    def _get_scores(self):
        return np.ones(self._curr_nimgs), np.arange(self._curr_nimgs), []


class ShuffledEPhysScorer(EPhysScorer):
    """
    shuffles scores returned by EPhysScorer as a control experiment
    """
    def __init__(self, *args, shuffle_first_n=None, match_imgfn_policy='loose', **kwargs):
        """
        :param shuffle_first_n: only shuffle first n; otherwise shuffle all
        :param match_imgfn_policy: 'strict', 'loose', 'gen_nat', or 'no_check'
        """
        super(ShuffledEPhysScorer, self).__init__(*args, **kwargs)
        # explicit random generator to stay separate from the random generator in EPhysScorer
        self._random_generator = np.random.RandomState(seed=self._random_seed)
        print('random seed (parallel) set to %d for ShuffledEPhysScorer' % self._random_seed)
        if shuffle_first_n is not None:
            self._first = int(shuffle_first_n)
        else:
            self._first = None
        assert match_imgfn_policy in self._supported_match_imgfn_policies,\
            'match_imgfn_policy %s not supported; must be one of %s'\
            % (match_imgfn_policy, str(self._supported_match_imgfn_policies))
        self._match_imgfn_policy = match_imgfn_policy

        self._verbose = False

        # specifically used in self._match_imgfn_2_imgid()
        self._matcher_istep = self._istep - 1
        self._matcher_curr_idc = None
        self._matcher_curr_gen_idc = None
        self._matcher_curr_nat_idc = None
        self._matcher_curr_igen = None
        self._matcher_curr_inat = None
        self._matcher_curr_iall = None

    @property
    def _supported_match_imgfn_policies(self):
        return 'strict', 'loose', 'gen_nat', 'no_check'

    def _match_imgfn_2_imgid(self, result_imgfn):
        if self._match_imgfn_policy == 'strict':
            return self._curr_imgfn_2_imgid[result_imgfn]
        elif self._match_imgfn_policy == 'loose':
            try:
                return self._curr_imgfn_2_imgid[result_imgfn]
            except KeyError:
                imgid = result_imgfn[:result_imgfn.rfind('.')]
                if imgid.find('block') == 0:
                    imgid = imgid[12:]
                return imgid
        elif self._match_imgfn_policy == 'gen_nat':
            if self._matcher_istep < self._istep:
                self._matcher_istep = self._istep
                self._matcher_curr_idc = np.arange(self._curr_nimgs)
                is_generated = np.array(['gen' in imgid for imgid in self._curr_imgids], dtype=bool)
                gen_idc = self._matcher_curr_idc[is_generated]
                nat_idc = self._matcher_curr_idc[~is_generated]
                gen_idc = gen_idc[np.argsort(self._curr_nscores[is_generated])]
                nat_idc = nat_idc[np.argsort(self._curr_nscores[~is_generated])]
                self._matcher_curr_gen_idc = gen_idc
                self._matcher_curr_nat_idc = nat_idc
                self._matcher_curr_igen = 0
                self._matcher_curr_inat = 0
            if 'gen' in result_imgfn:
                imgid = self._curr_imgids[self._matcher_curr_gen_idc[self._matcher_curr_igen]]
                self._matcher_curr_igen += 1
                self._matcher_curr_igen %= len(self._matcher_curr_gen_idc)
            else:
                imgid = self._curr_imgids[self._matcher_curr_nat_idc[self._matcher_curr_inat]]
                self._matcher_curr_inat += 1
                self._matcher_curr_inat %= len(self._matcher_curr_nat_idc)
            return imgid
        elif self._match_imgfn_policy == 'no_check':
            if self._matcher_istep < self._istep:
                self._matcher_istep = self._istep
                self._matcher_curr_idc = np.arange(self._curr_nimgs)
                self._matcher_curr_iall = 0
            imgid = self._curr_imgids[self._matcher_curr_idc[self._matcher_curr_iall]]
            self._matcher_curr_iall += 1
            self._matcher_curr_iall %= self._curr_nimgs
            return imgid

    def score(self, *args, **kwargs):
        scores = super(ShuffledEPhysScorer, self).score(*args, **kwargs)

        shuffled_view = np.arange(len(scores))
        if self._first is None:
            self._random_generator.shuffle(shuffled_view)
        else:
            self._random_generator.shuffle(shuffled_view[:self._first])
        shuffled_scores = scores[shuffled_view]
        return shuffled_scores
