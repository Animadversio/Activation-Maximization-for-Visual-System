import os

import net_utils
import utils
from Scorer import Scorer, WithIOScorer


class WithIOCNNScorer(WithIOScorer):
    '''Give the response to image of one neuron in a pretrained CNN. '''
    def __init__(self, target_neuron, writedir, backupdir, image_size, random_seed=None, record_pattern=False):
        """
        :param target_neuron: tuple of
            (str classifier_name, str net_layer, int neuron_index[, int neuron_x, int neuron_y])
        """
        super(WithIOCNNScorer, self).__init__(writedir, backupdir, image_size, random_seed)
        # parse the `target_neuron` parameter syntax i.e. `('caffe-net', 'fc8', 1)`
        self._classifier_name = str(target_neuron[0])
        self._net_layer = str(target_neuron[1])
        # `self._net_layer` is used to determine which layer to stop forwarding
        self._net_iunit = int(target_neuron[2])
        # this index is used to extract the scalar response `self._net_iunit`
        if len(target_neuron) == 5:
            self._net_unit_x = int(target_neuron[3])
            self._net_unit_y = int(target_neuron[4])
        else:
            self._net_unit_x = None
            self._net_unit_y = None

        self.record_pattern=record_pattern  # control if record the activation pattern of the whole layer

        self._classifier = None
        self._transformer = None

    def load_classifier(self):
        classifier = net_utils.load(self._classifier_name)
        transformer = net_utils.get_transformer(classifier, scale=1)
        self._classifier = classifier
        self._transformer = transformer

    def _get_scores(self):
        imgid_2_local_idx = {imgid: i for i, imgid in enumerate(self._curr_imgids)}
        # mapping from keys='imgid' (img filenames) to value=i (local_idx, in [0,40) )
        # i.e. the inverse mapping of `self._curr_imgids`.
        # Note the `local_idx` is the order number of the img in the order of `imgid`
        # And output scores are in the order of the `_curr_imgfn`
        organized_scores = []  # ?
        organized_pattern = []
        scores_local_idx = []  # ?
        novel_imgfns = []  # ?

        for imgfn in self._curr_imgfn_2_imgid.keys():
            im = utils.read_image(os.path.join(self._writedir, imgfn))  # shape=(83, 83, 3)
            tim = self._transformer.preprocess('data', im)  # shape=(3, 227, 227)
            self._classifier.blobs['data'].data[...] = tim
            self._classifier.forward(end=self._net_layer)
            score = self._classifier.blobs[self._net_layer].data[0, self._net_iunit]
            if self.record_pattern:  # record the whole layer's activation
                score_full = self._classifier.blobs[self._net_layer].data[0, :]
            # Use the `self._net_iunit` for indexing the output
            if self._net_unit_x is not None:
                # if `self._net_unit_x/y` are provided, then use them to slice the output score
                score = score[self._net_unit_x, self._net_unit_y]
            try:
                imgid = self._curr_imgfn_2_imgid[imgfn]
                local_idx = imgid_2_local_idx[imgid]
                organized_scores.append(score)
                scores_local_idx.append(local_idx)
                if self.record_pattern:
                    organized_pattern.append(score_full)
            except KeyError:
                novel_imgfns.append(imgfn)  # Record this `novel_imgfns` to report at the end of each gen.

        return organized_scores, scores_local_idx, novel_imgfns
        # Return the response for a generation of images

    def get_activation_pattern(self):
        '''Get full activation pattern of the whole layer'''
        # TODO: To complete and test this function.
        imgid_2_local_idx = {imgid: i for i, imgid in enumerate(self._curr_imgids)}
        organized_scores = []
        scores_local_idx = []
        novel_imgfns = []

        for imgfn in self._curr_imgfn_2_imgid.keys():
            im = utils.read_image(os.path.join(self._writedir, imgfn))
            tim = self._transformer.preprocess('data', im)
            self._classifier.blobs['data'].data[...] = tim
            self._classifier.forward(end=self._net_layer)
            score = self._classifier.blobs[self._net_layer].data[0, :]

            if self._net_unit_x is not None:
                score = score[self._net_unit_x, self._net_unit_y]
            try:
                imgid = self._curr_imgfn_2_imgid[imgfn]
                local_idx = imgid_2_local_idx[imgid]
                organized_scores.append(score)
                scores_local_idx.append(local_idx)
            except KeyError:
                novel_imgfns.append(imgfn)

        return organized_scores, scores_local_idx, novel_imgfns

#%%  Add noise to Scorer. Inherit most methods from CNNScorer

import numpy as np
import numpy.random as random
import matplotlib.pyplot as plt
from scipy.stats import norm, expon, gamma, lognorm, uniform

class WithIONoisyCNNScorer(WithIOCNNScorer):
    '''Give the response to image of one neuron in a pretrained CNN. Add noise to the unit's output.

    '''

    def __init__(self, target_neuron, writedir, backupdir, image_size, random_seed=None,
                 noise_scheme=None, noise_param=None, noise_rand_seed=0):
        """
        :param target_neuron: tuple of
            (str classifier_name, str net_layer, int neuron_index[, int neuron_x, int neuron_y])
        """
        super(WithIONoisyCNNScorer, self).__init__(target_neuron, writedir, backupdir, image_size, random_seed)
        self._AddNoise = None
        self.noise_dist = None
        self.noise_scheme = noise_scheme
        self.noise_param = noise_param
        self.init_noise_generator(noise_scheme, noise_param, noise_rand_seed)

    def init_noise_generator(self, noise_scheme, noise_param, noise_rand_seed):
        if noise_scheme is None:
            self._AddNoise = False
        else:
            self._AddNoise = True
            random.seed(seed=noise_rand_seed)
            distrib_dict = {'norm': norm, 'expon': expon,
                            'gamma': gamma, 'lognorm': lognorm,
                            'uniform': uniform}
            try:
                self.noise_dist = distrib_dict[noise_scheme]
                if type(noise_param) is dict:
                    self.noise_dist = self.noise_dist(**noise_param)
                elif type(noise_param) is tuple:
                    self.noise_dist = self.noise_dist(*noise_param)
                else:
                    raise (ValueError, "Input noise_param is not parseable")
            except KeyError:
                raise(KeyError, "Invalid `noise_scheme` name, cannot find corresponding distribution for noise")
        self.demo_noise_dist()

    def demo_noise_dist(self):
        dist = self.noise_dist
        if dist is not None:
            print("Noise scheme: {}, parameter: {} , \nmean: {:.3f}, median: {:.3f}, std: {:.3f}, [0.05-0.95] percentage range [{:.3f},{:.3f}]".format(self.noise_scheme, self.noise_param,
                                                                                           dist.mean(), dist.median(), dist.std(),
                                                                                           dist.ppf(0.05), dist.ppf(0.95),))

            LB=dist.ppf(0.05)
            HB=dist.ppf(0.95)
            support = np.arange(LB, HB, (HB-LB)/200)
            plt.figure()
            plt.plot(support, dist.pdf(support))
            plt.title("{} (param = {})".format(self.noise_scheme, self.noise_param))
            plt.show()
        else:
            print("Noise free mode!\n")
        input("Press any key to confirm the noise setting.")

    def noise_generator(self, shape=1):
        return self.noise_dist.rvs(size=shape)

    def _get_scores(self):
        organized_scores, scores_local_idx, novel_imgfns = super(WithIONoisyCNNScorer, self)._get_scores()
        if self._AddNoise and (not len(organized_scores) == 0):
            # noise = self.noise_generator(organized_scores.shape)
            # organized_scores = organized_scores + noise
            organized_scores = [score + self.noise_generator() for score in organized_scores]

        return organized_scores, scores_local_idx, novel_imgfns
        # Return the response for a generation of images


class NoIOCNNScorer(Scorer):
    # TODO
    # NOTE this module is more easy to use. Because, there is no point to handle blocks for CNN scorer.
    '''
        The only methods that a `scorer` module should get are
            `scorer.score(stimuli, stimuli_ids)`,
            `scorer.save_current_scores()`
            `scorer.load_classifier()`
        So to realize a `scorer` class just make these 3 functions. that's enough
    '''

    def __init__(self, target_neuron, writedir, record_pattern=False):
        """
        :param target_neuron: tuple of
            (str classifier_name, str net_layer, int neuron_index[, int neuron_x, int neuron_y])
        """
        super(NoIOCNNScorer, self).__init__(writedir)  # writedir goes to self._backupdir
        # parse `target_neuron` parameter syntax i.e. `('caffe-net', 'fc8', 1)`
        self._classifier_name = str(target_neuron[0])
        self._net_layer = str(target_neuron[1])
        # `self._net_layer` is used to determine which layer to stop forwarding
        self._net_iunit = int(target_neuron[2])
        # this index is used to extract the scalar response `self._net_iunit`
        if len(target_neuron) == 5:
            self._net_unit_x = int(target_neuron[3])
            self._net_unit_y = int(target_neuron[4])
        else:
            self._net_unit_x = None
            self._net_unit_y = None

        self.record_pattern = record_pattern  # control if record the activation pattern of the whole layer
        self._pattern_array = None  # record activation pattern

        self._istep = -1  # Instead of record block number in BlockWriter,

        self._classifier = None
        self._transformer = None

    def load_classifier(self):
        classifier = net_utils.load(self._classifier_name)
        transformer = net_utils.get_transformer(classifier, scale=1)
        self._classifier = classifier
        self._transformer = transformer

    def score(self, images, image_ids, ):
        '''input `images` is a list of nparray [256,256,3], image_ids is a list of str'''
        self._curr_imgids = image_ids
        self._curr_images = images  # take record

        # Check the input
        nimgs = len(images)
        assert len(image_ids) == nimgs

        self._istep += 1
        scores = np.zeros(nimgs)
        if self.record_pattern:
            self._pattern_array = [None] * nimgs

        for i, img in enumerate(images):
            # Note: now only support single repetition
            tim = self._transformer.preprocess('data', img)  # shape=(3, 227, 227)
            self._classifier.blobs['data'].data[...] = tim
            self._classifier.forward(end=self._net_layer)  # propagate the image the target layer

            if self.record_pattern:  # record the whole layer's activation
                score_full = self._classifier.blobs[self._net_layer].data[0, :]
                # self._pattern_array.append(score_full)
                self._pattern_array[i] = score_full.copy()

            # record only the neuron intended
            score = self._classifier.blobs[self._net_layer].data[0, self._net_iunit]

            if self._net_unit_x is not None:
                # if `self._net_unit_x/y` (inside dimension) are provided, then use them to slice the output score
                score = score[self._net_unit_x, self._net_unit_y]

            scores[i] = score
        self._curr_scores = scores
        if self.record_pattern:
            self._pattern_array = np.asarray(self._pattern_array)
            self.save_block_stats({'image_ids': self._curr_imgids, 'pattern_array': self._pattern_array}, "ActivPattArr")

        return scores

    def save_current_scores(self):
        savefpath = os.path.join(self._backupdir, 'scores_end_block%03d.npz' % self._istep)
        save_kwargs = {'image_ids': self._curr_imgids, 'scores': self._curr_scores}
        #               , 'scores_mat': self._curr_scores_mat, 'nscores': self._curr_nscores}
        print('saving scores to %s' % savefpath)
        utils.save_scores(savefpath, save_kwargs)

    def save_block_stats(self, save_kwargs, namestr):
        '''More general version of save_current_scores. save anything you like'''
        savefpath = os.path.join(self._backupdir, '%s_block%03d.npz' % (namestr, self._istep))
        print('saving %s to %s' % (namestr, savefpath))
        utils.save_scores(savefpath, save_kwargs)


