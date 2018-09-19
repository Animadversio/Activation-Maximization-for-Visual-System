import os

import net_utils
import utils
from Scorer import Scorer, WithIOScorer


class WithIOCNNScorer(WithIOScorer):
    '''Give the response to image of one neuron in a pretrained CNN. '''
    def __init__(self, target_neuron, writedir, backupdir, image_size, random_seed=None):
        """
        :param target_neuron: tuple of
            (str classifier_name, str net_layer, int neuron_index[, int neuron_x, int neuron_y])
        """
        super(WithIOCNNScorer, self).__init__(writedir, backupdir, image_size, random_seed)
        # parse the `target_neuron` parameter syntax i.e. `('caffe-net', 'fc8', 1)`
        self._classifier_name = str(target_neuron[0])
        self._net_layer = str(target_neuron[1])
        self._net_iunit = int(target_neuron[2])
        if len(target_neuron) == 5:
            self._net_unit_x = int(target_neuron[3])
            self._net_unit_y = int(target_neuron[4])
        else:
            self._net_unit_x = None
            self._net_unit_y = None

        self._classifier = None
        self._transformer = None

    def load_classifier(self):
        classifier = net_utils.load(self._classifier_name)
        transformer = net_utils.get_transformer(classifier, scale=1)
        self._classifier = classifier
        self._transformer = transformer

    def _get_scores(self):
        imgid_2_local_idx = {imgid: i for i, imgid in enumerate(self._curr_imgids)}
        organized_scores = []
        scores_local_idx = []
        novel_imgfns = []

        for imgfn in self._curr_imgfn_2_imgid.keys():
            im = utils.read_image(os.path.join(self._writedir, imgfn))
            tim = self._transformer.preprocess('data', im)
            self._classifier.blobs['data'].data[...] = tim
            self._classifier.forward(end=self._net_layer)
            score = self._classifier.blobs[self._net_layer].data[0, self._net_iunit]
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


class NoIOCNNScorer(Scorer):
    # TODO ???
    def __init__(self):
        pass
