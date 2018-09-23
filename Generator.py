import numpy as np
import net_utils


class Generator:
    '''Load CaffeNet generator

    Major use is to "visualize", detransform the code to the image
    '''
    def __init__(self):
        generator = net_utils.load('generator')
        detransformer = net_utils.get_detransformer(generator)
        self._GNN = generator
        self._detransformer = detransformer

    def visualize(self, code):
        x = self._GNN.forward(feat=code.reshape(1, 4096))['deconv0']
        x = self._detransformer.deprocess('data', x)
        x = np.clip(x, 0, 1)  # use clip to bound all the image output in interval [0,1]
        return (x * 255).astype('uint8')  # rescale to uint in [0,255]
