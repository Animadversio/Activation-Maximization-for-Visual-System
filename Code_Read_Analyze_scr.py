import utils
import numpy as np
import importlib
importlib.reload(utils)  # Reload the modules after changing them, or they will not affect the codes.

utils.load_codes("/home/poncelab/Documents/data/with_CNN/caffe-net_fc6_0001/backup/", 10)
utils.load_codes_search("/home/poncelab/Documents/data/with_CNN/caffe-net_fc6_0001/backup/", "gen100")

with np.load("/home/poncelab/Documents/data/with_CNN/caffe-net_fc6_0001/backup/scores_end_block298.npz") as data:
    for key in data.keys():
        print(key)
        print(data[key])
# ['image_ids', 'scores', 'scores_mat', 'nscores']
# data['scores']
# data['nscores']
