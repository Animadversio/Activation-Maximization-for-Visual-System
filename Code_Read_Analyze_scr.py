import utils
import numpy as np
import importlib
import os
importlib.reload(utils)  # Reload the modules after changing them, or they will not affect the codes.

utils.load_codes("/home/poncelab/Documents/data/with_CNN/caffe-net_fc6_0001/backup/", 10)
utils.load_codes_search("/home/poncelab/Documents/data/with_CNN/caffe-net_fc6_0001/backup/", "gen100")

CurDataDir = "/home/poncelab/Documents/data/with_CNN/caffe-net_fc6_0001/backup/"

with np.load(CurDataDir+"backup/scores_end_block298.npz") as data:
    for key in data.keys():
        print(key)
        print(data[key])
# Data Structure: ['image_ids', 'scores', 'scores_mat', 'nscores']
# 'image_ids' id, and name for files
# data['scores'], ['scores_mat'] are nearly the same.
# data['nscores']

with np.load(CurDataDir+"backup/scores_end_block298.npz") as data:
    for key in data.keys():
        print(key)
        print(data[key])

with np.load(os.path.join(CurDataDir, "genealogy_gen297.npz")) as data:
    print( data.keys() )
    for key in data.keys():
        print(key)
        print(data[key])
# ['image_ids', 'genealogy']
# In "genealogy", They trace the 2 parents of each new stimuli in this new generation.
# Such that a family tree can be reconstructed. And which imgs goes directly from last generation
# and comes to this generation.
#

# os.path.join(CurDataDir, "log.txt") is all the output to the console

