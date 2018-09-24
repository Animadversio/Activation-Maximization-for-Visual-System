import utils
import numpy as np
import matplotlib.pyplot as plt
import importlib
import os
importlib.reload(utils)  # Reload the modules after changing them, or they will not affect the codes.

utils.load_codes("/home/poncelab/Documents/data/with_CNN/caffe-net_fc6_0001/backup/", 10)
utils.load_codes_search("/home/poncelab/Documents/data/with_CNN/caffe-net_fc6_0001/backup/", "gen100")


CurDataDir = "/home/poncelab/Documents/data/with_CNN/caffe-net_fc6_0001/backup/"
with np.load(CurDataDir+"scores_end_block298.npz") as data:
    for key in data.keys():
        print(key)
        print(data[key])

# Data Structure: ['image_ids', 'scores', 'scores_mat', 'nscores']
# 'image_ids' id, and name for files
# data['scores'], ['scores_mat'] are nearly the same.
# data['nscores']

with np.load(os.path.join(CurDataDir, "genealogy_gen297.npz")) as data:
    print( data.keys() )
    for key in data.keys():
        print(key)
        print(data[key])

# Data Structure: ['image_ids', 'genealogy']
# In "genealogy", They trace the 2 parents of each new stimuli in this new generation.
# Such that a family tree can be reconstructed. And which imgs goes directly from last generation
# and comes to this generation.

# os.path.join(CurDataDir, "log.txt") is all the output to the console

# Point is to Record More response pattern during generation.

#%% Visualize Optimizing Trajectory
steps = 300
population_size = 40
ScoreEvolveTable = np.full((steps, population_size, ), np.NAN)

for stepi in range(steps):
    with np.load(CurDataDir+"scores_end_block{0:03}.npz".format(stepi)) as data:
        score_tmp = data['scores']
        ScoreEvolveTable[stepi, :len(score_tmp)] = score_tmp

gen_slice = np.arange(steps).reshape((-1, 1))
gen_num = np.repeat(gen_slice, population_size, 1)

AvgScore = np.nanmean(ScoreEvolveTable, axis=1)
MaxScore = np.nanmax(ScoreEvolveTable, axis=1)

plt.figure(1)
plt.scatter(gen_num, ScoreEvolveTable, s=16, alpha=0.7, label="all score")
plt.plot(gen_slice, AvgScore, color='black', label="Average score")
plt.plot(gen_slice, MaxScore, color='red', label="Max score")
plt.xlabel("generation #")
plt.ylabel("CNN unit score")
plt.legend()
plt.show()

