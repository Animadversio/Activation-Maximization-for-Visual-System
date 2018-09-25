'''
This code mainly deal with read the scores and codes and generating figures from them.

'''

import utils
import numpy as np
import matplotlib.pyplot as plt
import importlib
import os
importlib.reload(utils)  # Reload the modules after changing them, or they will not affect the codes.


#%%
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
# have been wrapped up in utils.visualize_score_trajectory

CurDataDir = "/home/poncelab/Documents/data/with_CNN_noisy_gauss_5/caffe-net_fc6_0001/backup/"
steps = 300
population_size = 40
title_str = "Noisy_CNN: gaussian Noise, std = 5"

ScoreEvolveTable = np.full((steps, population_size, ), np.NAN)
for stepi in range(steps):
    try:
        with np.load(CurDataDir+"scores_end_block{0:03}.npz".format(stepi)) as data:
            score_tmp = data['scores']
            ScoreEvolveTable[stepi, :len(score_tmp)] = score_tmp
    except FileNotFoundError:
        print("maximum steps is %d." % stepi)
        ScoreEvolveTable = ScoreEvolveTable[0:stepi, :]
        steps = stepi
        break

gen_slice = np.arange(steps).reshape((-1, 1))
gen_num = np.repeat(gen_slice, population_size, 1)

AvgScore = np.nanmean(ScoreEvolveTable, axis=1)
MaxScore = np.nanmax(ScoreEvolveTable, axis=1)

plt.figure(1)
plt.scatter(gen_num, ScoreEvolveTable, s=16, alpha=0.6, label="all score")
plt.plot(gen_slice, AvgScore, color='black', label="Average score")
plt.plot(gen_slice, MaxScore, color='red', label="Max score")
plt.xlabel("generation #")
plt.ylabel("CNN unit score")
plt.title("Optimization Trajectory of Score\n" + title_str)
plt.legend()
plt.show()
#%% Visualize the different modules

utils.visualize_score_trajectory("/home/poncelab/Documents/data/with_CNN_noisy_gauss_5/caffe-net_fc6_0001/backup/",
                                 title_str= "Noisy_CNN: gaussian Noise, std = 5")
utils.visualize_score_trajectory("/home/poncelab/Documents/data/with_CNN_noisy/caffe-net_fc6_0001/backup/",
                                 title_str= "Noisy_CNN: gaussian Noise, std = 3")
utils.visualize_score_trajectory("/home/poncelab/Documents/data/with_CNN/caffe-net_fc6_0001/backup/",
                                 title_str= "Normal_CNN: No noise")
utils.visualize_score_trajectory("/home/poncelab/Documents/data/with_humanscorer/backup/",
                                 title_str= "Human scorer")


#%% Visualize Score and Image for each generation.

CurDataDir = "/home/poncelab/Documents/data/with_CNN/caffe-net_fc6_0001/backup/"
block_num = 1
title_cmap = plt.cm.viridis
col_n = 6

fncatalog = os.listdir(CurDataDir)

fn_image_gen = [fn for fn in fncatalog if (".bmp" in fn) and ("block{0:03}".format(block_num) in fn)]
fn_score_gen = [fn for fn in fncatalog if (".npz" in fn) and ("score" in fn) and ("block{0:03}".format(block_num) in fn)]

fn_image_gen = sorted(fn_image_gen)
image_num = len(fn_image_gen)

row_n = np.ceil(image_num/col_n)
assert len(fn_score_gen) is 1, "not correct number of score files"
with np.load(os.path.join(CurDataDir, fn_score_gen[0])) as data:
    score_gen = data['scores']
assert len(score_gen) is image_num, "image and score number do not match"
lb = score_gen.min()
ub = score_gen.max()
if ub == lb:
    cmap_flag = False
else:
    cmap_flag = True

plt.figure(figsize=[12, 12])
for i, imagefn in enumerate(fn_image_gen):
    img_tmp = plt.imread(os.path.join(CurDataDir, imagefn) )
    score_tmp = score_gen[i]
    plt.subplot(row_n, col_n, i+1)
    plt.imshow(img_tmp)
    plt.axis('off')
    if cmap_flag:  # color the titles with a heatmap! 
        plt.title("{0:.2f}".format(score_tmp), color=title_cmap((score_tmp-lb)/(ub-lb)))  # normalize a value between [0,1]
    else:
        plt.title("{0:.2f}".format(score_tmp) )

plt.suptitle("Block{0:03}".format(block_num))
plt.tight_layout()
plt.show()

# plt.imread(os.path.join(CurDataDir, "block299_35_gen298_010752.bmp"))
