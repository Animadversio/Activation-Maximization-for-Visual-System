'''
This code mainly deal with read the scores and codes and generating figures from them.

Part of it is developping the code for visualization, and the other parts are

'''

import utils
import numpy as np
import matplotlib.pyplot as plt
import importlib
import os
import matplotlib as mpl
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
    print(data.keys())
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
block_num = 20
title_cmap = plt.cm.plasma
col_n = 6

fncatalog = os.listdir(CurDataDir)

fn_image_gen = [fn for fn in fncatalog if (".bmp" in fn) and ("block{0:03}".format(block_num) in fn)]
fn_score_gen = [fn for fn in fncatalog if (".npz" in fn) and ("score" in fn) and ("block{0:03}".format(block_num) in fn)]

fn_image_gen = sorted(fn_image_gen)
image_num = len(fn_image_gen)

row_n = int(np.ceil(image_num/col_n))
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

figs, axes = plt.subplots(row_n , col_n, figsize=[12, 13])
for i, imagefn in enumerate(fn_image_gen):
    img_tmp = plt.imread(os.path.join(CurDataDir, imagefn) )
    score_tmp = score_gen[i]
    plt.subplot(row_n, col_n, i+1)
    plt.imshow(img_tmp)
    plt.xticks([])
    plt.yticks([])
    plt.axis('off')
    if cmap_flag:  # color the titles with a heatmap!
        plt.title("{0:.2f}".format(score_tmp), fontsize=14,
                  color=title_cmap((score_tmp - lb) / (ub - lb)))  # normalize a value between [0,1]
    else:
        plt.title("{0:.2f}".format(score_tmp), fontsize=14)


# sm = plt.cm.ScalarMappable(cmap=title_cmap, norm=mpl.colors.Normalize(vmin=lb, vmax=ub))
# sm._A = []
# plt.colorbar(sm, ax=axes.ravel().tolist(), use_gridspec=True)

plt.tight_layout(pad=1, h_pad=0.1, w_pad=0, rect=(0.05, 0.2, 0.95, 0.9))

# norm = mpl.colors.Normalize(vmin=lb, vmax=ub)
# cb1 = mpl.colorbar.ColorbarBase(ax, cmap=title_cmap,
#                                norm=norm,
#                                orientation='vertical')
plt.suptitle("Block{0:03}".format(block_num), fontsize=14)
plt.show()

# plt.imread(os.path.join(CurDataDir, "block299_35_gen298_010752.bmp"))

#%% Code to visualize an experimental setting
CurDataDir = "/home/poncelab/Documents/data/with_CNN/caffe-net_fc6_0001/backup/"
SaveImgDir = os.path.join(CurDataDir, "sum_img/")
if not os.path.isdir(SaveImgDir):
    os.mkdir(SaveImgDir)
for num in range(1, 301):
    try:
        fig = utils.visualize_image_score_each_block(CurDataDir, block_num=num, save=True, savedir=SaveImgDir)
        fig.clf()
    except AssertionError:
        print("Show and Save %d number of image visualizations. " % (num) )
        break

utils.visualize_score_trajectory(CurDataDir, title_str="Normal_CNN: No noise")

#%% Visualization and testing block
CurDataDir = '/home/poncelab/Documents/data/with_CNN/caffe-net_fc6_0001/trial0/'

SaveImgDir = os.path.join(CurDataDir, "sum_img/")
if not os.path.isdir(SaveImgDir):
    os.mkdir(SaveImgDir)
utils.visualize_score_trajectory(CurDataDir, title_str="Normal_CNN: No noise",
                                 save=True, savedir=SaveImgDir)

#%% Activation pattern visualize
CurDataDir = "/home/poncelab/Documents/data/with_CNN/caffe-net_fc6_0001/trial1/"
with np.load(os.path.join(CurDataDir, "ActivPattArr_block001.npz")) as data:
    print(data.keys())
    for key in data.keys():
        print(key)
        print(data[key])
    Activation_Pattern = data['pattern_array']

#%% Compute Correlation between
with np.load(os.path.join(CurDataDir, "ActivPattArr_block001.npz")) as data:
    print(data.keys())
    for key in data.keys():
        print(key)
        print(data[key])
    Activation_Pattern = data['pattern_array']
#%% Load Activation Pattern Data
steps = 300
population_size = 40
layer_size = 4096
title_str = "CNN: No Noise"
ActivPatternEvolveTable = np.full((steps, population_size, layer_size), np.NAN)

for stepi in range(steps):
    try:
        with np.load(os.path.join(CurDataDir, "ActivPattArr_block{0:03}.npz".format(stepi))) as data:
            Activation_Pattern = data['pattern_array']
            pop_size, act_len = Activation_Pattern.shape
            ActivPatternEvolveTable[stepi, :pop_size, :act_len] = Activation_Pattern
    except FileNotFoundError:
        print("maximum steps is %d." % stepi)
        ActivPatternEvolveTable = ActivPatternEvolveTable[0:stepi, :]
        steps = stepi
        break

#%% Correlation between unit Calculation
ScorePool = ActivPatternEvolveTable[:, :36, :].reshape((steps * 36, layer_size))
CorrCoef = np.zeros(layer_size)
for uniti in range(layer_size):
    tmp = np.corrcoef(ScorePool[:, 1], ScorePool[:, uniti])
    CorrCoef[uniti] = tmp[0,1]
plt.figure(figsize=[12,5])
plt.scatter(np.arange(4096), CorrCoef,s=16, alpha=0.6,)
plt.xlabel("generation #")
plt.ylabel("Correlation Coefficient between units")
plt.show()

#%%
plt.figure(1, figsize= [15,10])
gen_slice = np.arange(steps).reshape((-1, 1))
gen_num = np.repeat(gen_slice, population_size, 1)
for uniti in [0, 1, 2, 2479, 2358, 2469]: # Add Unit id to inspect the activity pattern
    ScoreEvolveTable = ActivPatternEvolveTable[:, :, uniti]
    AvgScore = np.nanmean(ScoreEvolveTable, axis=1)
    MaxScore = np.nanmax(ScoreEvolveTable, axis=1)
    plt.scatter(gen_num, ScoreEvolveTable, s=16, alpha=0.5, label="all "+"Unit%d"%uniti)
    plt.plot(gen_slice, AvgScore, color='black', label="Average score"+"Unit%d"%uniti)
    plt.plot(gen_slice, MaxScore, color='red', label="Max score"+"Unit%d"%uniti)
plt.xlabel("generation #")
plt.ylabel("CNN unit score")
plt.title("Optimization Trajectory of Score\n" + title_str)
plt.legend()
plt.show()

#%%#%%
utils.visualize_score_trajectory('/home/poncelab/Documents/data/with_CNN/caffe-net_fc6_0001/backup/', save=True, title_str="baseline_genetic")

#%%
utils.visualize_all('/home/poncelab/Documents/data/with_CNN/caffe-net_fc6_0001/trial_cma',title_str="cma_trial1")
#%%#%%
utils.visualize_score_trajectory('/home/poncelab/Documents/data/with_CNN/caffe-net_fc6_0001/trial_cma2_sgm7', save=True, title_str="cma_trial2_sigma7")
#%%
utils.visualize_score_trajectory('/home/poncelab/Documents/data/with_CNN/caffe-net_fc6_0001/trial_cma3_no_eigdecom', save=True, title_str="cma_trial3_no_eig_decom")
#%%
utils.visualize_all('/home/poncelab/Documents/data/with_CNN/caffe-net_fc6_0001/trial_cma3_no_eigdecom', save=True, title_str="cma_trial3_no_eig_decom")

#%%
stepi = 1
CurDataDir = '/home/poncelab/Documents/data/with_CNN/caffe-net_fc6_0001/trial_cma3_no_eigdecom'
with np.load(os.path.join(CurDataDir, "scores_end_block{0:03}.npz".format(stepi))) as data:
    score_tmp = data['scores']

#%%
stepi = 1
CurDataDir = '/home/poncelab/Documents/data/with_CNN/caffe-net_fc6_0001/backup/'#trial_cma3_no_eigdecom/
fncatalog = os.listdir(CurDataDir)
with np.load(CurDataDir + "scores_end_block{0:03}.npz".format(stepi)) as data:
    print(data.keys())
    image_ids = data['image_ids']
    score_tmp = data['scores']
print(image_ids)
fn_image_gen=[]
for imgid in image_ids:
    fntmplist = [fn for fn in fncatalog if (imgid in fn) and '.bmp' in fn]
    assert len(fntmplist) is 1, "Image file not found"
    fn_image_gen.append(fntmplist[0])
print(fn_image_gen)
#%% Updated Version of visualization Oct. 7th
CurDataDir = '/home/poncelab/Documents/data/with_CNN/caffe-net_fc6_0001/trial_cma3_no_eigdecom/'#trial_cma3_no_eigdecom/
block_num = 10
col_n = 6
title_cmap = plt.cm.viridis
fncatalog = os.listdir(CurDataDir)
fn_score_gen = [fn for fn in fncatalog if
                (".npz" in fn) and ("score" in fn) and ("block{0:03}".format(block_num) in fn)]
assert len(fn_score_gen) is 1, "not correct number of score files"
with np.load(os.path.join(CurDataDir, fn_score_gen[0])) as data:
    score_gen = data['scores']
    image_ids = data['image_ids']
fn_image_gen = []
for imgid in image_ids:
    fn_tmp_list = [fn for fn in fncatalog if (imgid in fn) and '.bmp' in fn]
    assert len(fn_tmp_list) is 1, "Image file not found or wrong Image file number"
    fn_image_gen.append(fn_tmp_list[0])
image_num = len(fn_image_gen)

assert len(score_gen) is image_num, "image and score number do not match"
lb = score_gen.min()
ub = score_gen.max()
if ub == lb:
    cmap_flag = False
else:
    cmap_flag = True

row_n = np.ceil(image_num / col_n)
figW = 12
figH = figW / col_n * row_n + 1
# figs, axes = plt.subplots(int(row_n), col_n, figsize=[figW, figH])
fig = plt.figure(figsize=[figW, figH])
for i, imagefn in enumerate(fn_image_gen):
    img_tmp = plt.imread(os.path.join(CurDataDir, imagefn))
    score_tmp = score_gen[i]
    plt.subplot(row_n, col_n, i + 1)
    plt.imshow(img_tmp)
    plt.xticks([])
    plt.yticks([])
    plt.axis('off')
    if cmap_flag:  # color the titles with a heatmap!
        plt.title("{0:.2f}".format(score_tmp), fontsize=16,
                  color=title_cmap((score_tmp - lb) / (ub - lb)))  # normalize a value between [0,1]
    else:
        plt.title("{0:.2f}".format(score_tmp), fontsize=16)
plt.show(block=False)
#%%
CurDataDir = '/home/poncelab/Documents/data/with_CNN/caffe-net_fc6_0001/trial_cma3_no_eigdecom/'#trial_cma3_no_eigdecom/
utils.visualize_all(CurDataDir, title_str="cmaes_trial3_no_eigdecom", save=True)

