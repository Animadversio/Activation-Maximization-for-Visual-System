'''
This code mainly deal with read the scores and codes and generating figures from them.

Part of it is developping the code for visualization, and the other parts are

'''

import utils
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
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


#%% Trajectory Comparison between different algorithms
#%%
neuron_dir = "/home/poncelab/Documents/data/with_CNN/caffe-net_fc6_0005/"
trial_list = ['trial_cma5_noeig_sgm20', 'genetic_trial0', 'genetic_trial4']
utils.visualize_score_trajectory_cmp(CurDataDir_list=[neuron_dir+trial_title for trial_title in trial_list],
                                     title_str_list=trial_list, exp_title_str="caffe-net_fc6_0005",
                                     save=True, savedir=neuron_dir)
#%%
neuron_dir = "/home/poncelab/Documents/data/with_CNN/caffe-net_fc6_0100/"
trial_list = ['cma_trial4_noeig_sgm5', 'cma_trial8_noeig_sgm10', 'genetic_trial0', 'genetic_trial4']
utils.visualize_score_trajectory_cmp(CurDataDir_list=[neuron_dir+trial_title for trial_title in trial_list],
                                     title_str_list=trial_list, exp_title_str="caffe-net_fc6_0100",
                                     save=True, savedir=neuron_dir)
#%%
neuron_dir = "/home/poncelab/Documents/data/with_CNN/caffe-net_fc6_0010/"
trial_list = ['cma_trial0_noeig_sgm5', 'cma_trial5_noeig_sgm10', 'genetic_trial0', 'genetic_trial4']
utils.visualize_score_trajectory_cmp(CurDataDir_list=[neuron_dir+trial_title for trial_title in trial_list],
                                     title_str_list=trial_list, exp_title_str="caffe-net_fc6_0010",
                                     save=True, savedir=neuron_dir)
#%%
neuron_dir = "/home/poncelab/Documents/data/with_CNN/caffe-net_fc7_0001/"
trial_list = ['cma_trial0_noeig_sgm5', 'cma_trial5_noeig_sgm10', 'genetic_trial0', 'genetic_trial4']
utils.visualize_score_trajectory_cmp(CurDataDir_list=[neuron_dir+trial_title for trial_title in trial_list],
                                     title_str_list=trial_list, exp_title_str="caffe-net_fc7_0001",
                                     save=True, savedir=neuron_dir)
#%%
neuron_dir = "/home/poncelab/Documents/data/with_CNN/caffe-net_fc7_0010/"
trial_list = ['cma_trial0_noeig_sgm5', 'cma_trial5_noeig_sgm10', 'genetic_trial0', 'genetic_trial4']
utils.visualize_score_trajectory_cmp(CurDataDir_list=[neuron_dir+trial_title for trial_title in trial_list],
                                     title_str_list=trial_list, exp_title_str="caffe-net_fc7_0001",
                                     save=True, savedir=neuron_dir)

#%%
neuron_dir = "/home/poncelab/Documents/data/with_CNN/caffe-net_fc6_0010/"
trial_list = ['cma_trial0_noeig_sgm5', 'genetic_trial0','cma_noeig_sgm5_noise5_trial0','genetic_noise5_trial0','cma_noeig_sgm5_noise10_trial0','genetic_noise10_trial0' ,'cma_noeig_sgm5_noise20_trial0','genetic_noise20_trial0',]
utils.visualize_score_trajectory_cmp(CurDataDir_list=[neuron_dir+trial_title for trial_title in trial_list],
                                     title_str_list=trial_list, exp_title_str="caffe-net_fc6_0010_noise_level",
                                     save=True, savedir=neuron_dir)

#%%
neuron_dir = "/home/poncelab/Documents/data/with_CNN/caffe-net_fc6_0010/"
trial_list = ['cma_trial0_noeig_sgm5', 'genetic_trial0', 'cma_noeig_sgm5_noise20_trial0','genetic_noise20_trial0',]
utils.visualize_score_trajectory_cmp(CurDataDir_list=[neuron_dir+trial_title for trial_title in trial_list],
                                     title_str_list=trial_list, exp_title_str="caffe-net_fc6_0010_noise_level20",
                                     save=True, savedir=neuron_dir)

neuron_dir = "/home/poncelab/Documents/data/with_CNN/caffe-net_fc6_0010/"
trial_list = ['cma_trial0_noeig_sgm5', 'genetic_trial0', 'cma_noeig_sgm5_noise10_trial0','genetic_noise10_trial0',]
utils.visualize_score_trajectory_cmp(CurDataDir_list=[neuron_dir+trial_title for trial_title in trial_list],
                                     title_str_list=trial_list, exp_title_str="caffe-net_fc6_0010_noise_level10",
                                     save=True, savedir=neuron_dir)

neuron_dir = "/home/poncelab/Documents/data/with_CNN/caffe-net_fc6_0010/"
trial_list = ['cma_trial0_noeig_sgm5', 'genetic_trial0', 'cma_noeig_sgm5_noise5_trial0','genetic_noise5_trial0',]
utils.visualize_score_trajectory_cmp(CurDataDir_list=[neuron_dir+trial_title for trial_title in trial_list],
                                     title_str_list=trial_list, exp_title_str="caffe-net_fc6_0010_noise_level5",
                                     save=True, savedir=neuron_dir)
#%% Visualize Cholesky algorithm
neuron_dir = "/home/poncelab/Documents/data/with_CNN/caffe-net_fc6_0010/"
trial_list = ['cma_trial0_noeig_sgm5', 'genetic_trial0', 'choleskycma_sgm5_trial0', 'choleskycma_freqAupdate_sgm5_trial0']
utils.visualize_score_trajectory_cmp(CurDataDir_list=[neuron_dir+trial_title for trial_title in trial_list],
                                     title_str_list=trial_list, exp_title_str="caffe-net_fc6_0010_optimizer_cmp",
                                     save=True, savedir=neuron_dir)
#%%
neuron_dir = "/home/poncelab/Documents/data/with_CNN/caffe-net_fc6_0010/"
trial_list = ['choleskycma_sgm1_trial0', 'choleskycma_sgm3_trial0', 'choleskycma_sgm5_trial0', 'choleskycma_sgm3_uf10_trial0', 'choleskycma_sgm3_uf5_trial0']
utils.visualize_score_trajectory_cmp(CurDataDir_list=[neuron_dir+trial_title for trial_title in trial_list],
                                     title_str_list=trial_list, exp_title_str="caffe-net_fc6_0010_cholcma_param_cmp",
                                     save=True, savedir=neuron_dir)
#%%
CurDataDir = "/home/poncelab/Documents/data/with_CNN/caffe-net_fc6_0010/cma_trial0_noeig_sgm5"
utils.visualize_all(CurDataDir)
#%%
utils.visualize_image_score_each_block(CurDataDir, 299)

#%%
CurDataDir = "/home/poncelab/Documents/data/with_CNN/caffe-net_fc6_0010/"
#%% Visualize image code.

trial_title = 'choleskycma_sgm1_uf3_trial1'
utils.gen_visualize_image_score_each_block(CurDataDir+trial_title, block_num=298, exp_title_str=trial_title)
#%%
trial_title = "choleskycma_sgm1_uf3_trial0"
utils.gen_visualize_image_score_each_block(CurDataDir+trial_title, block_num=298, exp_title_str=trial_title)
#%%
trial_title = 'choleskycma_sgm3_trial0'
utils.gen_visualize_image_score_each_block(CurDataDir+trial_title, block_num=298, exp_title_str=trial_title,
                                           save=True, savedir=CurDataDir+trial_title)
trial_title = 'choleskycma_sgm3_trial1'
utils.gen_visualize_image_score_each_block(CurDataDir+trial_title, block_num=298, exp_title_str=trial_title,
                                           save=True, savedir=CurDataDir+trial_title)
trial_title = 'choleskycma_sgm3_trial2'
utils.gen_visualize_image_score_each_block(CurDataDir+trial_title, block_num=298, exp_title_str=trial_title,
                                           save=True, savedir=CurDataDir+trial_title)
#%%
trial_title = 'choleskycma_sgm3_uf5_trial0'
utils.gen_visualize_image_score_each_block(CurDataDir+trial_title, block_num=298, exp_title_str=trial_title,
                                           save=True, savedir=CurDataDir+trial_title)
trial_title = 'choleskycma_sgm3_uf5_trial1'
utils.gen_visualize_image_score_each_block(CurDataDir+trial_title, block_num=298, exp_title_str=trial_title,
                                           save=True, savedir=CurDataDir+trial_title)
trial_title = 'choleskycma_sgm3_uf5_trial2'
utils.gen_visualize_image_score_each_block(CurDataDir+trial_title, block_num=298, exp_title_str=trial_title,
                                           save=True, savedir=CurDataDir+trial_title)
#%%
trial_title = "choleskycma_sgm3_uf10_trial0"
utils.gen_visualize_image_score_each_block(CurDataDir+trial_title, block_num=298, exp_title_str=trial_title)
#%%
trial_title = "choleskycma_sgm3_uf10_trial0"
utils.gen_visualize_image_score_each_block(CurDataDir+trial_title, block_num=298, exp_title_str=trial_title,
                                           save=True, savedir=CurDataDir+trial_title)
#%%
trial_title = "choleskycma_freqAupdate_sgm5_trial0"
utils.gen_visualize_image_score_each_block(CurDataDir+trial_title, block_num=298, exp_title_str=trial_title,
                                           save=True, savedir=CurDataDir+trial_title)
#%%
trial_title = "genetic_trial0"
utils.visualize_image_score_each_block(CurDataDir+trial_title, block_num=298, exp_title_str=trial_title,
                                           save=True, savedir=CurDataDir+trial_title)

#%%  The Black Out trial code and image data!!!
# FIXME Why there is black out trial????
trial_title = "choleskycma_freqAupdate_sgm5_trial0"
imagefn = "gen296_011860.npy"
code_tmp = np.load(os.path.join(CurDataDir+trial_title, imagefn), allow_pickle=False).flatten()
img_tmp = utils.generator.visualize(code_tmp)
plt.imshow(img_tmp)
plt.show()
#%%
neuron_dir = "/home/poncelab/Documents/data/with_CNN/caffe-net_fc6_0010/"
trial_list = ['choleskycma_sgm1_trial0', 'choleskycma_sgm3_trial2', 'choleskycma_sgm5_trial0', 'choleskycma_sgm3_uf10_trial0', 'choleskycma_sgm3_uf5_trial0', 'choleskycma_sgm1_uf3_trial1', 'cma_trial0_noeig_sgm5', 'genetic_trial0']
fig=utils.visualize_score_trajectory_cmp(CurDataDir_list=[neuron_dir+trial_title for trial_title in trial_list],
                                     title_str_list=trial_list, exp_title_str="caffe-net_fc6_0010_cholcma_param_cmp2",
                                     save=True, savedir=neuron_dir)
#%%
neuron_dir = "/home/poncelab/Documents/data/with_CNN/caffe-net_fc6_0010/"
trial_list = ['choleskycma_sgm3_trial2', 'choleskycma_sgm3_uf10_trial0', 'choleskycma_sgm3_uf5_trial0', 'cma_trial0_noeig_sgm5', 'genetic_trial0']
fig=utils.visualize_score_trajectory_cmp(CurDataDir_list=[neuron_dir+trial_title for trial_title in trial_list],
                                     title_str_list=trial_list, exp_title_str="caffe-net_fc6_0010_cholcma_param_cmp3",
                                     save=True, savedir=neuron_dir)
fig.set_size_inches(14,10)
fig.show()
#%%
from Generator import Generator
generator = Generator()
#%% Compare Image Generation at certain block across trials
neuron_dir = "/home/poncelab/Documents/data/with_CNN/caffe-net_fc6_0010/"
# trial_list = ['choleskycma_sgm3_trial2', 'choleskycma_sgm1_trial2', 'choleskycma_sgm3_uf10_trial0', 'choleskycma_sgm3_uf5_trial0', 'cma_trial0_noeig_sgm5', 'genetic_trial0']
trial_list = ['choleskycma_sgm1_trial0',
 'choleskycma_sgm1_trial1',
 'choleskycma_sgm1_trial2',
 'choleskycma_sgm1_uf3_trial0',
 'choleskycma_sgm1_uf3_trial1',
 'choleskycma_sgm1_uf3_trial2',
 'choleskycma_sgm3_trial0',
 'choleskycma_sgm3_trial1',
 'choleskycma_sgm3_trial2',
 'choleskycma_sgm3_uf10_trial0',
 'choleskycma_sgm3_uf5_trial0',
 'choleskycma_sgm3_uf5_trial1',
 'choleskycma_sgm3_uf5_trial2',
 'choleskycma_sgm5_trial0',
 'genetic_trial0']
vis_image_num = 10
block_num = 299
exp_title_str = "Method evolving result compare"
figW = vis_image_num * 2.5
figH = len(trial_list) * 2.5 + 1
col_n = vis_image_num
row_n = len(trial_list)
fig = plt.figure(figsize=[figW, figH])
for trial_j, trial_title in enumerate(trial_list):
    CurDataDir = os.path.join(neuron_dir, trial_title)
    fncatalog = os.listdir(CurDataDir)
    fn_score_gen = [fn for fn in fncatalog if
                    (".npz" in fn) and ("score" in fn) and ("block{0:03}".format(block_num) in fn)]
    assert len(fn_score_gen) is 1, "not correct number of score files"
    with np.load(os.path.join(CurDataDir, fn_score_gen[0])) as data:
        score_gen = data['scores']
        image_ids = data['image_ids']
    idx = np.argsort( - score_gen)  # Note the minus sign for best scores sorting. positive sign for worst score sorting
    score_gen = score_gen[idx]
    image_ids = image_ids[idx]
    fn_image_gen = []
    use_img = not (len([fn for fn in fncatalog if (image_ids[0] in fn) and ('.bmp' in fn)])==0)
    # True, there is bmp rendered files. False, there is only code. we have to render it through Generator
    for imgid in image_ids[0: vis_image_num]:
        if use_img:
            fn_tmp_list = [fn for fn in fncatalog if (imgid in fn) and ('.bmp' in fn)]
            assert len(fn_tmp_list) is 1, "Code file not found or wrong Code file number"
            fn_image_gen.append(fn_tmp_list[0])
        if not use_img:
            fn_tmp_list = [fn for fn in fncatalog if (imgid in fn) and ('.npy' in fn)]
            assert len(fn_tmp_list) is 1, "Code file not found or wrong Code file number"
            fn_image_gen.append(fn_tmp_list[0])
    image_num = len(fn_image_gen)
    for i, imagefn in enumerate(fn_image_gen):
        if use_img:
            img_tmp = plt.imread(os.path.join(CurDataDir, imagefn))
        else:
            code_tmp = np.load(os.path.join(CurDataDir, imagefn), allow_pickle=False).flatten()
            img_tmp = generator.visualize(code_tmp)
        score_tmp = score_gen[i]
        plt.subplot(row_n, col_n, trial_j * col_n + i + 1)
        plt.imshow(img_tmp)
        plt.xticks([])
        plt.yticks([])
        if i == 0:
            plt.ylabel(trial_title)
        else:
            plt.axis('off')
        plt.title("{0:.2f}".format(score_tmp), fontsize=16)
        # if cmap_flag:  # color the titles with a heatmap!
        #     plt.title("{0:.2f}".format(score_tmp), fontsize=16,
        #               color=title_cmap((score_tmp - lb) / (ub - lb)))  # normalize a value between [0,1]
        # else:
        #     plt.title("{0:.2f}".format(score_tmp), fontsize=16)

plt.suptitle(exp_title_str + "\nBlock{0:03}".format(block_num), fontsize=16)
plt.show()

#%% the utility used
neuron_dir = "/home/poncelab/Documents/data/with_CNN/caffe-net_fc6_0010/"
# trial_list = ['choleskycma_sgm3_trial2', 'choleskycma_sgm1_trial2', 'choleskycma_sgm3_uf10_trial0', 'choleskycma_sgm3_uf5_trial0', 'cma_trial0_noeig_sgm5', 'genetic_trial0']
trial_list = ['choleskycma_sgm1_trial0',
 'choleskycma_sgm1_trial1',
 'choleskycma_sgm1_trial2',
 'choleskycma_sgm1_uf3_trial0',
 'choleskycma_sgm1_uf3_trial1',
 'choleskycma_sgm1_uf3_trial2',
 'choleskycma_sgm3_trial0',
 'choleskycma_sgm3_trial1',
 'choleskycma_sgm3_trial2',
 'choleskycma_sgm3_uf10_trial0',
 'choleskycma_sgm3_uf5_trial0',
 'choleskycma_sgm3_uf5_trial1',
 'choleskycma_sgm3_uf5_trial2',
 'choleskycma_sgm5_trial0',
 'genetic_trial0']
utils.cmp_image_score_across_trial(neuron_dir, trial_list, block_num=299, vis_image_num=10)

#%% score comparison across neurons
# neuron_list = [('caffe-net', 'fc6', 100), ('caffe-net', 'fc6', 5), ('caffe-net', 'fc7', 1),
#                    ('caffe-net', 'fc7', 10)]
neuron_list = ['caffe-net_fc6_0010', 'caffe-net_fc6_0100', 'caffe-net_fc6_0005', 'caffe-net_fc7_0001', 'caffe-net_fc7_0010']
# neuron_dir_list = [neuron_name + "/home/poncelab/Documents/data/with_CNN/" for neuron_name in neuron_list]
CurDataDir_list = [os.path.join("/home/poncelab/Documents/data/with_CNN/", neuron_name, "genetic_trial0") for neuron_name in neuron_list]
title_str_list = neuron_list
step_num = 300
population_size=40

cmp_trial_num = len(CurDataDir_list)
ScoreStatCmpTable = np.full((cmp_trial_num, step_num), np.NaN)
assert len(CurDataDir_list) == len(title_str_list)
figh, ax = plt.subplots()
for trial_j, (CurDataDir, title_str) in enumerate(zip(CurDataDir_list, title_str_list)):
    ScoreEvolveTable = np.full((step_num, population_size,), np.NAN)
    startnum=0
    steps=step_num
    for stepi in range(startnum, steps):
        try:
            with np.load(os.path.join(CurDataDir, "scores_end_block{0:03}.npz".format(stepi))) as data:
                score_tmp = data['scores']
                ScoreEvolveTable[stepi, :len(score_tmp)] = score_tmp
        except FileNotFoundError:
            if stepi == 0:
                startnum += 1
                steps += 1
                continue
            else:
                print("maximum steps is %d." % stepi)
                ScoreEvolveTable = ScoreEvolveTable[0:stepi, :]
                steps = stepi
                break

    gen_slice = np.arange(startnum, steps).reshape((-1, 1))
    gen_num = np.repeat(gen_slice, population_size, 1)

    AvgScore = np.nanmean(ScoreEvolveTable, axis=1)
    MaxScore = np.nanmax(ScoreEvolveTable, axis=1)
    ScoreStatCmpTable[trial_j, 0:len(AvgScore)] = AvgScore

img = ax.matshow(ScoreStatCmpTable, cmap=plt.cm.Blues)
ax.set_aspect(10)
ax.xaxis.tick_bottom()
# plt.plot(gen_slice, AvgScore, color='black', label="Average score")
# plt.plot(gen_slice, MaxScore, color='red', label="Max score")
plt.yticks(np.arange(cmp_trial_num), title_str_list)
plt.xlabel("generation #")
plt.ylabel("neurons")
plt.title("Optimization Heatmap of Score Comparison\n" + exp_title_str)
plt.colorbar(img, ax=ax, orientation='horizontal', fraction=.1)
# plt.legend()
# if save:
#     if savedir=='':
#         savedir = CurDataDir
#     plt.savefig(os.path.join(savedir, exp_title_str + "score_traj"))

plt.show()
# return figh


#%% Score Comparison for 2 methods

neuron_list = ['caffe-net_fc6_0005', 'caffe-net_fc6_0010', 'caffe-net_fc6_0030', 'caffe-net_fc6_0100', 'caffe-net_fc6_0150', 'caffe-net_fc6_0200', 'caffe-net_fc6_0250', 'caffe-net_fc7_0001', 'caffe-net_fc7_0010']
# neuron_dir_list = [neuron_name + "/home/poncelab/Documents/data/with_CNN/" for neuron_name in neuron_list]
method_list = ['Genetic', 'choleskycma_sgm3_uf10']
CurDataDir_list_gene = [os.path.join("/home/poncelab/Documents/data/with_CNN/", neuron_name, "genetic_trial0") for neuron_name in neuron_list]
CurDataDir_list_cma = [os.path.join("/home/poncelab/Documents/data/with_CNN/", neuron_name, "choleskycma_sgm3_uf10_trial0") for neuron_name in neuron_list]

exp_title_str = "Between Genetics and Cholesky CMA-ES algorithm"

title_str_list = neuron_list
step_num = 300
population_size = 40

# figh, ax = plt.subplots()
figh, axs = plt.subplots(1, 2, figsize=[10,4],)
imgs = []
StatCmpTable_list = []
for method_i, CurDataDir_list in enumerate([CurDataDir_list_gene, CurDataDir_list_cma]):
    # Loop through methods
    assert len(CurDataDir_list) == len(title_str_list)
    cmp_trial_num = len(CurDataDir_list)
    ScoreStatCmpTable = np.full((cmp_trial_num, step_num), np.NaN)
    for trial_j, (CurDataDir, title_str) in enumerate(zip(CurDataDir_list, title_str_list)):
        # Loop through trials / neurons
        ScoreEvolveTable = np.full((step_num, population_size,), np.NAN)
        startnum = 0
        steps = step_num
        for stepi in range(startnum, steps):
            try:
                with np.load(os.path.join(CurDataDir, "scores_end_block{0:03}.npz".format(stepi))) as data:
                    score_tmp = data['scores']
                    ScoreEvolveTable[stepi, :len(score_tmp)] = score_tmp
            except FileNotFoundError:
                if stepi == 0:
                    startnum += 1
                    steps += 1
                    continue
                else:
                    print("maximum steps is %d." % stepi)
                    ScoreEvolveTable = ScoreEvolveTable[0:stepi, :]
                    steps = stepi
                    break

        gen_slice = np.arange(startnum, steps).reshape((-1, 1))
        gen_num = np.repeat(gen_slice, population_size, 1)

        AvgScore = np.nanmean(ScoreEvolveTable, axis=1)
        MaxScore = np.nanmax(ScoreEvolveTable, axis=1)
        ScoreStatCmpTable[trial_j, 0:len(AvgScore)] = AvgScore
    StatCmpTable_list.append(ScoreStatCmpTable.copy())
    imgs.append(axs[method_i].matshow(ScoreStatCmpTable, cmap=plt.cm.Blues))
    axs[method_i].set_aspect(10)
    axs[method_i].xaxis.tick_bottom()
    if method_i == 0:
        axs[method_i].set_yticklabels(title_str_list)
        axs[method_i].set_ylabel("neurons")
    else:
        axs[method_i].set_yticklabels([])
    # fighs[method_i].yticks(np.arange(cmp_trial_num), title_str_list)
    axs[method_i].set_xlabel("generation #")
    axs[method_i].set_title(method_list[method_i])

# Find the min and max of all colors for use in setting the color scale.
vmin = min(img.get_array().min() for img in imgs)
vmax = max(img.get_array().max() for img in imgs)
norm = colors.Normalize(vmin=vmin, vmax=vmax)
for im in imgs:
    im.set_norm(norm)
cbaxes = figh.add_axes([0.1, 0.15, 0.8, 0.05])  # Very good method to add axes and colorbar for the figure.
plt.suptitle("Optimization Score Heatmap Comparison\n" + exp_title_str)
cbar = plt.colorbar(imgs[1], cax=cbaxes, orientation='horizontal')#, pad=0.2)#, fraction=.1)
cbar.set_label("Score")
# if save:
#     if savedir=='':
#         savedir = CurDataDir
#     plt.savefig(os.path.join(savedir, exp_title_str + "score_traj"))
# "CMA_Genetic_heatmap_cmp"
plt.show()
# return figh

#%% Normalized score
min_vec = np.minimum( StatCmpTable_list[0].min(axis=1), StatCmpTable_list[1].min(axis=1) )
max_vec = np.maximum( StatCmpTable_list[0].max(axis=1), StatCmpTable_list[1].max(axis=1) )
min_vec.shape = (-1,1)
max_vec.shape = (-1,1)
NormStatCmpTable_list = [0,0]
NormStatCmpTable_list[0] = ( StatCmpTable_list[0]-min_vec ) / (max_vec - min_vec)
NormStatCmpTable_list[1] = ( StatCmpTable_list[1]-min_vec ) / (max_vec - min_vec)

#%%
figh, axs = plt.subplots(1, 2, figsize=[10,4],)
imgs = []
for method_i in range(2):
    imgs.append(axs[method_i].matshow(NormStatCmpTable_list[method_i], cmap=plt.cm.Blues))
    axs[method_i].set_aspect(10)
    axs[method_i].xaxis.tick_bottom()
    if method_i == 0:
        axs[method_i].set_yticklabels(title_str_list)
        axs[method_i].set_ylabel("neurons")
    else:
        axs[method_i].set_yticklabels([])
    # fighs[method_i].yticks(np.arange(cmp_trial_num), title_str_list)
    axs[method_i].set_xlabel("generation #")
    axs[method_i].set_title(method_list[method_i])

# Find the min and max of all colors for use in setting the color scale.
vmin = min(img.get_array().min() for img in imgs)
vmax = max(img.get_array().max() for img in imgs)
norm = colors.Normalize(vmin=vmin, vmax=vmax)
for im in imgs:
    im.set_norm(norm)
cbaxes = figh.add_axes([0.1, 0.15, 0.8, 0.05])  # Very good method to add axes and colorbar for the figure.
plt.suptitle("Optimization (Normalized) Score Heatmap Comparison\n" + exp_title_str)
cbar = plt.colorbar(imgs[1], cax=cbaxes, orientation='horizontal')#, pad=0.2)#, fraction=.1)
cbar.set_label("Normalized Score for each unit")
# if save:
#     if savedir=='':
#         savedir = CurDataDir
#     plt.savefig(os.path.join(savedir, exp_title_str + "score_traj"))
# "CMA_Genetic_heatmap_cmp"
plt.show()

#%%
neuron_dir = "/home/poncelab/Documents/data/with_CNN/caffe-net_fc6_0010/"
trial_list = ['choleskycma_sgm3_uf10_trial%d' % i for i in range(5)] + ['genetic_trial%d' % i for i in range(5)]
fig=utils.visualize_score_trajectory_cmp(CurDataDir_list=[neuron_dir+trial_title for trial_title in trial_list],
                                     title_str_list=trial_list, exp_title_str="caffe-net_fc6_0010, Comparison between Genetic and Cholesky CMA",
                                     save=True, savedir=neuron_dir)
fig.set_size_inches(8,6)
fig.show()