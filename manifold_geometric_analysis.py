import os
from time import time
import numpy as np
import matplotlib.pyplot as plt
from CNNScorer import NoIOCNNScorer
import utils
from utils import add_neuron_subdir, add_trial_subdir, generator, simplex_interpolate
from sklearn.manifold import LocallyLinearEmbedding, MDS
from mpl_toolkits.mplot3d import Axes3D

import importlib
importlib.reload(utils)
#%%

exp_dir = "/home/poncelab/Documents/data/with_CNN/"
neuron = ('caffe-net', 'fc6', 10)
this_exp_dir = add_neuron_subdir(neuron, exp_dir)
trial_title = 'choleskycma_sgm3_uf10_trial%d' % 3
trialdir = add_trial_subdir(this_exp_dir, trial_title)

TestScorer = NoIOCNNScorer(target_neuron=neuron, writedir=this_exp_dir)
TestScorer.load_classifier()

#%% Title: Prepare the high level code set
exp_dir = "/home/poncelab/Documents/data/with_CNN/"
neuron = ('caffe-net', 'fc6', 10)
this_exp_dir = add_neuron_subdir(neuron, exp_dir)
code_total_list = []
score_total_list = []
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
for trial_title in trial_list:
    trialdir = add_trial_subdir(this_exp_dir, trial_title)
    # utils.scores_summary(trialdir, regenerate=True)
    code_array, score_list, _ = utils.select_image(trialdir, lb=205, ub=None)
    code_total_list += code_array
    score_total_list += list(score_list)
code_total_array = np.asarray(code_total_list)

#%% Tangent Map of GAN

#%% Gradient Map of Activation function


#%% Interpolate bridge between
img_tuple = 7303, 7310

img_tmp = [generator.visualize(simplex_interpolate( i/40, code_total_array[img_tuple, :])) for i in range(41)]
scores = TestScorer.test_score(img_tmp)
plt.figure()
plt.plot(np.linspace(0, 1, 41), scores)
plt.title("Interpolation Scores between %d-%d" % img_tuple)
pos = np.linspace(0,1,11)
labels = ["%.1f"%x for x in pos]
labels[0] += "\n"+str(img_tuple[0])
labels[-1] += "\n"+str(img_tuple[1])
plt.xticks(pos, labels)
plt.show()


#%%
from numpy.linalg import norm
def distance_metric(codes_array,):
    if type(codes_array) is list:
        codes_array = np.asarray(codes_array)
    # dist = np.sqrt(np.sum((codes_array[1,:]-codes_array[0,:])**2))
    dist = norm(codes_array[1, :]-codes_array[0, :])
    return dist

def interpolate_score_curve(codes_array, imgid_tuple, interp_n=40):
    '''Input the
    codes_array in ndarray form
    imgid_tuple: form of id tuple '''
    img_tmp = [generator.visualize(simplex_interpolate( i/interp_n, codes_array)) for i in range(interp_n + 1)]
    scores = TestScorer.test_score(img_tmp)
    dist = distance_metric(code_total_array[img_tuple, :])
    plt.figure()
    plt.plot(np.linspace(0, 1, interp_n + 1), scores)
    plt.title("Interpolation Scores between %d-%d\ndist:%.1f" % ( *imgid_tuple, dist))
    pos = np.linspace(0, 1, 11)
    labels = ["%.1f"%x for x in pos]
    labels[0] += "\n"+str(imgid_tuple[0])
    labels[-1] += "\n"+str(imgid_tuple[1])
    plt.xticks(pos, labels)
    plt.show()
    return scores

#%%
img_tuple = 1000,1002
interpolate_score_curve(code_total_array[img_tuple, :], img_tuple)


#%% The distance relationship between hot spots.
img_id_list = np.arange(0, 15000, 5, dtype=np.int)#[1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000]

img_n = len(img_id_list)
dist_mat = np.zeros((img_n, img_n))
for i in range(img_n):
    for j in range(i):
        id_tuple = (img_id_list[i], img_id_list[j])
        dist_mat[i, j] = distance_metric(code_total_array[id_tuple, :])
        dist_mat[j, i] = dist_mat[i, j]
#%% Visualize the distance matrix and the scores together!
import matplotlib.gridspec as gridspec
plt.figure(figsize=[6, 7])
gs = gridspec.GridSpec(2, 1, height_ratios=[3,1])
plt.subplot(gs[0])  #(2, 1, 1)
plt.matshow(dist_mat, fignum=False)
plt.colorbar()
score_total_array = np.asarray(score_total_list)
plt.subplot(gs[1])  # (2, 1, 2)
plt.plot(score_total_array[img_id_list])
plt.show()


#%%  Calculate the convexity and Concavity of linear interpolated bridge between nodes on manifold
interp_n = 20
img_n = len(img_id_list)
interp_max_score_mat = np.zeros((img_n, img_n), dtype=np.float)
interp_max_pos_mat = np.zeros((img_n, img_n), dtype=np.float)
interp_min_score_mat = np.zeros((img_n, img_n), dtype=np.float)
interp_min_pos_mat = np.zeros((img_n, img_n), dtype=np.float)
convex_flag_mat = np.zeros((img_n, img_n), dtype=np.bool)
concave_flag_mat = np.zeros((img_n, img_n), dtype=np.bool)
for i in range(img_n):
    for j in range(i):
        id_tuple = (img_id_list[i], img_id_list[j])
        img_tmp = [generator.visualize(simplex_interpolate(i / interp_n, code_total_array[id_tuple, :])) for i in range(interp_n + 1)]
        scores = TestScorer.test_score(img_tmp)
        pos_max = scores.argmax()
        score_max = scores[pos_max]
        pos_min = scores.argmin()
        score_min = scores[pos_min]
        if pos_max not in [0, interp_n]:
            # if maximum in the interval is achieved else where from tips, likely to be convex upward
            convex_flag_mat[i, j] = 1
            convex_flag_mat[j, i] = 1
            interp_max_score_mat[i, j] = score_max
            interp_max_score_mat[j, i] = score_max
            interp_max_pos_mat[i, j] = pos_max/interp_n
            interp_max_pos_mat[j, i] = 1 - pos_max/interp_n
        if pos_min not in [0, interp_n]:
            # if minimum in the interval is achieved else where from tips, likely to be concave downward
            concave_flag_mat[i, j] = 1
            concave_flag_mat[j, i] = 1
            interp_min_score_mat[i, j] = score_min
            interp_min_score_mat[j, i] = score_min
            interp_min_pos_mat[i, j] = pos_min / interp_n
            interp_min_pos_mat[j, i] = 1 - pos_min / interp_n

#%%
data_sets = {"dist_mat": dist_mat,
             "interp_max_score_mat": interp_max_score_mat ,
             "interp_max_pos_mat": interp_max_pos_mat ,
             "interp_min_score_mat": interp_min_score_mat ,
             "interp_min_pos_mat": interp_min_pos_mat ,
             "convex_flag_mat": convex_flag_mat ,
             "concave_flag_ma": concave_flag_mat}
np.savez(os.path.join(this_exp_dir, "Hotspot_dist_conv_mat.npz"), data_sets)

#%% ########################################################################
#  Title: Dimension Reduction of Image Islands
#   #######################################################################

#%% Manifold Embedding Visualization of the Code.

# embedding = LocallyLinearEmbedding(n_components=2, n_neighbors=100)
embedding2 = MDS(n_components=2)
#%%
# img_id_list = np.arange(0, 15000, 5, dtype=np.int)
DR_code_array = embedding2.fit_transform(code_total_array[img_id_list, :]) # Really time consuming! hard
#%% 2D dimension reduction visualize
plt.figure()
plt.scatter(DR_code_array[:,0],DR_code_array[:, 1], c=img_id_list)
plt.show()
#%% 3D dimension reduction visualize
embedding3 = MDS(n_components=3)
DR3_code_array = embedding3.fit_transform(code_total_array[img_id_list, :])
#%%
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(DR3_code_array[:, 0], DR3_code_array[:, 1], DR3_code_array[:, 2], c=img_id_list)
ax.view_init(80, 50)
plt.show()

#%% Title: Dimension Reduce the Optimization Trajectory
#%% Get all the code for one evolution
exp_dir = "/home/poncelab/Documents/data/with_CNN/"
neuron = ('caffe-net', 'fc6', 10)
this_exp_dir = add_neuron_subdir(neuron, exp_dir)
trial_title = "choleskycma_sgm3_uf10_trial0"
trialdir = add_trial_subdir(this_exp_dir, trial_title)
ScoreEvolveTable, ImagefnTable = utils.scores_summary(trialdir,)
code_list, score_list, imgid_list = utils.select_image(trialdir, trial_rng=(1, None))  # discard the 0 trial natural images
gen_num_array = np.ceil((np.arange(len(score_list)) - 36 + 1) / 40)
#%%
exp_dir = "/home/poncelab/Documents/data/with_CNN/"
neuron = ('caffe-net', 'fc6', 10)
this_exp_dir = add_neuron_subdir(neuron, exp_dir)
code_total_list = []
score_total_list = []
gen_total_list = []
method_total_list = []
trial_list = ['choleskycma_sgm3_uf10_trial0',
             'genetic_trial4']
for trial_title in trial_list:
    trialdir = add_trial_subdir(this_exp_dir, trial_title)
    # utils.scores_summary(trialdir, regenerate=True)
    code_list, score_list, _ = utils.select_image(trialdir, trial_rng=(1, None))
    if "cma" in trial_title:
        gen_num_array = np.ceil((np.arange(len(score_list)) - 36 + 1) / 40)
        method_num_array = [2] * len(score_list)
    elif "genetic" in trial_title:
        gen_num_array = np.ceil((np.arange(len(score_list)) - 36 + 1) / 36)
        method_num_array = [1] * len(score_list)
    code_total_list += code_list
    score_total_list += list(score_list)
    gen_total_list += list(gen_num_array)
    method_total_list += list(method_num_array)
code_total_array = np.asarray(code_total_list)
method_total_list = np.asarray(method_total_list)


#%%
embedding2 = MDS(n_components=2)
DR_traj_code_array = embedding2.fit_transform(code_total_array[slice(0, None, 8), :]) # Really time consuming! hard
#%% 2D dimension reduction visualize
color_array = np.asarray(gen_total_list)
size_array = ((method_total_list)**4)*4
plt.figure(figsize=[10, 8])
plt.scatter(DR_traj_code_array[:, 0], DR_traj_code_array[:, 1], c=color_array[slice(0, None, 8)], s=size_array[slice(0, None, 8)], alpha=0.5)
plt.title("Code Space Optimization Trajectory Cmp (MDS) \n"+str(trial_list))
cbar = plt.colorbar(orientation='vertical')#, pad=0.2)#, fraction=.1)
cbar.set_label("Generation Num")
plt.show()
#%%
color_array = score_total_list
size_array = ((method_total_list)**4)*4
plt.figure(figsize=[10, 8])
plt.scatter(DR_traj_code_array[:, 0], DR_traj_code_array[:, 1], c=color_array[slice(0, None, 8)], s=size_array[slice(0, None, 8)], alpha=0.5)
plt.title("Code Space Optimization Trajectory Cmp (MDS) \n"+str(trial_list))
cbar = plt.colorbar(orientation='vertical')#, pad=0.2)#, fraction=.1)
cbar.set_label("Score")
plt.show()

#%%
embedding3 = MDS(n_components=3)
DR3_code_array = embedding3.fit_transform(code_total_array[slice(0, None, 8), :])
#%%
color_array = np.asarray(gen_total_list)
size_array = ((method_total_list)**4)*4
fig = plt.figure(figsize=[10, 8])
ax = fig.add_subplot(111, projection='3d')
p = ax.scatter(DR3_code_array[:, 0], DR3_code_array[:, 1], DR3_code_array[:, 2], c=color_array[slice(0, None, 8)], s=size_array[slice(0, None, 8)], alpha=0.5)
ax.view_init(20, 30) #(10, -60)#
cbar = plt.colorbar(p, orientation='vertical')#, pad=0.2)#, fraction=.1)
cbar.set_label("Generation Num")
plt.title("Code Space Optimization Trajectory Cmp (MDS) \n"+str(trial_list))
plt.show()
#%%
color_array = score_total_list
size_array = ((method_total_list)**4)*4
fig = plt.figure(figsize=[10, 8])
ax = fig.add_subplot(111, projection='3d')
p = ax.scatter(DR3_code_array[:, 0], DR3_code_array[:, 1], DR3_code_array[:, 2], c=color_array[slice(0, None, 8)], s=size_array[slice(0, None, 8)], alpha=0.5)
ax.view_init(10, -60)#(20, 30) #
cbar = plt.colorbar(p, orientation='vertical')#, pad=0.2)#, fraction=.1)
cbar.set_label("Score")
plt.title("Code Space Optimization Trajectory Cmp (MDS) \n"+str(trial_list))
plt.show()

#%% Title: More Dimension reduced comparison for interpolation + optimization
exp_dir = "/home/poncelab/Documents/data/with_CNN/"
neuron = ('caffe-net', 'fc6', 10)
this_exp_dir = add_neuron_subdir(neuron, exp_dir)
code_total_list = []
score_total_list = []
gen_total_list = []
interp_total_list = []
slc = slice(0, None, 10)
for i in range(11):
    trial_title = 'choleskycma_sgm3_uf10_continopt_trial%d' % i
    trialdir = add_trial_subdir(this_exp_dir, trial_title)
    # utils.scores_summary(trialdir, regenerate=True)
    code_list, score_list, _ = utils.select_image(trialdir, trial_rng=(1, None))
    gen_num_array = np.ceil((np.arange(len(score_list)) - 36 + 1) / 40)
    interp_num_array = [i] * len(score_list)

    code_total_list += code_list[slc]
    score_total_list += list(score_list[slc])
    gen_total_list += list(gen_num_array[slc])
    interp_total_list += list(interp_num_array[slc])
code_total_array = np.asarray(code_total_list)
interp_total_list = np.asarray(interp_total_list)
#%%
embedding3 = MDS(n_components=3)
DR3_code_array = embedding3.fit_transform(code_total_array[slice(0, None, 2), :]) # Really time consuming! hard

#%%
color_array = np.asarray(gen_total_list)
size_array = 20 #((interp_total_list)**2)*4 [slice(0, None, 2)]
fig = plt.figure(figsize=[10, 8])
ax = fig.add_subplot(111, projection='3d')
p = ax.scatter(DR3_code_array[:, 0], DR3_code_array[:, 1], DR3_code_array[:, 2], c=color_array[slice(0, None, 2)], s=size_array, alpha=0.5)
ax.view_init(10, -60)#(20, 30) #
cbar = plt.colorbar(p, orientation='vertical')#, pad=0.2)#, fraction=.1)
cbar.set_label("Generation # ")
plt.title("Code Space Optimization Trajectory Cmp (MDS) \n" + "optimization from interpolated points")
plt.show()

#%%
color_array = np.asarray(interp_total_list)/10
size_array = 20 #((interp_total_list)**2)*4 [slice(0, None, 2)]
fig = plt.figure(figsize=[10, 8])
ax = fig.add_subplot(111, projection='3d')
p = ax.scatter(DR3_code_array[:, 0], DR3_code_array[:, 1], DR3_code_array[:, 2], c=color_array[slice(0, None, 2)], s=size_array, alpha=0.5)
ax.view_init(10, -60)#(20, 30) #
cbar = plt.colorbar(p, orientation='vertical')#, pad=0.2)#, fraction=.1)
cbar.set_label("Interpolation starting percentage")
plt.title("Code Space Optimization Trajectory Cmp (MDS) \n" + "optimization from interpolated points")
plt.show()

#%%
embedding2 = MDS(n_components=2)
DR_traj_code_array = embedding2.fit_transform(code_total_array[slice(0, None, 2), :]) # Really time consuming! hard
#%%
color_array = np.asarray(gen_total_list)[slice(0, None, 2)]
size_array = (((interp_total_list+1)**2)*2)[slice(0, None, 2)]
plt.figure(figsize=[10, 8])
plt.scatter(DR_traj_code_array[:, 0], DR_traj_code_array[:, 1], c=color_array, s=size_array, alpha=0.5)
plt.title("Code Space Optimization Trajectory Cmp (MDS) \n")
cbar = plt.colorbar(orientation='vertical')#, pad=0.2)#, fraction=.1)
cbar.set_label("Generation Num")
plt.show()
#%% 2D dimension reduction visualize
color_array = np.asarray(gen_total_list)[slice(0, None, 2)]
size_array = 20#((interp_total_list)**2)*2[slice(0, None, 2)]
plt.figure(figsize=[10, 8])
plt.scatter(DR_traj_code_array[:, 0], DR_traj_code_array[:, 1], c=color_array, s=size_array, alpha=0.5)
plt.title("Code Space Optimization Trajectory Cmp (MDS) \n")
cbar = plt.colorbar(orientation='vertical')#, pad=0.2)#, fraction=.1)
cbar.set_label("Generation Num")
plt.show()
#%% 2D dimension reduction visualize
color_array = np.asarray(interp_total_list)[slice(0, None, 2)]/10
size_array = 20#((interp_total_list)**2)*2[slice(0, None, 2)]
plt.figure(figsize=[10, 8])
plt.scatter(DR_traj_code_array[:, 0], DR_traj_code_array[:, 1], c=color_array, s=size_array, alpha=0.5)
plt.title("Code Space Optimization Trajectory Cmp (MDS) \n")
cbar = plt.colorbar(orientation='vertical')#, pad=0.2)#, fraction=.1)
cbar.set_label("Interpolation starting percentage")
plt.show()
#%%
color_array = np.asarray(score_total_list)[slice(0, None, 2)]
size_array = 20#((interp_total_list)**2)*4 [slice(0, None, 2)]
plt.figure(figsize=[10, 8])
plt.scatter(DR_traj_code_array[:, 0], DR_traj_code_array[:, 1], c=color_array, s=size_array, alpha=0.5)
plt.title("Code Space Optimization Trajectory Cmp (MDS) \n")
cbar = plt.colorbar(orientation='vertical')#, pad=0.2)#, fraction=.1)
cbar.set_label("Score")
plt.show()




#%% Title: Local Tangent Map of Gan and Score
