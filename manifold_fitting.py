import os
from time import time
import numpy as np
from CNNScorer import NoIOCNNScorer
import CNNScorer
from Optimizer import CMAES, Genetic, CholeskyCMAES
import utils
from utils import add_neuron_subdir, add_trial_subdir
import matplotlib.pyplot as plt
import importlib
importlib.reload(utils)  # Reload the modules after changing them, or they will not affect the codes.
importlib.reload(CNNScorer)

exp_dir = "/home/poncelab/Documents/data/with_CNN/"
neuron = ('caffe-net', 'fc6', 10)
this_exp_dir = add_neuron_subdir(neuron, exp_dir)
trial_title = 'choleskycma_sgm3_uf10_trial%d' % 3
trialdir = add_trial_subdir(this_exp_dir, trial_title)

#%% Score sorting routine
CurDataDir = trialdir
steps = 300
population_size = 40
# for CurDataDir, title_str in zip(CurDataDir_list, title_str_list):
ScoreEvolveTable = np.full((steps, population_size,), np.NAN)
ImageidTable = [[""]*population_size for i in range(steps)]
startnum = 0
for stepi in range(startnum, steps):
    try:
        with np.load(os.path.join(CurDataDir, "scores_end_block{0:03}.npz".format(stepi))) as data:
            score_tmp = data['scores']
            image_ids = data['image_ids']
            ScoreEvolveTable[stepi, :len(score_tmp)] = score_tmp
            ImageidTable[stepi][0:len(score_tmp)] = image_ids
    except FileNotFoundError:
        if stepi == 0:
            startnum += 1
            steps += 1
            continue
        else:
            print("maximum steps is %d." % stepi)
            ScoreEvolveTable = ScoreEvolveTable[0:stepi, :]
            ImageidTable = ImageidTable[0:stepi]
            steps = stepi
            break
ImageidTable = np.asarray(ImageidTable)

utils.savez(os.path.join(CurDataDir, "scores_summary_table.npz"),
         {"ScoreEvolveTable" : ScoreEvolveTable,"ImageidTable" : ImageidTable})



#%% Filter the Samples that has Score in a given range

def select_image(CurDataDir, lb = 200, ub = None):
    fncatalog = os.listdir(CurDataDir)
    if "scores_summary_table.npz" in fncatalog:
        # if the summary table exist, just read from it!
        with np.load(os.path.join(CurDataDir, "scores_summary_table.npz")) as data:
            ScoreEvolveTable = data['ScoreEvolveTable']
            ImageidTable = data['ImageidTable']
    else:
        ScoreEvolveTable, ImageidTable= utils.scores_summary(CurDataDir)
    if ub is None:
        ub = np.nanmax(ScoreEvolveTable)+1
    if lb is None:
        lb = np.nanmin(ScoreEvolveTable)-1
    imgid_list = ImageidTable[np.logical_and(ScoreEvolveTable > lb, ScoreEvolveTable < ub)]
    score_list = ScoreEvolveTable[np.logical_and(ScoreEvolveTable > lb, ScoreEvolveTable < ub)]
    image_fn= []
    for imgid in imgid_list:
        fn_tmp_list = [fn for fn in fncatalog if (imgid in fn) and '.npy' in fn]
        assert len(fn_tmp_list) is 1, "Code file not found or wrong Code file number"
        image_fn.append(fn_tmp_list[0])
    code_array = []
    for imagefn in image_fn:
        code = np.load(os.path.join(CurDataDir, imagefn), allow_pickle=False).flatten()
        code_array.append(code.copy())
        # img_tmp = utils.generator.visualize(code_tmp)
    return code_array, score_list, imgid_list
#%% Collect the high rated images across trial
exp_dir = "/home/poncelab/Documents/data/with_CNN/"
neuron = ('caffe-net', 'fc6', 100)
this_exp_dir = add_neuron_subdir(neuron, exp_dir)
code_total_list = []
score_total_list = []
for i in range(20):
    trial_title = 'choleskycma_sgm3_uf10_trial%d' % i
    trialdir = add_trial_subdir(this_exp_dir, trial_title)
    code_array, score_list, _ = select_image(trialdir, lb=210)
    code_total_list += code_array
    score_total_list += list(score_list)
#%% Do the manifold fitting through code space
from sklearn.manifold import LocallyLinearEmbedding
embedding = LocallyLinearEmbedding(n_components=10, n_neighbors=30)
code_total_array = np.asarray(code_total_list)
#%%
DR_code_array = embedding.fit_transform(code_total_array)

#%% Do manifold fitting through image space


#%%
def simplex_interpolate(wvec, code_array):
    if type(code_array) is list:
        code_array = np.asarray(code_array)
    code_n = code_array.shape[0]
    if np.isscalar(wvec):
        w_vec = np.asarray([wvec, 1-wvec])
    elif len(wvec) == code_n:
        w_vec = np.asarray(wvec)
    elif len(wvec) == code_n - 1:
        w_vec = np.zeros(code_n)
        w_vec[:-1] = wvec
        w_vec[-1] = 1 - sum(w_vec[:-1])
    else:
        raise ValueError
    code = w_vec @ code_array
    return code

#%% Visualize the interpolated code
img_num = 20000
plt.figure()
img_tmp = utils.generator.visualize(code_total_array[img_num,:])
plt.imshow(img_tmp)
plt.title(str(score_total_list[img_num]))
plt.show()
#%% Test the interpolated score
neuron = ('caffe-net', 'fc6', 100)
TestScorer = CNNScorer.NoIOCNNScorer(target_neuron=neuron, writedir=this_exp_dir)
TestScorer.load_classifier()

#%%
plt.subplots(1, 11, figsize = [10, 1.5])
img_nums = np.random.randint(70000,size=11)
for i in range(11):
    img_tmp = utils.generator.visualize(code_total_array[img_nums[i], :])
    plt.subplot(1, 11,i+1)
    plt.imshow(img_tmp)
    plt.xticks([])
    plt.yticks([])
    plt.axis('off')
    plt.title("%.2f" % (TestScorer.test_score(img_tmp)[0]) )
plt.suptitle(str(img_nums))
plt.tight_layout()
plt.show()

#%%
img_tuple = 14425, 43408
plt.subplots(1, 11, figsize = [10, 1.5])
for i in range(11):
    img_tmp = utils.generator.visualize(simplex_interpolate(i/10, code_total_array[img_tuple, :]))
    plt.subplot(1, 11,i+1)
    plt.imshow(img_tmp)
    plt.xticks([])
    plt.yticks([])
    plt.axis('off')
    plt.title("%.2f" % TestScorer.test_score(img_tmp)[0])
plt.tight_layout()
plt.show()


# img_tmp = utils.generator.visualize(code_tmp)
#%%
from experiment_with_CNN_Integrated import CNNExperiment_Simplify
neuron = ('caffe-net', 'fc6', 100)
this_exp_dir = add_neuron_subdir(neuron, exp_dir)
random_seed = int(time())
trial_title = 'choleskycma_sgm3_uf10_continopt_trial%d' % 0
trialdir = add_trial_subdir(this_exp_dir, trial_title)
experiment = CNNExperiment_Simplify(recorddir=trialdir, logdir=trialdir, target_neuron=neuron, max_steps=100,
                                    optimizer_name='cholcmaes', init_sigma=3, init_code=code_total_array[14425, :], Aupdate_freq=10,
                                    random_seed=random_seed, saveimg=False, record_pattern=False, )
experiment.run()
utils.visualize_score_trajectory(trialdir, save=True, title_str=trial_title)
