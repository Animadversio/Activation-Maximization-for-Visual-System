import os
from time import time
import re
import utils
from utils import add_neuron_subdir, add_trial_subdir
import utils
import numpy as np
import h5py
#%%
homedir = r"D:\Generator_DB_Windows" #os.path.expanduser('~')
initcodedir = os.path.join(homedir, 'stimuli/texture006')
natstimdir = os.path.join(homedir, 'stimuli/natimages-guapoCh9')
exp_dir = os.path.join(homedir, 'data/with_CNN')
neuron = ('caffe-net', 'fc8', 1)
this_exp_dir = add_neuron_subdir(neuron, exp_dir)
for i in range(5,10):
    trial_title = 'choleskycma_norm_trial%d' % (i)
    # "ActivPattArr_block.npz"
    trial_dir = add_trial_subdir(this_exp_dir, trial_title)
    activfns = sorted([fn for fn in os.listdir(trial_dir) if '.npz' in fn and 'ActivPattArr_block' in fn])
    # generations = []
    image_ids = []

    tmpscoref = np.load(os.path.join(trial_dir, activfns[0]), allow_pickle=True)
    layernames = list(tmpscoref["pattern_dict"].item().keys())
    activation_full = {}
    for key in layernames:
        activation_full[key] = []
        print(key)

    for activfn in activfns:
        geni = re.findall(r"ActivPattArr_block(\d+).npz", activfn)
        geni = int(geni[0])
        scoref = np.load(os.path.join(trial_dir, activfn), allow_pickle=True)
        image_ids.extend(list(scoref['image_ids']))
        activations = scoref["pattern_dict"].item()
        for key in layernames:
            activation_full[key].append(activations[key])
    for key in layernames:
        activation_full[key] = np.concatenate(activation_full[key], axis=0)
        # generations.extend([int(geni[0])] * len(cur_score))
    h = h5py.File(os.path.join(trial_dir, 'Activation_Summary.hdf5'))
    for k, v in activation_full.items():
        h.create_dataset(k, data=np.array(v))
    h.close()
    # for activfn in activfns:
    #     os.remove(os.path.join(trial_dir, activfn))
#%%
# h = h5py.File('Activation_Summary.hdf5')
# for k, v in activation_full.items():
#     h.create_dataset(k, data=np.array(v))
pattern_dict = tmpscoref["pattern_dict"]
#%%
for i in range(1):
    trial_title = 'choleskycma_norm_trial%d' % (i)
    trial_dir = add_trial_subdir(this_exp_dir, trial_title)
    h = h5py.File(os.path.join(trial_dir, 'Activation_Summary.hdf5'))
#%%
act_patt_fc8 = h["fc8"]
act_patt_fc7 = h["fc7"]
#%%

from cv2 import imread, imwrite
import matplotlib.pylab as plt
corr_coef = []
for i in range(act_patt_fc8.shape[1]):
    corr_coef.append(np.corrcoef(act_patt_fc8[:,0], act_patt_fc8[:,i])[0,1])
corr_coef = np.array(corr_coef)
#%%
plt.figure()
plt.scatter(np.arange(1000), corr_coef)
plt.show()