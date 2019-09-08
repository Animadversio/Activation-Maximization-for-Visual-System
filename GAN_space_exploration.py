from time import time, sleep
import os
import numpy as np
from  scipy.io import loadmat
from shutil import copy, copyfile
from utils import generator, add_trial_subdir
# from Optimizer import CMAES, Genetic, CholeskyCMAES
from cv2 import imread, imwrite
import matplotlib.pylab as plt
import re
#%%
matdata = loadmat(r"C:\Users\ponce\OneDrive\Documents\MATLAB\CodesEvolution1.mat")
codes_all = matdata['codes_all']
generations = matdata['generations']
del matdata
#%% Cropping effect
imgs = generator.visualize(codes_all[1000, :])
plt.figure()
plt.hist(np.reshape(imgs, (-1, 1)), 255, (0, 255))
plt.xlim((0, 255))
plt.show()

#%%
plt.imshow(imgs)
plt.show()
#%%
raw_img = generator.raw_output(0.1 * codes_all[4000, :])
deproc_raw_img = generator._detransformer.deprocess('data', raw_img)
plt.hist(deproc_raw_img.flatten(), 255)  # (0, 255))
#plt.xlim((0,255))
plt.show()
#%%
code_id = 4000
mult1 = 1 ; mult2 = 10 ;
code = codes_all[code_id, :]
raw_img = generator.raw_output(mult1 * code).copy()
deproc_raw_img = generator._detransformer.deprocess('data', raw_img)
raw_img2 = generator.raw_output(mult2 * code).copy()
deproc_raw_img2 = generator._detransformer.deprocess('data', raw_img2)
img1 = generator.visualize(mult1 * code)
img2 = generator.visualize(mult2 * code)
#%%
plt.figure(figsize=[10, 10])
plt.subplot(3,2,1)
plt.scatter(deproc_raw_img.flatten(), deproc_raw_img2.flatten())
plt.title("Deprocessed raw image 1 vs 2")
plt.subplot(3,2,2)
plt.scatter(raw_img.flatten(), raw_img2.flatten())
plt.title("Raw image 1 vs 2")
plt.subplot(3,2,3)
plt.hist(deproc_raw_img.flatten(), 255)
plt.title("Deprocessed raw image 1")
plt.subplot(3,2,5)
plt.hist(deproc_raw_img2.flatten(), 255)
plt.title("Deprocessed raw image 2")
plt.subplot(3,2,4)
plt.hist(raw_img.flatten(), 255)
plt.title("Raw image 1")
plt.subplot(3,2,6)
plt.hist(raw_img2.flatten(), 255)
plt.title("Raw image 2")
plt.suptitle("Code %d by multiplier 1: %.1f and 2: %.1f" % (code_id, mult1, mult2))
plt.show()
#%%
plt.figure(figsize=[10, 10])
plt.subplot(3,3,3)
plt.title("Code_id %d \n multiplied by  1: %.1f and 2: %.1f" % (code_id, mult1, mult2))
plt.axis('off')
plt.subplot(3,3,1)
plt.scatter(deproc_raw_img.flatten(), deproc_raw_img2.flatten())
plt.title("Deprocessed raw image 1 vs 2")
plt.subplot(3,3,2)
plt.scatter(raw_img.flatten(), raw_img2.flatten())
plt.title("Raw image 1 vs 2")
plt.subplot(3,3,4)
plt.hist(deproc_raw_img.flatten(), 255)
plt.title("Deprocessed raw image 1")
plt.subplot(3,3,7)
plt.hist(deproc_raw_img2.flatten(), 255)
plt.title("Deprocessed raw image 2")
plt.subplot(3,3,5)
plt.hist(raw_img.flatten(), 255)
plt.title("Raw image 1")
plt.subplot(3,3,8)
plt.hist(raw_img2.flatten(), 255)
plt.title("Raw image 2")
plt.subplot(3,3,6)
plt.imshow(img1)
plt.xticks([])
plt.yticks([])
plt.subplot(3,3,9)
plt.imshow(img2)
plt.xticks([])
plt.yticks([])
plt.show()
#%%
imgs = generator.visualize(0.01*codes_all[4000, :])
plt.imshow(imgs)
plt.show()
#%%
plt.scatter(codes_all[3500, :], codes_all[4000, :])
plt.show()
#%%
plt.figure(figsize=[8, 10])
plt.pcolor(codes_all.transpose())
plt.xlabel("Image id by generation")
plt.ylabel("code_entry")
plt.show()

#%%
exp_dir = r"D:\Generator_DB_Windows\data\with_CNN"
this_exp_dir = os.path.join(exp_dir, "purenoise")
trial_title = 'choleskycma_sgm3_uf10_cc%.2f_cs%.2f' % (0.00097, 0.0499)
trialdir = add_trial_subdir(this_exp_dir, trial_title)

#%%
def scores_imgname_summary(trialdir, savefile=True):
    """ """
    if "scores_all.npz" in os.listdir(trialdir):
        # if the summary table exist, just read from it!
        with np.load(os.path.join(trialdir, "scores_all.npz")) as data:
            scores = data["scores"]
            generations = data["generations"]
            image_ids = data["image_ids"]
        return scores, image_ids, generations

    scorefns = sorted([fn for fn in os.listdir(trialdir) if '.npz' in fn and 'scores_end_block' in fn])
    scores = []
    generations = []
    image_ids = []
    for scorefn in scorefns:
        geni = re.findall(r"scores_end_block(\d+).npz", scorefn)
        scoref = np.load(os.path.join(trialdir, scorefn), allow_pickle=False)
        cur_score = scoref['scores']
        scores.append(cur_score)
        image_ids.extend(list(scoref['image_ids']))
        generations.extend([int(geni[0])] * len(cur_score))
    scores = np.array(scores)
    generations = np.array(generations)
    if savefile:
        np.savez(os.path.join(trialdir, "scores_all.npz"), scores=scores, generations=generations, image_ids=image_ids)
    return scores, image_ids, generations


#%%
import utils
codes_all, generations = utils.codes_summary(trialdir)
