'''
Demo Code to generate samples in selected PC space from an experiment
depending on `utils` for code loading things
'''
from  scipy.io import loadmat
import os
import numpy as np
import re
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
import matplotlib.pylab as plt
from utils import generator
import utils
#%%
'''
Input the experimental backup folder containing the mat codes files. 
'''
backup_dir = r"\\storage1.ris.wustl.edu\crponce\Active\Stimuli\2019-06-Evolutions\beto-190909b\backup_09_09_2019_13_50_18"
newimg_dir = r"\\storage1.ris.wustl.edu\crponce\Active\Stimuli\2019-06-Evolutions\beto-190909b\backup_09_09_2019_13_50_18\PC_imgs"
#%%
os.makedirs(newimg_dir, exist_ok=True)
#%%
codes_all, generations = utils.load_codes_mat(backup_dir)
#%%
code_pca = PCA(n_components=50)
PC_Proj_codes = code_pca.fit_transform(codes_all)
PC_vectors = code_pca.components_
if PC_Proj_codes[-1, 0] < 0:
    inv_PC1 = True
    PC1_sign = -1
else:
    inv_PC1 = False
    PC1_sign = 1
# %% Spherical interpolation
# PC1_step = PC1_Amp / 10  # TODO: Control the step size and range of the images.
PC2_ang_step = 180 / 10
PC3_ang_step = 180 / 10
sphere_norm = 200

img_list = []
for j in range(-5, 6):
    for k in range(-5, 6):
        theta = PC2_ang_step * j / 180 * np.pi
        phi = PC3_ang_step * k / 180 * np.pi
        code_vec = np.array([[PC1_sign* np.cos(theta) * np.cos(phi),
                                        np.sin(theta) * np.cos(phi),
                                        np.sin(phi)]]) @ PC_vectors[0:3, :]
        code_vec = code_vec / np.sqrt((code_vec**2).sum()) * sphere_norm
        img = generator.visualize(code_vec)
        img_list.append(img.copy())
        # plt.imsave(os.path.join(newimg_dir, "PC1_%d_PC2_%d_PC3_%d.jpg" % (i, j, k)), img)

plt.figure(figsize=[30, 30])
for i, img in enumerate(img_list):
    plt.subplot(11, 11, i + 1)
    plt.imshow(img[:])
    plt.axis('off')
plt.show()





#%%
PC1_Amp = PC_Proj_codes[-1, 0] - PC_Proj_codes[0, 0] # can be negative
PC2_Amp = PC_Proj_codes[:, 1].max() - PC_Proj_codes[:, 1].min()
PC3_Amp = PC_Proj_codes[:, 2].max() - PC_Proj_codes[:, 2].min()

PC1_step = PC1_Amp / 10  # TODO: Control the step size and range of the images.
PC2_step = max(PC2_Amp, 100) / 10
PC3_step = max(PC3_Amp, 100) / 10
for i in range(5, 11):
    for j in range(-5, 6):
        for k in range(-5, 6):
            img = generator.visualize(np.array([[PC1_step * i, PC2_step * j, PC3_step * k]]) @ PC_vectors[:3, :])
            plt.imsave(os.path.join(newimg_dir, "PC1_%d_PC2_%d_PC3_%d.jpg" % (i, j, k)), img)
np.savez(os.path.join(newimg_dir, "PC_data.npz"), PC_vecs = PC_vectors[:3, :], PC1_step=PC1_step, PC2_step=PC2_step, PC3_step=PC3_step)
# %% Spherical interpolation
newimg_dir = os.path.join(backup_dir, "PC_sphere_imgs")
os.makedirs(newimg_dir, exist_ok=True)
PC1_step = PC1_Amp / 10  # TODO: Control the step size and range of the images.
PC2_step = max(PC2_Amp, 400) / 10
PC3_step = max(PC3_Amp, 400) / 10
sphere_norm = 200
i = 6
img_list = []
for j in range(-5, 6):
    for k in range(-5, 6):
        code_vec = np.array([[PC1_step * i, PC2_step * j, PC3_step * k]]) @ PC_vectors[:3, :]
        code_vec = code_vec / np.sqrt((code_vec**2).sum()) * sphere_norm
        img = generator.visualize(code_vec)
        plt.imsave(os.path.join(newimg_dir, "PC1_%d_PC2_%d_PC3_%d.jpg" % (i, j, k)), img)

#%% Try out different step size and the difference by eye.
import  matplotlib.pylab as plt
PC1_step = PC1_Amp / 10  # TODO: Control the step size and range of the images.
PC2_step = max(PC2_Amp, 300) / 10
PC3_step = max(PC3_Amp, 300) / 10
sphere_norm = 200
i = 6
img_list = []
for j in range(-5, 6):
    for k in range(-5, 6):
        code_vec = np.array([[PC1_step * i, PC2_step * j, PC3_step * k]]) @ PC_vectors[:3, :]
        code_vec / np.sqrt((code_vec**2).sum()) * sphere_norm  # Spherical normalization
        img = generator.visualize(np.array([[PC1_step * i, PC2_step * j, PC3_step * k]]) @ PC_vectors[:3, :])
        img_list.append(img.copy())
#%%
plt.figure(figsize=[30,30])
for i, img in enumerate(img_list):
    plt.subplot(11, 11, i+1)
    plt.imshow(img[:])
    plt.axis('off')
plt.show()
# plt.suptitle("Image Samples from %d Perturbations in Evolution %d\n %s space"%(magnitude, experiment_i, metric_space))


#%% Utility functions
def GAN_interp_sphere_ang(vectors, sphere_norm=200, theta_ang_step= 180/10, phi_ang_step=180/10, grid_shape=(11,11),
                          saveimg=False, savepath=None, inv_PC1 = True):
    img_list = []
    theta_n, phi_n = grid_shape
    for j in range(-int((theta_n-1)/2), int((theta_n+1)/2)):
        for k in range(-int((phi_n-1)/2), int((phi_n+1)/2)):
            theta = theta_ang_step * j / 180 * np.pi
            phi = phi_ang_step * k / 180 *np.pi
            if inv_PC1:
                code_vec = np.array([[-np.cos(theta) * np.cos(phi),
                                      np.sin(theta) * np.cos(phi),
                                      np.sin(phi)]]) @ vectors
            else:
                code_vec = np.array([[np.cos(theta) * np.cos(phi),
                                      np.sin(theta) * np.cos(phi),
                                      np.sin(phi)]]) @ vectors
            code_vec = code_vec / np.sqrt((code_vec**2).sum()) * sphere_norm
            img = generator.visualize(code_vec)
            img_list.append(img.copy())
            if saveimg:
                plt.imsave(os.path.join(savepath, "norm%d_theta_%d_phi_%d.jpg" % (sphere_norm, j, k)), img)
    return img_list


def visualize_img_list(img_list, scores=None, ncol=11, nrow=11, title_cmap=plt.cm.viridis):
    if scores is not None and not title_cmap == None:
        cmap_flag = True
        ub = scores.max()
        lb = scores.min()
    assert len(img_list) <= ncol * nrow
    figW = 30
    figH = figW / ncol * nrow + 1
    fig = plt.figure(figsize=[figW, figH])
    for i, img in enumerate(img_list):
        plt.subplot(ncol, nrow, i + 1)
        plt.imshow(img[:])
        plt.axis('off')
        if cmap_flag:  # color the titles with a heatmap!
            plt.title("{0:.2f}".format(scores[i]), fontsize=16,
                      color=title_cmap((scores[i] - lb) / (ub - lb)))  # normalize a value between [0,1]
        elif scores != None:
            plt.title("{0:.2f}".format(scores[i]), fontsize=16)
        else:
            pass
    plt.show()
    return fig
