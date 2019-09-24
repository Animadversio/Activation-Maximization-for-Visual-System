from  scipy.io import loadmat
import os
import numpy as np
import re
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
import  matplotlib.pylab as plt
from utils import generator
#%%
backup_dir=r"\\storage1.ris.wustl.edu\crponce\Active\Stimuli\2019-06-Evolutions\beto-190909b\backup_09_09_2019_13_50_18"
newimg_dir = r"\\storage1.ris.wustl.edu\crponce\Active\Stimuli\2019-06-Evolutions\beto-190909b\backup_09_09_2019_13_50_18\PC_imgs"

os.makedirs(newimg_dir, exist_ok=True)
codes_fns = sorted([fn for fn in os.listdir(backup_dir) if "_code.mat" in fn])
#%%
codes_all = []
img_ids = []
for i, fn in enumerate(codes_fns[:]):
    matdata = loadmat(os.path.join(backup_dir, fn))
    codes_all.append(matdata["codes"])
    img_ids.extend(list(matdata["ids"]))

#%%
codes_all = np.concatenate(tuple(codes_all), axis=0)
img_ids = np.concatenate(tuple(img_ids), axis=0)
img_ids = [img_ids[i][0] for i in range(len(img_ids))]
generations = [int(re.findall("gen(\d+)", img_id)[0])  if 'gen' in img_id else -1 for img_id in img_ids]
#%%
code_pca = PCA(n_components=50)
PC_Proj_codes = code_pca.fit_transform(codes_all)
PC_vectors = code_pca.components_
#%%
PC1_Amp = PC_Proj_codes[-1, 0] - PC_Proj_codes[0, 0] # can be negative
PC2_Amp = PC_Proj_codes[:, 1].max() - PC_Proj_codes[:, 1].min()
PC3_Amp = PC_Proj_codes[:, 2].max() - PC_Proj_codes[:, 2].min()
# if not os.path.exists(newimg_dir):
# generator()
PC1_step = PC1_Amp / 10  # TODO: Control the step size and range of the images.
PC2_step = max(PC2_Amp, 100) / 10
PC3_step = max(PC3_Amp, 100) / 10
for i in range(5, 11):
    for j in range(-5, 6):
        for k in range(-5, 6):
            img = generator.visualize(np.array([[PC1_step * i, PC2_step * j, PC3_step * k]]) @ PC_vectors[:3, :])
            plt.imsave(os.path.join(newimg_dir, "PC1_%d_PC2_%d_PC3_%d.jpg" % (i, j, k)), img)
np.savez(os.path.join(newimg_dir, "PC_data.npz"), PC_vecs = PC_vectors[:3, :], PC1_step=PC1_step, PC2_step=PC2_step, PC3_step=PC3_step)
# %%
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

plt.figure(figsize=[30,30])
for i, img in enumerate(img_list):
    plt.subplot(11, 11, i+1)
    plt.imshow(img[:])
    plt.axis('off')
plt.show()
# plt.suptitle("Image Samples from %d Perturbations in Evolution %d\n %s space"%(magnitude, experiment_i, metric_space))
