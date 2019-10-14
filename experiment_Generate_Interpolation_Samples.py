# -*- coding: utf-8 -*-
'''
Experimental Code to generate samples in selected PC space from an experiment
depending on `utils` for code loading things
'''
from  scipy.io import loadmat
import os
import numpy as np
from numpy.linalg import norm
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
backup_dir = r"\\storage1.ris.wustl.edu\crponce\Active\Stimuli\2019-06-Evolutions\beto-190625b\backup_06_25_2019_14_56_50"
backup_dir = r"C:\Users\Poncelab-ML2a\Documents\Python\Generator_DB_Windows\shared\backup_10_14_2019_12_51_15"
newimg_dir = os.path.join(backup_dir, "PC_imgs")
#%%
os.makedirs(newimg_dir, exist_ok=True)
print("Save new images to folder %s", newimg_dir)
#%%
print("Loading the codes from experiment folder %s", backup_dir)
codes_all_col, generations_col = utils.load_multithread_codes_mat(backup_dir, thread_num=2)
for thread in range(2):
    print("Shape of code for thread %d" % thread, codes_all_col[thread].shape)
#%%
final_gen_norm1 = norm(codes_all_col[0][generations_col[0] == generations_col[0].max(), :], axis=1)
final_gen_norm2 = norm(codes_all_col[1][generations_col[1] == generations_col[1].max(), :], axis=1)
mean_final_norm = np.mean([final_gen_norm1.mean(), final_gen_norm2.mean()])
#%%
final_gen_vec1 = np.mean(codes_all_col[0][generations_col[0] == generations_col[0].max(), :], axis=0)
final_gen_vec2 = np.mean(codes_all_col[1][generations_col[1] == generations_col[1].max(), :], axis=0)
unit_final_vec1 = final_gen_vec1/norm(final_gen_vec1)
unit_final_vec2 = final_gen_vec2/norm(final_gen_vec2)
ortho_final_vec2 = unit_final_vec2 - np.dot(unit_final_vec1, unit_final_vec2) * unit_final_vec1
ortho_final_vec2 = ortho_final_vec2/norm(ortho_final_vec2)
unit_final_vec1 = unit_final_vec1[np.newaxis, :]
ortho_final_vec2 = ortho_final_vec2[np.newaxis, :]
#%%
codes_all = np.concatenate((codes_all_col[0], codes_all_col[1]), axis=0)
# print("Computing PCs")
# code_pca = PCA(n_components=50)
# code_pca.fit(codes_all)
# PC_Proj_codes1 = code_pca.transform(codes_all_col[0])
# PC_Proj_codes2 = code_pca.transform(codes_all_col[1])
# PC_vectors = code_pca.components_
#%%
ortho_codes_all = codes_all - codes_all @ unit_final_vec1.T @ unit_final_vec1 \
                  - codes_all @ ortho_final_vec2.T @ ortho_final_vec2
#%%
print("Computing PCs of the orthogonal space")
ortho_code_pca = PCA(n_components=50)
ortho_code_pca.fit(ortho_codes_all)
ortho_PC_vectors = ortho_code_pca.components_
# ortho_code_pca.transform()
#%%
plt.figure()
# plt.imshow(generator.visualize((ortho_final_vec2 + ortho_PC_vectors[0:1, :])*400))
plt.imshow(generator.visualize((unit_final_vec2)*400))
plt.axis("off")
plt.show()
#%%
coord_vectors = np.concatenate((unit_final_vec1, ortho_final_vec2, ortho_PC_vectors[0:1, :]), axis=0)
interp_ang_step = 180 / 10
PC3_ang_step = 180 / 10
sphere_norm = mean_final_norm
print("Generating images on interpolating  sphere (rad = %d)" % sphere_norm)
img_list = []
for k in range(-5, 6):
    for j in range(-5, 11):
        if PC3_ang_step * k == -90 and j != 0:
            continue
        if PC3_ang_step * k ==  90 and j != 0:
            continue
        theta = interp_ang_step * j / 180 * np.pi
        phi = PC3_ang_step * k / 180 * np.pi
        code_vec = np.array([[np.cos(theta) * np.cos(phi),
                              np.sin(theta) * np.cos(phi),
                              np.sin(phi)]]) @ coord_vectors
        code_vec = code_vec / np.sqrt((code_vec**2).sum()) * sphere_norm
        img = generator.visualize(code_vec)
        if k == 5 or k ==-5:
            img_list.extend([img.copy()] * 16)
        else:
            img_list.append(img.copy())
        plt.imsave(os.path.join(newimg_dir, "norm_%d_interp_%d_PC3_%d.jpg" % (sphere_norm, interp_ang_step * j, PC3_ang_step* k)), img)

fig1 = utils.visualize_img_list(img_list, ncol=16, nrow=11)  # rows are PC3(PC50,RND_vec2) direction, columns are PC2(PC49 RND_vec1) directions
#%%
coord_vectors = np.concatenate((unit_final_vec1, ortho_final_vec2, ortho_PC_vectors[1:2, :]), axis=0)
interp_ang_step = 180 / 10
PC3_ang_step = 180 / 10
sphere_norm = mean_final_norm
print("Generating images on interpolating  sphere (rad = %d)" % sphere_norm)
img_list = []
for k in range(-5, 6):
    for j in range(-5, 11):
        if PC3_ang_step * k == -90 and j != 0:
            continue
        if PC3_ang_step * k ==  90 and j != 0:
            continue
        theta = interp_ang_step * j / 180 * np.pi
        phi = PC3_ang_step * k / 180 * np.pi
        code_vec = np.array([[np.cos(theta) * np.cos(phi),
                              np.sin(theta) * np.cos(phi),
                              np.sin(phi)]]) @ coord_vectors
        code_vec = code_vec / np.sqrt((code_vec**2).sum()) * sphere_norm
        img = generator.visualize(code_vec)
        if k == 5 or k ==-5:
            img_list.extend([img.copy()] * 16)
        else:
            img_list.append(img.copy())
        plt.imsave(os.path.join(newimg_dir, "norm_%d_interp_%d_PC4_%d.jpg" % (sphere_norm, interp_ang_step * j, PC3_ang_step* k)), img)

fig2 = utils.visualize_img_list(img_list, ncol=16, nrow=11)  # rows are PC3(PC50,RND_vec2) direction, columns are PC2(PC49 RND_vec1) directions

np.savez(os.path.join(newimg_dir, "PC_vector_data.npz"), coord_vectors=coord_vectors, ortho_PC_vectors=ortho_PC_vectors[:2, :],
             sphere_norm=sphere_norm, interp_ang_step=interp_ang_step, PC3_ang_step=PC3_ang_step)