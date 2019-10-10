'''
Experimental Code to generate samples in selected PC space from an experiment
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
backup_dir = r"\\storage1.ris.wustl.edu\crponce\Active\backup_10_09_2019_13_49_17"
#r"\\storage1.ris.wustl.edu\crponce\Active\Stimuli\2019-06-Evolutions\beto-190925a\backup_09_25_2019_16_05_18"
newimg_dir = os.path.join(backup_dir, "PC_imgs")
#%%
os.makedirs(newimg_dir, exist_ok=True)
print("Save new images to folder %s", newimg_dir)
#%%
print("Loading the codes from experiment folder %s", backup_dir)
codes_all, generations = utils.load_codes_mat(backup_dir)
print("Shape of code", codes_all.shape)
generations = np.array(generations)
#%%
code_last_gen = codes_all[generations==generations.max(), :]
norm_last_gen = np.sqrt(np.sum(code_last_gen**2, axis=1))
sphere_norm = norm_last_gen.mean()
#%%
import math, random

def fibonacci_sphere(samples=121, randomize=False):
    rnd = 1.
    if randomize:
        rnd = random.random() * samples

    points = []
    offset = 2./samples
    increment = math.pi * (3. - math.sqrt(5.))

    for i in range(samples):
        y = ((i * offset) - 1) + (offset / 2)
        r = math.sqrt(1 - pow(y,2))

        phi = ((i + rnd) % samples) * increment

        x = math.cos(phi) * r
        z = math.sin(phi) * r

        points.append([x,y,z])

    return points
points = fibonacci_sphere()
#%%
import mpl_toolkits.mplot3d
points = np.array(points)
plt.figure().add_subplot(111, projection='3d').scatter(points[:, 0], points[:, 1], points[:, 2]);
plt.show()
#%%
print("Computing PCs")
code_pca = PCA(n_components=50)
PC_Proj_codes = code_pca.fit_transform(codes_all)
PC_vectors = code_pca.components_
if PC_Proj_codes[-1, 0] < 0:  # decide which is the positive direction for PC1
    inv_PC1 = True
    PC1_sign = -1
else:
    inv_PC1 = False
    PC1_sign = 1
# %% Spherical interpolation
print("Generating images on PC1, PC2, PC3 sphere (rad = %d)" % sphere_norm)
img_list = []
for i, point in enumerate(points):
    point = np.array(point)[np.newaxis, :]
    code_vec = point @ PC_vectors[0:3, :]
    code_vec = code_vec * sphere_norm
    img = generator.visualize(code_vec)
    img_list.append(img.copy())
    plt.imsave(
        os.path.join(newimg_dir, "norm_%d_PC1_%.2f_PC2_%.2f_PC3_%.2f.jpg" % (sphere_norm, point[0,0], point[0,1], point[0,2])), img)
fig1 = utils.visualize_img_list(img_list)  # rows are PC3(PC50,RND_vec2) direction, columns are PC2(PC49 RND_vec1) directions
# %% Spherical interpolation
PC2_ang_step = 180 / 10
PC3_ang_step = 180 / 10
sphere_norm = 300
print("Generating images on PC1, PC49, PC50 sphere (rad = %d)" % sphere_norm)
PC_nums = [0, 48, 49]  # can tune here to change the selected PC to generate images
img_list = []
for j in range(-5, 6):
    for k in range(-5, 6):
        theta = PC2_ang_step * j / 180 * np.pi
        phi = PC3_ang_step * k / 180 * np.pi
        code_vec = np.array([[PC1_sign* np.cos(theta) * np.cos(phi),
                                        np.sin(theta) * np.cos(phi),
                                        np.sin(phi)]]) @ PC_vectors[PC_nums, :]
        code_vec = code_vec / np.sqrt((code_vec**2).sum()) * sphere_norm
        img = generator.visualize(code_vec)
        img_list.append(img.copy())
        plt.imsave(os.path.join(newimg_dir, "norm_%d_PC%d_%d_PC%d_%d.jpg" % (sphere_norm,
                                                                            PC_nums[1] + 1, PC2_ang_step * j,
                                                                            PC_nums[2] + 1, PC3_ang_step * k)), img)
fig2 = utils.visualize_img_list(img_list) # rows are PC3(PC50,RND_vec2) direction, columns are PC2(PC49 RND_vec1) directions

# %% Spherical interpolation
PC2_ang_step = 180 / 10
PC3_ang_step = 180 / 10
sphere_norm = 300
print("Generating images on PC1, Random vector1, Random vector2 sphere (rad = %d)" % sphere_norm)
# Random select and orthogonalize the vectors to form the sphere
rand_vec2 = np.random.randn(2, 4096)
rand_vec2 = rand_vec2 - (rand_vec2 @ PC_vectors.T) @ PC_vectors
rand_vec2 = rand_vec2 / np.sqrt((rand_vec2**2).sum(axis=1))[:, np.newaxis]
vectors = np.concatenate((PC_vectors[0:1, :], rand_vec2))
img_list = []
for j in range(-5, 6):
    for k in range(-5, 6):
        theta = PC2_ang_step * j / 180 * np.pi
        phi = PC3_ang_step * k / 180 * np.pi
        code_vec = np.array([[PC1_sign* np.cos(theta) * np.cos(phi),
                                        np.sin(theta) * np.cos(phi),
                                        np.sin(phi)]]) @ vectors
        code_vec = code_vec / np.sqrt((code_vec**2).sum()) * sphere_norm
        img = generator.visualize(code_vec)
        img_list.append(img.copy())
        plt.imsave(os.path.join(newimg_dir, "norm_%d_RND1_%d_RND2_%d.jpg" % (sphere_norm, PC2_ang_step * j, PC3_ang_step * k)), img)
fig3 = utils.visualize_img_list(img_list)  # rows are PC3(PC50,RND_vec2) direction, columns are PC2(PC49 RND_vec1) directions

np.savez(os.path.join(newimg_dir, "PC_vector_data.npz"), PC_vecs=PC_vectors, rand_vec2=rand_vec2,
         sphere_norm=sphere_norm, PC2_ang_step=PC2_ang_step, PC3_ang_step=PC3_ang_step)
