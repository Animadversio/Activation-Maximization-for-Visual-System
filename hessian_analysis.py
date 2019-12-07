
import sys
import os
from os.path import join
from time import time
from importlib import reload
import re
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from cv2 import imread, imwrite
import matplotlib.pylab as plt
sys.path.append("D:\Github\pytorch-caffe")
sys.path.append("D:\Github\pytorch-receptive-field")
from torch_receptive_field import receptive_field, receptive_field_for_unit
from caffenet import *
from hessian import hessian
#%% plot the spectrum of the matrix
output_dir = r"D:\Generator_DB_Windows\data\with_CNN\hessian"
unit_arr = [
            ('caffe-net', 'conv1', 10, 10, 10),
            ('caffe-net', 'conv2', 5, 10, 10),
            ('caffe-net', 'conv3', 5, 10, 10),
            ('caffe-net', 'conv4', 5, 10, 10),
            ('caffe-net', 'conv5', 5, 10, 10),
            ('caffe-net', 'fc6', 1),
            ('caffe-net', 'fc6', 2),
            ('caffe-net', 'fc6', 3),
            ('caffe-net', 'fc7', 1),
            ('caffe-net', 'fc7', 2),
            ('caffe-net', 'fc8', 1),
            ('caffe-net', 'fc8', 10),
            ]
plt.figure()
for unit in unit_arr:
    # unit = unit_arr[1]
    data = np.load(join(output_dir, "hessian_result_%s_%d.npz" % (unit[1], unit[2])))
    pos1_nums = (data["heig"] > 1).sum()
    pos_nums = (data["heig"] > 0.1).sum()
    num01 = (np.logical_and(data["heig"] < 0.1, data["heig"] > -0.1)).sum()
    num001 = (np.logical_and(data["heig"] < 0.01, data["heig"] > -0.01)).sum()
    num0001 = (np.logical_and(data["heig"] < 0.001, data["heig"] > -0.001)).sum()
    print("%s [1, inf]:%d, [0.1, inf]:%d, [-0.1,0.1]: %d; [-0.01,0.00]: %d; [-0.001,0.001]: %d; " % (unit, pos1_nums, pos_nums, num01, num001, num0001))
    plt.plot(data["heig"][:100], label="%s-%s" % (unit[0], unit[1]), alpha=0.5, lw=2)

plt.legend()
plt.show()
#%%
def vec_cos(v1, v2):
    return np.vdot(v1,v2)/np.linalg.norm(v1)/np.linalg.norm(v2)
#%%
output_dir = r"D:\Generator_DB_Windows\data\with_CNN\hessian"
unit_arr = [
            ('caffe-net', 'conv1', 10, 10, 10),
            ('caffe-net', 'conv2', 5, 10, 10),
            ('caffe-net', 'conv3', 5, 10, 10),
            ('caffe-net', 'conv4', 5, 10, 10),
            ('caffe-net', 'conv5', 5, 10, 10),
            ('caffe-net', 'fc6', 1),
            ('caffe-net', 'fc6', 2),
            ('caffe-net', 'fc6', 3),
            ('caffe-net', 'fc7', 1),
            ('caffe-net', 'fc7', 2),
            ('caffe-net', 'fc8', 1),
            ('caffe-net', 'fc8', 10),
            ]
for unit in unit_arr[-2:-1]:
    data = np.load(join(output_dir, "hessian_result_%s_%d.npz" % (unit[1], unit[2])))
    z = data["z"]
    G = data["grad"]
    Heig = data["heig"]
    Heigvec = data["heigvec"]
    vec_cos(z, Heigvec[:, 1]), vec_cos(z, Heigvec[:, 2])
    # plt.plot(data["heig"][:100], label="%s-%s" % (unit[0], unit[1]), alpha=0.5, lw=2)

# z=feat.detach().numpy(),
# activation=-neg_activ.detach().numpy(),
# grad=gradient.numpy(),H=H.detach().numpy(),
# heig=eigval,heigvec=eigvec
#%%
import utils
from utils import generator
#%%
from insilico_Exp import CNNmodel
unit = ('caffe-net', 'fc8', 1)
CNNmodel = CNNmodel(unit[0])  # 'caffe-net'
CNNmodel.select_unit(unit)
#%%
def perturb_images_sphere(cent_vec, perturb_vec, PC2_ang_step = 18, PC3_ang_step = 18):
    sphere_norm = np.linalg.norm(cent_vec)
    vectors = np.zeros((3, cent_vec.size))
    vectors[  0, :] = cent_vec / sphere_norm
    vectors[1:3, :] = perturb_vec
    img_list = []
    for j in range(-5, 6):
        for k in range(-5, 6):
            theta = PC2_ang_step * j / 180 * np.pi
            phi = PC3_ang_step * k / 180 * np.pi
            code_vec = np.array([[np.cos(theta) * np.cos(phi),
                                  np.sin(theta) * np.cos(phi),
                                  np.sin(phi)]]) @ vectors[0:3, :]
            code_vec = code_vec / np.sqrt((code_vec ** 2).sum()) * sphere_norm
            img = generator.visualize(code_vec)
            img_list.append(img.copy())
    return img_list

def perturb_images_arc(cent_vec, perturb_vec, PC2_ang_step = 18):
    sphere_norm = np.linalg.norm(cent_vec)
    vectors = np.zeros((2, cent_vec.size))
    vectors[  0, :] = cent_vec / sphere_norm
    vectors[1:2, :] = perturb_vec
    img_list = []
    for j in range(-5, 6):
        theta = PC2_ang_step * j / 180 * np.pi
        code_vec = np.array([[np.cos(theta),
                              np.sin(theta)]]) @ vectors[0:2, :]
        code_vec = code_vec / np.sqrt((code_vec ** 2).sum()) * sphere_norm
        img = generator.visualize(code_vec)
        img_list.append(img.copy())
    return img_list

def perturb_images_line(cent_vec, perturb_vec, PC2_step = 18):
    sphere_norm = np.linalg.norm(cent_vec)
    img_list = []
    for j in range(-5, 6):
        L = PC2_step * j
        code_vec = cent_vec + L * perturb_vec
        # code_vec = code_vec / np.sqrt((code_vec ** 2).sum()) * sphere_norm
        img = generator.visualize(code_vec)
        img_list.append(img.copy())
    return img_list

#%%
img_list = perturb_images_line(z, Heigvec[:,1000])
figh = utils.visualize_img_list(img_list, nrow=1)
#%%
img_list = perturb_images_line(z, Heigvec[:, 1] * 1)
scores = CNNmodel.score(img_list)
figh = utils.visualize_img_list(img_list, nrow=1, scores=scores)
# Note the maximum found by exact gradient descent has smaller norm than those found by CMA-ES optimization.
