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
from caffenet import *
from hessian import hessian
print(torch.cuda.current_device())
# print(torch.cuda.device(0))
if torch.cuda.is_available():
    print(torch.cuda.device_count(), " GPU is available:", torch.cuda.get_device_name(0))

output_dir = join(r"D:\Generator_DB_Windows\data\with_CNN", "hessian")
os.makedirs(output_dir,exist_ok=True)
#%%
basedir = r"D:\Generator_DB_Windows\nets"
protofile = os.path.join(basedir, r"caffenet\caffenet.prototxt") # 'resnet50/deploy.prototxt'
weightfile = os.path.join(basedir, 'bvlc_reference_caffenet.caffemodel') # 'resnet50/resnet50.caffemodel'
save_path = os.path.join(basedir, r"caffenet\caffenet_state_dict.pt")
net = CaffeNet(protofile)
print(net)
if os.path.exists(save_path):
    net.load_state_dict(torch.load(save_path))
else:
    net.load_weights(weightfile)
    torch.save(net.state_dict(), save_path)
net.eval()

basedir = r"D:/Generator_DB_Windows/nets"
save_path = os.path.join(basedir, r"upconv/fc6/generator_state_dict.pt")
protofile = os.path.join(basedir, r"upconv/fc6/generator.prototxt") # 'resnet50/deploy.prototxt'
weightfile = os.path.join(basedir, r'upconv/fc6/generator.caffemodel') # 'resnet50/resnet50.caffemodel'
Generator = CaffeNet(protofile)
print(Generator)
if os.path.exists(save_path):
    Generator.load_state_dict(torch.load(save_path))
else:
    Generator.load_weights(weightfile)
    Generator.save(Generator.state_dict(), save_path)
Generator.eval()

net.verbose = False
Generator.verbose = False
net.requires_grad_(requires_grad=False)
Generator.requires_grad_(requires_grad=False)
for param in net.parameters():
    param.requires_grad = False
for param in Generator.parameters():
    param.requires_grad = False

import net_utils
detfmr = net_utils.get_detransformer(net_utils.load('generator'))
tfmr = net_utils.get_transformer(net_utils.load('caffe-net'))
#%%
unit_arr = [('caffe-net', 'fc6', 1),
            ('caffe-net', 'fc7', 1),
            ('caffe-net', 'fc8', 1),
        ('caffe-net', 'conv1', 5, 10, 10),
        ('caffe-net', 'conv2', 5, 10, 10),
            ('caffe-net', 'conv4', 5, 10, 10),
            ('caffe-net', 'conv3', 5, 10, 10),
            ('caffe-net', 'conv5', 5, 10, 10),
            ]
unit = unit_arr[7]
#%%
# def display_image(ax, out_img):
#     deproc_img = detfmr.deprocess('data', out_img.data.numpy())
#     ax.imshow(np.clip(deproc_img, 0, 1))
for unit in unit_arr:
    print(unit)
    feat = 0.05 * np.random.rand(1, 4096)
    feat = torch.from_numpy(np.float32(feat))
    feat = Variable(feat, requires_grad=True)
    offset = 16
    pipe_optimizer = optim.SGD([feat], lr=0.05)  # Seems Adam is not so good, Adagrad ... is not so
    score = []
    feat_norm = []
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111)
    ax.set_axis_off()
    for step in range(200):
        if step % 5 == 0:
            pipe_optimizer.zero_grad()
        blobs = Generator(feat)  # forward the feature vector through the GAN
        out_img = blobs['deconv0']  # get raw output image from GAN
        resz_out_img = F.interpolate(out_img, (224, 224), mode='bilinear', align_corners=True)
        blobs_CNN = net(resz_out_img)
        if len(unit) == 5:
            neg_activ = - blobs_CNN[unit[1]][0, unit[2], unit[3], unit[4]]
        elif len(unit) == 3:
            neg_activ = - blobs_CNN[unit[1]][0, unit[2]]
        else:
            neg_activ = - blobs_CNN['fc8'][0, 1]
        neg_activ.backward()
        pipe_optimizer.step()
        score.append(- neg_activ.data.item())
        feat_norm.append(feat.norm(p=2).data.item())
        if step % 10 == 0:
            # display_image(ax, out_img)
            # display.clear_output(wait=True)
            # display.display(fig)
            print("%d steps, Neuron activation %.3f" % (step, - neg_activ.data.item()))
    deproc_img = detfmr.deprocess('data', out_img.data.numpy())
    plt.figure(figsize=[6, 6])
    plt.imshow(np.clip(deproc_img, 0, 1))# .view([224,224,3])
    plt.axis('off')
    plt.savefig(join(output_dir, "%s_%s_%d_final_image.png"%(unit[0], unit[1], unit[2])))
    plt.show()

    plt.figure(figsize=[12, 4])
    plt.subplot(1, 2, 1)
    plt.plot(np.arange(len(score)), score)
    plt.title("Score Trajectory")
    plt.subplot(1, 2, 2)
    plt.plot(np.arange(len(score)), feat_norm)
    plt.title("Norm Trajectory")
    plt.savefig(join(output_dir, "%s_%s_%d_trajectory.png"%(unit[0], unit[1], unit[2])))
    plt.show()

    t0 = time()
    blobs = Generator(feat) # forward the feature vector through the GAN
    out_img = blobs['deconv0'] # get raw output image from GAN
    resz_out_img = F.interpolate(out_img, (224, 224), mode='bilinear', align_corners=True) # Differentiable resizing
    blobs_CNN = net(resz_out_img)
    if len(unit) == 5:
        neg_activ = - blobs_CNN[unit[1]][0, unit[2], unit[3], unit[4]]
    elif len(unit) == 3:
        neg_activ = - blobs_CNN[unit[1]][0, unit[2]]
    else:
        neg_activ = - blobs_CNN['fc8'][0, 1]
    gradient = torch.autograd.grad(neg_activ,feat,retain_graph=True)[0] # First order gradient
    H = hessian(neg_activ,feat, create_graph=False) # Second order gradient
    t1 = time()
    print(t1-t0, " sec, computing Hessian") # Each Calculation may take 1050s esp for deep layer in the network!
    eigval, eigvec = np.linalg.eigh(H.detach().numpy()) # eigen decomposition for a symmetric array! ~ 5.7 s
    g = gradient.numpy()
    g = np.sort(g)
    t2 = time()
    print(t2-t1, " sec, eigen factorizing hessian")
    np.savez(join(output_dir, "hessian_result_%s_%d.npz"%(unit[1], unit[2])),
             z=feat.detach().numpy(),
             activation=-neg_activ.detach().numpy(),
             grad=gradient.numpy(),H=H.detach().numpy(),
             heig=eigval,heigvec=eigvec)
    plt.figure(figsize=[12,6])
    plt.subplot(211)
    plt.plot(g[0,::-1])
    plt.xticks(np.arange(0,4200,200))
    plt.ylabel("Gradient")
    plt.subplot(212)
    plt.plot(eigval[::-1])
    plt.xticks(np.arange(0,4200,200))
    plt.ylabel("EigenVal of Hessian")
    plt.suptitle("Gradient and Hessian Spectrum of Hotspot")
    plt.savefig(join(output_dir, "%s_%s_%d_hessian_eig.png"%(unit[0], unit[1], unit[2])))
    plt.show()