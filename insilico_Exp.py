"""Supporting classes and experimental code for in-silico experiment"""
# Manifold_experiment
import utils
import net_utils
from utils import generator
from time import time
import numpy as np
from Optimizer import CholeskyCMAES
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import sys
import os
from os.path import join
from sys import platform
#%%
if platform == "linux":
    recorddir = "/scratch/binxu/CNN_data/"
else:
    if os.environ['COMPUTERNAME'] == 'DESKTOP-9DDE2RH':  # PonceLab-Desktop 3
        recorddir = r"D:\Generator_DB_Windows\data\with_CNN"
    elif os.environ['COMPUTERNAME'] == 'DESKTOP-MENSD6S':  ## Home_WorkStation
        recorddir = r"D:\Monkey_Data\Generator_DB_Windows\data\with_CNN"

code_length = 4096
init_sigma = 3
Aupdate_freq = 10
#%% Simplified in silico experiment modules
class CNNmodel:
    def __init__(self, model_name):
        self._classifier = net_utils.load(model_name)
        self._transformer = net_utils.get_transformer(self._classifier, scale=1)
        self.artiphys = False

    def select_unit(self, unit_tuple):
        self._classifier_name = str(unit_tuple[0])
        self._net_layer = str(unit_tuple[1])
        # `self._net_layer` is used to determine which layer to stop forwarding
        self._net_iunit = int(unit_tuple[2])
        # this index is used to extract the scalar response `self._net_iunit`
        if len(unit_tuple) == 5:
            self._net_unit_x = int(unit_tuple[3])
            self._net_unit_y = int(unit_tuple[4])
        else:
            self._net_unit_x = None
            self._net_unit_y = None

    def set_recording(self, record_layers):
        self.artiphys = True  # flag to record the neural activity in one layer
        self.record_layers = record_layers
        self.recordings = {}
        for layername in record_layers:  # will be arranged in a dict of lists
            self.recordings[layername] = []

    # def forward(self, imgs):
    #     return recordings

    def score(self, images):
        scores = np.zeros(len(images))
        for i, img in enumerate(images):
            # Note: now only support single repetition
            tim = self._transformer.preprocess('data', img)  # shape=(3, 227, 227)
            self._classifier.blobs['data'].data[...] = tim
            self._classifier.forward(end=self._net_layer)  # propagate the image the target layer
            # record only the neuron intended
            score = self._classifier.blobs[self._net_layer].data[0, self._net_iunit]
            if self._net_unit_x is not None:
                # if `self._net_unit_x/y` (inside dimension) are provided, then use them to slice the output score
                score = score[self._net_unit_x, self._net_unit_y]
            scores[i] = score
            if self.artiphys:  # record the whole layer's activation
                for layername in self.record_layers:
                    score_full = self._classifier.blobs[layername].data[0, :]
                    # self._pattern_array.append(score_full)
                    self.recordings[layername].append(score_full.copy())
        if self.artiphys:
            return scores, self.recordings
        else:
            return scores

def render(codes):
    '''Render a list of codes to list of images'''
    if type(codes) is list:
        images = [generator.visualize(codes[i]) for i in range(len(codes))]
    else:
        images = [generator.visualize(codes[i, :]) for i in range(codes.shape[0])]
    return images

class ExperimentEvolve:
    def __init__(self, model_unit, max_step=200):
        self.recording = []
        self.scores_all = []
        self.codes_all = []
        self.generations = []
        self.CNNmodel = CNNmodel(model_unit[0])  # 'caffe-net'
        self.CNNmodel.select_unit(model_unit)
        self.optimizer = CholeskyCMAES(recorddir=recorddir, space_dimen=code_length, init_sigma=init_sigma,
                                       init_code=np.zeros([1, code_length]), Aupdate_freq=Aupdate_freq) # , optim_params=optim_params
        self.max_steps = max_step

    def run(self, init_code=None):
        self.recording = []
        self.scores_all = []
        self.codes_all = []
        self.generations = []
        for self.istep in range(self.max_steps):
            if self.istep == 0:
                if init_code is None:
                    codes = np.zeros([1, code_length])
                else:
                    codes = init_code
            print('\n>>> step %d' % self.istep)
            t0 = time()
            self.current_images = render(codes)
            t1 = time()  # generate image from code
            synscores = self.CNNmodel.score(self.current_images)
            t2 = time()  # score images
            codes_new = self.optimizer.step_simple(synscores, codes)
            t3 = time()  # use results to update optimizer
            self.codes_all.append(codes)
            self.scores_all = self.scores_all + list(synscores)
            self.generations = self.generations + [self.istep] * len(synscores)
            codes = codes_new
            # summarize scores & delays
            print('synthetic img scores: mean {}, all {}'.format(np.nanmean(synscores), synscores))
            print(('step %d time: total %.2fs | ' +
                   'code visualize %.2fs  score %.2fs  optimizer step %.2fs')
                  % (self.istep, t3 - t0, t1 - t0, t2 - t1, t3 - t2))
        self.codes_all = np.concatenate(tuple(self.codes_all), axis=0)
        self.scores_all = np.array(self.scores_all)
        self.generations = np.array(self.generations)

    def visualize_exp(self, show=False):
        idx_list = []
        for geni in range(min(self.generations), max(self.generations)+1):
            rel_idx = np.argmax(self.scores_all[self.generations == geni])
            idx_list.append(np.nonzero(self.generations == geni)[0][rel_idx])
        idx_list = np.array(idx_list)
        select_code = self.codes_all[idx_list, :]
        score_select = self.scores_all[idx_list]
        img_select = render(select_code)
        fig = utils.visualize_img_list(img_select, score_select, show=show)
        return fig

    def visualize_best(self, show=False):
        idx = np.argmax(self.scores_all)
        select_code = self.codes_all[idx:idx+1, :]
        score_select = self.scores_all[idx]
        img_select = render(select_code)
        fig = plt.figure(figsize=[3, 3])
        plt.imshow(img_select[0])
        plt.axis('off')
        plt.title("{0:.2f}".format(score_select), fontsize=16)
        if show:
            plt.show()
        return fig

    def visualize_trajectory(self, show=True):
        gen_slice = np.arange(min(self.generations), max(self.generations)+1)
        AvgScore = np.zeros_like(gen_slice)
        MaxScore = np.zeros_like(gen_slice)
        for i, geni in enumerate(gen_slice):
            AvgScore[i] = np.mean(self.scores_all[self.generations == geni])
            MaxScore[i] = np.max(self.scores_all[self.generations == geni])
        figh = plt.figure()
        plt.scatter(self.generations, self.scores_all, s=16, alpha=0.6, label="all score")
        plt.plot(gen_slice, AvgScore, color='black', label="Average score")
        plt.plot(gen_slice, MaxScore, color='red', label="Max score")
        plt.xlabel("generation #")
        plt.ylabel("CNN unit score")
        plt.title("Optimization Trajectory of Score\n")# + title_str)
        plt.legend()
        if show:
            plt.show()
        return figh

from cv2 import resize
import cv2
sys.path.append("D:\Github\pytorch-receptive-field")
from torch_receptive_field import receptive_field, receptive_field_for_unit
def resize_and_pad(img_list, size, coord, canvas_size=(227, 227)):
    '''Render a list of codes to list of images'''
    resize_img = []
    for img in img_list:
        pad_img = np.ones(canvas_size + (3,)) * 0.5
        pad_img[coord[0]:coord[0]+size[0], coord[1]:coord[1]+size[1], :] = resize(img, size, cv2.INTER_AREA)
        resize_img.append(pad_img.copy())
    return resize_img

class ExperimentResizeEvolve:
    """Resize the evolved image before feeding into CNN and see how the evolution goes. """
    def __init__(self, model_unit, max_step=200):
        self.recording = []
        self.scores_all = []
        self.codes_all = []
        self.generations = []
        self.CNNmodel = CNNmodel(model_unit[0])  # 'caffe-net'
        self.CNNmodel.select_unit(model_unit)
        self.optimizer = CholeskyCMAES(recorddir=recorddir, space_dimen=code_length, init_sigma=init_sigma,
                                       init_code=np.zeros([1, code_length]), Aupdate_freq=Aupdate_freq) # , optim_params=optim_params
        self.max_steps = max_step

    def run(self, init_code=None):
        self.recording = []
        self.scores_all = []
        self.codes_all = []
        self.generations = []
        for self.istep in range(self.max_steps):
            if self.istep == 0:
                if init_code is None:
                    codes = np.zeros([1, code_length])
                else:
                    codes = init_code
            print('\n>>> step %d' % self.istep)
            t0 = time()
            self.current_images = render(codes)
            self.current_images = resize_and_pad(self.current_images, (50, 50), (100, 100))
            t1 = time()  # generate image from code
            synscores = self.CNNmodel.score(self.current_images)
            t2 = time()  # score images
            codes_new = self.optimizer.step_simple(synscores, codes)
            t3 = time()  # use results to update optimizer
            self.codes_all.append(codes)
            self.scores_all = self.scores_all + list(synscores)
            self.generations = self.generations + [self.istep] * len(synscores)
            codes = codes_new
            # summarize scores & delays
            print('synthetic img scores: mean {}, all {}'.format(np.nanmean(synscores), synscores))
            print(('step %d time: total %.2fs | ' +
                   'code visualize %.2fs  score %.2fs  optimizer step %.2fs')
                  % (self.istep, t3 - t0, t1 - t0, t2 - t1, t3 - t2))
        self.codes_all = np.concatenate(tuple(self.codes_all), axis=0)
        self.scores_all = np.array(self.scores_all)
        self.generations = np.array(self.generations)

    def visualize_exp(self, show=False):
        idx_list = []
        for geni in range(min(self.generations), max(self.generations)+1):
            rel_idx = np.argmax(self.scores_all[self.generations == geni])
            idx_list.append(np.nonzero(self.generations == geni)[0][rel_idx])
        idx_list = np.array(idx_list)
        select_code = self.codes_all[idx_list, :]
        score_select = self.scores_all[idx_list]
        img_select = render(select_code)
        fig = utils.visualize_img_list(img_select, score_select, show=show)
        return fig

    def visualize_best(self, show=False):
        idx = np.argmax(self.scores_all)
        select_code = self.codes_all[idx:idx+1, :]
        score_select = self.scores_all[idx]
        img_select = render(select_code)
        fig = plt.figure(figsize=[3, 3])
        plt.imshow(img_select[0])
        plt.axis('off')
        plt.title("{0:.2f}".format(score_select), fontsize=16)
        if show:
            plt.show()
        return fig

    def visualize_trajectory(self, show=True):
        gen_slice = np.arange(min(self.generations), max(self.generations)+1)
        AvgScore = np.zeros_like(gen_slice)
        MaxScore = np.zeros_like(gen_slice)
        for i, geni in enumerate(gen_slice):
            AvgScore[i] = np.mean(self.scores_all[self.generations == geni])
            MaxScore[i] = np.max(self.scores_all[self.generations == geni])
        figh = plt.figure()
        plt.scatter(self.generations, self.scores_all, s=16, alpha=0.6, label="all score")
        plt.plot(gen_slice, AvgScore, color='black', label="Average score")
        plt.plot(gen_slice, MaxScore, color='red', label="Max score")
        plt.xlabel("generation #")
        plt.ylabel("CNN unit score")
        plt.title("Optimization Trajectory of Score\n")# + title_str)
        plt.legend()
        if show:
            plt.show()
        return figh

class ExperimentManifold:
    def __init__(self, model_unit, max_step=100, savedir="", explabel=""):
        self.recording = []
        self.scores_all = []
        self.codes_all = []
        self.generations = []
        self.pref_unit = model_unit
        self.CNNmodel = CNNmodel(model_unit[0])  # 'caffe-net'
        self.CNNmodel.select_unit(model_unit)
        self.optimizer = CholeskyCMAES(recorddir=recorddir, space_dimen=code_length, init_sigma=init_sigma,
                                       init_code=np.zeros([1, code_length]),
                                       Aupdate_freq=Aupdate_freq)  # , optim_params=optim_params
        self.max_steps = max_step
        self.savedir = savedir
        self.explabel = explabel
        self.Perturb_vec = []

    def run(self, init_code=None):
        self.recording = []
        self.scores_all = []
        self.codes_all = []
        self.generations = []
        for self.istep in range(self.max_steps):
            if self.istep == 0:
                if init_code is None:
                    codes = np.zeros([1, code_length])
                else:
                    codes = init_code
            print('\n>>> step %d' % self.istep)
            t0 = time()
            self.current_images = render(codes)
            t1 = time()  # generate image from code
            synscores = self.CNNmodel.score(self.current_images)
            t2 = time()  # score images
            codes_new = self.optimizer.step_simple(synscores, codes)
            t3 = time()  # use results to update optimizer
            self.codes_all.append(codes)
            self.scores_all = self.scores_all + list(synscores)
            self.generations = self.generations + [self.istep] * len(synscores)
            codes = codes_new
            # summarize scores & delays
            print('synthetic img scores: mean {}, all {}'.format(np.nanmean(synscores), synscores))
            print(('step %d time: total %.2fs | ' +
                   'code visualize %.2fs  score %.2fs  optimizer step %.2fs')
                  % (self.istep, t3 - t0, t1 - t0, t2 - t1, t3 - t2))
        self.codes_all = np.concatenate(tuple(self.codes_all), axis=0)
        self.scores_all = np.array(self.scores_all)
        self.generations = np.array(self.generations)

    def analyze_traj(self):
        '''Get the trajectory and the PCs and the structures of it'''
        final_gen_norms = np.linalg.norm(self.codes_all[self.generations == max(self.generations), :], axis=1)
        self.sphere_norm = final_gen_norms.mean()
        code_pca = PCA(n_components=50)
        PC_Proj_codes = code_pca.fit_transform(self.codes_all)
        self.PC_vectors = code_pca.components_
        if PC_Proj_codes[-1, 0] < 0:  # decide which is the positive direction for PC1
            inv_PC1 = True
            self.PC_vectors[0, :] = - self.PC_vectors[0, :]
            self.PC1_sign = -1
        else:
            inv_PC1 = False
            self.PC1_sign = 1
            pass

    def run_manifold(self, subspace_list, interval=9):
        '''Generate examples on manifold and run'''
        self.score_sum = []
        figsum = plt.figure(figsize=[16.7, 4])
        for spi, subspace in enumerate(subspace_list):
            if subspace == "RND":
                title = "Norm%dRND%dRND%d" % (self.sphere_norm, 0 + 1, 1 + 1)
                print("Generating images on PC1, Random vector1, Random vector2 sphere (rad = %d)" % self.sphere_norm)
                rand_vec2 = np.random.randn(2, 4096)
                rand_vec2 = rand_vec2 - (rand_vec2 @ self.PC_vectors.T) @ self.PC_vectors
                rand_vec2 = rand_vec2 / np.sqrt((rand_vec2 ** 2).sum(axis=1))[:, np.newaxis]
                rand_vec2[1, :] = rand_vec2[1, :] - (rand_vec2[1, :] @ rand_vec2[0, :].T) * rand_vec2[0, :]
                rand_vec2[1, :] = rand_vec2[1, :] / np.linalg.norm(rand_vec2[1, :])
                vectors = np.concatenate((self.PC_vectors[0:1, :], rand_vec2), axis=0)
                self.Perturb_vec.append(vectors)
                img_list = []
                interv_n = int(90 / interval)
                for j in range(-interv_n, interv_n + 1):
                    for k in range(-interv_n, interv_n + 1):
                        theta = interval * j / 180 * np.pi
                        phi = interval * k / 180 * np.pi
                        code_vec = np.array([[np.cos(theta) * np.cos(phi),
                                              np.sin(theta) * np.cos(phi),
                                              np.sin(phi)]]) @ vectors
                        code_vec = code_vec / np.sqrt((code_vec ** 2).sum()) * self.sphere_norm
                        img = generator.visualize(code_vec)
                        img_list.append(img.copy())
            else:
                PCi, PCj = subspace
                title = "Norm%dPC%dPC%d" % (self.sphere_norm, PCi + 1, PCj + 1)
                print("Generating images on PC1, PC%d, PC%d sphere (rad = %d)" % (PCi + 1, PCj + 1, self.sphere_norm, ))
                img_list = []
                interv_n = int(90 / interval)
                self.Perturb_vec.append(self.PC_vectors[[0, PCi, PCj], :])
                for j in range(-interv_n, interv_n + 1):
                    for k in range(-interv_n, interv_n + 1):
                        theta = interval * j / 180 * np.pi
                        phi = interval * k / 180 * np.pi
                        code_vec = np.array([[np.cos(theta) * np.cos(phi),
                                              np.sin(theta) * np.cos(phi),
                                              np.sin(phi)]]) @ self.PC_vectors[[0, PCi, PCj], :]
                        code_vec = code_vec / np.sqrt((code_vec ** 2).sum()) * self.sphere_norm
                        img = generator.visualize(code_vec)
                        img_list.append(img.copy())
                        # plt.imsave(os.path.join(newimg_dir, "norm_%d_PC2_%d_PC3_%d.jpg" % (
                        # self.sphere_norm, interval * j, interval * k)), img)
            scores = self.CNNmodel.score(img_list)
            fig = utils.visualize_img_list(img_list, scores=scores, ncol=2*interv_n+1, nrow=2*interv_n+1, )
            fig.savefig(os.path.join(self.savedir, "%s_%s.png" % (title, self.explabel)))
            scores = np.array(scores).reshape((2*interv_n+1, 2*interv_n+1))
            self.score_sum.append(scores)
            ax = figsum.add_subplot(1, len(subspace_list), spi + 1)
            im = ax.imshow(scores)
            plt.colorbar(im, ax=ax)
            ax.set_xticks([0, interv_n / 2, interv_n, 1.5 * interv_n, 2*interval]); ax.set_xticklabels([-90,45,0,45,90])
            ax.set_yticks([0, interv_n / 2, interv_n, 1.5 * interv_n, 2*interval]); ax.set_yticklabels([-90,45,0,45,90])
            ax.set_title(title+"_Hemisphere")
        figsum.suptitle("%s-%s-unit%03d  %s" % (self.pref_unit[0], self.pref_unit[1], self.pref_unit[2], self.explabel))
        figsum.savefig(os.path.join(self.savedir, "Manifold_summary_%s_norm%d.png" % (self.explabel, self.sphere_norm)))
        self.Perturb_vec = np.concatenate(tuple(self.Perturb_vec), axis=0)
        return self.score_sum, figsum
#%%
from scipy.stats import ortho_group, special_ortho_group
import math
def make_orthonormal_matrix(n):
    """
    Makes a square matrix which is orthonormal by concatenating
    random Householder transformations
    """
    A = np.identity(n)
    d = np.zeros(n)
    d[n-1] = np.random.choice([-1.0, 1.0])
    for k in range(n-2, -1, -1):
        # generate random Householder transformation
        x = np.random.randn(n-k)
        s = np.sqrt((x**2).sum()) # norm(x)
        sign = math.copysign(1.0, x[0])
        s *= sign
        d[k] = -sign
        x[0] += s
        beta = s * x[0]
        # apply the transformation
        y = np.dot(x,A[k:n,:]) / beta
        A[k:n,:] -= np.outer(x,y)
    # change sign of rows
    A *= d.reshape(n,1)
    return A

class ExperimentGANAxis:
    """ Tuning w.r.t. all the major axis in the GAN or the randomly generated O(n) frame set. """
    def __init__(self, model_unit, savedir="", explabel=""):
        self.recording = []
        self.scores_all = []
        self.scores_all_rnd = []
        self.codes_all = []
        self.pref_unit = model_unit
        self.CNNmodel = CNNmodel(model_unit[0])  # 'caffe-net'
        self.CNNmodel.select_unit(model_unit)
        self.savedir = savedir
        self.explabel = explabel

    def run_axis(self, Norm, orthomat=None):
        '''Generate examples on manifold and run'''
        self.score_sum = []
        figsum = plt.figure(figsize=[16.7, 8])

        BATCH_SIZE = 128
        BATCH_N = int(4096 / BATCH_SIZE)
        print("Test the tuning on all the axis in GAN space (Norm %d)"%Norm)
        code_mat = np.eye(4096, 4096)
        scores_all = []
        scores_all_neg = []
        for bi in range(BATCH_N):
            img_list = []
            for j in range(BATCH_SIZE):
                img = generator.visualize(Norm * code_mat[bi * BATCH_N + j, :])
                img_list.append(img.copy())
            scores = self.CNNmodel.score(img_list)
            scores_all.extend(list(scores))
            img_list = []
            for j in range(BATCH_SIZE):
                img = generator.visualize(- Norm * code_mat[bi * BATCH_N + j, :])
                img_list.append(img.copy())
            scores = self.CNNmodel.score(img_list)
            scores_all_neg.extend(list(scores))
            print("Finished batch %02d/%02d"%( bi+1, BATCH_N))
        self.scores_all = np.array(scores_all + scores_all_neg)
        ax = figsum.add_subplot(2, 1, 1)
        ax.scatter(np.arange(4096), scores_all, alpha=0.5)
        ax.scatter(np.arange(4096), scores_all_neg, alpha=0.4)
        ax.plot(sorted(scores_all), color='orange')
        ax.plot(sorted(scores_all_neg), color='green')
        ax.set_xlim(-50, 4150)
        if orthomat is None:
            code_mat = make_orthonormal_matrix(4096)# ortho_group.rvs(4096)
        else:
            code_mat = orthomat
        scores_all = []
        scores_all_neg = []
        print("Test the tuning on a random O(N) in GAN space (Norm %d)" % Norm)
        for bi in range(BATCH_N):
            img_list = []
            for j in range(BATCH_SIZE):
                img = generator.visualize(Norm * code_mat[bi * BATCH_N + j, :])
                img_list.append(img.copy())
            scores = self.CNNmodel.score(img_list)
            scores_all.extend(list(scores))
            img_list = []
            for j in range(BATCH_SIZE):
                img = generator.visualize(- Norm * code_mat[bi * BATCH_N + j, :])
                img_list.append(img.copy())
            scores = self.CNNmodel.score(img_list)
            scores_all_neg.extend(list(scores))
            print("Finished batch %02d/%02d"% (bi + 1, BATCH_N))
        self.scores_all_rnd = np.array(scores_all + scores_all_neg)
        ax = figsum.add_subplot(2, 1, 2)
        ax.scatter(np.arange(4096), scores_all, alpha=0.5)
        ax.plot(sorted(scores_all), color='orange')
        ax.scatter(np.arange(4096), scores_all_neg, alpha=0.4)
        ax.plot(sorted(scores_all_neg), color='green')
        ax.set_xlim(-50, 4150)
        # ax = figsum.add_subplot(1, len(subspace_list), spi + 1)
        # im = ax.imshow(scores)
        # plt.colorbar(im, ax=ax)
        # ax.set_xticks([0, interv_n / 2, interv_n, 1.5 * interv_n, 2*interval]); ax.set_xticklabels([-90,45,0,45,90])
        # ax.set_yticks([0, interv_n / 2, interv_n, 1.5 * interv_n, 2*interval]); ax.set_yticklabels([-90,45,0,45,90])
        # ax.set_title(title+"_Hemisphere")
        # figsum.suptitle("%s-%s-unit%03d  %s" % (self.pref_unit[0], self.pref_unit[1], self.pref_unit[2], self.explabel))
        figsum.savefig(os.path.join(self.savedir, "Axis_summary_%s_norm%d.png" % (self.explabel, Norm)))
        return self.scores_all, self.scores_all_rnd, figsum
#%%
class ExperimentRestrictEvolve:
    """Evolution in a restricted linear subspace with subspace_d """
    def __init__(self, subspace_d, model_unit, max_step=200):
        self.sub_d = subspace_d
        self.recording = []
        self.scores_all = []
        self.codes_all = []
        self.generations = []
        self.CNNmodel = CNNmodel(model_unit[0])  # 'caffe-net'
        self.CNNmodel.select_unit(model_unit)  # ('caffe-net', 'fc8', 1)
        self.optimizer = CholeskyCMAES(recorddir=recorddir, space_dimen=subspace_d, init_sigma=init_sigma,
                                       init_code=np.zeros([1, subspace_d]),
                                       Aupdate_freq=Aupdate_freq)  # , optim_params=optim_params
        self.max_steps = max_step

    def get_basis(self):
        self.basis = np.zeros([self.sub_d, code_length])
        for i in range(self.sub_d):
            tmp_code = np.random.randn(1, code_length)
            tmp_code = tmp_code - (tmp_code @ self.basis.T) @ self.basis
            self.basis[i, :] = tmp_code / np.linalg.norm(tmp_code)
        return self.basis

    def run(self, init_code=None):
        self.recording = []
        self.scores_all = []
        self.codes_all = []
        self.coords_all = []
        self.generations = []
        for self.istep in range(self.max_steps):
            if self.istep == 0:
                if init_code is None:
                    coords = np.zeros([1, self.sub_d])
                else:
                    coords = init_code
            codes = coords @ self.basis
            print('\n>>> step %d' % self.istep)
            t0 = time()
            self.current_images = render(codes)
            t1 = time()  # generate image from code
            synscores = self.CNNmodel.score(self.current_images)
            t2 = time()  # score images
            coords_new = self.optimizer.step_simple(synscores, coords)
            t3 = time()  # use results to update optimizer
            self.coords_all.append(coords)
            self.codes_all.append(codes)
            self.scores_all = self.scores_all + list(synscores)
            self.generations = self.generations + [self.istep] * len(synscores)
            coords = coords_new
            # summarize scores & delays
            print('synthetic img scores: mean {}, all {}'.format(np.nanmean(synscores), synscores))
            print(('step %d time: total %.2fs | ' +
                   'code visualize %.2fs  score %.2fs  optimizer step %.2fs')
                  % (self.istep, t3 - t0, t1 - t0, t2 - t1, t3 - t2))
        self.coords_all = np.concatenate(tuple(self.coords_all), axis=0)
        self.codes_all = np.concatenate(tuple(self.codes_all), axis=0)
        self.scores_all = np.array(self.scores_all)
        self.generations = np.array(self.generations)

    def visualize_exp(self, show=False):
        idx_list = []
        for geni in range(min(self.generations), max(self.generations)+1):
            rel_idx = np.argmax(self.scores_all[self.generations == geni])
            idx_list.append(np.nonzero(self.generations == geni)[0][rel_idx])
        idx_list = np.array(idx_list)
        select_code = self.codes_all[idx_list, :]
        score_select = self.scores_all[idx_list]
        img_select = render(select_code)
        fig = utils.visualize_img_list(img_select, score_select, show=show)
        return fig

    def visualize_best(self, show=False):
        idx = np.argmax(self.scores_all)
        select_code = self.codes_all[idx:idx+1, :]
        score_select = self.scores_all[idx]
        img_select = render(select_code)
        fig = plt.figure(figsize=[3, 3])
        plt.imshow(img_select[0])
        plt.axis('off')
        plt.title("{0:.2f}".format(score_select), fontsize=16)
        if show:
            plt.show()
        return fig

    def visualize_trajectory(self, show=True):
        gen_slice = np.arange(min(self.generations), max(self.generations)+1)
        AvgScore = np.zeros_like(gen_slice)
        MaxScore = np.zeros_like(gen_slice)
        for i, geni in enumerate(gen_slice):
            AvgScore[i] = np.mean(self.scores_all[self.generations == geni])
            MaxScore[i] = np.max(self.scores_all[self.generations == geni])
        figh = plt.figure()
        plt.scatter(self.generations, self.scores_all, s=16, alpha=0.6, label="all score")
        plt.plot(gen_slice, AvgScore, color='black', label="Average score")
        plt.plot(gen_slice, MaxScore, color='red', label="Max score")
        plt.xlabel("generation #")
        plt.ylabel("CNN unit score")
        plt.title("Optimization Trajectory of Score\n")# + title_str)
        plt.legend()
        if show:
            plt.show()
        return figh
# experiment = ExperimentEvolve()
# experiment.run()
#%%
if __name__ == "__main__":
    # subspace_d = 50
    # for triali in range(100):
    #     experiment = ExperimentRestrictEvolve(subspace_d, ('caffe-net', 'fc8', 1))
    #     experiment.get_basis()
    #     experiment.run()
    #     fig = experiment.visualize_trajectory(show=False)
    #     fig.savefig(os.path.join(recorddir, "Subspc%dScoreTrajTrial%03d" % (subspace_d, triali) + ".png"))
    #     fig2 = experiment.visualize_exp(show=False)
    #     fig2.savefig(os.path.join(recorddir, "Subspc%dEvolveTrial%03d"%(subspace_d, triali) + ".png"))
    #
    # #%%
    # #%% Restricted evolution for the 5 examplar layerse
    # subspace_d = 50
    # unit = ('caffe-net', 'conv5', 5, 10, 10)
    # savedir = os.path.join(recorddir, "%s_%s_%d" % (unit[0], unit[1], unit[2]))
    # os.makedirs(savedir, exist_ok=True)
    # best_scores_col = []
    # for triali in range(100):
    #     experiment = ExperimentRestrictEvolve(subspace_d, unit, max_step=200)
    #     experiment.get_basis()
    #     experiment.run()
    #     fig0 = experiment.visualize_best(show=False)
    #     fig0.savefig(join(savedir, "Subspc%dBestImgTrial%03d.png" % (subspace_d, triali)))
    #     fig = experiment.visualize_trajectory(show=False)
    #     fig.savefig(join(savedir, "Subspc%dScoreTrajTrial%03d.png" % (subspace_d, triali)))
    #     fig2 = experiment.visualize_exp(show=False)
    #     fig2.savefig(join(savedir, "Subspc%dEvolveTrial%03d.png" % (subspace_d, triali)))
    #     plt.close("all")
    #     np.savez(join(savedir, "scores_subspc%dtrial%03d.npz" % (subspace_d, triali)),
    #              generations=experiment.generations,
    #              scores_all=experiment.scores_all)
    #     lastgen_max = [experiment.scores_all[experiment.generations == geni].max() for geni in
    #      range(experiment.generations.max() - 10, experiment.generations.max() + 1)]
    #     best_scores_col.append(lastgen_max)
    # best_scores_col = np.array(best_scores_col)
    # np.save(join(savedir, "best_scores.npy"), best_scores_col)
    # #%%
    # subspace_d = 50
    # unit = ('caffe-net', 'conv3', 5, 10, 10)
    # savedir = os.path.join(recorddir, "%s_%s_%d" % (unit[0], unit[1], unit[2]))
    # os.makedirs(savedir, exist_ok=True)
    # best_scores_col = []
    # for triali in range(0, 100):
    #     experiment = ExperimentRestrictEvolve(subspace_d, unit, max_step=200)
    #     experiment.get_basis()
    #     experiment.run()
    #     fig0 = experiment.visualize_best(show=False)
    #     fig0.savefig(join(savedir, "Subspc%dBestImgTrial%03d.png" % (subspace_d, triali)))
    #     fig = experiment.visualize_trajectory(show=False)
    #     fig.savefig(join(savedir, "Subspc%dScoreTrajTrial%03d.png" % (subspace_d, triali)))
    #     fig2 = experiment.visualize_exp(show=False)
    #     fig2.savefig(join(savedir, "Subspc%dEvolveTrial%03d.png" % (subspace_d, triali)))
    #     plt.close("all")
    #     np.savez(join(savedir, "scores_subspc%dtrial%03d.npz" % (subspace_d, triali)),
    #              generations=experiment.generations,
    #              scores_all=experiment.scores_all)
    #     lastgen_max = [experiment.scores_all[experiment.generations == geni].max() for geni in
    #      range(experiment.generations.max() - 10, experiment.generations.max() + 1)]
    #     best_scores_col.append(lastgen_max)
    # best_scores_col = np.array(best_scores_col)
    # np.save(join(savedir, "best_scores.npy"), best_scores_col)
    #
    # subspace_d = 50
    # unit = ('caffe-net', 'fc6', 1)
    # savedir = os.path.join(recorddir, "%s_%s_%d" % (unit[0], unit[1], unit[2]))
    # os.makedirs(savedir, exist_ok=True)
    # best_scores_col = []
    # for triali in range(100):
    #     experiment = ExperimentRestrictEvolve(subspace_d, unit, max_step=200)
    #     experiment.get_basis()
    #     experiment.run()
    #     fig0 = experiment.visualize_best(show=False)
    #     fig0.savefig(join(savedir, "Subspc%dBestImgTrial%03d.png" % (subspace_d, triali)))
    #     fig = experiment.visualize_trajectory(show=False)
    #     fig.savefig(join(savedir, "Subspc%dScoreTrajTrial%03d.png" % (subspace_d, triali)))
    #     fig2 = experiment.visualize_exp(show=False)
    #     fig2.savefig(join(savedir, "Subspc%dEvolveTrial%03d.png" % (subspace_d, triali)))
    #     np.savez(join(savedir, "scores_subspc%dtrial%03d.npz" % (subspace_d, triali)),
    #              generations=experiment.generations,
    #              scores_all=experiment.scores_all)
    #     lastgen_max = [experiment.scores_all[experiment.generations == geni].max() for geni in
    #      range(experiment.generations.max() - 10, experiment.generations.max() + 1)]
    #     best_scores_col.append(lastgen_max)
    # best_scores_col = np.array(best_scores_col)
    # np.save(join(savedir, "best_scores.npy"), best_scores_col)
    #
    # subspace_d = 50
    # unit = ('caffe-net', 'fc7', 1)
    # savedir = os.path.join(recorddir, "%s_%s_%d" % (unit[0], unit[1], unit[2]))
    # os.makedirs(savedir, exist_ok=True)
    # best_scores_col = []
    # for triali in range(100):
    #     experiment = ExperimentRestrictEvolve(subspace_d, unit, max_step=200)
    #     experiment.get_basis()
    #     experiment.run()
    #     fig0 = experiment.visualize_best(show=False)
    #     fig0.savefig(join(savedir, "Subspc%dBestImgTrial%03d.png" % (subspace_d, triali)))
    #     fig = experiment.visualize_trajectory(show=False)
    #     fig.savefig(join(savedir, "Subspc%dScoreTrajTrial%03d.png" % (subspace_d, triali)))
    #     fig2 = experiment.visualize_exp(show=False)
    #     fig2.savefig(join(savedir, "Subspc%dEvolveTrial%03d.png" % (subspace_d, triali)))
    #     np.savez(join(savedir, "scores_subspc%dtrial%03d.npz" % (subspace_d, triali)),
    #              generations=experiment.generations,
    #              scores_all=experiment.scores_all)
    #     lastgen_max = [experiment.scores_all[experiment.generations == geni].max() for geni in
    #      range(experiment.generations.max() - 10, experiment.generations.max() + 1)]
    #     best_scores_col.append(lastgen_max)
    # best_scores_col = np.array(best_scores_col)
    # np.save(join(savedir, "best_scores.npy"), best_scores_col)

    #%% Baseline Full Evolution
    unit_arr = [('caffe-net', 'conv3', 5, 10, 10),
                ('caffe-net', 'conv5', 5, 10, 10),
                ('caffe-net', 'fc6', 1),
                ('caffe-net', 'fc7', 1),
                ('caffe-net', 'fc8', 1)]
    # unit = ('caffe-net', 'fc7', 1)
    # for unit in unit_arr:
    #     savedir = os.path.join(recorddir, "%s_%s_%d_full" % (unit[0], unit[1], unit[2]))
    #     os.makedirs(savedir, exist_ok=True)
    #     best_scores_col = []
    #     for triali in range(20):
    #         experiment = ExperimentEvolve(unit, max_step=200)
    #         experiment.run()
    #         fig0 = experiment.visualize_best(show=False)
    #         fig0.savefig(join(savedir, "FullBestImgTrial%03d.png" % (triali)))
    #         fig = experiment.visualize_trajectory(show=False)
    #         fig.savefig(join(savedir, "FullScoreTrajTrial%03d.png" % (triali)))
    #         fig2 = experiment.visualize_exp(show=False)
    #         fig2.savefig(join(savedir, "EvolveTrial%03d.png" % (triali)))
    #         plt.close('all')
    #         np.savez(join(savedir, "scores_trial%03d.npz" % (triali)),
    #                  generations=experiment.generations,
    #                  scores_all=experiment.scores_all)
    #         lastgen_max = [experiment.scores_all[experiment.generations == geni].max() for geni in
    #          range(experiment.generations.max() - 10, experiment.generations.max() + 1)]
    #         best_scores_col.append(lastgen_max)
    #     best_scores_col = np.array(best_scores_col)
    #     np.save(join(savedir, "best_scores.npy"), best_scores_col)

    #%%
    unit_arr = [('caffe-net', 'conv1', 5, 10, 10),
                ('caffe-net', 'conv2', 5, 10, 10),
                ('caffe-net', 'conv3', 5, 10, 10),
                #('caffe-net', 'conv1', 5, 10, 10),
                ]
    subspace_d = 20
                #  [('caffe-net', 'conv3', 5, 10, 10),
                # ('caffe-net', 'conv5', 5, 10, 10),
                # ('caffe-net', 'fc6', 1),
                # ('caffe-net', 'fc7', 1),
                # ('caffe-net', 'fc8', 1)]
    # unit = ('caffe-net', 'fc7', 1)
    # for unit in unit_arr:
    #     savedir = os.path.join(recorddir, "%s_%s_%d_full" % (unit[0], unit[1], unit[2]))
    #     os.makedirs(savedir, exist_ok=True)
    #     best_scores_col = []
    #     for triali in range(20):
    #         experiment = ExperimentEvolve(unit, max_step=200)
    #         experiment.run()
    #         fig0 = experiment.visualize_best(show=False)
    #         fig0.savefig(join(savedir, "FullBestImgTrial%03d.png" % (triali)))
    #         fig = experiment.visualize_trajectory(show=False)
    #         fig.savefig(join(savedir, "FullScoreTrajTrial%03d.png" % (triali)))
    #         fig2 = experiment.visualize_exp(show=False)
    #         fig2.savefig(join(savedir, "EvolveTrial%03d.png" % (triali)))
    #         plt.close('all')
    #         np.savez(join(savedir, "scores_trial%03d.npz" % (triali)),
    #                  generations=experiment.generations,
    #                  scores_all=experiment.scores_all)
    #         lastgen_max = [experiment.scores_all[experiment.generations == geni].max() for geni in
    #          range(experiment.generations.max() - 10, experiment.generations.max() + 1)]
    #         best_scores_col.append(lastgen_max)
    #     best_scores_col = np.array(best_scores_col)
    #     np.save(join(savedir, "best_scores.npy"), best_scores_col)

    # for unit in unit_arr:
    #     savedir = os.path.join(recorddir, "%s_%s_%d_subspac%d" % (unit[0], unit[1], unit[2], subspace_d))
    #     os.makedirs(savedir, exist_ok=True)
    #     best_scores_col = []
    #     for triali in range(100):
    #         experiment = ExperimentRestrictEvolve(subspace_d, unit, max_step=200)
    #         experiment.get_basis()
    #         experiment.run()
    #         fig0 = experiment.visualize_best(show=False)
    #         fig0.savefig(join(savedir, "Subspc%dBestImgTrial%03d.png" % (subspace_d, triali)))
    #         fig = experiment.visualize_trajectory(show=False)
    #         fig.savefig(join(savedir, "Subspc%dScoreTrajTrial%03d.png" % (subspace_d, triali)))
    #         fig2 = experiment.visualize_exp(show=False)
    #         fig2.savefig(join(savedir, "Subspc%dEvolveTrial%03d.png" % (subspace_d, triali)))
    #         plt.close('all')
    #         np.savez(join(savedir, "scores_subspc%dtrial%03d.npz" % (subspace_d, triali)),
    #                  generations=experiment.generations,
    #                  scores_all=experiment.scores_all)
    #         lastgen_max = [experiment.scores_all[experiment.generations == geni].max() for geni in
    #          range(experiment.generations.max() - 10, experiment.generations.max() + 1)]
    #         best_scores_col.append(lastgen_max)
    #     best_scores_col = np.array(best_scores_col)
    #     np.save(join(savedir, "best_scores.npy"), best_scores_col)
    #%%

    unit_arr = [('caffe-net', 'conv1', 5, 10, 10),
                ('caffe-net', 'conv2', 5, 10, 10),
                ('caffe-net', 'conv4', 5, 10, 10),]
                # ('caffe-net', 'conv3', 5, 10, 10),
                # ('caffe-net', 'conv5', 5, 10, 10),
                # ('caffe-net', 'fc6', 1),
                # ('caffe-net', 'fc7', 1),
                # ('caffe-net', 'fc8', 1),
                # ]
    for unit in unit_arr:
        savedir = os.path.join(r"D:\Generator_DB_Windows\data\with_CNN", "%s_%s_manifold" % (unit[0], unit[1]))
        os.makedirs(savedir, exist_ok=True)
        for chan in range(50):
            if len(unit) == 3:
                unit = (unit[0], unit[1], chan)
            else:
                unit = (unit[0], unit[1], chan, 10, 10)
            experiment = ExperimentManifold(unit, max_step=100, savedir=savedir, explabel="chan%03d" % chan)
            experiment.run()
            experiment.analyze_traj()
            score_sum, _ = experiment.run_manifold([(1, 2), (24, 25), (48, 49), "RND"])
            np.savez(os.path.join(savedir, "score_map_chan%d.npz" % chan), score_sum=score_sum,
                     Perturb_vectors=experiment.Perturb_vec, sphere_norm=experiment.sphere_norm)
            plt.close("all")

    for unit in unit_arr:
        savedir = os.path.join(r"D:\Generator_DB_Windows\data\with_CNN", "%s_%s_manifold_25gen" % (unit[0], unit[1]))
        os.makedirs(savedir, exist_ok=True)
        for chan in range(50):
            if len(unit) == 3:
                unit = (unit[0], unit[1], chan)
            else:
                unit = (unit[0], unit[1], chan, 10, 10)
            experiment = ExperimentManifold(unit, max_step=25, savedir=savedir, explabel="step25_chan%03d" % chan)
            experiment.run()
            experiment.analyze_traj()
            score_sum, _ = experiment.run_manifold([(1, 2), (24, 25), (48, 49), "RND"])
            np.savez(os.path.join(savedir, "score_map_step25_chan%d.npz" % chan), score_sum=score_sum,
                     Perturb_vectors=experiment.Perturb_vec, sphere_norm=experiment.sphere_norm)
            plt.close("all")

    for unit in unit_arr:
        savedir = os.path.join(r"D:\Generator_DB_Windows\data\with_CNN", "%s_%s_manifold_50gen" % (unit[0], unit[1]))
        os.makedirs(savedir, exist_ok=True)
        for chan in range(50):
            if len(unit) == 3:
                unit = (unit[0], unit[1], chan)
            else:
                unit = (unit[0], unit[1], chan, 10, 10)
            experiment = ExperimentManifold(unit, max_step=50, savedir=savedir, explabel="step50_chan%03d" % chan)
            experiment.run()
            experiment.analyze_traj()
            score_sum, _ = experiment.run_manifold([(1, 2), (24, 25), (48, 49), "RND"])
            np.savez(os.path.join(savedir, "score_map_step50_chan%d.npz" % chan), score_sum=score_sum,
                     Perturb_vectors=experiment.Perturb_vec, sphere_norm=experiment.sphere_norm)
            plt.close("all")
#%%
omat = np.load("ortho4096.npy")
savedir = join(recorddir, "axis_data")
unit_arr = [('caffe-net', 'conv1', 5, 10, 10),
            ('caffe-net', 'conv2', 5, 10, 10),
            ('caffe-net', 'conv3', 5, 10, 10),
            ('caffe-net', 'conv4', 5, 10, 10),
            ('caffe-net', 'conv5', 5, 10, 10),
            ('caffe-net', 'fc6', 1),
            ('caffe-net', 'fc7', 1),
            ('caffe-net', 'fc8', 1),
            ]
for unit in unit_arr:
    exp = ExperimentGANAxis(unit, savedir=savedir,
                            explabel="%s_%d" % (unit[1],unit[2]))
    exp.run_axis(350, orthomat=omat)
    np.savez(join(savedir, "axis_score_%s_%d" % (unit[1],unit[2])), scores_all=exp.scores_all, scores_all_rnd=exp.scores_all_rnd)


#%%
savedir = join(recorddir, "resize_data")
os.makedirs(savedir, exist_ok=True)
unit_arr = [
            ('caffe-net', 'conv5', 5, 10, 10),
            ('caffe-net', 'conv1', 5, 10, 10),
            ('caffe-net', 'conv2', 5, 10, 10),
            ('caffe-net', 'conv3', 5, 10, 10),
            ('caffe-net', 'conv4', 5, 10, 10),
            ('caffe-net', 'fc6', 1),
            ('caffe-net', 'fc7', 1),
            ('caffe-net', 'fc8', 1),
            ]
for unit in unit_arr:
    exp = ExperimentResizeEvolve(unit, )
                            #explabel="%s_%d" % (unit[1],unit[2]))
    exp.run()
    exp.visualize_best()
    exp.visualize_trajectory()
    exp.visualize_exp()