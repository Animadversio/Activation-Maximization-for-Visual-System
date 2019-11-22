#%% Preparation for RF computation.
import torchvision
import sys
sys.path.append(r"D:\Github\pytorch-receptive-field")
from torch_receptive_field import receptive_field, receptive_field_for_unit
from importlib import reload
alexnet = torchvision.models.AlexNet() # using the pytorch alexnet as proxy for caffenet.
rf_dict = receptive_field(alexnet.features, (3,227,227), device="cpu")
layer_name_map = {"conv1": "1", "conv2": "4", "conv3": "7", "conv4": "9", "conv5": "11"}  # how names in unit tuple maps to the
#%%
from insilico_Exp import *
plt.ioff()
import matplotlib
matplotlib.use('Agg')
savedir = join(recorddir, "resize_data")
os.makedirs(savedir, exist_ok=True)
unit_arr = [#('caffe-net', 'conv5', 5, 10, 10),
            ('caffe-net', 'conv5', 6, 10, 10),
            ('caffe-net', 'conv5', 7, 10, 10),
            ('caffe-net', 'conv5', 8, 10, 10),
            ('caffe-net', 'conv5', 9, 10, 10),
            ('caffe-net', 'conv5', 10, 10, 10),
            # ('caffe-net', 'conv4', 5, 10, 10),
            # ('caffe-net', 'conv3', 5, 10, 10),
            # ('caffe-net', 'conv2', 5, 10, 10),
            # ('caffe-net', 'conv1', 5, 10, 10),
            # ('caffe-net', 'fc6', 1),
            # ('caffe-net', 'fc7', 1),
            # ('caffe-net', 'fc8', 1),
            ]
for channel in range(1, 51):
    unit = ('caffe-net', 'conv5', channel, 10, 10)
#for unit in unit_arr:
    if "conv" in unit[1]:
        rf_pos = receptive_field_for_unit(rf_dict, (3, 227, 227), layer_name_map[unit[1]], (unit[3], unit[4]))
        imgsize = (int(rf_pos[0][1] - rf_pos[0][0]), int(rf_pos[1][1] - rf_pos[1][0]))
        corner = (int(rf_pos[0][0]), int(rf_pos[1][0]))
    else:
        rf_pos = [(0, 227), (0, 227)]
        imgsize = (227, 227)
        corner = (0, 0)
    exp = ExperimentResizeEvolve(unit, imgsize=imgsize, corner=corner,
                                     max_step=100, savedir=savedir, explabel="%s_%d_rf_fit" % (unit[1], unit[2]))
    exp.run()
    exp.visualize_best()
    exp.visualize_trajectory()
    exp.visualize_exp()
    np.savez(join(savedir, "Evolv_%s_%d_%d_%d_rf_fit.npz" % (unit[1], unit[2], unit[3], unit[4])), scores_all=exp.scores_all,
             codes_all=exp.codes_all, generations=exp.generations)

    expo = ExperimentResizeEvolve(unit, imgsize=(227, 227), corner=(0, 0),
                                 max_step=100, savedir=savedir, explabel="%s_%d_origin" % (unit[1], unit[2]))
    expo.run()
    expo.visualize_best()
    expo.visualize_trajectory()
    expo.visualize_exp()
    np.savez(join(savedir, "Evolv_%s_%d_%d_%d_orig.npz" % (unit[1], unit[2], unit[3], unit[4])), scores_all=expo.scores_all,
             codes_all=exp.codes_all, generations=expo.generations)