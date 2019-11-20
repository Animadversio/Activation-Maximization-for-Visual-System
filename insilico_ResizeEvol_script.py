from insilico_Exp import *
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
    exp = ExperimentResizeEvolve(unit, imgsize=(227, 227), corner=(0, 0),
                 max_step=200, savedir="", explabel="")
                            #explabel="%s_%d" % (unit[1],unit[2]))
    exp.run()
    exp.visualize_best()
    exp.visualize_trajectory()
    exp.visualize_exp()