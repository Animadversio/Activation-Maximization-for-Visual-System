#%%
import h5py
import os
from os.path import join
from glob import glob
from scipy.io import loadmat
import torch
import numpy as np
import matplotlib.pylab as plt
import csv
from time import time
#%%
stimPath = r"N:\Stimuli\2019-Manifold\beto-191030a\backup_10_30_2019_10_15_31"
allfns = os.listdir(stimPath)
matfns = sorted(glob(join(stimPath, "*.mat")))  # [fn if ".mat" in fn else [] for fn in allfns]
imgfns = sorted(glob(join(stimPath, "*.jpg")))  # [fn if ".mat" in fn else [] for fn in allfns]
#%% Load the Codes mat file
data = loadmat(matfns[1])
codes = data['codes']
img_id = [arr[0] for arr in list(data['ids'][0])] # list of ids
#%%
from torch_net_utils import load_caffenet,load_generator, visualize
# net = load_caffenet()
Generator = load_generator()
#%%
rspPath = r"D:\Network_Data_Sync\Data-Ephys-MAT"#r"N:\Data-Ephys-MAT"
EphsFN = "Beto64chan-30102019-001"
Rspfns = sorted(glob(join(rspPath, EphsFN+"*")))
rspData = h5py.File(Rspfns[1])
spikeID = rspData['meta']['spikeID']
rsp = rspData['rasters']
#%%
prefchan_idx = np.nonzero(spikeID[0,:]==26)[0] - 1
prefrsp = rsp[:, :, prefchan_idx]  # Dataset reading takes time
scores = prefrsp[:, 50:, :].mean(axis=1) - prefrsp[:, :40, :].mean(axis=1)
#%%
vis_img = visualize(Generator, codes[3,:])
plt.imshow(vis_img)
plt.show()
#%%
# imgnmdata = loadmat(join(rspPath, EphsFN+"_imgName.mat")) # doesn't work so well
#%%
imgnms = [] # the one stored in *.mat file. Depleted of .jpg suffix
with open(join(rspPath, EphsFN+"_imgName.csv"), newline='\n') as csvfile:
    reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    for row in reader:
        imgnms.append(row[0])
# imgnmref = rspData['Trials']['imageName']
# https://docs.python.org/3/library/csv.html
#%% Find the rows for generated images.
gen_rows = ['gen' in fn and 'block' in fn and not fn[:2].isnumeric() for fn in imgnms]
nat_rows = [not i for i in gen_rows]
#%
gen_rows_idx = [i for i, b in enumerate(gen_rows) if b]
nat_rows_idx = [i for i, b in enumerate(nat_rows) if b]
gen_fns = [imgnms[i] for i, b in enumerate(gen_rows) if b]
nat_fns = [imgnms[i] for i, b in enumerate(nat_rows) if b]
#%%
import torchvision.models as models
# resnet18 = models.resnet18()
# vgg16 = models.vgg16()
alexnet = models.alexnet(pretrained=True)
conv4net = alexnet.features[0:9]
for param in conv4net.parameters():
    param.requires_grad = False
#%%
from torchvision import transforms
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])  # Note without normalization, the
denormalize = transforms.Normalize(
    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255],
    std=[1/0.229, 1/0.224, 1/0.255])
#%%
t0 = time()
Bnum = 10
out_feats_all = torch.tensor([], dtype=torch.float32)
idx_csr = 0
while True:
    nimg_B = torch.tensor([], dtype=torch.float32)
    for i in range(Bnum):
        if idx_csr == len(gen_fns):
            break
        cur_imgfn = [fn for fn in imgfns if gen_fns[idx_csr] in fn][0]  # fetch the file name in the Stimulus folder containing the
        img = plt.imread(cur_imgfn).astype(np.float32) / 255
        img = torch.from_numpy(img).permute(2, 0, 1)  # [C,H,W]
        nimg = normalize(img) #.permute([2,0,1])
        nimg_B = torch.cat((nimg_B, nimg.unsqueeze(0)), 0)
        idx_csr = idx_csr + 1
    out = conv4net(nimg_B)  # add 1 to first axis
    FeatTsrShape = out.shape[1:]
    out_feats_all = torch.cat((out_feats_all, out.reshape(out.shape[0], -1)), 0)
    if idx_csr == len(gen_fns):
        break
t1 = time()
print("%d s"% t1-t0) # 203.1430070400238 s for 3000 samples # 344s for larger batch
#%%
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
model = Lasso()
#%%
y = scores[gen_rows, 0]
model.fit(out_feats_all.numpy(), y)
fitY = model.predict(out_feats_all.numpy())
np.savez("LASSO_weights.npz", weights = model.coef_, bias = model.intercept_, L1=model.l1_ratio)
#%%
plt.figure(2)
plt.clf()
plt.plot(y)
plt.plot(fitY)
plt.show()
#%%
weightTsr = np.reshape(model.coef_,np.array(FeatTsrShape))
wightMap = np.abs(weightTsr).sum(axis=0)
#%%
figh = plt.figure(3)
plt.matshow(wightMap)
plt.colorbar()
figh.show()
#%%
import cv2
from torch.autograd import Variable
class SaveFeatures():
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)
    def hook_fn(self, module, input, output):
        self.features = torch.tensor(output,requires_grad=True).cuda()
    def close(self):
        self.hook.remove()

def val_tfms(img_np):
    img = torch.from_numpy(img_np.astype(np.float32)).permute(2, 0, 1)
    nimg = normalize(img).unsqueeze(0).cuda()
    return nimg

def val_detfms(img_tsr):
    img = denormalize(img_tsr.squeeze()).permute(1,2,0)
    return img

class FilterVisualizer():
    def __init__(self, size=56, upscaling_steps=12, upscaling_factor=1.2):
        self.size, self.upscaling_steps, self.upscaling_factor = size, upscaling_steps, upscaling_factor
        self.model = alexnet.features.cuda().eval()
        # set_trainable(self.model, False)
        for param in self.model.parameters():
            param.requires_grad = False

    def visualize(self, layer, filter, lr=0.1, opt_steps=20, blur=None):
        sz = self.size
        img = np.uint8(np.random.uniform(150, 180, (sz, sz, 3))) / 255  # generate random image
        activations = SaveFeatures(list(self.model.children())[layer])  # register hook

        for _ in range(self.upscaling_steps):  # scale the image up upscaling_steps times
            # train_tfms, val_tfms = tfms_from_model(vgg16, sz)
            img_var = Variable(val_tfms(img), requires_grad=True) # convert image to Variable that requires grad
            optimizer = torch.optim.Adam([img_var], lr=lr, weight_decay=1e-6)
            for n in range(opt_steps):  # optimize pixel values for opt_steps times
                optimizer.zero_grad()
                self.model(img_var)
                loss = -activations.features[0, filter].mean()
                loss.backward()
                optimizer.step()
            img = val_detfms(img_var.data.cpu()).numpy()
            self.output = img
            sz = int(self.upscaling_factor * sz)  # calculate new image size
            img = cv2.resize(img, (sz, sz), interpolation=cv2.INTER_CUBIC)  # scale image up
            if blur is not None: img = cv2.blur(img, (blur, blur))  # blur image to reduce high frequency patterns
        self.save(layer, filter)
        activations.close()

    def save(self, layer, filter):
        plt.imsave("layer_" + str(layer) + "_filter_" + str(filter) + ".jpg", np.clip(self.output, 0, 1))

#%%
FVis = FilterVisualizer(size=224, upscaling_steps=10, upscaling_factor=1.2)
FVis.visualize(8, 20, blur=5)