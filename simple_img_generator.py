
from time import time, sleep
import os
import numpy as np
import h5py
from  scipy.io import loadmat
# import utils
# from utils import generator
#%%
def load_block_mat_code(matfpath):
    attempts = 0
    while True:
        try:
            data = loadmat(matfpath) # need the mat file to be saved in a older version
            codes = data['codes']
            ids = data['ids']
            imgids = []
            for id in ids[0]:
                imgids.append(id[0])
            return imgids, codes
        except (KeyError, IOError, OSError):    # if broken mat file or unable to access
            attempts += 1
            if attempts % 100 == 0:
                print('%d failed attempts to read .mat file' % attempts)
            sleep(0.001)
#%%

respdir = r'\\128.252.37.224\shared';
iblock = 0
t0 = time()
# wait for matf
matfn = 'block%03d_code.mat' % iblock
matfpath = os.path.join(respdir, matfn)

matfpath = r'\\128.252.37.224\shared\block%03d.mat' % iblock
print('waiting for %s' % matfn)
while not os.path.isfile(matfpath):
    sleep(0.001)
sleep(0.5)    # ensures mat file finish writing
t1 = time()
# load .mat file results
imgid_list, codes = load_block_mat_code(matfpath)
#%%
names = ['block%03d_' % (iblock) +id for id in imgid_list]
#%%
from Generator import Generator
generator = Generator()
imgs = [generator.visualize(code) for code in codes]
#%%
utils.write_images(imgs, names, respdir)
#%%
def write_block(self):
    assert self._images is not None, 'no images loaded'
    if self._blocksize is not None:
        blocksize = self._blocksize
    else:
        blocksize = len(self._images)

    self._iblock += 1
    self._iloop += 1

    view = self._random_generator.permutation(self._nimgs)
    prioritized_view = np.argsort(self._remaining_times_toshow[view])[::-1][:blocksize]
    block_images = self._images[view[prioritized_view]]
    block_imgids = self._imgids[view[prioritized_view]]
    block_ids = ['block%03d_%02d' % (self._iblock, i) for i in range(blocksize)]
    block_imgfns = ['%s_%s.bmp' % (blockid, imgid) for blockid, imgid in zip(block_ids, block_imgids)]
    imgfn_2_imgid = {name: imgid for name, imgid in zip(block_imgfns, block_imgids)}
    utils.write_images(block_images, block_imgfns, self._writedir, self._imsize)
    #  the order of `block_ids` is not the same as `imgid` !

    self._curr_block_imgfns = block_imgfns
    self._imgfn_2_imgid = imgfn_2_imgid
    self._remaining_times_toshow[view[prioritized_view]] -= 1
    return imgfn_2_imgid