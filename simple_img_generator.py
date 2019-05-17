
from time import time, sleep
import os
import numpy as np
import h5py
from  scipy.io import loadmat
import utils
import os
from shutil import copy, copyfile
from utils import generator
from Optimizer import CMAES, Genetic, CholeskyCMAES
from cv2 import imread, imwrite

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
def copy_nature_image(natstimdir, expdir, size=None, reformat=False, prefix='', num_mark=False):
    '''Adapt from attach_nature_image code'''
    natstimfns = utils.sort_nicely([fn for fn in os.listdir(natstimdir) if
                                    '.bmp' in fn or '.jpg' in fn or '.png' in fn or '.jepg' in fn or '.JPEG' in fn])
    natstimnames = [fn[:fn.rfind('.')] for fn in natstimfns]
    nstimuli = len(natstimnames)
    if nstimuli == 0:
        raise Exception('no images found in natual stimulus directory %s' % natstimdir)
    if size is None:
        size = nstimuli
    else:
        size = int(size)

    # choose images to load (if nstimuli != size)
    toshow_args = np.arange(nstimuli)
    if nstimuli < size:
        print('note: number of natural images (%d) < requested (%d); repeating images'
              % (nstimuli, size))
        toshow_args = np.repeat(toshow_args, int(np.ceil(size / float(nstimuli))))[:size]
    elif nstimuli > size:
        print('note: number of natural images (%d) > requested (%d); taking first %d images'
              % (nstimuli, size, size))
        toshow_args = toshow_args[:size]
    natstimnames = [natstimnames[arg] for arg in toshow_args]
    natstimfns = [natstimfns[arg] for arg in toshow_args]
    # make ids
    natstimids = natstimnames[:]
    #   strip off initial [], if any
    for i, id_ in enumerate(natstimids):
        if id_[0] == '[' and ']' in id_:
            natstimids[i] = id_[id_.find(']') + 1:]
    #   resolve nonunique ids, if any
    if nstimuli < size:
        for i, id_ in enumerate(natstimids):
            icopy = 1
            new_id = id_
            while new_id in natstimids[:i]:
                new_id = '%s_copy%02d' % (id_, icopy)
            natstimids[i] = new_id
    print('showing the following %d natural stimuli loaded from %s:' % (size, natstimdir))
    print(natstimids)
    # if no image of other format, then use copy directly; if not, use reformat to make them 'bmp'
    if reformat:
        for fn, id in zip(natstimfns, natstimids):
            img = imread(os.path.join(natstimdir, fn))
            imwrite(os.path.join(expdir, prefix + id + '.bmp'), img)
    else:
        for fn, id in zip(natstimfns, natstimids):
            # copy(os.path.join(natstimdir, fn), expdir)
            copyfile(os.path.join(natstimdir, fn), os.path.join(expdir, prefix + id + '.bmp'))

def remove_bmp(dir_):
    catalog = os.listdir(dir_)
    for item in catalog:
        if item.endswith(".bmp"):
            os.remove(os.path.join(dir_, item))

#%%
homedir = os.path.expanduser('~')
initcodedir = os.path.join(homedir, 'Documents/stimuli/texture006')
initmat_path = '/home/poncelab/shared/backup/init_population/init_code.mat'
natstimdir = os.path.join(homedir, 'Documents/stimuli/natimages-guapoCh9')
respdir = '/home/poncelab/shared'  # in windows use r'\\128.252.37.224\shared'
expdir = respdir
backupdir = '/home/poncelab/shared/BACKUP_3'
for dir_ in (initcodedir, natstimdir):
    if not os.path.isdir(dir_):
        raise OSError('directory not found: %s' % dir_)
if not os.path.isfile(initmat_path):
    raise OSError('Initial code .mat file not found: %s' % initmat_path)
# check existence of file and direcrtory
for dir_ in (initcodedir, backupdir, respdir, expdir):
    if not os.path.isdir(dir_):
        os.mkdir(dir_)
#%%
# matfpath = '\\128.252.37.224\shared\block%03d.mat' % iblock
iblock = -1
# Add code to clear up images beforehand
while True:
    if iblock != -1:
        t0 = time()
        # wait for matfile
        matfn = 'block%03d_code.mat' % iblock
        matfpath = os.path.join(respdir, matfn)
        print('waiting for %s' % matfn)
        while not os.path.isfile(matfpath):
            sleep(0.001)
        remove_bmp(respdir)
        copy_nature_image(natstimdir, expdir)
        sleep(0.9)    # ensures mat file finish writing
        t1 = time()
        # load .mat file results
        imgid_list, codes = load_block_mat_code(matfpath)
    else:
        matfpath = initmat_path
        imgid_list, codes = load_block_mat_code(matfpath)
        imgid_list = ["gen_"+imgid for imgid in imgid_list]  # use gen as marker to distinguish from natural images
        copy_nature_image(natstimdir, expdir)  # , prefix='block000')
    iblock += 1
    # TODO Permutation and mixing can be added here ! But not needed!
    names = ['block%03d_' % (iblock) + id for id in imgid_list]
    # TODO More complex naming rule can be applied
    imgs = [generator.visualize(code) for code in codes]
    utils.write_images(imgs, names, respdir, format='bmp')  # use jpg to compress size, use bmp to speed up
    utils.write_images(imgs, names, backupdir, format='jpg')
    copy(matfpath, backupdir)


