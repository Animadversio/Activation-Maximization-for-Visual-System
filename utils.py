import os
import re
from time import time, sleep

import h5py
import numpy as np
from PIL import Image
from cv2 import imread, resize, INTER_CUBIC, INTER_AREA


def read_image(image_fpath):
    # BGR is flipped to RGB. why BGR?:
    #     Note In the case of color images, the decoded images will have the channels stored in B G R order.
    #     https://docs.opencv.org/2.4/modules/highgui/doc/reading_and_writing_images_and_video.html
    imarr = imread(image_fpath)[:, :, ::-1]
    return imarr


def write_images(imgs, names, path, size=None, timeout=0.5):
    """
    Saves images as 24-bit bmp files to given path with given names
    :param imgs: list of images as numpy arrays with shape (w, h, c) and dtype uint8
    :param names: filenames of images including or excluding '.bmp'
    :param path: path to save to
    :param size: size (pixels) to resize image to; default is unchanged
    :param timeout: timeout for trying to write each image
    :return: None
    """
    for im_arr, name in zip(imgs, names):
        if size is not None and im_arr.shape[1] != size:
            if im_arr.shape[1] < size:    # upsampling
                im_arr = resize(im_arr, (size, size), interpolation=INTER_CUBIC)
            else:                         # downsampling
                im_arr = resize(im_arr, (size, size), interpolation=INTER_AREA)
        img = Image.fromarray(im_arr)
        trying = True
        t0 = time()
        if name.rfind('.bmp') != len(name) - 4:
            name += '.bmp'
        while trying and time() - t0 < timeout:
            try:
                img.save(os.path.join(path, name))
                trying = False
            except IOError as e:
                if e.errno != 35:
                    raise
                sleep(0.01)


def write_codes(codes, names, path, timeout=0.5):
    """
    Saves codes as npy files (1 in each file) to given path with given names
    :param codes: list of images as numpy arrays with shape (w, h, c) and dtype uint8. NOTE only thing in a .npy file is a single code.
    :param names: filenames of images, excluding extension. number of names should be paired with codes.
    :param path: path to save to
    :param timeout: timeout for trying to write each code
    :return: None
    """
    for name, code in zip(names, codes):
        trying = True
        t0 = time()
        while trying and time() - t0 < timeout:
            try:
                np.save(os.path.join(path, name), code, allow_pickle=False)
                trying = False
    #         File "/Users/wuxiao/Documents/MCO/Rotations/Kreiman Lab/scripts/Playtest6/utils.py", line
    #         56, in write_codes
    #         np.save(os.path.join(path, name), code, allow_pickle=False)
    #     File "/usr/local/lib/python3.6/site-packages/numpy/lib/npyio.py", line 514, in save
    #         fid.close()
    #     OSError: [Errno 89] Operation canceled
            except (OSError, IOError) as e:
                if e.errno != 35 and e.errno != 89:
                    raise
                sleep(0.01)


def savez(fpath, save_kwargs, timeout=1):
    """
    wraps numpy.savez, implementing OSError tolerance within timeout
    "Save several arrays into a single file in uncompressed ``.npz`` format." DUMP EVERYTHING!
    """
    trying = True
    t0 = time()
    while trying and time() - t0 < timeout:
        try:
            np.savez(fpath, **save_kwargs)
        except IOError as e:
            if e.errno != 35:
                raise
            sleep(0.01)


save_scores = savez    # a synonym for backwards compatibility


def load_codes(codedir, size):
    """ load all the *.npy files in the `codedir`. and randomly sample # `size` of them.
    make sure enough codes for requested size
    """
    codefns = sorted([fn for fn in os.listdir(codedir) if '.npy' in fn])
    assert size <= len(codefns), 'not enough codes (%d) to satisfy size (%d)' % (len(codefns), size)
    # load codes
    codes = []
    for codefn in np.random.choice(codefns, size=min(len(codefns), size), replace=False):
        code = np.load(os.path.join(codedir, codefn), allow_pickle=False).flatten()
        codes.append(code)
    codes = np.array(codes)
    return codes


def load_codes2(codedir, size):
    """ unlike load_codes, also returns name of load """
    # make sure enough codes for requested size
    codefns = sorted([fn for fn in os.listdir(codedir) if '.npy' in fn])
    assert size <= len(codefns), 'not enough codes (%d) to satisfy size (%d)' % (len(codefns), size)
    # load codes
    codefns = list(np.random.choice(codefns, size=min(len(codefns), size), replace=False))
    codes = []
    for codefn in codefns:
        code = np.load(os.path.join(codedir, codefn), allow_pickle=False).flatten()
        codes.append(code)
    codes = np.array(codes)
    return codes, codefns


def load_codes_search(codedir, srckey, size=None):
    """Load the code files with `srckey` in its name.

    :param codedir:
    :param srckey: keyword to identify / filter the code. e.g. "gen298_010760.npy", "gen298_010760", "gen298"
    :param size: Defaultly None. if there is too many codes, one can use this to specify the sample size
    :return: codes and corresponding file names `codes, codefns`
    Added @sep.19
    """
    # make sure enough codes for requested size
    codefns = sorted([fn for fn in os.listdir(codedir) if ('.npy' in fn) and (srckey in fn)])

    if not size is None: # input size parameter indicates to select the codes.
        assert size <= len(codefns), 'not enough codes (%d) to satisfy size (%d)' % (len(codefns), size)
        codefns = list(np.random.choice(codefns, size=min(len(codefns), size), replace=False))

    # load codes by the codefns
    codes = []
    for codefn in codefns:
        code = np.load(os.path.join(codedir, codefn), allow_pickle=False).flatten()
        codes.append(code)
    codes = np.array(codes)
    return codes, codefns


def load_block_mat(matfpath):
    attempts = 0
    while True:
        try:
            with h5py.File(matfpath, 'r') as f:
                imgids_refs = np.array(f['stimulusID'])[0]
                imgids = []
                for ref in imgids_refs:
                    imgpath = ''.join(chr(i) for i in f[ref])
                    imgids.append(imgpath.split('\\')[-1])
                imgids = np.array(imgids)
                scores = np.array(f['tEvokedResp'])    # shape = (imgs, channels)
            return imgids, scores
        except (KeyError, IOError, OSError):    # if broken mat file or unable to access
            attempts += 1
            if attempts % 100 == 0:
                print('%d failed attempts to read .mat file' % attempts)
            sleep(0.001)


def set_dynamic_parameters_by_file(fpath, dynamic_parameters):
    try:
        with open(fpath, 'r') as file:
            line = 'placeholder'
            while len(line) > 0:
                line = file.readline()
                if ':' not in line:
                    continue
                if '#' in line:
                    line = line[:line.find('#')]
                if len(line.split(':')) != 2:
                    continue
                key, val = line.split(':')
                key = key.strip()
                val = val.strip()
                try:
                    # if key is not in dynamic_parameter.keys(), will throw KeyError
                    # if val (a str literal) cannot be converted to dynamic_parameter.type, will throw ValueError
                    dynamic_parameters[key].set_value(val)
                except (KeyError, ValueError):
                    continue
    except IOError:
        print('cannot open dynamic parameters file %s' % fpath)


def write_dynamic_parameters_to_file(fpath, dynamic_parameters):
    with open(fpath, 'w') as file:
        for key in sorted(list(dynamic_parameters.keys())):
            file.write('%s:\t%s\t# %s\n' % (key, str(dynamic_parameters[key].value), dynamic_parameters[key].description))


# https://nedbatchelder.com/blog/200712/human_sorting.html
def tryint(s):
    try:
        return int(s)
    except ValueError:
        return s


def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    return [tryint(c) for c in re.split('([0-9]+)', s)]


def sort_nicely(l):
    """ Return the given list sorted in the way that humans expect.
    """
    newl = l[:]
    newl.sort(key=alphanum_key)
    return newl

import matplotlib.pyplot as plt


def visualize_score_trajectory(CurDataDir, steps=300, population_size=40, title_str=""):
    ScoreEvolveTable = np.full((steps, population_size,), np.NAN)
    for stepi in range(steps):
        try:
            with np.load(CurDataDir + "scores_end_block{0:03}.npz".format(stepi)) as data:
                score_tmp = data['scores']
                ScoreEvolveTable[stepi, :len(score_tmp)] = score_tmp
        except FileNotFoundError:
            print("maximum steps is %d." % stepi)
            ScoreEvolveTable = ScoreEvolveTable[0:stepi, :]
            steps = stepi
            break;

    gen_slice = np.arange(steps).reshape((-1, 1))
    gen_num = np.repeat(gen_slice, population_size, 1)

    AvgScore = np.nanmean(ScoreEvolveTable, axis=1)
    MaxScore = np.nanmax(ScoreEvolveTable, axis=1)

    figh = plt.figure()
    plt.scatter(gen_num, ScoreEvolveTable, s=16, alpha=0.6, label="all score")
    plt.plot(gen_slice, AvgScore, color='black', label="Average score")
    plt.plot(gen_slice, MaxScore, color='red', label="Max score")
    plt.xlabel("generation #")
    plt.ylabel("CNN unit score")
    plt.title("Optimization Trajectory of Score\n" + title_str)
    plt.legend()
    plt.show()
    return figh