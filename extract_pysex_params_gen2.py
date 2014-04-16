import load_data
import pysex

import numpy as np

import multiprocessing as mp
import cPickle as pickle


"""
Extract a bunch of extra info to get a better idea of the size of objects
"""


SUBSETS = ['train', 'test']
TARGET_PATTERN = "data/pysex_params_gen2_%s.npy.gz"
SIGMA2 = 5000 # 5000 # std of the centrality weighting (Gaussian)
DETECT_THRESH = 2.0 # 10.0 # detection threshold for sextractor
NUM_PROCESSES = 8


def estimate_params(img):
    img_green = img[..., 1] # supposedly using the green channel is a good idea. alternatively we could use luma.
    # this seems to work well enough.

    out = pysex.run(img_green, params=[
            'X_IMAGE', 'Y_IMAGE', # barycenter
            # 'XMIN_IMAGE', 'XMAX_IMAGE', 'YMIN_IMAGE', 'YMAX_IMAGE', # enclosing rectangle
            # 'XPEAK_IMAGE', 'YPEAK_IMAGE', # location of maximal intensity
            'A_IMAGE', 'B_IMAGE', 'THETA_IMAGE', # ellipse parameters
            'PETRO_RADIUS',
            # 'KRON_RADIUS', 'PETRO_RADIUS', 'FLUX_RADIUS', 'FWHM_IMAGE', # various radii
        ], conf_args={ 'DETECT_THRESH': DETECT_THRESH })

    # x and y are flipped for some reason.
    # theta should be 90 - theta.
    # we convert these here so we can plot stuff with matplotlib easily.
    try:
        ys = out['X_IMAGE'].tonumpy()
        xs = out['Y_IMAGE'].tonumpy()
        as_ = out['A_IMAGE'].tonumpy()
        bs = out['B_IMAGE'].tonumpy()
        thetas = 90 - out['THETA_IMAGE'].tonumpy()
        # kron_radii = out['KRON_RADIUS'].tonumpy()
        petro_radii = out['PETRO_RADIUS'].tonumpy()
        # flux_radii = out['FLUX_RADIUS'].tonumpy()
        # fwhms = out['FWHM_IMAGE'].tonumpy()

        # detect the most salient galaxy
        # take in account size and centrality
        surface_areas = np.pi * (as_ * bs)
        centralities = np.exp(-((xs - 211.5)**2 + (ys - 211.5)**2)/SIGMA2) # 211.5, 211.5 is the center of the image

        # salience is proportional to surface area, with a gaussian prior on the distance to the center.
        saliences = surface_areas * centralities
        most_salient_idx = np.argmax(saliences)

        x = xs[most_salient_idx]
        y = ys[most_salient_idx]
        a = as_[most_salient_idx]
        b = bs[most_salient_idx]
        theta = thetas[most_salient_idx]
        # kron_radius = kron_radii[most_salient_idx]
        petro_radius = petro_radii[most_salient_idx]
        # flux_radius = flux_radii[most_salient_idx]
        # fwhm = fwhms[most_salient_idx]

    except TypeError: # sometimes these are empty (no objects found), use defaults in that case
        x = 211.5
        y = 211.5
        a = np.nan # dunno what this has to be, deal with it later
        b = np.nan # same
        theta = np.nan # same
        # kron_radius = np.nan
        petro_radius = np.nan
        # flux_radius = np.nan
        # fwhm = np.nan


    # return (x, y, a, b, theta, flux_radius, kron_radius, petro_radius, fwhm)
    return (x, y, a, b, theta, petro_radius)



for subset in SUBSETS:
    print "SUBSET: %s" % subset
    print

    if subset == 'train':
        num_images = load_data.num_train
        ids = load_data.train_ids
    elif subset == 'test':
        num_images = load_data.num_test
        ids = load_data.test_ids
    

    def process(k):
        print "image %d/%d (%s)" % (k + 1, num_images, subset)
        img_id = ids[k]
        img = load_data.load_image(img_id, from_ram=True, subset=subset)
        return estimate_params(img)

    pool = mp.Pool(NUM_PROCESSES)

    estimated_params = pool.map(process, xrange(num_images), chunksize=100)
    pool.close()
    pool.join()

    # estimated_params = map(process, xrange(num_images)) # no mp for debugging

    params_array = np.array(estimated_params)

    target_path = TARGET_PATTERN % subset
    print "Saving to %s..." % target_path
    load_data.save_gz(target_path, params_array)
