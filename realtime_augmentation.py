"""
Generator that augments data in realtime
"""

import numpy as np
import skimage
import multiprocessing as mp
import time

import load_data


NUM_PROCESSES = 6
CHUNK_SIZE = 25000


IMAGE_WIDTH = 424
IMAGE_HEIGHT = 424
IMAGE_NUM_CHANNELS = 3


y_train = np.load("data/solutions_train.npy")

# split training data into training + a small validation set
num_train = y_train.shape[0]

num_valid = num_train // 10 # integer division
num_train -= num_valid

y_valid = y_train[num_train:]
y_train = y_train[:num_train]

valid_ids = load_data.train_ids[num_train:]
train_ids = load_data.train_ids[:num_train]



## UTILITIES ##

def select_indices(num, num_selected):
    selected_indices = np.arange(num)
    np.random.shuffle(selected_indices)
    selected_indices = selected_indices[:num_selected]
    return selected_indices


def fast_warp(img, tf, output_shape=(53,53), mode='reflect'):
    """
    This wrapper function is about five times faster than skimage.transform.warp, for our use case.
    """
    m = tf._matrix
    img_wf = np.empty((output_shape[0], output_shape[1], 3), dtype='float32')
    for k in xrange(3):
        img_wf[..., k] = skimage.transform._warps_cy._warp_fast(img[..., k], m, output_shape=output_shape, mode=mode)
    return img_wf




## TRANSFORMATIONS ##

center_shift = np.array((IMAGE_HEIGHT, IMAGE_WIDTH)) / 2. - 0.5
tform_center = skimage.transform.SimilarityTransform(translation=-center_shift)
tform_uncenter = skimage.transform.SimilarityTransform(translation=center_shift)

def build_augmentation_transform(zoom=1.0, rotation=0, shear=0, translation=(0, 0)):
    tform_augment = skimage.transform.AffineTransform(scale=(1/zoom, 1/zoom), rotation=np.deg2rad(rotation), shear=np.deg2rad(shear), translation=translation)
    tform = tform_center + tform_augment + tform_uncenter # shift to center, augment, shift back (for the rotation/shearing)
    return tform

def build_ds_transform_old(ds_factor=1.0, target_size=(53, 53)):
    tform_ds = skimage.transform.SimilarityTransform(scale=ds_factor)
    shift_x = IMAGE_WIDTH / (2.0 * ds_factor) - target_size[0] / 2.0
    shift_y = IMAGE_HEIGHT / (2.0 * ds_factor) - target_size[1] / 2.0
    tform_shift_ds = skimage.transform.SimilarityTransform(translation=(shift_x, shift_y))
    return tform_shift_ds + tform_ds


def build_ds_transform(ds_factor=1.0, orig_size=(424, 424), target_size=(53, 53), do_shift=True, subpixel_shift=False):
    """
    This version is a bit more 'correct', it mimics the skimage.transform.resize function.
    """
    rows, cols = orig_size
    trows, tcols = target_size
    col_scale = row_scale = ds_factor
    src_corners = np.array([[1, 1], [1, rows], [cols, rows]]) - 1
    dst_corners = np.zeros(src_corners.shape, dtype=np.double)
    # take into account that 0th pixel is at position (0.5, 0.5)
    dst_corners[:, 0] = col_scale * (src_corners[:, 0] + 0.5) - 0.5
    dst_corners[:, 1] = row_scale * (src_corners[:, 1] + 0.5) - 0.5

    tform_ds = skimage.transform.AffineTransform()
    tform_ds.estimate(src_corners, dst_corners)

    if do_shift:
        if subpixel_shift: 
            # if this is true, we add an additional 'arbitrary' subpixel shift, which 'aligns'
            # the grid of the target image with the original image in such a way that the interpolation
            # is 'cleaner', i.e. groups of <ds_factor> pixels in the original image will map to
            # individual pixels in the resulting image.
            #
            # without this additional shift, and when the downsampling factor does not divide the image
            # size (like in the case of 424 and 3.0 for example), the grids will not be aligned, resulting
            # in 'smoother' looking images that lose more high frequency information.
            #
            # technically this additional shift is not 'correct' (we're not looking at the very center
            # of the image anymore), but it's always less than a pixel so it's not a big deal.
            #
            # in practice, we implement the subpixel shift by rounding down the orig_size to the
            # nearest multiple of the ds_factor. Of course, this only makes sense if the ds_factor
            # is an integer.

            cols = (cols // int(ds_factor)) * int(ds_factor)
            rows = (rows // int(ds_factor)) * int(ds_factor)
            # print "NEW ROWS, COLS: (%d,%d)" % (rows, cols)


        shift_x = cols / (2 * ds_factor) - tcols / 2.0
        shift_y = rows / (2 * ds_factor) - trows / 2.0
        tform_shift_ds = skimage.transform.SimilarityTransform(translation=(shift_x, shift_y))
        return tform_shift_ds + tform_ds
    else:
        return tform_ds



def random_perturbation_transform(zoom_range, rotation_range, shear_range, translation_range, do_flip=False):
    # random shift [-4, 4] - shift no longer needs to be integer!
    shift_x = np.random.uniform(*translation_range)
    shift_y = np.random.uniform(*translation_range)
    translation = (shift_x, shift_y)

    # random rotation [0, 360]
    rotation = np.random.uniform(*rotation_range) # there is no post-augmentation, so full rotations here!

    # random shear [0, 5]
    shear = np.random.uniform(*shear_range)

    # # flip
    if do_flip and (np.random.randint(2) > 0): # flip half of the time
        shear += 180
        rotation += 180
        # shear by 180 degrees is equivalent to rotation by 180 degrees + flip.
        # So after that we rotate it another 180 degrees to get just the flip.

    # random zoom [0.9, 1.1]
    # zoom = np.random.uniform(*zoom_range)
    log_zoom_range = [np.log(z) for z in zoom_range]
    zoom = np.exp(np.random.uniform(*log_zoom_range)) # for a zoom factor this sampling approach makes more sense.
    # the range should be multiplicatively symmetric, so [1/1.1, 1.1] instead of [0.9, 1.1] makes more sense.

    return build_augmentation_transform(zoom, rotation, shear, translation)


def perturb_and_dscrop(img, ds_transforms, augmentation_params, target_sizes=None):
    if target_sizes is None: # default to (53,53) for backwards compatibility
        target_sizes = [(53, 53) for _ in xrange(len(ds_transforms))]

    tform_augment = random_perturbation_transform(**augmentation_params)
    # return [skimage.transform.warp(img, tform_ds + tform_augment, output_shape=target_size, mode='reflect').astype('float32') for tform_ds in ds_transforms]

    result = []
    for tform_ds, target_size in zip(ds_transforms, target_sizes):
        result.append(fast_warp(img, tform_ds + tform_augment, output_shape=target_size, mode='reflect').astype('float32'))

    return result




tform_ds_8x = build_ds_transform(8.0, target_size=(53, 53))
tform_ds_cropped33 = build_ds_transform(3.0, target_size=(53, 53))
tform_ds_cc = build_ds_transform(1.0, target_size=(53, 53))

tform_identity = skimage.transform.AffineTransform() # this is an identity transform by default


ds_transforms_default = [tform_ds_cropped33, tform_ds_8x]
ds_transforms_381 = [tform_ds_cropped33, tform_ds_8x, tform_ds_cc]

ds_transforms = ds_transforms_default # CHANGE THIS LINE to select downsampling transforms to be used

## REALTIME AUGMENTATION GENERATOR ##

def load_and_process_image(img_index, ds_transforms, augmentation_params, target_sizes=None):
    # start_time = time.time()
    img_id = load_data.train_ids[img_index]
    img = load_data.load_image(img_id, from_ram=True)
    # load_time = (time.time() - start_time) * 1000
    # start_time = time.time()
    img_a = perturb_and_dscrop(img, ds_transforms, augmentation_params, target_sizes)
    # augment_time = (time.time() - start_time) * 1000
    # print "load: %.2f ms\taugment: %.2f ms" % (load_time, augment_time)
    return img_a


class LoadAndProcess(object):
    """
    UGLY HACK:

    pool.imap does not allow for extra arguments to be passed to the called function.
    This is a problem because we want to pass in the augmentation parameters.
    As a workaround, we could use a lambda or a locally defined function, but that
    doesn't work, because it cannot be pickled properly.

    The solution is to use a callable object instead, which is picklable.
    """
    def __init__(self, ds_transforms, augmentation_params, target_sizes=None):
        self.ds_transforms = ds_transforms
        self.augmentation_params = augmentation_params
        self.target_sizes = target_sizes

    def __call__(self, img_index):
        return load_and_process_image(img_index, self.ds_transforms, self.augmentation_params, self.target_sizes)


default_augmentation_params = {
    'zoom_range': (1.0, 1.1),
    'rotation_range': (0, 360),
    'shear_range': (0, 0),
    'translation_range': (-4, 4),
}


def realtime_augmented_data_gen(num_chunks=None, chunk_size=CHUNK_SIZE, augmentation_params=default_augmentation_params,
                                ds_transforms=ds_transforms_default, target_sizes=None, processor_class=LoadAndProcess):
    """
    new version, using Pool.imap instead of Pool.map, to avoid the data structure conversion
    from lists to numpy arrays afterwards.
    """
    if target_sizes is None: # default to (53,53) for backwards compatibility
        target_sizes = [(53, 53) for _ in xrange(len(ds_transforms))]

    n = 0 # number of chunks yielded so far
    while True:
        if num_chunks is not None and n >= num_chunks:
            # print "DEBUG: DATA GENERATION COMPLETED"
            break

        # start_time = time.time()
        selected_indices = select_indices(num_train, chunk_size)
        labels = y_train[selected_indices]

        process_func = processor_class(ds_transforms, augmentation_params, target_sizes)

        target_arrays = [np.empty((chunk_size, size_x, size_y, 3), dtype='float32') for size_x, size_y in target_sizes]
        pool = mp.Pool(NUM_PROCESSES)
        gen = pool.imap(process_func, selected_indices, chunksize=100) # lower chunksize seems to help to keep memory usage in check

        for k, imgs in enumerate(gen):
            # print ">>> converting data: %d" % k
            for i, img in enumerate(imgs):
                target_arrays[i][k] = img

        pool.close()
        pool.join()

        # TODO: optionally do post-augmentation here

        target_arrays.append(labels)

        # duration = time.time() - start_time
        # print "chunk generation took %.2f seconds" % duration

        yield target_arrays, chunk_size

        n += 1





### Fixed test-time augmentation ####


def augment_fixed_and_dscrop(img, ds_transforms, augmentation_transforms, target_sizes=None):
    if target_sizes is None: # default to (53,53) for backwards compatibility
        target_sizes = [(53, 53) for _ in xrange(len(ds_transforms))]

    augmentations_list = []
    for tform_augment in augmentation_transforms:
        augmentation = [fast_warp(img, tform_ds + tform_augment, output_shape=target_size, mode='reflect').astype('float32') for tform_ds, target_size in zip(ds_transforms, target_sizes)]
        augmentations_list.append(augmentation)

    return augmentations_list


def load_and_process_image_fixed(img_index, subset, ds_transforms, augmentation_transforms, target_sizes=None):
    if subset == 'train':
        img_id = load_data.train_ids[img_index]
    elif subset == 'test':
        img_id = load_data.test_ids[img_index]

    img = load_data.load_image(img_id, from_ram=True, subset=subset)
    img_a = augment_fixed_and_dscrop(img, ds_transforms, augmentation_transforms, target_sizes)
    return img_a


class LoadAndProcessFixed(object):
    """
    Same ugly hack as before
    """
    def __init__(self, subset, ds_transforms, augmentation_transforms, target_sizes=None):
        self.subset = subset
        self.ds_transforms = ds_transforms
        self.augmentation_transforms = augmentation_transforms
        self.target_sizes = target_sizes

    def __call__(self, img_index):
        return load_and_process_image_fixed(img_index, self.subset, self.ds_transforms, self.augmentation_transforms, self.target_sizes)




def realtime_fixed_augmented_data_gen(selected_indices, subset, ds_transforms=ds_transforms_default, augmentation_transforms=[tform_identity],
                                        chunk_size=CHUNK_SIZE, target_sizes=None, processor_class=LoadAndProcessFixed):
    """
    by default, only the identity transform is in the augmentation list, so no augmentation occurs (only ds_transforms are applied).
    """
    num_ids_per_chunk = (chunk_size // len(augmentation_transforms)) # number of datapoints per chunk - each datapoint is multiple entries!
    num_chunks = int(np.ceil(len(selected_indices) / float(num_ids_per_chunk)))

    if target_sizes is None: # default to (53,53) for backwards compatibility
        target_sizes = [(53, 53) for _ in xrange(len(ds_transforms))]

    process_func = processor_class(subset, ds_transforms, augmentation_transforms, target_sizes)

    for n in xrange(num_chunks):
        indices_n = selected_indices[n * num_ids_per_chunk:(n+1) * num_ids_per_chunk]
        current_chunk_size = len(indices_n) * len(augmentation_transforms) # last chunk will be shorter!

        target_arrays = [np.empty((chunk_size, size_x, size_y, 3), dtype='float32') for size_x, size_y in target_sizes]

        pool = mp.Pool(NUM_PROCESSES)
        gen = pool.imap(process_func, indices_n, chunksize=100) # lower chunksize seems to help to keep memory usage in check

        for k, imgs_aug in enumerate(gen):
            for j, imgs in enumerate(imgs_aug):
                for i, img in enumerate(imgs):
                    idx = k * len(augmentation_transforms) + j # put all augmented versions of the same datapoint after each other
                    target_arrays[i][idx] = img

        pool.close()
        pool.join()

        yield target_arrays, current_chunk_size


def post_augment_chunks(chunk_list, gamma_range=(1.0, 1.0)):
    """
    post augmentation MODIFIES THE CHUNKS IN CHUNK_LIST IN PLACE to save memory where possible!

    post_augmentation_params:
        - gamma_range: range of the gamma correction exponents
    """
    chunk_size = chunk_list[0].shape[0]

    if gamma_range != (1.0, 1.0):
        gamma_min, gamma_max = gamma_range
        lgamma_min = np.log(gamma_min)
        lgamma_max = np.log(gamma_max)
        gammas = np.exp(np.random.uniform(lgamma_min, lgamma_max, (chunk_size,)))
        gammas = gammas.astype('float32').reshape(-1, 1, 1, 1)

        for i in xrange(len(chunk_list)):
            chunk_list[i] **= gammas



def post_augment_gen(data_gen, post_augmentation_params):
    for target_arrays, chunk_size in data_gen:
        # start_time = time.time()
        post_augment_chunks(target_arrays[:-1], **post_augmentation_params)
        # print "post augmentation took %.4f seconds" % (time.time() - start_time)
        # target_arrays[:-1], don't augment the labels!
        yield target_arrays, chunk_size



colour_channel_weights = np.array([-0.0148366, -0.01253134, -0.01040762], dtype='float32')


def post_augment_brightness_gen(data_gen, std=0.5):
    for target_arrays, chunk_size in data_gen:
        labels = target_arrays.pop()
        
        stds = np.random.randn(chunk_size).astype('float32') * std
        noise = stds[:, None] * colour_channel_weights[None, :]

        target_arrays = [np.clip(t + noise[:, None, None, :], 0, 1) for t in target_arrays]
        target_arrays.append(labels)

        yield target_arrays, chunk_size



def post_augment_gaussian_noise_gen(data_gen, std=0.1):
    """
    Adds gaussian noise. Note that this is not entirely correct, the correct way would be to do it
    before downsampling, so the regular image and the rot45 image have the same noise pattern.
    But this is easier.
    """
    for target_arrays, chunk_size in data_gen:
        labels = target_arrays.pop()
        
        noise = np.random.randn(*target_arrays[0].shape).astype('float32') * std

        target_arrays = [np.clip(t + noise, 0, 1) for t in target_arrays]
        target_arrays.append(labels)

        yield target_arrays, chunk_size


def post_augment_gaussian_noise_gen_separate(data_gen, std=0.1):
    """
    Adds gaussian noise. Note that this is not entirely correct, the correct way would be to do it
    before downsampling, so the regular image and the rot45 image have the same noise pattern.
    But this is easier.

    This one generates separate noise for the different channels but is a lot slower
    """
    for target_arrays, chunk_size in data_gen:
        labels = target_arrays.pop()

        new_target_arrays = []

        for target_array in target_arrays:
            noise = np.random.randn(*target_array.shape).astype('float32') * std
            new_target_arrays.append(np.clip(target_array + noise, 0, 1))

        new_target_arrays.append(labels)

        yield new_target_arrays, chunk_size


### Alternative image loader and processor which does pysex centering

# pysex_params_train = load_data.load_gz("data/pysex_params_extra_train.npy.gz")
# pysex_params_test = load_data.load_gz("data/pysex_params_extra_test.npy.gz")


pysex_params_train = load_data.load_gz("data/pysex_params_gen2_train.npy.gz")
pysex_params_test = load_data.load_gz("data/pysex_params_gen2_test.npy.gz")

pysexgen1_params_train = load_data.load_gz("data/pysex_params_extra_train.npy.gz")
pysexgen1_params_test = load_data.load_gz("data/pysex_params_extra_test.npy.gz")


center_x, center_y = (IMAGE_WIDTH - 1) / 2.0, (IMAGE_HEIGHT - 1) / 2.0

# def build_pysex_center_transform(img_index, subset='train'):
#     if subset == 'train':
#         x, y, a, b, theta, flux_radius, kron_radius, petro_radius, fwhm = pysex_params_train[img_index]
#     elif subset == 'test':
#         x, y, a, b, theta, flux_radius, kron_radius, petro_radius, fwhm = pysex_params_test[img_index]

#     return build_augmentation_transform(translation=(x - center_x, y - center_y))  


# def build_pysex_center_rescale_transform(img_index, subset='train', target_radius=170.0): # target_radius=160.0):
#     if subset == 'train':
#         x, y, a, b, theta, flux_radius, kron_radius, petro_radius, fwhm = pysex_params_train[img_index]
#     elif subset == 'test':
#         x, y, a, b, theta, flux_radius, kron_radius, petro_radius, fwhm = pysex_params_test[img_index]
    
#     scale_factor_limit = 1.5 # scale up / down by this fraction at most

#     scale_factor = target_radius / (petro_radius * a) # magic constant, might need some tuning

#     if np.isnan(scale_factor):
#         scale_factor = 1.0 # no info
    
#     scale_factor = max(min(scale_factor, scale_factor_limit), 1.0 / scale_factor_limit) # truncate for edge cases

#     return build_augmentation_transform(translation=(x - center_x, y - center_y), zoom=scale_factor)  


def build_pysex_center_transform(img_index, subset='train'):
    if subset == 'train':
        x, y, a, b, theta, petro_radius = pysex_params_train[img_index]
    elif subset == 'test':
        x, y, a, b, theta, petro_radius = pysex_params_test[img_index]

    return build_augmentation_transform(translation=(x - center_x, y - center_y))  


def build_pysex_center_rescale_transform(img_index, subset='train', target_radius=160.0):
    if subset == 'train':
        x, y, a, b, theta, petro_radius = pysex_params_train[img_index]
    elif subset == 'test':
        x, y, a, b, theta, petro_radius = pysex_params_test[img_index]
    
    scale_factor_limit = 1.5 # scale up / down by this fraction at most

    scale_factor = target_radius / (petro_radius * a) # magic constant, might need some tuning

    if np.isnan(scale_factor):
        scale_factor = 1.0 # no info
    
    scale_factor = max(min(scale_factor, scale_factor_limit), 1.0 / scale_factor_limit) # truncate for edge cases

    return build_augmentation_transform(translation=(x - center_x, y - center_y), zoom=scale_factor)  



def build_pysexgen1_center_rescale_transform(img_index, subset='train', target_radius=160.0):
    if subset == 'train':
        x, y, a, b, theta, flux_radius, kron_radius, petro_radius, fwhm = pysexgen1_params_train[img_index]
    elif subset == 'test':
        x, y, a, b, theta, flux_radius, kron_radius, petro_radius, fwhm = pysexgen1_params_test[img_index]
    
    scale_factor_limit = 1.5 # scale up / down by this fraction at most

    scale_factor = target_radius / (petro_radius * a) # magic constant, might need some tuning

    if np.isnan(scale_factor):
        scale_factor = 1.0 # no info
    
    scale_factor = max(min(scale_factor, scale_factor_limit), 1.0 / scale_factor_limit) # truncate for edge cases

    return build_augmentation_transform(translation=(x - center_x, y - center_y), zoom=scale_factor)  




def perturb_and_dscrop_with_prepro(img, ds_transforms, augmentation_params, target_sizes=None, prepro_transform=tform_identity):
    """
    This version supports a preprocessing transform which is applied before anything else
    """
    if target_sizes is None: # default to (53,53) for backwards compatibility
        target_sizes = [(53, 53) for _ in xrange(len(ds_transforms))]

    tform_augment = random_perturbation_transform(**augmentation_params)
    # return [skimage.transform.warp(img, tform_ds + tform_augment, output_shape=target_size, mode='reflect').astype('float32') for tform_ds in ds_transforms]

    result = []
    for tform_ds, target_size in zip(ds_transforms, target_sizes):
        result.append(fast_warp(img, tform_ds + tform_augment + prepro_transform, output_shape=target_size, mode='reflect').astype('float32'))

    return result


def load_and_process_image_pysex_centering(img_index, ds_transforms, augmentation_params, target_sizes=None):
    # start_time = time.time()
    img_id = load_data.train_ids[img_index]
    img = load_data.load_image(img_id, from_ram=True)
    # load_time = (time.time() - start_time) * 1000
    # start_time = time.time()
    tf_center = build_pysex_center_transform(img_index)

    img_a = perturb_and_dscrop_with_prepro(img, ds_transforms, augmentation_params, target_sizes, prepro_transform=tf_center)
    # augment_time = (time.time() - start_time) * 1000
    # print "load: %.2f ms\taugment: %.2f ms" % (load_time, augment_time)
    return img_a


class LoadAndProcessPysexCentering(object):
    def __init__(self, ds_transforms, augmentation_params, target_sizes=None):
        self.ds_transforms = ds_transforms
        self.augmentation_params = augmentation_params
        self.target_sizes = target_sizes

    def __call__(self, img_index):
        return load_and_process_image_pysex_centering(img_index, self.ds_transforms, self.augmentation_params, self.target_sizes)


def load_and_process_image_pysex_centering_rescaling(img_index, ds_transforms, augmentation_params, target_sizes=None):
    # start_time = time.time()
    img_id = load_data.train_ids[img_index]
    img = load_data.load_image(img_id, from_ram=True)
    # load_time = (time.time() - start_time) * 1000
    # start_time = time.time()
    tf_center_rescale = build_pysex_center_rescale_transform(img_index)

    img_a = perturb_and_dscrop_with_prepro(img, ds_transforms, augmentation_params, target_sizes, prepro_transform=tf_center_rescale)
    # augment_time = (time.time() - start_time) * 1000
    # print "load: %.2f ms\taugment: %.2f ms" % (load_time, augment_time)
    return img_a


class LoadAndProcessPysexCenteringRescaling(object):
    def __init__(self, ds_transforms, augmentation_params, target_sizes=None):
        self.ds_transforms = ds_transforms
        self.augmentation_params = augmentation_params
        self.target_sizes = target_sizes

    def __call__(self, img_index):
        return load_and_process_image_pysex_centering_rescaling(img_index, self.ds_transforms, self.augmentation_params, self.target_sizes)


def load_and_process_image_pysexgen1_centering_rescaling(img_index, ds_transforms, augmentation_params, target_sizes=None):
    # start_time = time.time()
    img_id = load_data.train_ids[img_index]
    img = load_data.load_image(img_id, from_ram=True)
    # load_time = (time.time() - start_time) * 1000
    # start_time = time.time()
    tf_center_rescale = build_pysexgen1_center_rescale_transform(img_index)

    img_a = perturb_and_dscrop_with_prepro(img, ds_transforms, augmentation_params, target_sizes, prepro_transform=tf_center_rescale)
    # augment_time = (time.time() - start_time) * 1000
    # print "load: %.2f ms\taugment: %.2f ms" % (load_time, augment_time)
    return img_a


class LoadAndProcessPysexGen1CenteringRescaling(object):
    def __init__(self, ds_transforms, augmentation_params, target_sizes=None):
        self.ds_transforms = ds_transforms
        self.augmentation_params = augmentation_params
        self.target_sizes = target_sizes

    def __call__(self, img_index):
        return load_and_process_image_pysexgen1_centering_rescaling(img_index, self.ds_transforms, self.augmentation_params, self.target_sizes)







def augment_fixed_and_dscrop_with_prepro(img, ds_transforms, augmentation_transforms, target_sizes=None, prepro_transform=tform_identity):
    if target_sizes is None: # default to (53,53) for backwards compatibility
        target_sizes = [(53, 53) for _ in xrange(len(ds_transforms))]

    augmentations_list = []
    for tform_augment in augmentation_transforms:
        augmentation = [fast_warp(img, tform_ds + tform_augment + prepro_transform, output_shape=target_size, mode='reflect').astype('float32') for tform_ds, target_size in zip(ds_transforms, target_sizes)]
        augmentations_list.append(augmentation)

    return augmentations_list


def load_and_process_image_fixed_pysex_centering(img_index, subset, ds_transforms, augmentation_transforms, target_sizes=None):
    if subset == 'train':
        img_id = load_data.train_ids[img_index]
    elif subset == 'test':
        img_id = load_data.test_ids[img_index]

    tf_center = build_pysex_center_transform(img_index, subset)
    
    img = load_data.load_image(img_id, from_ram=True, subset=subset)
    img_a = augment_fixed_and_dscrop_with_prepro(img, ds_transforms, augmentation_transforms, target_sizes, prepro_transform=tf_center)
    return img_a


class LoadAndProcessFixedPysexCentering(object):
    """
    Same ugly hack as before
    """
    def __init__(self, subset, ds_transforms, augmentation_transforms, target_sizes=None):
        self.subset = subset
        self.ds_transforms = ds_transforms
        self.augmentation_transforms = augmentation_transforms
        self.target_sizes = target_sizes

    def __call__(self, img_index):
        return load_and_process_image_fixed_pysex_centering(img_index, self.subset, self.ds_transforms, self.augmentation_transforms, self.target_sizes)



def load_and_process_image_fixed_pysex_centering_rescaling(img_index, subset, ds_transforms, augmentation_transforms, target_sizes=None):
    if subset == 'train':
        img_id = load_data.train_ids[img_index]
    elif subset == 'test':
        img_id = load_data.test_ids[img_index]

    tf_center_rescale = build_pysex_center_rescale_transform(img_index, subset)
    
    img = load_data.load_image(img_id, from_ram=True, subset=subset)
    img_a = augment_fixed_and_dscrop_with_prepro(img, ds_transforms, augmentation_transforms, target_sizes, prepro_transform=tf_center_rescale)
    return img_a


class LoadAndProcessFixedPysexCenteringRescaling(object):
    """
    Same ugly hack as before
    """
    def __init__(self, subset, ds_transforms, augmentation_transforms, target_sizes=None):
        self.subset = subset
        self.ds_transforms = ds_transforms
        self.augmentation_transforms = augmentation_transforms
        self.target_sizes = target_sizes

    def __call__(self, img_index):
        return load_and_process_image_fixed_pysex_centering_rescaling(img_index, self.subset, self.ds_transforms, self.augmentation_transforms, self.target_sizes)



def load_and_process_image_fixed_pysexgen1_centering_rescaling(img_index, subset, ds_transforms, augmentation_transforms, target_sizes=None):
    if subset == 'train':
        img_id = load_data.train_ids[img_index]
    elif subset == 'test':
        img_id = load_data.test_ids[img_index]

    tf_center_rescale = build_pysexgen1_center_rescale_transform(img_index, subset)
    
    img = load_data.load_image(img_id, from_ram=True, subset=subset)
    img_a = augment_fixed_and_dscrop_with_prepro(img, ds_transforms, augmentation_transforms, target_sizes, prepro_transform=tf_center_rescale)
    return img_a


class LoadAndProcessFixedPysexGen1CenteringRescaling(object):
    """
    Same ugly hack as before
    """
    def __init__(self, subset, ds_transforms, augmentation_transforms, target_sizes=None):
        self.subset = subset
        self.ds_transforms = ds_transforms
        self.augmentation_transforms = augmentation_transforms
        self.target_sizes = target_sizes

    def __call__(self, img_index):
        return load_and_process_image_fixed_pysexgen1_centering_rescaling(img_index, self.subset, self.ds_transforms, self.augmentation_transforms, self.target_sizes)






### Processor classes for brightness normalisation ###

def load_and_process_image_brightness_norm(img_index, ds_transforms, augmentation_params, target_sizes=None):
    img_id = load_data.train_ids[img_index]
    img = load_data.load_image(img_id, from_ram=True)
    img = img / img.max() # normalise
    img_a = perturb_and_dscrop(img, ds_transforms, augmentation_params, target_sizes)
    return img_a


class LoadAndProcessBrightnessNorm(object):
    def __init__(self, ds_transforms, augmentation_params, target_sizes=None):
        self.ds_transforms = ds_transforms
        self.augmentation_params = augmentation_params
        self.target_sizes = target_sizes

    def __call__(self, img_index):
        return load_and_process_image_brightness_norm(img_index, self.ds_transforms, self.augmentation_params, self.target_sizes)


def load_and_process_image_fixed_brightness_norm(img_index, subset, ds_transforms, augmentation_transforms, target_sizes=None):
    if subset == 'train':
        img_id = load_data.train_ids[img_index]
    elif subset == 'test':
        img_id = load_data.test_ids[img_index]

    img = load_data.load_image(img_id, from_ram=True, subset=subset)
    img = img / img.max() # normalise
    img_a = augment_fixed_and_dscrop(img, ds_transforms, augmentation_transforms, target_sizes)
    return img_a


class LoadAndProcessFixedBrightnessNorm(object):
    """
    Same ugly hack as before
    """
    def __init__(self, subset, ds_transforms, augmentation_transforms, target_sizes=None):
        self.subset = subset
        self.ds_transforms = ds_transforms
        self.augmentation_transforms = augmentation_transforms
        self.target_sizes = target_sizes

    def __call__(self, img_index):
        return load_and_process_image_fixed_brightness_norm(img_index, self.subset, self.ds_transforms, self.augmentation_transforms, self.target_sizes)





class CallableObj(object):
    """
    UGLY HACK:

    pool.imap does not allow for extra arguments to be passed to the called function.
    This is a problem because we want to pass in the augmentation parameters.
    As a workaround, we could use a lambda or a locally defined function, but that
    doesn't work, because it cannot be pickled properly.

    The solution is to use a callable object instead, which is picklable.
    """
    def __init__(self, func, *args, **kwargs):
        self.func = func # the function to call
        self.args = args # additional arguments
        self.kwargs = kwargs # additional keyword arguments

    def __call__(self, index):
        return self.func(index, *self.args, **self.kwargs)
        

