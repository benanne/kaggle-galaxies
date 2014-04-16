import numpy as np 
from scipy import ndimage
import glob
import itertools
import threading
import time
import skimage.transform
import skimage.io
import skimage.filter
import gzip
import os
import Queue
import multiprocessing as mp


num_train = 61578 # 70948
num_test = 79975 # 79971


train_ids = np.load("data/train_ids.npy")
test_ids = np.load("data/test_ids.npy")



def load_images_from_jpg(subset="train", downsample_factor=None, normalise=True, from_ram=False):
    if from_ram:
        pattern = "/dev/shm/images_%s_rev1/*.jpg"
    else:
        pattern = "data/raw/images_%s_rev1/*.jpg"
    paths = glob.glob(pattern % subset)
    paths.sort() # alphabetic ordering is used everywhere.
    for path in paths:
        # img = ndimage.imread(path)
        img = skimage.io.imread(path)
        if normalise:
            img = img.astype('float32') / 255.0 # normalise and convert to float

        if downsample_factor is None:
            yield img
        else:
            yield img[::downsample_factor, ::downsample_factor]


load_images = load_images_from_jpg



### data loading, chunking ###

def images_gen(id_gen, *args, **kwargs):
    for img_id in id_gen:
        yield load_image(img_id, *args, **kwargs)


def load_image(img_id, subset='train', normalise=True, from_ram=False):
        if from_ram:
            path = "/dev/shm/images_%s_rev1/%d.jpg" % (subset, img_id)
        else:
            path = "data/raw/images_%s_rev1/%d.jpg" % (subset, img_id)
        # print "loading %s" % path # TODO DEBUG
        img = skimage.io.imread(path)
        if normalise:
            img = img.astype('float32') / 255.0 # normalise and convert to float
        return img


def cycle(l, shuffle=True): # l should be a NUMPY ARRAY of ids
    l2 = list(l) # l.copy() # make a copy to avoid changing the input
    while True:
        if shuffle:
            np.random.shuffle(l2)
        for i in l2:
            yield i


def chunks_gen(images_gen, shape=(100, 424, 424, 3)):
    """
    specify images_gen(cycle(list(train_ids))) as the ids_gen to loop through the training set indefinitely in random order.

    The shape parameter is (chunk_size, imsize1, imsize2, ...)
    So the size of the resulting images needs to be known in advance for efficiency reasons.
    """
    chunk = np.zeros(shape)
    size = shape[0]

    k = 0
    for image in images_gen: 
        chunk[k] = image
        k += 1

        if k >= size:
            yield chunk, size # return the chunk as well as its size (this is useful because the final chunk may be smaller)
            chunk = np.zeros(shape)
            k = 0

    # last bit of chunk
    if k > 0: # there is leftover data
        yield chunk, k # the chunk is a fullsize array, but only the first k entries are valid.



### threaded generator with a buffer ###

def _generation_thread(source_gen, buffer, buffer_lock, buffer_size=2, sleep_time=1):
    while True:
        # print "DEBUG: loader: acquiring lock"-
        with buffer_lock:
            # print "DEBUG: loader: lock acquired, checking if buffer is full"
            buffer_is_full = (len(buffer) >= buffer_size)
            # print "DEBUG: loader: buffer length is %d" % len(buffer)
            
        if buffer_is_full:
            # buffer is full, wait.
            # this if-clause has to be outside the with-clause, else the lock is held for no reason!
            # print "DEBUG: loader: buffer is full, waiting"
            
            #print "buffer is full, exiting (DEBUG)"
            #break
            time.sleep(sleep_time)
        else:
            try:
                data = source_gen.next()
            except StopIteration:
                break # no more data. STAHP.
            # print "DEBUG: loader: loading %s" % current_path
     
            # stuff the data in the buffer as soon as it is free
            # print "DEBUG: loader: acquiring lock"
            with buffer_lock:
                # print "DEBUG: loader: lock acquired, adding data to buffer"
                buffer.append(data)
                # print "DEBUG: loader: buffer length went from %d to %d" % (len(buffer) - 1, len(buffer))

            
    
    
def threaded_gen(source_gen, buffer_size=2, sleep_time=1):
    """
    Generator that runs a slow source generator in a separate thread.
    buffer_size: the maximal number of items to pre-generate (length of the buffer)
    """
    buffer_lock = threading.Lock()
    buffer = []
    
    thread = threading.Thread(target=_generation_thread, args=(source_gen, buffer, buffer_lock, buffer_size, sleep_time))
    thread.setDaemon(True)
    thread.start()
    
    while True:
        # print "DEBUG: generator: acquiring lock"
        with buffer_lock:
            # print "DEBUG: generator: lock acquired, checking if buffer is empty"
            buffer_is_empty = (len(buffer) == 0)
            # print "DEBUG: generator: buffer length is %d" % len(buffer)
            
        if buffer_is_empty:
            # there's nothing in the buffer, so wait a bit.
            # this if-clause has to be outside the with-clause, else the lock is held for no reason!
            # print "DEBUG: generator: buffer is empty, waiting"

            if not thread.isAlive():
                print "buffer is empty and loading thread is finished, exiting"
                break

            print "buffer is empty, waiting!"
            time.sleep(sleep_time)
        else:
            # print "DEBUG: generator: acquiring lock"
            with buffer_lock:
                # print "DEBUG: generator: lock acquired, removing data from buffer, yielding"
                data = buffer.pop(0)
                # print "DEBUG: generator: buffer length went from %d to %d" % (len(buffer) + 1, len(buffer))
            yield data


### perturbation and preprocessing ###
# use these with imap to apply them to a generator and return a generator

def im_rotate(img, angle):
    return skimage.transform.rotate(img, angle, mode='reflect')


def im_flip(img, flip_h, flip_v):
    if flip_h:
        img = img[::-1]
    if flip_v:
        img = img[:, ::-1]
    return img


# this old version uses ndimage, which is a bit unreliable (lots of artifacts)
def im_rotate_old(img, angle):
    # downsampling afterwards is recommended
    return ndimage.rotate(img, angle, axes=(0,1), mode='reflect', reshape=False)


def im_translate(img, shift_x, shift_y):
    ## this could probably be a lot easier... meh.
    # downsampling afterwards is recommended
    translate_img = np.zeros_like(img, dtype=img.dtype)

    if shift_x >= 0:
        slice_x_src = slice(None, img.shape[0] - shift_x, None)
        slice_x_tgt = slice(shift_x, None, None)
    else:
        slice_x_src = slice(- shift_x, None, None)
        slice_x_tgt = slice(None, img.shape[0] + shift_x, None)

    if shift_y >= 0:
        slice_y_src = slice(None, img.shape[1] - shift_y, None)
        slice_y_tgt = slice(shift_y, None, None)
    else:
        slice_y_src = slice(- shift_y, None, None)
        slice_y_tgt = slice(None, img.shape[1] + shift_y, None)

    translate_img[slice_x_tgt, slice_y_tgt] = img[slice_x_src, slice_y_src]

    return translate_img


def im_rescale(img, scale_factor):
    zoomed_img = np.zeros_like(img, dtype=img.dtype)
    zoomed = skimage.transform.rescale(img, scale_factor)

    if scale_factor >= 1.0:
        shift_x = (zoomed.shape[0] - img.shape[0]) // 2
        shift_y = (zoomed.shape[1] - img.shape[1]) // 2
        zoomed_img[:,:] = zoomed[shift_x:shift_x+img.shape[0], shift_y:shift_y+img.shape[1]]
    else:
        shift_x = (img.shape[0] - zoomed.shape[0]) // 2
        shift_y = (img.shape[1] - zoomed.shape[1]) // 2
        zoomed_img[shift_x:shift_x+zoomed.shape[0], shift_y:shift_y+zoomed.shape[1]] = zoomed

    return zoomed_img


# this old version uses ndimage zoom which is unreliable
def im_rescale_old(img, scale_factor):
    zoomed_img = np.zeros_like(img, dtype=img.dtype)

    if img.ndim == 2:
        z = (scale_factor, scale_factor)
    elif img.ndim == 3:
        z = (scale_factor, scale_factor, 1)
    # else fail
    zoomed = ndimage.zoom(img, z)

    if scale_factor >= 1.0:
        shift_x = (zoomed.shape[0] - img.shape[0]) // 2
        shift_y = (zoomed.shape[1] - img.shape[1]) // 2
        zoomed_img[:,:] = zoomed[shift_x:shift_x+img.shape[0], shift_y:shift_y+img.shape[1]]
    else:
        shift_x = (img.shape[0] - zoomed.shape[0]) // 2
        shift_y = (img.shape[1] - zoomed.shape[1]) // 2
        zoomed_img[shift_x:shift_x+zoomed.shape[0], shift_y:shift_y+zoomed.shape[1]] = zoomed

    return zoomed_img


def im_downsample(img, ds_factor):
    return img[::ds_factor, ::ds_factor]

def im_downsample_smooth(img, ds_factor):
    return skimage.transform.rescale(img, 1.0/ds_factor)
    # ndimage is unreliable, don't use it
    # channels = [ndimage.zoom(img[:,:, k], 1.0/ds_factor) for k in range(3)]
    # return np.dstack(channels)


def im_crop(img, ds_factor):
    size_x = img.shape[0]
    size_y = img.shape[1]

    cropped_size_x = img.shape[0] // ds_factor
    cropped_size_y = img.shape[1] // ds_factor

    shift_x = (size_x - cropped_size_x) // 2
    shift_y = (size_y - cropped_size_y) // 2

    return img[shift_x:shift_x+cropped_size_x, shift_y:shift_y+cropped_size_y]


def im_lcn(img, sigma_mean, sigma_std):
    """
    based on matlab code by Guanglei Xiong, see http://www.mathworks.com/matlabcentral/fileexchange/8303-local-normalization
    """
    means = ndimage.gaussian_filter(img, sigma_mean)
    img_centered = img - means
    stds = np.sqrt(ndimage.gaussian_filter(img_centered**2, sigma_std))
    return img_centered / stds



rgb2yuv = np.array([[0.299, 0.587, 0.114],
                    [-0.147, -0.289, 0.436],
                    [0.615, -0.515, -0.100]])

yuv2rgb = np.linalg.inv(rgb2yuv)



def im_rgb_to_yuv(img):
    return np.tensordot(img, rgb2yuv, [[2], [0]])

def im_yuv_to_rgb(img):
    return np.tensordot(img, yuv2rgb, [[2], [0]])


def im_lcn_color(img, sigma_mean, sigma_std, std_bias):
    img_yuv = im_rgb_to_yuv(img)
    img_luma = img_yuv[:, :, 0]
    img_luma_filtered = im_lcn_bias(img_luma, sigma_mean, sigma_std, std_bias)
    img_yuv[:, :, 0] = img_luma_filtered
    return im_yuv_to_rgb(img_yuv)


def im_norm_01(img): # this is just for visualisation
    return (img - img.min()) / (img.max() - img.min())


def im_lcn_bias(img, sigma_mean, sigma_std, std_bias):
    """
    LCN with an std bias to avoid noise amplification
    """
    means = ndimage.gaussian_filter(img, sigma_mean)
    img_centered = img - means
    stds = np.sqrt(ndimage.gaussian_filter(img_centered**2, sigma_std) + std_bias)
    return img_centered / stds


def im_luma(img):
    return np.tensordot(img, np.array([0.299, 0.587, 0.114], dtype='float32'), [[2], [0]])


def chunk_luma(chunk): # faster than doing it per image, probably
    return np.tensordot(chunk, np.array([0.299, 0.587, 0.114], dtype='float32'), [[3], [0]])


def im_normhist(img, num_bins=256): # from http://www.janeriksolem.net/2009/06/histogram-equalization-with-python-and.html
    # this function only makes sense for grayscale images.
    img_flat = img.flatten()
    imhist, bins = np.histogram(img_flat, num_bins, normed=True)
    cdf = imhist.cumsum() #cumulative distribution function
    cdf = 255 * cdf / cdf[-1] #normalize

    #use linear interpolation of cdf to find new pixel values
    im2 = np.interp(img_flat, bins[:-1], cdf)

    return im2.reshape(img.shape)



def chunk_lcn(chunk, sigma_mean, sigma_std, std_bias=0.0, rescale=1.0):
    """
    based on matlab code by Guanglei Xiong, see http://www.mathworks.com/matlabcentral/fileexchange/8303-local-normalization
    assuming chunk.shape == (num_examples, x, y, channels)

    'rescale' is an additional rescaling constant to get the variance of the result in the 'right' range.
    """
    means = np.zeros(chunk.shape, dtype=chunk.dtype)
    for k in xrange(len(chunk)):
        means[k] = skimage.filter.gaussian_filter(chunk[k], sigma_mean, multichannel=True)

    chunk = chunk - means # centering
    del means # keep memory usage in check

    variances = np.zeros(chunk.shape, dtype=chunk.dtype)
    chunk_squared = chunk**2
    for k in xrange(len(chunk)):
        variances[k] = skimage.filter.gaussian_filter(chunk_squared[k], sigma_std, multichannel=True)

    chunk = chunk / np.sqrt(variances + std_bias)

    return chunk / rescale

    # TODO: make this 100x faster lol. otherwise it's not usable.



def chunk_gcn(chunk, rescale=1.0):
    means = chunk.reshape(chunk.shape[0], chunk.shape[1] * chunk.shape[2], chunk.shape[3]).mean(1).reshape(chunk.shape[0], 1, 1, chunk.shape[3])
    chunk -= means

    stds = chunk.reshape(chunk.shape[0], chunk.shape[1] * chunk.shape[2], chunk.shape[3]).std(1).reshape(chunk.shape[0], 1, 1, chunk.shape[3])
    chunk /= stds

    return chunk





def array_chunker_gen(data_list, chunk_size, loop=True, truncate=True, shuffle=True):
    while True:
        if shuffle:
            rs = np.random.get_state()
            for data in data_list:
                np.random.set_state(rs)
                np.random.shuffle(data)

        if truncate:
            num_chunks = data_list[0].shape[0] // chunk_size # integer division, we only want whole chunks
        else:
            num_chunks = int(np.ceil(data_list[0].shape[0] / float(chunk_size)))

        for k in xrange(num_chunks):
            idx_range = slice(k * chunk_size, (k+1) * chunk_size, None)
            chunks = []
            for data in data_list:
                c = data[idx_range]
                current_size = c.shape[0]
                if current_size < chunk_size: # incomplete chunk, pad zeros
                    cs = list(c.shape)
                    cs[0] = chunk_size
                    c_full = np.zeros(tuple(cs), dtype=c.dtype)
                    c_full[:current_size] = c
                else:
                    c_full = c
                chunks.append(c_full)
            yield tuple(chunks), current_size

        if not loop:
            break



def load_gz(path): # load a .npy.gz file
    if path.endswith(".gz"):
        f = gzip.open(path, 'rb')
        return np.load(f)
    else:
        return np.load(path)


def save_gz(path, arr): # save a .npy.gz file
    tmp_path = os.path.join("/tmp", os.path.basename(path) + ".tmp.npy")
    # tmp_path = path + ".tmp.npy" # temp file needs to end in .npy, else np.load adds it!
    np.save(tmp_path, arr)
    os.system("gzip -c %s > %s" % (tmp_path, path))
    os.remove(tmp_path)



def numpy_loader_gen(paths_gen, shuffle=True):
    for paths in paths_gen:
        # print "loading " + str(paths)
        data_list = [load_gz(p) for p in paths]

        if shuffle:
            rs = np.random.get_state()
            for data in data_list:
                np.random.set_state(rs)
                np.random.shuffle(data)

        yield data_list, data_list[0].shape[0] # 'chunk' length needs to be the last entry


def augmented_data_gen(path_patterns):
    paths = [sorted(glob.glob(pattern)) for pattern in path_patterns]
    assorted_paths = zip(*paths)
    paths_gen = cycle(assorted_paths, shuffle=True)
    return numpy_loader_gen(paths_gen)


def post_augmented_data_gen(path_patterns):
    paths = [sorted(glob.glob(pattern)) for pattern in path_patterns]
    assorted_paths = zip(*paths)
    paths_gen = cycle(assorted_paths, shuffle=True)
    for data_list, chunk_length in numpy_loader_gen(paths_gen):
        # print "DEBUG: post augmenting..."
        start_time = time.time()
        data_list = post_augment_chunk(data_list)
        # print "DEBUG: post augmenting done. took %.4f seconds." % (time.time() - start_time)
        yield data_list, chunk_length


def post_augment_chunk(data_list):
    """
    perform fast augmentation that can be applied directly to the chunks in realtime.
    """
    chunk_size = data_list[0].shape[0]

    rotations = np.random.randint(0, 4, chunk_size)
    flip_h = np.random.randint(0, 2, chunk_size).astype('bool')
    flip_v = np.random.randint(0, 2, chunk_size).astype('bool')

    for x in data_list:
        if x.ndim <= 3:
            continue # don't apply the transformations to anything that isn't an image

        for k in xrange(chunk_size):
            x_k = np.rot90(x[k], k=rotations[k])

            if flip_h[k]:
                x_k = x_k[::-1]

            if flip_v[k]:
                x_k = x_k[:, ::-1]

            x[k] = x_k

    return data_list



### better threaded/buffered generator using the Queue class ###

### threaded generator with a buffer ###

def buffered_gen(source_gen, buffer_size=2, sleep_time=1):
    """
    Generator that runs a slow source generator in a separate thread.
    buffer_size: the maximal number of items to pre-generate (length of the buffer)
    """
    buffer = Queue.Queue(maxsize=buffer_size)

    def _buffered_generation_thread(source_gen, buffer):
        while True:
            # we block here when the buffer is full. There's no point in generating more data
            # when the buffer is full, it only causes extra memory usage and effectively
            # increases the buffer size by one.
            while buffer.full():
                print "DEBUG: buffer is full, waiting to generate more data."
                time.sleep(sleep_time)

            try:
                data = source_gen.next()
            except StopIteration:
                break

            buffer.put(data)
    
    thread = threading.Thread(target=_buffered_generation_thread, args=(source_gen, buffer))
    thread.setDaemon(True)
    thread.start()
    
    while True:
        yield buffer.get()
        buffer.task_done()


### better version using multiprocessing, because the threading module acts weird,
# the background thread seems to slow down significantly. When the main thread is
# busy, i.e. computation time is not divided fairly.

def buffered_gen_mp(source_gen, buffer_size=2, sleep_time=1):
    """
    Generator that runs a slow source generator in a separate process.
    buffer_size: the maximal number of items to pre-generate (length of the buffer)
    """
    buffer = mp.Queue(maxsize=buffer_size)

    def _buffered_generation_process(source_gen, buffer):
        while True:
            # we block here when the buffer is full. There's no point in generating more data
            # when the buffer is full, it only causes extra memory usage and effectively
            # increases the buffer size by one.
            while buffer.full():
                # print "DEBUG: buffer is full, waiting to generate more data."
                time.sleep(sleep_time)

            try:
                data = source_gen.next()
            except StopIteration:
                # print "DEBUG: OUT OF DATA, CLOSING BUFFER"
                buffer.close() # signal that we're done putting data in the buffer
                break

            buffer.put(data)
    
    process = mp.Process(target=_buffered_generation_process, args=(source_gen, buffer))
    process.start()
    
    while True:
        try:
            # yield buffer.get()
            # just blocking on buffer.get() here creates a problem: when get() is called and the buffer
            # is empty, this blocks. Subsequently closing the buffer does NOT stop this block.
            # so the only solution is to periodically time out and try again. That way we'll pick up
            # on the 'close' signal.
            try:
                yield buffer.get(True, timeout=sleep_time)
            except Queue.Empty:
                if not process.is_alive():
                    break # no more data is going to come. This is a workaround because the buffer.close() signal does not seem to be reliable.

                # print "DEBUG: queue is empty, waiting..."
                pass # ignore this, just try again.

        except IOError: # if the buffer has been closed, calling get() on it will raise IOError.
            # this means that we're done iterating.
            # print "DEBUG: buffer closed, stopping."
            break




def hms(seconds):
    seconds = np.floor(seconds)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)

    return "%02d:%02d:%02d" % (hours, minutes, seconds)
