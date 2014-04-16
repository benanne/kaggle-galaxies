"""
Load an analysis file and redo the predictions on the validation set / test set,
this time with augmented data and averaging. Store them as numpy files.
"""

import numpy as np
# import pandas as pd
import theano
import theano.tensor as T
import layers
import cc_layers
import custom
import load_data
import realtime_augmentation as ra
import time
import csv
import os
import cPickle as pickle


BATCH_SIZE = 32 # 16
NUM_INPUT_FEATURES = 3

CHUNK_SIZE = 8000 # 10000 # this should be a multiple of the batch size

# ANALYSIS_PATH = "analysis/try_convnet_cc_multirot_3x69r45_untied_bias.pkl"
ANALYSIS_PATH = "analysis/final/try_convnet_cc_multirotflip_3x69r45_maxout2048_extradense_dup3.pkl"

DO_VALID = True # disable this to not bother with the validation set evaluation
DO_TEST = True # disable this to not generate predictions on the testset



target_filename = os.path.basename(ANALYSIS_PATH).replace(".pkl", ".npy.gz")
target_path_valid = os.path.join("predictions/final/augmented/valid", target_filename)
target_path_test = os.path.join("predictions/final/augmented/test", target_filename)


print "Loading model data etc."
analysis = np.load(ANALYSIS_PATH)

input_sizes = [(69, 69), (69, 69)]

ds_transforms = [
    ra.build_ds_transform(3.0, target_size=input_sizes[0]),
    ra.build_ds_transform(3.0, target_size=input_sizes[1]) + ra.build_augmentation_transform(rotation=45)]

num_input_representations = len(ds_transforms)

# split training data into training + a small validation set
num_train = load_data.num_train
num_valid = num_train // 10 # integer division
num_train -= num_valid
num_test = load_data.num_test

valid_ids = load_data.train_ids[num_train:]
train_ids = load_data.train_ids[:num_train]
test_ids = load_data.test_ids

train_indices = np.arange(num_train)
valid_indices = np.arange(num_train, num_train+num_valid)
test_indices = np.arange(num_test)

y_valid = np.load("data/solutions_train.npy")[num_train:]


print "Build model"
l0 = layers.Input2DLayer(BATCH_SIZE, NUM_INPUT_FEATURES, input_sizes[0][0], input_sizes[0][1])
l0_45 = layers.Input2DLayer(BATCH_SIZE, NUM_INPUT_FEATURES, input_sizes[1][0], input_sizes[1][1])

l0r = layers.MultiRotSliceLayer([l0, l0_45], part_size=45, include_flip=True)

l0s = cc_layers.ShuffleBC01ToC01BLayer(l0r) 

l1a = cc_layers.CudaConvnetConv2DLayer(l0s, n_filters=32, filter_size=6, weights_std=0.01, init_bias_value=0.1, dropout=0.0, partial_sum=1, untie_biases=True)
l1 = cc_layers.CudaConvnetPooling2DLayer(l1a, pool_size=2)

l2a = cc_layers.CudaConvnetConv2DLayer(l1, n_filters=64, filter_size=5, weights_std=0.01, init_bias_value=0.1, dropout=0.0, partial_sum=1, untie_biases=True)
l2 = cc_layers.CudaConvnetPooling2DLayer(l2a, pool_size=2)

l3a = cc_layers.CudaConvnetConv2DLayer(l2, n_filters=128, filter_size=3, weights_std=0.01, init_bias_value=0.1, dropout=0.0, partial_sum=1, untie_biases=True)
l3b = cc_layers.CudaConvnetConv2DLayer(l3a, n_filters=128, filter_size=3, pad=0, weights_std=0.1, init_bias_value=0.1, dropout=0.0, partial_sum=1, untie_biases=True)
l3 = cc_layers.CudaConvnetPooling2DLayer(l3b, pool_size=2)

l3s = cc_layers.ShuffleC01BToBC01Layer(l3)

j3 = layers.MultiRotMergeLayer(l3s, num_views=4) # 2) # merge convolutional parts


l4a = layers.DenseLayer(j3, n_outputs=4096, weights_std=0.001, init_bias_value=0.01, dropout=0.5, nonlinearity=layers.identity)
l4b = layers.FeatureMaxPoolingLayer(l4a, pool_size=2, feature_dim=1, implementation='reshape')
l4c = layers.DenseLayer(l4b, n_outputs=4096, weights_std=0.001, init_bias_value=0.01, dropout=0.5, nonlinearity=layers.identity)
l4 = layers.FeatureMaxPoolingLayer(l4c, pool_size=2, feature_dim=1, implementation='reshape')

# l5 = layers.DenseLayer(l4, n_outputs=37, weights_std=0.01, init_bias_value=0.0, dropout=0.5, nonlinearity=custom.clip_01) #  nonlinearity=layers.identity)
l5 = layers.DenseLayer(l4, n_outputs=37, weights_std=0.01, init_bias_value=0.1, dropout=0.5, nonlinearity=layers.identity)

# l6 = layers.OutputLayer(l5, error_measure='mse')
l6 = custom.OptimisedDivGalaxyOutputLayer(l5) # this incorporates the constraints on the output (probabilities sum to one, weighting, etc.)



xs_shared = [theano.shared(np.zeros((1,1,1,1), dtype=theano.config.floatX)) for _ in xrange(num_input_representations)]

idx = T.lscalar('idx')

givens = {
    l0.input_var: xs_shared[0][idx*BATCH_SIZE:(idx+1)*BATCH_SIZE],
    l0_45.input_var: xs_shared[1][idx*BATCH_SIZE:(idx+1)*BATCH_SIZE],
}

compute_output = theano.function([idx], l6.predictions(dropout_active=False), givens=givens)


print "Load model parameters"
layers.set_param_values(l6, analysis['param_values'])

print "Create generators"
# set here which transforms to use to make predictions
augmentation_transforms = []
for zoom in [1 / 1.2, 1.0, 1.2]:
    for angle in np.linspace(0, 360, 10, endpoint=False):
        augmentation_transforms.append(ra.build_augmentation_transform(rotation=angle, zoom=zoom))
        augmentation_transforms.append(ra.build_augmentation_transform(rotation=(angle + 180), zoom=zoom, shear=180)) # flipped

print "  %d augmentation transforms." % len(augmentation_transforms)


augmented_data_gen_valid = ra.realtime_fixed_augmented_data_gen(valid_indices, 'train', augmentation_transforms=augmentation_transforms, chunk_size=CHUNK_SIZE, target_sizes=input_sizes, ds_transforms=ds_transforms)
valid_gen = load_data.buffered_gen_mp(augmented_data_gen_valid, buffer_size=1)


augmented_data_gen_test = ra.realtime_fixed_augmented_data_gen(test_indices, 'test', augmentation_transforms=augmentation_transforms, chunk_size=CHUNK_SIZE, target_sizes=input_sizes, ds_transforms=ds_transforms)
test_gen = load_data.buffered_gen_mp(augmented_data_gen_test, buffer_size=1)


approx_num_chunks_valid = int(np.ceil(num_valid * len(augmentation_transforms) / float(CHUNK_SIZE)))
approx_num_chunks_test = int(np.ceil(num_test * len(augmentation_transforms) / float(CHUNK_SIZE)))

print "Approximately %d chunks for the validation set" % approx_num_chunks_valid
print "Approximately %d chunks for the test set" % approx_num_chunks_test


if DO_VALID:
    print
    print "VALIDATION SET"
    print "Compute predictions"
    predictions_list = []
    start_time = time.time()

    for e, (chunk_data, chunk_length) in enumerate(valid_gen):
        print "Chunk %d" % (e + 1)
        xs_chunk = chunk_data

        # need to transpose the chunks to move the 'channels' dimension up
        xs_chunk = [x_chunk.transpose(0, 3, 1, 2) for x_chunk in xs_chunk]

        print "  load data onto GPU"
        for x_shared, x_chunk in zip(xs_shared, xs_chunk):
            x_shared.set_value(x_chunk)
        num_batches_chunk = int(np.ceil(chunk_length / float(BATCH_SIZE)))

        # make predictions, don't forget to cute off the zeros at the end
        predictions_chunk_list = []
        for b in xrange(num_batches_chunk):
            if b % 1000 == 0:
                print "  batch %d/%d" % (b + 1, num_batches_chunk)

            predictions = compute_output(b)
            predictions_chunk_list.append(predictions)

        predictions_chunk = np.vstack(predictions_chunk_list)
        predictions_chunk = predictions_chunk[:chunk_length] # cut off zeros / padding

        print "  compute average over transforms"
        predictions_chunk_avg = predictions_chunk.reshape(-1, len(augmentation_transforms), 37).mean(1)

        predictions_list.append(predictions_chunk_avg)

        time_since_start = time.time() - start_time
        print "  %s since start" % load_data.hms(time_since_start)


    all_predictions = np.vstack(predictions_list)

    print "Write predictions to %s" % target_path_valid
    load_data.save_gz(target_path_valid, all_predictions)

    print "Evaluate"
    rmse_valid = analysis['losses_valid'][-1]
    rmse_augmented = np.sqrt(np.mean((y_valid - all_predictions)**2))
    print "  MSE (last iteration):\t%.6f" % rmse_valid
    print "  MSE (augmented):\t%.6f" % rmse_augmented



if DO_TEST:
    print
    print "TEST SET"
    print "Compute predictions"
    predictions_list = []
    start_time = time.time()

    for e, (chunk_data, chunk_length) in enumerate(test_gen):
        print "Chunk %d" % (e + 1)
        xs_chunk = chunk_data

        # need to transpose the chunks to move the 'channels' dimension up
        xs_chunk = [x_chunk.transpose(0, 3, 1, 2) for x_chunk in xs_chunk]

        print "  load data onto GPU"
        for x_shared, x_chunk in zip(xs_shared, xs_chunk):
            x_shared.set_value(x_chunk)
        num_batches_chunk = int(np.ceil(chunk_length / float(BATCH_SIZE)))

        # make predictions, don't forget to cute off the zeros at the end
        predictions_chunk_list = []
        for b in xrange(num_batches_chunk):
            if b % 1000 == 0:
                print "  batch %d/%d" % (b + 1, num_batches_chunk)

            predictions = compute_output(b)
            predictions_chunk_list.append(predictions)

        predictions_chunk = np.vstack(predictions_chunk_list)
        predictions_chunk = predictions_chunk[:chunk_length] # cut off zeros / padding

        print "  compute average over transforms"
        predictions_chunk_avg = predictions_chunk.reshape(-1, len(augmentation_transforms), 37).mean(1)

        predictions_list.append(predictions_chunk_avg)

        time_since_start = time.time() - start_time
        print "  %s since start" % load_data.hms(time_since_start)

    all_predictions = np.vstack(predictions_list)


    print "Write predictions to %s" % target_path_test
    load_data.save_gz(target_path_test, all_predictions)

    print "Done!"
