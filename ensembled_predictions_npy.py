"""
Given a set of predictions for the validation and testsets (as .npy.gz), this script computes
the optimal linear weights on the validation set, and then computes the weighted predictions on the testset.
"""

import sys
import os
import glob

import theano 
import theano.tensor as T

import numpy as np 

import scipy

import load_data


TARGET_PATH = "predictions/final/blended/blended_predictions.npy.gz"
TARGET_PATH_SEPARATE = "predictions/final/blended/blended_predictions_separate.npy.gz"
TARGET_PATH_UNIFORM = "predictions/final/blended/blended_predictions_uniform.npy.gz"

predictions_valid_dir = "predictions/final/augmented/valid"
predictions_test_dir = "predictions/final/augmented/test"


y_train = np.load("data/solutions_train.npy")
train_ids = load_data.train_ids
test_ids = load_data.test_ids

# split training data into training + a small validation set
num_train = len(train_ids)
num_test = len(test_ids)

num_valid = num_train // 10 # integer division
num_train -= num_valid

y_valid = y_train[num_train:]
y_train = y_train[:num_train]

valid_ids = train_ids[num_train:]
train_ids = train_ids[:num_train]

train_indices = np.arange(num_train)
valid_indices = np.arange(num_train, num_train + num_valid)
test_indices = np.arange(num_test)



# paths of all the files to blend.
predictions_test_paths = glob.glob(os.path.join(predictions_test_dir, "*.npy.gz"))
predictions_valid_paths = [os.path.join(predictions_valid_dir, os.path.basename(path)) for path in predictions_test_paths]

print "Loading validation set predictions"
predictions_list = [load_data.load_gz(path) for path in predictions_valid_paths]
predictions_stack = np.array(predictions_list).astype(theano.config.floatX) # num_sources x num_datapoints x 37
del predictions_list
print

print "Compute individual prediction errors"
individual_prediction_errors = np.sqrt(((predictions_stack - y_valid[None])**2).reshape(predictions_stack.shape[0], -1).mean(1))
print

print "Compiling Theano functions"
X = theano.shared(predictions_stack) # source predictions
t = theano.shared(y_valid) # targets

W = T.vector('W')


# shared weights for all answers
s = T.nnet.softmax(W).reshape((W.shape[0], 1, 1))

weighted_avg_predictions = T.sum(X * s, axis=0) #  T.tensordot(X, s, [[0], [0]])

error = T.mean((weighted_avg_predictions - t) ** 2)
grad = T.grad(error, W)

f = theano.function([W], error)
g = theano.function([W], grad)


# separate weights for all answers
s2 = T.nnet.softmax(W.reshape((37, predictions_stack.shape[0]))).dimshuffle(1, 'x', 0) # (num_prediction_sets, 1, num_answers)

weighted_avg_predictions2 = T.sum(X * s2, axis=0) #  T.tensordot(X, s, [[0], [0]])

error2 = T.mean((weighted_avg_predictions2 - t) ** 2)
grad2 = T.grad(error2, W)

f2 = theano.function([W], error2)
g2 = theano.function([W], grad2)

print

print "Optimizing blending weights: shared"
w_init = np.random.randn(predictions_stack.shape[0]).astype(theano.config.floatX) * 0.01
w_zero = np.zeros(predictions_stack.shape[0], dtype=theano.config.floatX)
out, res, _ = scipy.optimize.fmin_l_bfgs_b(f, w_init, fprime=g, pgtol=1e-09, epsilon=1e-08, maxfun=10000)

rmse = np.sqrt(res)
out_s = np.exp(out)
out_s /= out_s.sum()
rmse_uniform = np.sqrt(f(w_zero))
print

print "Optimizing blending weights: separate"
w_init2 = np.random.randn(predictions_stack.shape[0] * 37).astype(theano.config.floatX) * 0.01
out2, res2, _ = scipy.optimize.fmin_l_bfgs_b(f2, w_init2, fprime=g2, pgtol=1e-09, epsilon=1e-08, maxfun=10000)

rmse2 = np.sqrt(res2)
out_s2 = np.exp(out2).reshape(37, predictions_stack.shape[0]).T
out_s2 /= out_s2.sum(0)[None, :]
print

print "Individual prediction errors:"
for path, error in zip(predictions_valid_paths, individual_prediction_errors):
    print "  %.6f\t%s" % (error, os.path.basename(path))

print
print "Resulting weights (shared):"
for path, weight in zip(predictions_valid_paths, out_s):
    print "  %.5f\t%s" % (weight, os.path.basename(path))

print
print "Resulting error (shared):\t\t%.6f" % rmse
print "Resulting error (separate):\t\t%.6f" % rmse2
print "Uniform weighting error:\t%.6f" % rmse_uniform

print
print "Blending testset predictions"
# we only load one testset predictions file at a time to save memory.

blended_predictions = None
blended_predictions_separate = None
blended_predictions_uniform = None

for path, weight, weights_separate in zip(predictions_test_paths, out_s, out_s2):
    # print "  %s" % os.path.basename(path)
    predictions = load_data.load_gz(path)
    predictions_uniform = predictions * (1.0 / len(predictions_test_paths))
    predictions_separate = predictions * weights_separate[None, :]
    predictions *= weight # inplace scaling

    if blended_predictions is None:
        blended_predictions = predictions
        blended_predictions_separate = predictions_separate
        blended_predictions_uniform = predictions_uniform
    else:
        blended_predictions += predictions
        blended_predictions_separate += predictions_separate
        blended_predictions_uniform += predictions_uniform


print
print "Storing blended predictions (shared) in %s" % TARGET_PATH
load_data.save_gz(TARGET_PATH, blended_predictions)

print
print "Storing blended predictions (separate) in %s" % TARGET_PATH_SEPARATE
load_data.save_gz(TARGET_PATH_SEPARATE, blended_predictions_separate)

print
print "Storing uniformly blended predictions in %s" % TARGET_PATH_UNIFORM
load_data.save_gz(TARGET_PATH_UNIFORM, blended_predictions_uniform)

    
print
print "Done!"