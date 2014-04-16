# TRAIN_LABELS_PATH = "data/raw/solutions_training.csv"
TRAIN_LABELS_PATH = "data/raw/training_solutions_rev1.csv"
TARGET_PATH = "data/solutions_train.npy"

import pandas as pd 
import numpy as np 




d = pd.read_csv(TRAIN_LABELS_PATH)
targets = d.as_matrix()[:, 1:].astype('float32')


print "Saving %s" % TARGET_PATH
np.save(TARGET_PATH, targets)

