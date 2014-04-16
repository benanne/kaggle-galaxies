TRAIN_IDS_PATH = "data/train_ids.npy"
# TRAIN_LABELS_PATH = "data/raw/solutions_training.csv"
TRAIN_LABELS_PATH = "data/raw/training_solutions_rev1.csv"

import numpy as np
import os
import csv

with open(TRAIN_LABELS_PATH, 'r') as f:
    reader = csv.reader(f, delimiter=",")
    train_ids = []
    for k, line in enumerate(reader):
        if k == 0: continue # skip header
        train_ids.append(int(line[0]))

train_ids = np.array(train_ids)
print "Saving %s" % TRAIN_IDS_PATH
np.save(TRAIN_IDS_PATH, train_ids)