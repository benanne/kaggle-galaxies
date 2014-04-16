"""
This file evaluates all constraints on the training labels as stipulated on the 'decision tree' page, and reports when they are violated.
It uses only the source CSV file for the sake of reproducibility.
"""

import numpy as np
import pandas as pd 

TOLERANCE = 0.00001 # 0.01 # only absolute errors greater than this are reported.

# TRAIN_LABELS_PATH = "data/raw/solutions_training.csv"
TRAIN_LABELS_PATH = "data/raw/training_solutions_rev1.csv"

d = pd.read_csv(TRAIN_LABELS_PATH)
targets = d.as_matrix()[:, 1:].astype('float32')
ids = d.as_matrix()[:, 0].astype('int32')


# separate out the questions for convenience
questions = [
    targets[:, 0:3], # 1.1 - 1.3,
    targets[:, 3:5], # 2.1 - 2.2
    targets[:, 5:7], # 3.1 - 3.2
    targets[:, 7:9], # 4.1 - 4.2
    targets[:, 9:13], # 5.1 - 5.4
    targets[:, 13:15], # 6.1 - 6.2
    targets[:, 15:18], # 7.1 - 7.3
    targets[:, 18:25], # 8.1 - 8.7
    targets[:, 25:28], # 9.1 - 9.3
    targets[:, 28:31], # 10.1 - 10.3
    targets[:, 31:37], # 11.1 - 11.6
]

# there is one constraint for each question.
# the sums of all answers for each of the questions should be equal to these numbers.
sums = [
    np.ones(targets.shape[0]), # 1, # Q1
    questions[0][:, 1], # Q2
    questions[1][:, 1], # Q3
    questions[1][:, 1], # Q4
    questions[1][:, 1], # Q5
    np.ones(targets.shape[0]), # 1, # Q6
    questions[0][:, 0], # Q7
    questions[5][:, 0], # Q8
    questions[1][:, 0], # Q9
    questions[3][:, 0], # Q10
    questions[3][:, 0], # Q11
]

num_total_violations = 0
affected_ids = set()

for k, desired_sums in enumerate(sums):
    print "QUESTION %d" % (k + 1)
    actual_sums = questions[k].sum(1)
    difference = abs(desired_sums - actual_sums)
    indices_violated = difference > TOLERANCE
    ids_violated = ids[indices_violated]
    num_violations = len(ids_violated)
    if num_violations > 0:
        print "%d constraint violations." % num_violations
        num_total_violations += num_violations
        for id_violated, d_s, a_s in zip(ids_violated, desired_sums[indices_violated], actual_sums[indices_violated]):
            print "violated by %d, sum should be %.6f but it is %.6f" % (id_violated, d_s, a_s)
            affected_ids.add(id_violated)
    else:
        print "No constraint violations."

    print

print
print "%d violations in total." % num_total_violations
print "%d data points violate constraints." % len(affected_ids)