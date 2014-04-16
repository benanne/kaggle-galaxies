TEST_IDS_PATH = "data/test_ids.npy"

import numpy as np
import glob
import os

filenames = glob.glob("data/raw/images_test_rev1/*.jpg")

test_ids = [int(os.path.basename(s).replace(".jpg", "")) for s in filenames]
test_ids.sort()
test_ids = np.array(test_ids)
print "Saving %s" % TEST_IDS_PATH
np.save(TEST_IDS_PATH, test_ids)