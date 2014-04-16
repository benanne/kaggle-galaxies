import os
import time

paths = ["data/raw/images_train_rev1", "data/raw/images_test_rev1"]

for path in paths:
    if os.path.exists(os.path.join("/dev/shm", os.path.basename(path))):
        print "%s exists in /dev/shm, skipping." % path
        continue

    print "Copying %s to /dev/shm..." % path
    start_time = time.time()
    os.system("cp -R %s /dev/shm/" % path)
    print "  took %.2f seconds." % (time.time() - start_time)
