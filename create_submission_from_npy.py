import sys 
import os
import csv
import load_data


if len(sys.argv) != 2:
    print "Creates a gzipped CSV submission file from a gzipped numpy file with testset predictions."
    print "Usage: create_submission_from_npy.py <input.npy.gz>"
    sys.exit()

src_path = sys.argv[1]
src_dir = os.path.dirname(src_path)
src_filename = os.path.basename(src_path)
tgt_filename = src_filename.replace(".npy.gz", ".csv")
tgt_path = os.path.join(src_dir, tgt_filename)


test_ids = load_data.test_ids


print "Loading %s" % src_path

data = load_data.load_gz(src_path)
assert data.shape[0] == load_data.num_test

print "Saving %s" % tgt_path

with open(tgt_path, 'wb') as csvfile:
    writer = csv.writer(csvfile) # , delimiter=',', quoting=csv.QUOTE_MINIMAL)

    # write header
    writer.writerow(['GalaxyID', 'Class1.1', 'Class1.2', 'Class1.3', 'Class2.1', 'Class2.2', 'Class3.1', 'Class3.2', 'Class4.1', 'Class4.2', 'Class5.1', 'Class5.2', 'Class5.3', 'Class5.4', 'Class6.1', 'Class6.2', 'Class7.1', 'Class7.2', 'Class7.3', 'Class8.1', 'Class8.2', 'Class8.3', 'Class8.4', 'Class8.5', 'Class8.6', 'Class8.7', 'Class9.1', 'Class9.2', 'Class9.3', 'Class10.1', 'Class10.2', 'Class10.3', 'Class11.1', 'Class11.2', 'Class11.3', 'Class11.4', 'Class11.5', 'Class11.6'])

    # write data
    for k in xrange(load_data.num_test):
        row = [test_ids[k]] + data[k].tolist()
        writer.writerow(row)


print "Gzipping..."
os.system("gzip %s" % tgt_path)

print "Done!"