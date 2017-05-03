##
## python util function to create dataset from Serial Observations
##
import sys
import numpy as np
import SlidingWindowGenerator


def loadfile(infile):
    data = []
    with open(infile, 'r') as inf:
        data = inf.readlines()
    return data

def savefile(outfile, dataset, ids):
    with open(outfile,'w') as of:
        for id in ids:
            of.write(dataset[id])

def saveListsToCsvfile(outfile, dataset, ids):
    pass
	
def perform_sliding_winow(infile, ratio):
    dataset = loadfileWithStrippedNewline(infile)
    dataset = SlidingWindowGenerator(dataset, window_size=5).runSlidingWindow(max_dataset_size = 0)
    total = len(dataset)
    if len(dataset) <= 0:
        print('load empty file, quit')
        sys.exit(-1)

    if ratio < 0:
        print('ratio should be in (0,1), reset it to 0.2')
        ratio = 0.2
        testcnt = int(total*ratio)
    elif ratio > 1:
        print('set absolute test number as %d'%ratio)
        testcnt = int(ratio)
    else:
        testcnt = int(total*ratio)
    

    id = np.arange(total)
    perm = np.random.permutation(id)
    
    test = perm[:testcnt]
    train = perm[testcnt:]
    saveListsToCsvfile('train-' + infile, dataset, train)
    saveListsToCsvfile('test-' + infile, dataset, test)

def process_split(infile, ratio):

    dataset = loadfile(infile)
    total = len(dataset)
    if len(dataset) <= 0:
        print('load empty file, quit')
        sys.exit(-1)

    if ratio < 0:
        print('ratio should be in (0,1), reset it to 0.2')
        ratio = 0.2
        testcnt = int(total*ratio)
    elif ratio > 1:
        print('set absolute test number as %d'%ratio)
        testcnt = int(ratio)
    else:
        testcnt = int(total*ratio)
    

    id = np.arange(total)
    perm = np.random.permutation(id)
    
    test = perm[:testcnt]
    train = perm[testcnt:]
    savefile('train-' + infile, dataset, train)
    savefile('test-' + infile, dataset, test)

def process_split_X_Y_to_train_test(infile_X, infile_Y, ratio):

    dataset_X = loadfile(infile_X)
    dataset_Y = loadfile(infile_Y)

    total = len(dataset_X)
    if len(dataset_X) <= 0:
        print('load empty file, quit')
        sys.exit(-1)

    if ratio < 0:
        print('ratio should be in (0,1), reset it to 0.2')
        ratio = 0.2
        testcnt = int(total*ratio)
    elif ratio > 1:
        print('set absolute test number as %d'%ratio)
        testcnt = int(ratio)
    else:
        testcnt = int(total*ratio)

    permutated_indices = np.random.permutation(total)

    savefile('train_X-' + infile, dataset_X, permutated_indices[testcnt:])
    savefile('test_X-' + infile, dataset_X, permutated_indices[:testcnt])
    savefile('train_Y-' + infile, dataset_Y, permutated_indices[testcnt:])
    savefile('test_Y-' + infile, dataset_Y, permutated_indices[:testcnt])

if __name__ == '__main__':
    if len(sys.argv) < 2 :
        print('usage: datasplit.py <ratio> <infile1> <[infile2]> <[seed]>')
        print('ratio ex : 0.2 would create test data with  0.2*rows, train data with 0.8*rows ')
        sys.exit(-1)
    
    infile = sys.argv[2]
    ratio = float(sys.argv[1]) 

    if len(sys.argv) <= 4:
        print('fix random seed to 123')
        np.random.seed(seed= 123)

    if len(sys.argv) >= 4:
        infile2 = sys.argv[3]
        process_split_X_Y_to_train_test(infile, infile2, ratio)
    else:
        process_split(infile, ratio)
