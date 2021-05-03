from sklearn.metrics.pairwise import pairwise_distances
import sys
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import numpy as np
import pickle

def main():
    parser = ArgumentParser("Compute Similarities",
                            formatter_class=ArgumentDefaultsHelpFormatter,
                            conflict_handler='resolve')
    parser.add_argument('--filename', type=str, help='embedding file name')
    parser.add_argument('--workers', type=int, help='embedding file name')
    args = parser.parse_args()

    filename = args.filename

    data = []
    with open(filename, 'r') as f:
        for i, line in enumerate(f):
            if i == 0:
                continue
            st = line.split()
            data.append([float(s) for s in st[1:]])

    data = np.array(data, dtype=np.float)

    points = list(range(0, 81000, 5000)) + [data.shape[0]]

    for i in range(len(points)-1):
        ind = list(range(points[i], points[i+1]))
        sub_data = data[ind, :]
        sim = 0 - pairwise_distances(sub_data, metric="euclidean", n_jobs=args.workers)
        with open('similarities/' + filename + '--' + str(points[i]) + '-' + str(points[i + 1]) + ',' + str(points[i]) + '-' + str(
                points[i + 1]) + '.pkl', 'wb') as f:
            b = points[i + 1] - points[i]
            pickle.dump(sim[0:b, 0:b], f, protocol=4)

    for i in range(len(points)-1):
        for j in range(i+1, len(points)-1):
            ind = list(range(points[i], points[i+1])) + list(range(points[j], points[j+1]))
            sub_data = data[ind, :]
            sim = 0 - pairwise_distances(sub_data, metric="euclidean", n_jobs=args.workers)
            with open('similarities/' + filename + '--' + str(points[i]) + '-' + str(points[i+1]) + ',' + str(points[j]) + '-' + str(points[j+1]) + '.pkl', 'wb') as f:
                b = points[i+1] - points[i]
                pickle.dump(sim[0:b, b:], f, protocol=4)



if __name__ == '__main__':

    sys.exit(main())
