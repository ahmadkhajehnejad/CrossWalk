from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import numpy as np

if __name__ == '__main__':
    parser = ArgumentParser("Synthesize Graph",
                            formatter_class=ArgumentDefaultsHelpFormatter,
                            conflict_handler='resolve')

    parser.add_argument('--nodes', type=int, help='Number nodes')
    parser.add_argument('--Pred', type=float, help='Probability of being red for each node')
    parser.add_argument('--Phom', type=float, help='Probability of within group connections')
    parser.add_argument('--Phet', type=float, help='Probability of cross group connections')

    args = parser.parse_args()

    n = args.nodes
    n_red = int(n * args.Pred)
    n_blue = n - n_red

    edges = []
    for i in range(1, n_red+1):
        for j in range(i+1, n_red+1):
            if np.random.rand() < args.Phom:
                edges.append((i, j))
                edges.append((j, i))
    for i in range(n_red+1, n+1):
        for j in range(i+1, n+1):
            if np.random.rand() < args.Phom:
                edges.append((i, j))
                edges.append((j, i))
    for i in range(1, n_red+1):
        for j in range(n_red+1, n+1):
            if np.random.rand() < args.Phet:
                edges.append((i, j))
                edges.append((j, i))


    filename = 'synthetic/synthetic_n' + str(n) + '_Pred' + str(args.Pred) + \
               '_Phom' + str(args.Phom) + '_Phet' + str(args.Phet)

    with open(filename + '.attr', 'w') as f:
        for i in range(1, n_red+1):
            f.write(str(i) + ' 1' + '\n')
        for i in range(n_red+1, n+1):
            f.write(str(i) + ' 0' + '\n')

    with open(filename + '.links', 'w') as f:
        for e in edges:
            f.write(str(e[0]) + ' ' + str(e[1]) + '\n')
