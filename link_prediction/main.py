from sklearn.linear_model.logistic import LogisticRegression
import numpy as np
import pickle
import os.path

def read_embeddings(emb_file):
    emb = dict()
    with open(emb_file, 'r') as fin:
        for i_l, line in enumerate(fin):
            s = line.split()
            if i_l == 0:
                dim = int(s[1])
                continue
            emb[int(s[0])] = [float(x) for x in s[1:]]
    return emb, dim

def read_sensitive_attr(sens_attr_file, emb):
    sens_attr = dict()
    with open(sens_attr_file, 'r') as fin:
        for line in fin:
            s = line.split()
            id = int(s[0])
            if id in emb:
                sens_attr[id] = int(s[1])
    return sens_attr

def read_links(links_file, emb):
    with open(links_file, 'rb') as f:
        links = pickle.load(f)
    return [l for l in links if l[0] in emb.keys() and l[1] in emb.keys()]

def extract_features(u,v):
    return (u-v)**2
    # return np.array([np.sqrt(np.sum((u-v)**2))])

if __name__ == '__main__':

    # all_labels = [0,1,2]
    all_labels = [0, 1]
    label_pairs = [str(all_labels[i]) + ',' + str(all_labels[j]) for i in range(len(all_labels)) for j in range(i, len(all_labels))]
    accuracy_keys = label_pairs + ['max_diff', 'var', 'total']

    accuracy = {k : [] for k in accuracy_keys}

    for iter in [str(k) for k in range(1,6)]:

        print('iter: ', iter)

        # filename = 'sample_4000_connected_subset/sample_4000_connected_subset'
        filename = 'rice_subset/rice_subset'
        # emb_file = filename + '.embeddings_unweighted_d32_' + iter
        # emb_file = filename + '.embeddings_fairwalk_d32_' + iter
        # emb_file = filename + '.randomembedding_d32_' + iter
        emb_file = filename + '.embeddings_random_walk_5_bndry_0.7_exp_2.0_d32_' + iter
        sens_attr_file = filename + '.attr'
        train_links_file = filename + '_' + iter + '_trainlinks'
        test_links_file = filename + '_' + iter + '_testlinks'

        emb, dim = read_embeddings(emb_file)
        sens_attr = read_sensitive_attr(sens_attr_file, emb)
        train_links = read_links(train_links_file, emb)
        test_links = read_links(test_links_file, emb)


        for key in label_pairs + ['total']:
            if key == 'total':
                valid_edge_pairs = [(all_labels[i],all_labels[j]) for i in range(len(all_labels)) for j in range(len(all_labels))]
            else:
                l1, l2 = [int(l) for l in key.split(',')]
                valid_edge_pairs = [(l1, l2)]
                if l1 != l2:
                    valid_edge_pairs.append((l2,l1))

            filtered_train_links = [l for l in train_links if (l[2], l[3]) in valid_edge_pairs]
            filtered_test_links = [l for l in test_links if (l[2],l[3]) in valid_edge_pairs]

            filtered_train_links = train_links

            clf = LogisticRegression(solver='lbfgs')
            x_train = np.array([extract_features(np.array(emb[l[0]]), np.array(emb[l[1]])) for l in filtered_train_links])
            y_train = np.array([l[4] for l in filtered_train_links])
            x_test = np.array([extract_features(np.array(emb[l[0]]), np.array(emb[l[1]])) for l in filtered_test_links])
            y_test = np.array([l[4] for l in filtered_test_links])
            clf.fit(x_train, y_train)
            y_pred = clf.predict(x_test)
            accuracy[key].append(100 * np.sum(y_test == y_pred) / x_test.shape[0])

        last_accs = [accuracy[k][-1] for k in label_pairs]
        accuracy['max_diff'].append(np.max(last_accs) - np.min(last_accs))
        accuracy['var'].append(np.var(last_accs))

        print(accuracy)
        print()

    print(accuracy)
    print()
    for k in accuracy_keys:
        print(k + ':', np.mean(accuracy[k]), '(' + str(np.std(accuracy[k])) + ')')
