import numpy as np


def config_diagonal(M, D, x=1):
    import copy
    k = M.shape[0]  # number of classes
    M_ = copy.deepcopy(M)
    D_ = copy.deepcopy(D)
    if x != 0:
        for i in range(k):  # for each diagonal element
            # for i in range(int(k/2)+1,k):
            for j in range(k):
                if i == j:
                    M_[i][j] -= 0.1 * x
                else:
                    M_[i][j] += (0.1 * x) / (k - 1)
            M_[M_ < 0] = 0
        for i in range(k):
            # for i in range(int(k/2)+1,k):
            for j in range(k):
                if i == j:
                    D_[i][j] = D_[i][j] * (M_[i][j] / (M_[i][j] + 0.1 * x))
                else:
                    D_[i][j] = D_[i][j] * (M_[i][j] / (M_[i][j] - (0.1) / (k - 1)))
        for i in range(k):
            M_[i] = M_[i] / sum(M_[i])
        D_[D_ <= 0] = 0
    return M_, D_


def feature_extraction(S, X, Label):
    k = max(Label) + 1
    M, D = calc_class_features(S, k, Label)
    H = calc_attr_cor(X, Label)

    partition = []
    for i in range(k):
        partition.append([])
    for i in range(len(Label)):
        partition[Label[i]].append(i)

    class_size = []
    for i in partition:
        class_size.append(len(i))
    class_size = np.array(class_size) / sum(class_size)

    # node degree
    theta = np.zeros(len(Label))
    nnz = S.nonzero()
    for i in range(len(nnz[0])):
        if nnz[0][i] < nnz[1][i]:
            theta[nnz[0][i]] += 1
            theta[nnz[1][i]] += 1

    return M, D, list(class_size), H, sorted(theta, reverse=True)


def calc_class_features(S, k, Label):
    pref = np.zeros((len(Label), k))
    nnz = S.nonzero()
    for i in range(len(nnz[0])):
        if nnz[0][i] < nnz[1][i]:
            pref[nnz[0][i]][Label[nnz[1][i]]] += 1
            pref[nnz[1][i]][Label[nnz[0][i]]] += 1
    for i in range(len(Label)):
        pref[i] /= sum(pref[i])
    pref = np.nan_to_num(pref)

    partition = []
    for i in range(k):
        partition.append([])
    for i in range(len(Label)):
        partition[Label[i]].append(i)

    # caluculate average and deviation of class preference
    from statistics import mean, median, variance, stdev
    class_pref_mean = np.zeros((k, k))
    class_pref_dev = np.zeros((k, k))
    for i in range(k):
        pref_tmp = []
        for j in partition[i]:
            pref_tmp.append(pref[j])
        pref_tmp = np.array(pref_tmp).transpose()
        for h in range(k):
            class_pref_mean[i, h] = mean(pref_tmp[h])
            if len(pref_tmp[h]) > 1:
                class_pref_dev[i, h] = stdev(pref_tmp[h])
            else:
                class_pref_dev[i, h] = 0
    return class_pref_mean, class_pref_dev


def calc_attr_cor(X, Label):
    k = max(Label) + 1
    n = X.shape[0]
    d = X.shape[1]

    partition = []
    for i in range(k):
        partition.append([])
    for i in range(len(Label)):
        partition[Label[i]].append(i)

    from statistics import mean
    attr_cor = np.zeros((d, k))
    for i in range(k):
        tmp = np.zeros(d)
        for j in partition[i]:
            tmp += X[j]
        attr_cor[:, i] = tmp / len(partition[i])
    return attr_cor
