#!/usr/bin/python
# coding=utf8

import sys
import numpy as np
import pickle as pk
from struct import unpack

print(sys.argv)

fName = sys.argv[1]

with open(fName, 'rb') as f:
    info = f.readline().split(bytes(' '.encode('utf8')))
    wordNum = int(info[0])
    embSize = int(info[1])

    l = []
    vocab = {}
    count = 0
    buf = ''
    first = False

    while True:
        ch = f.read(1).decode('utf8')

        if ch == '':
            break
        elif ch == ' ':
            ll = [unpack('f', f.read(4))[0] for _ in range(embSize)]
            l.append(ll)
            vocab[buf.lower()] = count
            count += 1
        elif ch == '\n':
            buf = ''
        else:
            buf += str(ch)

    matrix = np.array(l, dtype=np.float32)

    avgNorm = np.linalg.norm(matrix, axis = 1).reshape([len(vocab), 1])

    matrix = matrix / avgNorm

    # Read Vectors

# WordSim-353
# with open('wordsim353.pkl', 'rb') as f:
#     testData = pk.load(f)
#     w1Idx = []
#     w2Idx = []
#     labels = []

#     totalList = []

#     for p, c in testData.items():
#         w1 = p[0]
#         w2 = p[1]

#         if w1 in vocab and w2 in vocab:
#             w1Idx.append(vocab[w1])
#             w2Idx.append(vocab[w2])
#             labels.append(float(c))

#             totalList.append((float(c), (vocab[w1], vocab[w2])))

# SemLex-999
# with open('SimLex-999.txt', 'r') as f:
#     w1Idx = []
#     w2Idx = []
#     labels = []
#     totalList = []
#     l = f.readline()

#     for line in f.readlines():
#         line = line.split('\t')
#         w1 = line[0]
#         w2 = line[1]

#         if w1 in vocab and w2 in vocab:
#             w1Idx.append(vocab[w1])
#             w2Idx.append(vocab[w2])
#             labels.append(float(line[3]))

#             totalList.append((float(line[3]), (vocab[w1], vocab[w2])))

# MEN
with open('MEN_dataset_lemma_form_full', 'r') as f:
    w1Idx = []
    w2Idx = []
    labels = []
    totalList = []

    for line in f.readlines():
        line = line.split(' ')
        w1 = line[0]
        w2 = line[1]

        if w1 in vocab and w2 in vocab:
            w1Idx.append(vocab[w1])
            w2Idx.append(vocab[w2])
            labels.append(float(line[2]))

            totalList.append((float(line[2]), (vocab[w1], vocab[w2])))

    # norm = np.absolute(np.maximum(0, np.sum(matrix[w1Idx, :] * matrix[w2Idx, :], axis = 1)) - np.array(labels, dtype = np.float32) / 10)
    # print("Avg Loss:", np.sum(norm) / len(labels), "\nData Count:", len(labels))

    totalList.sort(key = lambda x: x[0])
    rankDict = {}

    for i, v in enumerate(totalList):
        rankDict[v[1]] = i

    cosines = np.maximum(0, np.sum(matrix[w1Idx, :] * matrix[w2Idx, :], axis = 1))

    totalList = []
    for i in range(len(w1Idx)):
        totalList.append((cosines[i], (w1Idx[i], w2Idx[i])))

    totalList.sort(key = lambda x: x[0])

    summ = 0
    n = len(w1Idx)
    for i, v in enumerate(totalList):
        summ += (rankDict[v[1]] - i)**2

    print('Spearman\'s Correlation:', 1 - (6 * summ / n / (n**2 - 1)))
