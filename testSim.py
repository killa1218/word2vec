#!/usr/bin/python
# coding=utf8

import sys
import numpy as np
import pickle as pk

print(sys.argv)

fName = sys.argv[1]

with open(fName, 'rb') as f:
    info = f.readline().split(bytes(' '.encode('utf8')))
    wordNum = int(info[0])
    embSize = int(info[1])

    l = []
    vocab = {}
    count = 0

    for line in f.readlines():
        arr = line.split(bytes(' '.encode('utf8')))

        print(arr)

        token = str(arr[0]).lower()
        vocab[token] = count
        count += 1
        l.append(arr[1:embSize + 1])

    # print(l)

    matrix = np.array(l, dtype=np.float32)

    avgNorm = np.sqrt(np.sum(matrix**2) / len(vocab))

    matrix = matrix / avgNorm

    # Read Vectors

with open('wordsim353.pkl', 'rb') as f:
    testData = pk.load(f)
    w1Idx = []
    w2Idx = []
    labels = []

    for p, c in testData.items():
        w1 = p[0]
        w2 = p[1]

        if w1 in vocab and w2 in vocab:
            w1Idx.append(vocab[w1])
            w2Idx.append(vocab[w2])
            labels.append(float(c))

    norm = (np.sum(matrix[w1Idx, :] * matrix[w2Idx, :], axis = 1) - np.array(labels, dtype = np.float32))**2

    print("Avg Loss:", norm / len(labels))
