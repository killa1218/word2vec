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

    norm = np.absolute(np.sum(matrix[w1Idx, :] * matrix[w2Idx, :], axis = 1) + 1 - np.array(labels, dtype = np.float32) / 5)

    print("Avg Loss:", np.sum(norm) / len(labels), "\nData Count:", len(labels))
