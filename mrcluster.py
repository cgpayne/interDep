#!/usr/bin/env python
#  mrcluster = cluster analysis of initial data
#  head -n 14 bigread.py
#  python3 mrcluster.py
#  By:  Charlie Payne
#  License: n/a
# DESCRIPTION
#  not sure yet
# NOTES
#  [none]
# KNOWN BUGS
#  [none]
# DESIRED FEATURES
#  [none]

import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
# from matplotlib import cm
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

# input and output file names
finSta = 'data/out_mrclean/states.csv'
foutTmp = 'mytmp.o'


def labtally(labels, check):
    count = 0
    for i in range(len(labels)):
        if labels[i] == check:
            count += 1
    return count


with open(finSta, 'r', newline='\n') as fin:
    reader = csv.reader(fin)
    indata = list(reader)
fin.close()
liACH = []
liSta = []
dlen = len(indata)
for i in range(dlen):
    liACH.append(indata[i][0])
    liSta.append(indata[i][1])

# split all the words up, account for overlaps (set), then sort
vocabi = sorted(set(word for sentence in liSta for word in sentence.split()))
print(len(vocabi), vocabi)

for i in range(dlen):
    # HMMM (lack of domain knowledge): Antigen, Cell, Clear,
    liSta[i] = liSta[i].replace('Basal ', 'Basal_')
    liSta[i] = liSta[i].replace('Non ', 'Non_')
    liSta[i] = liSta[i].replace('Soft ', 'Soft_')
    liSta[i] = liSta[i].replace(' Amp', '_Amp')  # HMMM....
    liSta[i] = liSta[i].replace('Central Nervous System', 'CNS')
    liSta[i] = liSta[i].replace('Peripheral Nervous System', 'PNS')
    liSta[i] = liSta[i].replace(' Grade', '_Grade')  # HMMM....
    liSta[i] = liSta[i].replace('Upper ', 'Upper_')
    liSta[i] = liSta[i].replace('Urinary Tract', 'Urinary_Tract')

vocabf = sorted(set(word for sentence in liSta for word in sentence.split()))
print(len(vocabf), vocabf)

# relevant formulae for TF-IDF in: https://en.wikipedia.org/wiki/Tf–idf
vec = TfidfVectorizer(stop_words=None)
vec.fit(liSta)
# print(len(vec.vocabulary_.keys()), [key for key in sorted(vec.vocabulary_.keys())])
# features = vec.transform(liSta)
# print(features)
# print('GAH')
# print(features.toarray())
# exit()

dfVoc = pd.DataFrame(vec.transform(liSta).toarray(),
                     columns=sorted(vec.vocabulary_.keys()))

print(dfVoc.values)

# print(dfVoc.keys())
# exit()

# check the running of the PCA!
pca = PCA(n_components=2, random_state=202108)
liRF = pca.fit_transform(dfVoc.values)  # reduced features
dfRF = pd.DataFrame(liRF, columns=['x', 'y'])
# print(dfRF)
# exit()

# neigh = NearestNeighbors(n_neighbors=2)
# nbrs = neigh.fit(dfRF)
# distances, indices = nbrs.kneighbors(dfRF)
#
# distances = np.sort(distances, axis=0)
# distances = distances[:, 1]
# plt.figure(figsize=(7, 7))
# plt.plot(distances)
# plt.title('K-distance Graph', fontsize=18)
# plt.xlabel('Data Points sorted by distance', fontsize=12)
# plt.ylabel('Epsilon', fontsize=12)
# plt.show()
# exit()

imin = 0.003
imax = 0.012
istep = 0.001
jmin = 4
jmax = 10
for epi in np.arange(imin, imax+istep, istep):
    if epi == imin:
        print('x.xxx:  ', end='')
    else:
        print('{0:.3f}:  '.format(epi), end='')
    for msj in range(jmin, jmax+1):
        if epi == imin:
            print(msj, end='\t\t')
            continue
        dbscan = DBSCAN(eps=epi, min_samples=msj)
        dbscan.fit(dfRF)
        print('({0:d}, {1:d})'.format(max(dbscan.labels_) + 1,
              labtally(dbscan.labels_, -1)), end='\t')
        # print('{'+str(msj)+'}= '+str(max(dbscan.labels_) + 1), end='\t')
        # print('{\{0:d\}}= {1:d}'.format(msj, max(dbscan.labels_) + 1), end=',')
    print('')

# (6, 4), (7, 5), (8, 6), ...
dbscan = DBSCAN(eps=0.008, min_samples=6)  # pretty good!
dbscan.fit(dfRF)
dfRF['DBSCAN_labels'] = dbscan.labels_

print(dfRF)

# mycol = ['gray', 'red', 'green', 'blue']
# mycol = ['gray', 'red', 'orange', 'green', 'blue', 'indigo', 'violet']
acmap = plt.cm.get_cmap('summer')
mycol = acmap(np.arange(acmap.N))
mycol = np.concatenate(([np.array([0.3, 0.3, 0.3, 1.0])], mycol))
# print(mycol)

plt.figure(figsize=(7, 7))
plt.scatter(dfRF['x'], dfRF['y'], c=dfRF['DBSCAN_labels'],
            cmap=matplotlib.colors.ListedColormap(mycol), s=15)
            # cmap=matplotlib.colors.ListedColormap(mycol), s=15)
plt.title('DBSCAN Clustering', fontsize=18)
# plt.legend(fontsize=20, loc='lower right', fancybox=False, edgecolor='black')
plt.colorbar(ticks=[i for i in range(-1, dlen)])
plt.xlabel('Feature 1', fontsize=12)
plt.ylabel('Feature 2', fontsize=12)
plt.show()

with open(foutTmp, 'w') as fout:
    for i in range(dlen):
        outstr = (liACH[i] + ', ' + str(dfRF['DBSCAN_labels'][i])
                  + ', ' + liSta[i] + '\n')
        fout.write(outstr)

print('FIN')
exit(0)
# FIN
