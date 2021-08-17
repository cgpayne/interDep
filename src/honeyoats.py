#!/usr/bin/env python
#  honeyoats = cluster analysis of initial data (honey + oats = clusters)
#  head -n 14 honeyoats.py
#  python3 honeyoats.py
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

# import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
# from matplotlib import cm
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

from intdep_util import feaSt2
from intdep_util import eprint, uncsvip

# input and output file names
foutTmp = 'tmp.o'


# def uncsvip(filename):
#     with open(filename, 'r', newline='\n') as fin:
#         reader = csv.reader(fin)
#         indata = list(reader)
#     fin.close()
#     lefty = []
#     righty = []
#     dlen = len(indata)
#     for i in range(dlen):
#         lefty.append(indata[i][0])
#         righty.append(indata[i][1])
#     return lefty, righty, dlen


def labtally(labels, check):
    count = 0
    for i in range(len(labels)):
        if labels[i] == check:
            count += 1
    return count


def secordcen(i, fndat):
    if i <= 1:
        eprint('ERROR 041: i is too low!')
        eprint('i =', i)
        eprint('exiting...')
        exit(1)
    elif i >= len(fndat)-1:
        eprint('ERROR 052: i is too high!')
        eprint('i =', i)
        eprint('exiting...')
        exit(1)
    else:
        return fndat[i+1] - 2*fndat[i] + fndat[i]  # NOTE: h**2 = 1**2 = 1


# # in:   X = real value, Xdat = array of ordered real values, Xlen = length of Xdat
# # out:  i = i-th place, where X can be found between Xdat[i] and Xdat[i+1] (left justified), via bisection method
# #           if X is too far left => return 'L', if X is too far right => return 'R'
# def iXfind(X, Xdat):
#   Xlen = len(Xdat)
#   if X < Xdat[0]:
#     i = 'L'
#   elif Xdat[Xlen-1] == X:
#     i = Xlen-2  # close off the right-most bracket, for completeness
#   elif Xdat[Xlen-1] < X:
#     i = 'R'
#   else:
#     a = 0
#     c = Xlen-1
#     hit = 0
#     while hit == 0:
#       b = (a + c)/2
#       if (Xdat[a] <= X) and (X < Xdat[b]):  # it's in the left bracket
#         if a == b-1:
#           i = a  # left justified
#           hit = 1
#         else:
#           c = b
#       else:  # it's in the right bracket
#         if b == c-1:
#           i = b  # left justified
#           hit = 1
#         else:
#           a = b
#   return i
#
#
# def trapdat(x, lidat):
#     i = iXfind(x, lidat)
#     yy = 0
#     if i != 'L' and i != 'R':
#         # slope = (lidat[i+1] - lidat[i])/(Nvec[i+1] - Nvec[i])
#         slope = (lidat[i+1] - lidat[i])
#         # yy = slope*(N - Nvec[i]) + lidat[i]
#         yy = slope*(x - x//1) + lidat[i]


ykk, stlen = uncsvip(feaSt2)
liACH = ykk[0]
liSta = ykk[1]

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
print(dfVoc.values.shape)
# print(dfVoc.keys())
# exit()

# check the running of the PCA!
pca_tot = PCA(n_components=None, random_state=202108)
pca_tot.fit_transform(dfVoc.values)
print('variance captured by total = {0:.1f}'
      .format(100*sum(pca_tot.explained_variance_ratio_)))
plt.plot(np.cumsum(pca_tot.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('captured variance')
plt.show()
# exit()

# pca = PCA(n_components=2, random_state=202108)
# liRF = pca.fit_transform(dfVoc.values)  # reduced features
# dfRF = pd.DataFrame(liRF, columns=['x', 'y'])
# # print(dfRF)
pca_new = PCA(n_components=0.95, random_state=202108)
liRF = pca_new.fit_transform(dfVoc.values)  # reduced features
dfRF = pd.DataFrame(liRF)
print(dfRF)
print(dfRF.values.shape)
print('variance captured by {0:d} = {1:.1f}%'
      .format(dfRF.values.shape[1],
              100*sum(pca_new.explained_variance_ratio_)))
# exit()

# should maybe test this?
# https://stackabuse.com/k-nearest-neighbors-algorithm-in-python-and-scikit-learn/
neigh = NearestNeighbors(n_neighbors=2)
nbrs = neigh.fit(dfRF)
distances, indices = nbrs.kneighbors(dfRF)

distances = np.sort(distances, axis=0)
distances = list(distances[:, 1])  # HMMM...
print(distances)
print(len(distances))
for i in range(390, 448):
    print(i, secordcen(i, distances))
lolz = max([secordcen(i, distances) for i in range(1, 448)])
# print(lolz, int(np.where(distances == lolz)[0]))
print(lolz, distances.index(lolz))
plt.figure(figsize=(7, 7))
plt.plot(distances)
plt.title('K-distance Graph', fontsize=18)
plt.xlabel('Data Points sorted by distance', fontsize=12)
plt.ylabel('Epsilon', fontsize=12)
plt.show()
exit()

imin = 0.01
imax = 0.04
istep = 0.001
jmin = 4
jmax = 12
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
# dbscan = DBSCAN(eps=0.008, min_samples=6)  # pretty good!
dbscan = DBSCAN(eps=0.03, min_samples=10)
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
plt.colorbar(ticks=[i for i in range(-1, stlen)])
plt.xlabel('Feature 1', fontsize=12)
plt.ylabel('Feature 2', fontsize=12)
plt.show()

with open(foutTmp, 'w') as fout:
    for i in range(stlen):
        outstr = (liACH[i] + ',' + str(dfRF['DBSCAN_labels'][i])
                  + ',' + liSta[i] + '\n')
        fout.write(outstr)


print('FIN')
exit(0)
# FIN
