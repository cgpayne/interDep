#!/usr/bin/env python
#  honeyoats = cluster analysis of initial data (honey + oats = clusters)
#  head -n 14 honeyoats.py
#  python3 honeyoats.py
#  By:  Charlie Payne
#  License: n/a
# DESCRIPTION
#  not sure yet
# NOTES
#  -- it seems that scaling the TI-FIDF data worsens the PCA
# KNOWN BUGS
#  [none]
# DESIRED FEATURES
#  -- somehow split the state strings?

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
# from sklearn.preprocessing import StandardScaler

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


def secordcen(i, fndat, di):
    if i < 1:
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
        slopeL = fndat[i] - fndat[i-1]  # run = i - (i-1) = 1
        yyL = fndat[i] - slopeL*di
        slopeR = fndat[i+1] - fndat[i]  # run = (i+1) - i = 1
        yyR = slopeR*di + fndat[i]
        return (yyR - 2*fndat[i] + yyL)/di**2


def posifizer(alist):
    for i in range(len(alist)):
        if alist[i] < 0:
            alist[i] = 0
    return alist


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
# dfVoc = pd.DataFrame(vec.transform(liSta).toarray(),
#                      columns=sorted(vec.vocabulary_.keys()))
# dfVoc = pd.DataFrame(vec.transform(liSta).toarray())
# print(dfVoc)
# print(type(dfVoc.values))
# print(dfVoc.values.shape)
# print('yyeeeee')
# features = vec.transform(liSta).toarray()
# print(type(features))
# print(features)
# print('WTF')
# print(np.array(features))
# print(type(np.array(features)))
# # print(dfVoc.keys())
nati_states = vec.transform(liSta).toarray()
# exit()

# made it worse!?
# scaler = StandardScaler()
# scaler.fit(nati_states)
# natisc_states = scaler.transform(nati_states)
# print(natisc_states)
# # print(scX.shape)

# check the running of the PCA!
pca_tot = PCA(n_components=None, random_state=202108)
# pca_tot.fit_transform(dfVoc.values)
pca_tot.fit_transform(nati_states)
print('variance captured by total = {0:.1f}'
      .format(100*sum(pca_tot.explained_variance_ratio_)))
plt.plot(np.cumsum(pca_tot.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('captured variance')
plt.show()
# exit()

# pca = PCA(n_components=2, random_state=202108)
# liRF = pca.fit_transform(dfVoc.values)  # reduced features
# dfred_states = pd.DataFrame(liRF, columns=['x', 'y'])
# # print(dfred_states)
pca_new = PCA(n_components=0.33, random_state=202108)
# liRF = pca_new.fit_transform(dfVoc.values)  # reduced features
nared_states = pca_new.fit_transform(nati_states)  # reduced features
dfred_states = pd.DataFrame(nared_states)
print(dfred_states)
print(dfred_states.values.shape)
print('variance captured by {0:d} = {1:.1f}%'
      .format(dfred_states.values.shape[1],
              100*sum(pca_new.explained_variance_ratio_)))
# exit()

# should maybe test this?
# https://stackabuse.com/k-nearest-neighbors-algorithm-in-python-and-scikit-learn/
neigh = NearestNeighbors(n_neighbors=2)
nbrs = neigh.fit(dfred_states)
distances, indices = nbrs.kneighbors(dfred_states)

distances = np.sort(distances, axis=0)
distances = list(distances[:, 1])  # HMMM...
print(distances)
print(len(distances))
for i in range(390, 447):
    print(i, secordcen(i, distances, 1e-2))
# lolz = max([secordcen(i, distances) for i in range(1, 447)])
# print(lolz, int(np.where(distances == lolz)[0]))
ohmahgawd = posifizer([secordcen(i, distances, 1e-2) for i in range(1, 447)])
ohmahgawd.insert(0, 0.0)
ohmahgawd.append(0)
print(ohmahgawd)
dmax, di = max((dmax, di) for (di, dmax) in enumerate(ohmahgawd))
print(dmax, di)
plt.figure(figsize=(7, 7))
plt.plot(distances)
plt.title('K-distance Graph', fontsize=18)
plt.xlabel('Data Points sorted by distance', fontsize=12)
plt.ylabel('Epsilon', fontsize=12)
plt.show()
print(distances[400:430])
# exit()

imin = 0.005
imax = 0.02
istep = 0.001
jmin = 4
jmax = 10
for epi in np.arange(imin, imax+istep, istep):
    if epi == imin:
        print('x.xxx:  ', end='')
    else:
        print('{0:.4f}:  '.format(epi), end='')
    for msj in range(jmin, jmax+1):
        if epi == imin:
            print(msj, end='\t\t')
            continue
        dbscan = DBSCAN(eps=epi, min_samples=msj)
        dbscan.fit(dfred_states)
        print('({0:d}, {1:d})'.format(max(dbscan.labels_) + 1,
              labtally(dbscan.labels_, -1)), end='\t')
        # print('{'+str(msj)+'}= '+str(max(dbscan.labels_) + 1), end='\t')
        # print('{\{0:d\}}= {1:d}'.format(msj, max(dbscan.labels_) + 1), end=',')
    print('')
# exit()

# (6, 4), (7, 5), (8, 6), ...
# dbscan = DBSCAN(eps=0.008, min_samples=6)  # pretty good!
dbscan = DBSCAN(eps=0.015, min_samples=8)
dbscan.fit(dfred_states)
dfred_states['DBSCAN_labels'] = dbscan.labels_

print(dfred_states)

# mycol = ['gray', 'red', 'green', 'blue']
# mycol = ['gray', 'red', 'orange', 'green', 'blue', 'indigo', 'violet']
acmap = plt.cm.get_cmap('summer')
mycol = acmap(np.arange(acmap.N))
mycol = np.concatenate(([np.array([0.3, 0.3, 0.3, 1.0])], mycol))
# print(mycol)

# plt.figure(figsize=(7, 7))
# plt.scatter(dfred_states['x'], dfred_states['y'], c=dfred_states['DBSCAN_labels'],
#             cmap=matplotlib.colors.ListedColormap(mycol), s=15)
#             # cmap=matplotlib.colors.ListedColormap(mycol), s=15)
# plt.title('DBSCAN Clustering', fontsize=18)
# # plt.legend(fontsize=20, loc='lower right', fancybox=False, edgecolor='black')
# plt.colorbar(ticks=[i for i in range(-1, stlen)])
# plt.xlabel('Feature 1', fontsize=12)
# plt.ylabel('Feature 2', fontsize=12)
# plt.show()

with open(foutTmp, 'w') as fout:
    for i in range(stlen):
        outstr = (liACH[i] + ',' + str(dfred_states['DBSCAN_labels'][i])
                  + ',' + liSta[i] + '\n')
        fout.write(outstr)


print('FIN')
exit(0)
# FIN
