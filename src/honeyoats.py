#!/usr/bin/env python
#  honeyoats = cluster analysis of initial data (honey + oats = clusters)
#  head -n 14 honeyoats.py
#  python3 honeyoats.py
#  By:  Charlie Payne
#  License: n/a
# DESCRIPTION
#  not sure yet
# NOTES
#  -- it seems that scaling the TI-FIDF data worsens the PCA, hmm...
#  -- I fear the clustering has become no more advantageous than grep
#  -- taking less variance from the PCA seems to cluster "better" for this
# KNOWN BUGS
#  [none]
# DESIRED FEATURES
#  [none]

import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
from scipy.signal import savgol_filter
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
# from sklearn.preprocessing import StandardScaler

from intdep_util import fsi_st2, fho_stf
from intdep_util import eprint, uncsvip


seedpca = 202108  # random seed for PCA if svd_solver='auto' -> 'arpack'/etc
npca = 2          # n(PCA) = << # of components OR % of variance >>


# ~~~ function definitions ~~~

# labtally: make a tally of a label's appearance in a list
#  in:   labels = a list of (repeating) labels,
#        check = the label to check for in the tally
#  out:  count = the number of times <check> appeared in <labels>
def labtally(labels, check):
    count = 0
    for i in range(len(labels)):
        if labels[i] == check:
            count += 1
    return count


# secordcen: take a second order derivative using a central finite difference
#  in:   fndat = a list of floats representing a function
#        i = the index to centre the differentiation on
#        di = the spacing of the finite difference
#  out:  the central second order finite difference of <fndat> at <i> by <di>
def secordcen(fndat, i, di):
    # parse the input: i must range from 1 to -2 (python indexing)
    if i < 1:
        eprint('ERROR 041: i is too low!')
        eprint('i =', i)
        eprint('exiting...')
        sys.exit(1)
    elif i >= len(fndat)-1:
        eprint('ERROR 052: i is too high!')
        eprint('i =', i)
        eprint('exiting...')
        sys.exit(1)
    else:
        # calculate the finite difference
        slopeL = fndat[i] - fndat[i-1]  # run = i - (i-1) = 1
        yyL = fndat[i] - slopeL*di
        slopeR = fndat[i+1] - fndat[i]  # run = (i+1) - i = 1
        yyR = slopeR*di + fndat[i]
        return (yyR - 2*fndat[i] + yyL)/di**2


# nonnegz: zero all negative elements of a list of floats
#  in:   alist = a list of floats
#  out:  alist = the same list, but with x -> 0 for all x < 0
def nonnegz(alist):
    for i in range(len(alist)):
        if alist[i] < 0:
            alist[i] = 0
    return alist


# --------- execute the code ---------

# read in the data
ykk, stlen = uncsvip(fsi_st2)
li_ACH = ykk[0]     # the ACH-depmap_id's
li_sites = ykk[1]   # the corresponding cancer sites
li_cancer = ykk[2]  # the corresponding cancers

# run a Term Frequency–Inverse Document Frequency (TF-IDF) on <li_cancer>
# relevant formulae for TF-IDF in: https://en.wikipedia.org/wiki/Tf–idf
vec = TfidfVectorizer(stop_words=None)
vec.fit(li_cancer)
npti_cancer = vec.transform(li_cancer).toarray()  # TF-IDF = 'ti'

# scale the data -- made it worse!?
# scaler = StandardScaler()
# scaler.fit(npti_cancer)
# nptisc_cancer = scaler.transform(npti_cancer)
# print(nptisc_cancer)

# check the running of the Principal Component Analysis (PCA)!
pca_tot = PCA(n_components=None, random_state=seedpca)
pca_tot.fit_transform(npti_cancer)
print('variance captured by total PCA = {0:.1f}'
      .format(100*sum(pca_tot.explained_variance_ratio_)))
plt.plot(np.cumsum(pca_tot.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('captured variance')
plt.show()
# sys.exit()

# do the PCA for n_components = npca
pca_new = PCA(n_components=npca, random_state=seedpca)  # HERE!
nprf_cancer = pca_new.fit_transform(npti_cancer)  # reduced features
dfrf_cancer = pd.DataFrame(nprf_cancer, columns=['x', 'y'])
print(dfrf_cancer)
print(dfrf_cancer.values.shape)
print('variance captured by {0:d} = {1:.1f}%'
      .format(dfrf_cancer.values.shape[1],
              100*sum(pca_new.explained_variance_ratio_)))
sys.exit()

# should maybe test this?
# https://stackabuse.com/k-nearest-neighbors-algorithm-in-python-and-scikit-learn/
neigh = NearestNeighbors(n_neighbors=2)
nbrs = neigh.fit(dfrf_cancer)
distances, indices = nbrs.kneighbors(dfrf_cancer)

distances = np.sort(distances, axis=0)
# distances = list(distances[:, 1])  # HMMM...
distances = np.array(distances[:, 1])  # HMMM...
distances = savgol_filter(distances, 11, 3)
print(distances)
print(len(distances))
for i in range(390, 447):
    print(i, secordcen(distances, i, 1e-2))
# lolz = max([secordcen(i, distances) for i in range(1, 447)])
# print(lolz, int(np.where(distances == lolz)[0]))
ohmahgawd = nonnegz([secordcen(distances, i, 1e-2) for i in range(1, 447)])
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
# sys.exit()

imin = 0.002
imax = 0.03
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
        dbscan.fit(dfrf_cancer)
        print('({0:d}, {1:d})'.format(max(dbscan.labels_) + 1,
              labtally(dbscan.labels_, -1)), end='\t')
        # print('{'+str(msj)+'}= '+str(max(dbscan.labels_) + 1), end='\t')
        # print('{\{0:d\}}= {1:d}'.format(msj, max(dbscan.labels_) + 1), end=',')
    print('')
# sys.exit()

# (6, 4), (7, 5), (8, 6), ...
# dbscan = DBSCAN(eps=0.008, min_samples=6)  # pretty good!
dbscan = DBSCAN(eps=0.015, min_samples=6)
dbscan.fit(dfrf_cancer)
dfrf_cancer['DBSCAN_labels'] = dbscan.labels_

print(dfrf_cancer)

# mycol = ['gray', 'red', 'green', 'blue']
# mycol = ['gray', 'red', 'orange', 'green', 'blue', 'indigo', 'violet']
acmap = plt.cm.get_cmap('summer')
mycol = acmap(np.arange(acmap.N))
mycol = np.concatenate(([np.array([0.3, 0.3, 0.3, 1.0])], mycol))
# print(mycol)

plt.figure(figsize=(7, 7))
plt.scatter(dfrf_cancer['x'], dfrf_cancer['y'], c=dfrf_cancer['DBSCAN_labels'],
            cmap=matplotlib.colors.ListedColormap(mycol), s=15)
            # cmap=matplotlib.colors.ListedColormap(mycol), s=15)
plt.title('DBSCAN Clustering', fontsize=18)
# plt.legend(fontsize=20, loc='lower right', fancybox=False, edgecolor='black')
plt.colorbar(ticks=[i for i in range(-1, stlen)])
plt.xlabel('Feature 1', fontsize=12)
plt.ylabel('Feature 2', fontsize=12)
plt.show()

print('wtf1')
dfnew = dfrf_cancer[['DBSCAN_labels']]  # [['key']] -> get as a dataframe
# dfnew.loc[:, 'Can'] = li_cancer  # tac on li_cancer (np.array(...) for numbers)
print('wtf2')
# dfnew.insert(1, 'Can', li_cancer, True)
dfnew = pd.concat([dfnew, pd.DataFrame(li_cancer, columns=['Can'])], axis=1)
# print(dfnew)
# sys.exit()
print('wtf3')
dfnew = dfnew.sort_values(by='DBSCAN_labels', ignore_index=True)
# print(dfnew)
# sys.exit()
print('wtf4')
for i in range(dfnew.shape[0]):
    dblab = dfnew.loc[i, 'DBSCAN_labels']
    print('{0:2d}  {1:s}'.format(dblab, dfnew.loc[i, 'Can']))
    if i == dfnew.shape[0]-1:
        continue
    if dblab == dfnew.loc[i+1, 'DBSCAN_labels'] - 1:
        print('')
print('wtf5')

with open(fho_stf, 'w') as fout:
    for i in range(stlen):
        outstr = (li_ACH[i] + ',' + str(dfrf_cancer['DBSCAN_labels'][i])
                  + ',' + li_cancer[i] + '\n')
        fout.write(outstr)


print('FIN')
sys.exit(0)
# FIN
