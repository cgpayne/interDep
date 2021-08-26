#!/usr/bin/env python
#  mrclean = initial data cleaning, PART I (Mr. Clean, Mr. Clean, Mr. Clean!)
#  head -n 15 mrclean.py
#  python3 mrclean.py
#  By:  Charlie Payne
#  License: n/a
# DESCRIPTION
#  reorganize the input data to a format that makes sense (to me at least)
#  output the overlapping data and some header info for later processing
# NOTES
#  [none]
# KNOWN BUGS
#  [none]
# DESIRED FEATURES
#  -- finish printing all results to file

import sys

import pandas as pd

from intdep_util import fin_dru, fin_dep, fin_eff
from intdep_util import fea_dru, fea_dep, fea_eff, fcl_st1


ldchr = 4             # to remove leading string 'ACH-' in depmap_id's
depid1 = 'depmap_id'  # string for df keys
depid2 = 'DepMap_ID'  # alt -^-


# ~~~ function definitions ~~~

# ACHorg: check for mutually exlusive ACH-depmap_id's
#  in:   listA = a list of ACH-depmap_id's,
#        listB = a different list of ACH-depmap_id's
#  out:  answer = a list of common ACH-depmap_id's
def ACHorg(listA, listB):
    answer = []
    for i in range(len(listA)):
        for j in range(len(listB)):
            if listA[i] == listB[j]:
                answer.append(listA[i])
    return answer


# # OBSOLETE
# def printvert(alist):
#     for i in range(len(alist)):
#         print(alist[i])


# ACHtoint:  convert an ACH-depmap_id to an int (nice and straightforward)
#  in:   alist = a list of ACH-depmap_id's
#  out:  adict = a dictionary of the ACH-depmap_id keys() and int values()
def ACHtoint(alist):
    adict = {}
    for i in range(len(alist)):
        x = alist[i]
        adict[alist[i]] = int(x[ldchr:])
    return adict


# --------- execute the code ---------

# read in the data and output to screen to check
with open(fin_dru, 'r') as fin:
    df_dru = pd.read_csv(fin_dru, keep_default_na=False, low_memory=False)
fin.close()
with open(fin_dep, 'r') as fin:
    df_dep = pd.read_csv(fin_dep, keep_default_na=False, low_memory=False)
fin.close()
with open(fin_eff, 'r') as fin:
    df_eff = pd.read_csv(fin_eff, keep_default_na=False, low_memory=False)
fin.close()
print('~~~~ data from: {0:s}'.format(fin_dru))
print(df_dru)
print('\n')
print('~~~~ data from: {0:s}'.format(fin_dep))
print(df_dep)
print('\n')
print('~~~~ data from: {0:s}'.format(fin_eff))
print(df_eff)
print('\n')

# find the mutually exclusive depmap_id's
hit_drudep = ACHorg(df_dru[depid1], df_dep[depid2])
print('number of common ACH\'s found from df_dru and df_dep = {0:d}'
      .format(len(hit_drudep)))
print('resulting loss = {0:d}'
      .format(abs(len(df_dru[depid1]) - len(df_dep[depid2]))))
hitf = ACHorg(hit_drudep, df_eff[depid2])
print('number of common ACH\'s found from above result and df_eff = {0:d}'
      .format(len(hitf)))
print('resulting = {0:d}\n'.format(abs(len(hitf) - len(df_eff[depid2]))))

# convert/sort the depmap_id's to ints
hitdict = ACHtoint(hitf)
hitdict = dict(sorted(hitdict.items(), key=lambda x: x[1]))
hitkeys = list(hitdict.keys())
# printvert(hitkeys)
print(hitkeys[0])
print('...')
print(hitkeys[-1])
print('')

# reduce the df's to only overlapping depmap_id's
ordf_dru = pd.DataFrame(df_dru.values.T[1:, :], columns=df_dru.values.T[0, :])
ordf_dru = ordf_dru[hitkeys]
# print(ordf_dru)
ordf_dep = pd.DataFrame(df_dep.values.T[1:, :], columns=df_dep.values.T[0, :])
ordf_dep = ordf_dep[hitkeys]
# print(ordf_dep)
ordf_eff = pd.DataFrame(df_eff.values.T[1:, :], columns=df_eff.values.T[0, :])
ordf_eff = ordf_eff[hitkeys]
# print(ordf_eff)

# print results to file
print('writing everything to file...')
with open(fcl_st1, 'w') as fout:
    for i in range(len(ordf_dru.iloc[0, :])):
        outstr = (str(ordf_dru.keys()[i]) + ',' + str(ordf_dru.iloc[1, i])
                  + ',' + str(ordf_dru.iloc[2, i])
                  + ' ' + str(ordf_dru.iloc[3, i])
                  + ' ' + str(ordf_dru.iloc[4, i]) + '\n')
        fout.write(outstr)
print('...done!\n')


print('FIN')
sys.exit(0)
# FIN
