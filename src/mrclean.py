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
#  [none]

import pandas as pd

from intdep_util import finDru, finDep, finEff, feaDru, feaDep, feaEff, fclSt1

ldchr = 4             # to remove leading string 'ACH-' in depmap_id's
depid1 = 'depmap_id'  # string for df keys
depid2 = 'DepMap_ID'  # alt -^-


# ~~~ function definitions ~~~

# ACHorg: check for mutually exlusive ACH-depmap_id's
#  in:   listA = a list of ACH-depmap_id's
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
with open(finDru, 'r') as fin:
    dfDru = pd.read_csv(finDru, keep_default_na=False, low_memory=False)
fin.close()
with open(finDep, 'r') as fin:
    dfDep = pd.read_csv(finDep, keep_default_na=False, low_memory=False)
fin.close()
with open(finEff, 'r') as fin:
    dfEff = pd.read_csv(finEff, keep_default_na=False, low_memory=False)
fin.close()
print('~~~~ data from: {0:s}'.format(finDru))
print(dfDru)
print('\n')
print('~~~~ data from: {0:s}'.format(finDep))
print(dfDep)
print('\n')
print('~~~~ data from: {0:s}'.format(finEff))
print(dfEff)
print('\n')

# find the mutually exclusive depmap_id's
hitDruDep = ACHorg(dfDru[depid1], dfDep[depid2])
print('number of common ACH\'s found from dfDru and dfDep = {0:d}'
      .format(len(hitDruDep)))
print('loss = {0:d}'.format(abs(len(dfDru[depid1]) - len(dfDep[depid2]))))
hitf = ACHorg(hitDruDep, dfEff[depid2])
print('number of common ACH\'s found from above result and dfEff = {0:d}'
      .format(len(hitf)))
print('loss = {0:d}\n'.format(abs(len(hitf) - len(dfEff[depid2]))))

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
ordfDru = pd.DataFrame(dfDru.values.T[1:, :], columns=dfDru.values.T[0, :])
ordfDru = ordfDru[hitkeys]
# print(ordfDru)
ordfDep = pd.DataFrame(dfDep.values.T[1:, :], columns=dfDep.values.T[0, :])
ordfDep = ordfDep[hitkeys]
# print(ordfDep)
ordfEff = pd.DataFrame(dfEff.values.T[1:, :], columns=dfEff.values.T[0, :])
ordfEff = ordfEff[hitkeys]
# print(ordfEff)

# hmmm
print('writing everything to file...')
with open(fclSt1, 'w') as fout:
    for i in range(len(ordfDru.iloc[0, :])):
        outstr = (str(ordfDru.keys()[i]) + ',' + str(ordfDru.iloc[1, i])
                  + ',' + str(ordfDru.iloc[2, i])
                  + ' ' + str(ordfDru.iloc[3, i])
                  + ' ' + str(ordfDru.iloc[4, i]) + '\n')
        fout.write(outstr)
print('...done!\n')


print('FIN')
exit(0)
# FIN
