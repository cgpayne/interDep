import pandas as pd
import numpy as np

finDru = 'data/Drug_sensitivity_(PRISM_Repurposing_Primary_Screen)_19Q4.csv'
finDep = 'data/CRISPR_gene_dependency_Chronos.csv'
finEff = 'data/CRISPR_gene_effect.csv'

ldchr = 4  # to remove leading string 'ACH-' in depmap_id's


def ACHorg(listA, listB):
    answer = []
    for i in range(len(listA)):
        for j in range(len(listB)):
            if listA[i] == listB[j]:
                answer.append(listA[i])
    return answer


def printvert(alist):
    for i in range(len(alist)):
        print(alist[i])


def ACHtoint(alist):
    adict = {}
    for i in range(len(alist)):
        x = alist[i]
        adict[alist[i]] = int(x[ldchr:])
    return adict


dfDru = pd.read_csv(finDru, keep_default_na=False, low_memory=False)
dfDep = pd.read_csv(finDep, keep_default_na=False, low_memory=False)
dfEff = pd.read_csv(finEff, keep_default_na=False, low_memory=False)
# print(dfDru)
# print(dfDep)
# print(dfEff)

# print(dfDru['depmap_id'])
# print(dfDep['DepMap_ID'])

hitDruDep = ACHorg(dfDru['depmap_id'], dfDep['DepMap_ID'])
hitf = ACHorg(hitDruDep, dfEff['DepMap_ID'])
# print(len(hitDruDep))
# print(len(hitf))
hitdict = ACHtoint(hitf)
hitdict = dict(sorted(hitdict.items(), key=lambda x: x[1]))
# printvert(list(hitdict.keys()))

ordfDru = pd.DataFrame(dfDru.values.T[1:, :], columns=dfDru.values.T[0, :])
ordfDru = ordfDru[list(hitdict.keys())]
# print(ordfDru)
# print(ordfDru.values[1:5, 0:5])
for i in range(1, len(ordfDru.iloc[0, :])):
    print((ordfDru.iloc[1, i]+ordfDru.iloc[2, i]+ordfDru.iloc[3, i]).replace(' ', ''))
# print(type(ordfDru.iloc[1, 0]))
exit()
ordfDep = pd.DataFrame(dfDep.values.T[1:, :], columns=dfDep.values.T[0, :])
ordfDep = ordfDep[list(hitdict.keys())]
# print(ordfDep)
ordfEff = pd.DataFrame(dfEff.values.T[1:, :], columns=dfEff.values.T[0, :])
ordfEff = ordfEff[list(hitdict.keys())]
# print(ordfEff)


exit(0)
# FIN
