import pandas as pd
import numpy as np

finDru = 'data/Drug_sensitivity_(PRISM_Repurposing_Primary_Screen)_19Q4.csv'
finDep = 'data/CRISPR_gene_dependency_Chronos.csv'
finEff = 'data/CRISPR_gene_effect.csv'


def achorg(listA, listB):
    answer = []
    for i in range(len(listA)):
        for j in range(len(listB)):
            if listA[i] == listB[j]:
                answer.append(listA[i])
    return answer


def printvert(alist):
    for i in range(len(alist)):
        print(alist[i])


dfDru = pd.read_csv(finDru, low_memory=False)
dfDep = pd.read_csv(finDep, low_memory=False)
dfEff = pd.read_csv(finEff, low_memory=False)
# print(dfDru)
# print(dfDep)
# print(dfEff)

# print(dfDru['depmap_id'])
# print(dfDep['DepMap_ID'])

hitDruDep = achorg(dfDru['depmap_id'], dfDep['DepMap_ID'])
hitf = achorg(hitDruDep, dfEff['DepMap_ID'])
print(len(hitDruDep))
print(len(hitf))
printvert(hitf)


exit(0)
# FIN
