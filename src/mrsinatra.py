#!/usr/bin/env python
#  mrsinatra = initial data cleaning, PART II
#  head -n 14 mrsinatra.py
#  python3 mrsinatra.py
#  By:  Charlie Payne
#  License: n/a
# DESCRIPTION
#  do a bit more tailoring on the strings describing cancer type and site
# NOTES
#  [none]
# KNOWN BUGS
#  [none]
# DESIRED FEATURES
#  [none]

import sys

from intdep_util import fcl_st1, fsi_st2
from intdep_util import uncsvip


# ~~~ function definitions ~~~

# medcompound: change revelant medical words into compound words
#  in:   alist = a list of medical words
#  out:  alist = the same list, but with compound words connected by an _
def medcompound(alist):
    for i in range(len(alist)):
        # HMMM (lack of domain knowledge): Antigen, Cell, Clear,
        alist[i] = alist[i].replace('Basal ', 'Basal_')
        alist[i] = alist[i].replace('Non ', 'Non_')
        alist[i] = alist[i].replace('Soft ', 'Soft_')
        alist[i] = alist[i].replace(' Amp', '_Amp')  # HMMM....
        alist[i] = alist[i].replace('Central Nervous System', 'CNS')
        alist[i] = alist[i].replace('Peripheral Nervous System', 'PNS')
        alist[i] = alist[i].replace(' Grade', '_Grade')  # HMMM....
        alist[i] = alist[i].replace('Upper ', 'Upper_')
        alist[i] = alist[i].replace('Urinary Tract', 'Urinary_Tract')
    return alist


# vocit: split all the words up, account for overlaps (set), then sort
#  in:   alist = a list of sentences
#  out:  a sorted list of all the words in alist (without repeats)
def vocit(alist):
    return sorted(set(word for sentence in alist for word in sentence.split()))


# --------- execute the code ---------

# read in the data
ykk, stlen = uncsvip(fcl_st1)
li_ACH = ykk[0]     # the ACH-depmap_id's
li_sites = ykk[1]   # the corresponding cancer sites
li_cancer = ykk[2]  # the corresponding cancers

# collapse into words, then convert them with medcompound, and print
vocabi = vocit(li_cancer)
print('-- initial vocabulary --')
print(vocabi)
print('length =', len(vocabi))
print('')

li_sites = medcompound(li_sites)
li_cancer = medcompound(li_cancer)

vocabf = vocit(li_cancer)
print('-- converted vocabulary --')
print(vocabf)
print('length =', len(vocabf))
print('')

# print the results to file
with open(fsi_st2, 'w') as fout:
    for i in range(stlen):
        fout.write(li_ACH[i] + ',' + li_sites[i] + ',' + li_cancer[i] + '\n')
fout.close()


print('FIN')
sys.exit(0)
# FIN
