#!/usr/bin/env python
#  intdep_util = interDep utilities module
#  head -n 14 intdep_util.py
#  python3: import intdep_util as idu
#  By:  Charlie Payne
#  License: n/a
# DESCRIPTION
#  contains common values/lists/etc and functions used throughout interDep
# NOTES
#  [none]
# KNOWN BUGS
#  [none]
# DESIRED FEATURES
#  [none]

from __future__ import print_function
import csv
import sys

# data directory names
ddin = '../data/input/'
ddea = '../data/out_mister/'


# data file names
finDru = ddin + 'Drug_sensitivity_(PRISM_Repurposing_Primary_Screen)_19Q4.csv'
finDep = ddin + 'CRISPR_gene_dependency_Chronos.csv'
finEff = ddin + 'CRISPR_gene_effect.csv'
feaDru = ddea + 'drug.csv'
feaDep = ddea + 'dependency.csv'
feaEff = ddea + 'effect.csv'
fclSt1 = ddea + 'states_mrclean.csv'
fsiSt2 = ddea + 'states_mrsinatra.csv'


# ~~~ function definitions ~~~

def eprint(*args, **kwargs):
    sys.stdout.flush()  # so stederr doesn't redirect to top of file with 2>&1
    print(*args, file=sys.stderr, **kwargs)


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


def uncsvip(filename):
    with open(filename, 'r', newline='\n') as fin:
        reader = csv.reader(fin)
        indata = list(reader)
    fin.close()
    outlist = []
    dlen = len(indata)
    for i in range(len(indata[0])):
        outlist.append([])
        for j in range(dlen):
            outlist[i].append(indata[j][i])
    return outlist, dlen
