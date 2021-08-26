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
ddmr = '../data/out_mister/'
ddho = '../data/out_ho/'

# data file names
fin_dru = ddin + 'Drug_sensitivity_(PRISM_Repurposing_Primary_Screen)_19Q4.csv'
fin_dep = ddin + 'CRISPR_gene_dependency_Chronos.csv'
fin_eff = ddin + 'CRISPR_gene_effect.csv'
fea_dru = ddmr + 'drug.csv'
fea_dep = ddmr + 'dependency.csv'
fea_eff = ddmr + 'effect.csv'
fcl_st1 = ddmr + 'states_mrclean.csv'
fsi_st2 = ddmr + 'states_mrsinatra.csv'
fho_stf = ddho + 'states_final.csv'


# ~~~ function definitions ~~~

# eprint: print to stderr
#  in:   << same as print >>
#  out:  << print to stderr >>
def eprint(*args, **kwargs):
    sys.stdout.flush()  # so stederr doesn't redirect to top of file with 2>&1
    print(*args, file=sys.stderr, **kwargs)


# uncsvip: unzip (so to speak) csv columns into rows of array
#  in:   filename = name of input file
#  out:  outlist = list where rows are csv columns from filename,
#        dlen = the length of the columns from filename
def uncsvip(filename):
    # read in data to get started
    with open(filename, 'r', newline='\n') as fin:
        reader = csv.reader(fin)
        indata = list(reader)
    fin.close()
    
    outlist = []
    dlen = len(indata)
    # add empty list to outlist to be filled as rows from columns of filename
    for i in range(len(indata[0])):
        outlist.append([])
        for j in range(dlen):
            outlist[i].append(indata[j][i])
    return outlist, dlen
