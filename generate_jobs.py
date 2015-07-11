__author__ = 'kostia'

import subprocess
import os

MORPH = "2014-12-17-MorphIntervalAnnot"
LPIP = "2015-02-04-LPIP"

SHORTCUT= {MORPH: "morph", LPIP: "lpip"}

FOLDER_PATH = "/datagrid/personal/antonkos/experiments/vilma"

def makedir_p(dirname):
    try:
        os.makedirs(dirname, mode=0777)
    except:
        pass

def isCaculated(folder_path, data_name, oracle_name, supervised_num, year_range, fraction, perm_id, lmbda):
    output_path = folder_path + ("/%s/%s-%d/year-%d/fraction-%d/" % (data_name, oracle_name, supervised_num, year_range, fraction))
    file_path = output_path + ("%d-%.4f.bin" % (perm_id, lmbda))
    return os.path.exists(file_path)

def print_sh_job_file(data_name, oracle_id, supervised_num, lmbda, permid, year_range, fraction):

    input_path = FOLDER_PATH + "/%s/%s/range%d/perm-%d/%s" % (data_name, SHORTCUT[data_name], year_range, permid,SHORTCUT[data_name])
    if supervised_num == fraction:
      output_path = FOLDER_PATH + ("/%s/%s-baseline/year-%d/fraction-%d/" % (data_name, oracle_id,
                                                                     year_range, fraction))
    else:
        output_path = FOLDER_PATH + ("/%s/%s-%d/year-%d/fraction-%d/" % (data_name, oracle_id,
                                                                     supervised_num, year_range, fraction))

    makedir_p(output_path)

    output_filename = output_path + ("%d-%.4f" % (permid, lmbda))

    filename = "job-%s-Year%d-Sup%d-Frac%d-Perm%d-L%.5f" % \
           (oracle_id, year_range, supervised_num, fraction, permid, lmbda)

    file_body = """#!/bin/bash
YEAR=%d
BIN_PATH=/home.dokt/antonkos/code/vilma/build/main
DATA_PATH=/datagrid/personal/antonkos/experiments/vilma/%s

""" % (year_range, data_name)

    file_body += "${BIN_PATH}  %s %s %s %d %d %.3f" % \
                 (input_path, output_filename, oracle_id, supervised_num, fraction, lmbda)

    if not isCaculated(FOLDER_PATH, data_name, oracle_id, supervised_num, year_range, fraction, permid, lmbda):
        f = file(filename + ".sh", 'w')
        print >> f, file_body
        f.close()

if __name__ == "__main__":
    target_oracles = ["SingleGenderAgeBmrmOracle",
                      "SingleGenderNoBetaBmrmOracle"]

    Loss = "ZOLoss"
    data_name = MORPH
    for supervised_num in [3300]:
        #for fraction in [3300, 6600, 11000, 16000, 21000]:
        for fraction in [3300, 6600, 10000, 13000, 23000, 33000]:
            if fraction < supervised_num:
                continue
            for oracle_id in ["SingleGenderNoBetaBmrmOracle"]:
                for year_range in [5]:
                    for lmbda in [0.1, 0.01]:
                        for permid in [1, 2, 3]:
                            print_sh_job_file(data_name, oracle_id + "-" + Loss, supervised_num, 
                                                      lmbda, permid, year_range, fraction)