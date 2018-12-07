using_ipython = True
try:
    shell = get_ipython().__class__.__name__
except NameError:
    using_ipython = False

import subprocess, os, re, tempfile, zipfile, gzip, io, shutil, string, random, itertools, pickle, json
from time import time
from glob import glob
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
if not using_ipython:
    import matplotlib
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.stats
import scipy as sp
from scipy.stats import pearsonr as pearson, spearmanr as spearman, kendalltau
from sklearn.metrics import roc_auc_score as auroc, average_precision_score as auprc, roc_curve as roc, precision_recall_curve as prc, accuracy_score as accuracy

def get_name(path, ext=True):
    name = os.path.basename(path)
    if ext: return name
    else: return os.path.splitext(name)[0]

def load_json(path):
    with open(path, 'r+') as f:
        return json.load(f)

def save_json(dict_, path):
    with open(path, 'wb') as f:
        json.dump(dict_, codecs.getwriter('utf-8')(f), indent=4, sort_keys=True)

def load_text(path):
    with open(path, 'r+') as f:
        return f.read()

def save_text(string, path):
    with open(path, 'w+') as f:
        f.write(string)

def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def save_pickle(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)

def remove(path):
    if not os.path.exists(path):
        return
    elif os.path.isfile(path):
        os.remove(path)
    else:
        shutil.rmtree(path)

def wget(link, output_directory):
    cmd = 'wget %s -P %s' % (path, output_directory)
    shell(cmd)
    output_path = os.path.join(os.path.basename(link))
    if not os.path.exists(output_path): raise RuntimeError('Failed to run %s' % cmd)
    return output_path

def extract(input_path, output_path=None):
    if input_path[-3:] == '.gz':
        if not output_path:
            output_path = input_path[:-3]
        with gzip.open(input_path, 'rb') as f_in:
            with open(output_path, 'wb') as f_out:
                f_out.write(f_in.read())
    else:
        raise RuntimeError('Don\'t know file extension for ' + input_path)

def shell(cmd, wait=True, ignore_error=2):
    if type(cmd) != str:
        cmd = ' '.join(cmd)
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if not wait:
        return process
    out, err = process.communicate()
    return out.decode(), err.decode() if err else None

def parallel_execution(process_fn, n_jobs, error_msg=''):
    '''
    process_fn: fn that takes in the thread index (0 to n_job - 1) returns a process and one other output
    '''
    processes, outputs = zip(*[process_fn(i) for i in range(n_jobs)])
    outs, errs = zip(*[process.communicate() for process in processes])
    error = False
    for err in errs:
        if err:
            print(err.decode('UTF-8'))
            error = True
    if error:
        raise RuntimeError(error_msg)
    return outputs

def attributes(obj):
    import inspect, pprint
    pprint.pprint(inspect.getmembers(obj, lambda a: not inspect.isroutine(a)))

def reindex(df, order=None, rename=None, level=[], axis=0, squeeze=True):
    assert axis in [0, 1]
    if type(level) is not list:
        if order is not None: order = [order]
        if rename is not None: rename = [rename]
        level = [level]
    if order is None: order = [[]] * len(level)
    if rename is None: rename = [{}] * len(level)
    assert len(level) == len(rename) == len(order)
    multiindex = df.index
    if axis == 1:
        multiindex = df.columns
    for i, (o, lev) in enumerate(zip(order, level)):
        if len(o) == 0:
            seen = set()
            new_o = []
            for k in multiindex.get_level_values(lev):
                if k in seen: continue
                new_o.append(k)
                seen.add(k)
            order[i] = new_o
    assert len(set(level) - set(multiindex.names)) == 0, 'Levels %s not in index %s along axis %s' % (level, axis, multiindex.names)
    lev_order = dict(zip(level, order))
    level_map = {}
    for lev in multiindex.names:
        if lev in level:
            level_map[lev] = { name : i for i, name in enumerate(lev_order[lev]) }
        else:
            index_map = {}
            for x in multiindex.get_level_values(lev):
                if x in index_map: continue
                index_map[x] = len(index_map)
            level_map[lev] = index_map
    tuples = list(multiindex)
    def get_numerical(tup):
        return tuple(level_map[lev][t] for t, lev in zip(tup, multiindex.names))
    filtered_tuples = [tup for tup in tuples if all(t in level_map[lev] for t, lev in zip(tup, multiindex.names))]
    new_tuples = sorted(filtered_tuples, key=get_numerical)
    lev_rename = dict(zip(level, rename))
    renamed_tuples = [tuple(lev_rename.get(lev, {}).get(t, t) for t, lev in zip(tup, multiindex.names)) for tup in new_tuples]
    new_index = pd.MultiIndex.from_tuples(new_tuples, names=multiindex.names)
    renamed_index = pd.MultiIndex.from_tuples(renamed_tuples, names=multiindex.names)
    if squeeze:
        single_levels = [i for i, level in enumerate(renamed_index.levels) if len(level) == 1]
        renamed_index = renamed_index.droplevel(single_levels)
    if axis == 0:
        new_df = df.loc[new_index]
        new_df.index = renamed_index
    else:
        new_df = df.loc[:, new_index]
        new_df.columns = renamed_index
    return new_df
