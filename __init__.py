from __future__ import absolute_import, print_function

import subprocess, sys, os, re, tempfile, zipfile, gzip, io, shutil, string, random, itertools, pickle, json
from datetime import datetime
from time import time
from glob import glob
from collections import OrderedDict, defaultdict, Counter
import pdb
d = d_ = pdb.set_trace
import warnings
warnings.filterwarnings('ignore')

if sys.version_info[0] < 3: 
    from StringIO import StringIO
else:
    from io import StringIO

def load_json(path):
    with open(path, 'r+') as f:
        return json.load(f)

def save_json(path, dict_):
    with open(path, 'w+') as f:
        json.dump(dict_, f, indent=4, sort_keys=True)

def format_json(dict_):
    return json.dumps(dict_, indent=4, sort_keys=True)

def load_text(path):
    with open(path, 'r+') as f:
        return f.read()

def save_text(path, string):
    with open(path, 'w+') as f:
        f.write(string)

def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def save_pickle(path, obj):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)

def wget(link, output_dir):
    cmd = 'wget %s -P %s' % (path, output_dir)
    shell(cmd)
    output_path = Path(output_dir) / os.path.basename(link)
    if not output_path.exists(): raise RuntimeError('Failed to run %s' % cmd)
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

def attributes(obj):
    import inspect, pprint
    pprint.pprint(inspect.getmembers(obj, lambda a: not inspect.isroutine(a)))

def import_module(module_name, module_path):
    import imp
    module = imp.load_source(module_name, module_path)
    return module

class Path(str):
    @classmethod
    def get_project(cls, path=None):
        cwd = Path(path or os.getcwd())
        while cwd and cwd != '/':
            proj_path = cwd / 'project.py'
            if proj_path.exists():
                break
            cwd = cwd._up
        else:
            return None
        project = import_module('project', str(proj_path))
        return project.project
    
    def __init__(self, path):
        pass
        
    def __add__(self, subpath):
        return Path(str(self) + str(subpath))
    
    def __truediv__(self, subpath):
        return Path(os.path.join(str(self), str(subpath)))
    
    def __floordiv__(self, subpath):
        return (self / subpath)._
    
    def ls(self, show_hidden=True, dir_only=False, file_only=False):
        subpaths = [Path(self / subpath) for subpath in os.listdir(self) if show_hidden or not subpath.startswith('.')]
        isdirs = [os.path.isdir(subpath) for subpath in subpaths]
        subdirs = [subpath for subpath, isdir in zip(subpaths, isdirs) if isdir]
        files = [subpath for subpath, isdir in zip(subpaths, isdirs) if not isdir]
        if dir_only:
            return subdirs
        if file_only:
            return files
        return subdirs, files
    
    def recurse(self, dir_fn=None, file_fn=None):
        if dir_fn is not None:
            dir_fn(self)
        dirs, files = self.ls()
        if file_fn is not None:
            list(map(file_fn, files))
        for dir in dirs:
            dir.recurse(dir_fn=dir_fn, file_fn=file_fn)
        
    def mk(self):
        if not self.exists():
            os.makedirs(self)
        return self
    
    def rm(self):
        if self.isfile() or self.islink():
            os.remove(self)
        elif self.isdir():
            shutil.rmtree(self)
        return self
    
    def mv(self, dest):
        shutil.move(self, dest)

    def mv_from(self, src):
        shutil.move(src, self)
    
    def cp(self, dest):
        shutil.copy(self, dest)
    
    def cp_from(self, src):
        shutil.copy(src, self)
    
    def link(self, target, force=False):
        if self.exists():
            if not force:
                return
            else:
                self.rm()
        os.symlink(target, self)

    def exists(self):
        return os.path.exists(self)
    
    def isfile(self):
        return os.path.isfile(self)
    
    def isdir(self):
        return os.path.isdir(self)

    def islink(self):
        return os.path.islink(self)
    
    def rel(self, start=None):
        return Path(os.path.relpath(self, start=start))
    
    @property
    def _(self):
        return str(self)

    @property
    def _real(self):
        return Path(os.path.realpath(self))
    
    @property
    def _up(self):
        return Path(os.path.dirname(self))
    
    @property
    def _name(self):
        return os.path.basename(self)
    
    @property
    def _ext(self):
        frags = self._name.split('.', 1)
        if len(frags) == 1:
            return ''
        return frags[1]

    load_json = load_json
    save_json = save_json
    load_text = load_text
    save_text = save_text
    load_pickle = load_pickle
    save_pickle = save_pickle
    extract = extract
    
    def load(self):
        return eval('self.load_%s' % self._ext)()

class Namespace(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
    
    def var(self, *args, **kwargs):
        for a in args:
            kwargs[a] = True
        self.__dict__.update(kwargs)
        return self
    
    def unvar(self, *args):
        for a in args:
            self.__dict__.pop(a)
        return self
    
    def get(self, key, default=None):
        return self.__dict__.get(key, default)


##### Functions for compute

using_ipython = True
try:
    _ = get_ipython().__class__.__name__
except NameError:
    using_ipython = False

try:
    import numpy as np
    import pandas as pd

    import scipy.stats
    import scipy as sp
    from scipy.stats import pearsonr as pearson, spearmanr as spearman, kendalltau

    if not using_ipython:
        import matplotlib
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except ImportError:
    pass
try:
    from sklearn.metrics import roc_auc_score as auroc, average_precision_score as auprc, roc_curve as roc, precision_recall_curve as prc, accuracy_score as accuracy
except ImportError:
    pass

def recurse(x, fn):
    T = type(x)
    if T in [dict, OrderedDict]:
        return T((k, recurse(v, fn)) for k, v in x.items())
    elif T in [list, tuple]:
        return T(recurse(v, fn) for v in x)
    return fn(x)

def from_numpy(x):
    def helper(x):
        if type(x).__module__ == np.__name__:
            if type(x) == np.ndarray:
                return recurse(list(x), helper)
            return np.asscalar(x)
        return x
    return recurse(x, helper)

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

def get_gpu_info():
    nvidia_str, _ = shell('nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,nounits')
    nvidia_str = nvidia_str.replace('[Not Supported]', '100').replace(', ', ',')
    nvidia_str_io = StringIO(nvidia_str)

    gpu_df = pd.read_csv(nvidia_str_io, index_col=0)
    devices_str = os.environ.get('CUDA_VISIBLE_DEVICES')
    if devices_str:
        devices = list(map(int, devices_str.split(',')))
        gpu_df = gpu_df.loc[devices]
        gpu_df.index = gpu_df.index.map({k: i for i, k in enumerate(devices)})
    
    gpu_df['memory'] = gpu_df['memory.total [MiB]'] - gpu_df['memory.used [MiB]']
    gpu_df['utilization'] = 1 - gpu_df['utilization.gpu [%]'] / 100
    return gpu_df

def get_gpu_process_info(pid=None):
    nvidia_str, _ = shell('nvidia-smi --query-compute-apps=pid,gpu_name,used_gpu_memory --format=csv,nounits')
    nvidia_str_io = StringIO(nvidia_str.replace(', ', ','))

    gpu_df = pd.read_csv(nvidia_str_io, index_col=0)
    if pid is None:
        return gpu_df
    if pid == -1:
        pid = os.getpid()
    return gpu_df.loc[pid]


##### torch functions

try:
    import torch
    import torch.nn as nn

    def to_torch(x, device='cuda'):
        def helper(x):
            if x is None:
                return None
            elif type(x) == torch.Tensor:
                return x.to(device)
            elif type(x) in [str, bool, int, float]:
                return x
            return torch.from_numpy(x).to(device)
        return recurse(x, helper)

    def from_torch(t):
        def helper(t):
            x = t.detach().cpu().numpy()
            if x.size == 1 or np.isscalar(x):
                return np.asscalar(x)
            return x
        return recurse(t, helper)

    class Flatten(nn.Module):
        def forward(self, input):
            return input.view(input.size(0), -1)
except ImportError:
    pass
