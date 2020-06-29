from __future__ import absolute_import, print_function

import subprocess, sys, os, re, tempfile, zipfile, gzip, io, shutil, string, random, itertools, pickle, json, yaml, gc
from datetime import datetime
from time import time
from fnmatch import fnmatch
from glob import glob
from tqdm import tqdm
from copy import copy, deepcopy
from collections import OrderedDict, defaultdict, Counter
import q
qq = q
import warnings
warnings.filterwarnings('ignore')

version = sys.version_info
if version[0] < 3:
    from StringIO import StringIO
else:
    from io import StringIO

def lrange(*args, **kwargs):
    return list(range(*args, **kwargs))

class Dict(dict if version.major == 3 and version.minor >= 6 else OrderedDict):
    def __add__(self, d):
        return Dict(**self).merge(d)

    def merge(self, *dicts, **kwargs):
        for d in dicts:
            self.update(d)
        self.update(kwargs)
        return self

    def filter(self, keys):
        try: # check for iterable
            keys = set(keys)
            return Dict((k, v) for k, v in self.items() if k in keys)
        except TypeError: # function key
            f = keys
            return Dict((k, v) for k, v in self.items() if f(k, v))

    def map(self, mapper):
        if callable(mapper): # function mapper
            return Dict((k, mapper(v)) for k, v in self.items())
        else: # dictionary mapper
            return Dict((k, mapper[v]) for k, v in self.items())

def parse_dot(d):
    ks = [(k, v) for k, v in d.items() if '.' in k]
    for k, v in ks:
        del d[k]
        curr = d
        *fronts, back = k.split('.')
        for k_ in fronts:
            curr = curr.setdefault(k_, {})
        curr[back] = v
    return d

def load_json(path):
    with open(path, 'r+') as f:
        return json.load(f)

def save_json(path, dict_):
    with open(path, 'w+') as f:
        json.dump(dict_, f, indent=4, sort_keys=True)

def format_json(dict_):
    return json.dumps(dict_, indent=4, sort_keys=True)

def format_yaml(dict_):
    dict_ = recurse(dict_, lambda x: x._ if type(x) is Path else dict(x) if type(x) is Dict else x)
    return yaml.dump(dict_)

def load_text(path, encoding='utf-8'):
    with open(path, 'r', encoding=encoding) as f:
        return f.read()

def save_text(path, string):
    with open(path, 'w') as f:
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

def rand_string(length):
    import string
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(length))

nexti = nextk = lambda iterable: next(iter(iterable))
nextv = lambda dict: next(iter(dict.values()))
nextkv = lambda dict: next(iter(dict.items()))

def tmux_window(cmd, session='', window='', directory=None):
    def flag(cmds, flag, value):
        if value:
            cmds.extend([flag, value])
        return cmds

    cmds = []
    # if window exists, skip everything
    if window:
        cmds.extend(['tmux', 'list-panes', '-t', '%s:%s' % (session, window)])
        cmds.append('||')

    # else if session exists
    subcmds = ['tmux', 'has-session']
    flag(subcmds, '-t', session)
    subcmds.append('&&')

    # then call new-window
    subcmds.extend(['tmux', 'new-window', '-d'])
    flag(subcmds, '-t', session)
    flag(subcmds, '-n', window)
    flag(subcmds, '-c', directory)
    subcmds.append("'%s'" % cmd)

    cmds.append('(%s)' % ' '.join(subcmds))
    cmds.append('||')

    # else new-session
    cmds.extend(['tmux', 'new-session', '-d'])
    flag(cmds, '-s', session)
    flag(cmds, '-n', window)
    flag(cmds, '-c', directory)

    cmds.append("'%s'" % cmd)
    return ' '.join(cmds)

def ssh(user, host, cmd, key=None, password=None, terminal=False):
    cmds = ['ssh']
    if key is not None:
        cmds.extend(['-i', key])
    if password is not None:
        cmds = ['sshpass', '-p', password] + cmds
    if terminal:
        cmds.append('-t')
    cmds.append('%s@%s' % (user, host))
    cmds.append('"%s"' % cmd)
    return ' '.join(cmds)

def shell(cmd, wait=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE):
    stdout = stdout or subprocess.DEVNULL
    stderr = stderr or subprocess.DEVNULL
    if type(cmd) != str:
        cmd = ' '.join(cmd)
    process = subprocess.Popen(cmd, shell=True, stdout=stdout, stderr=stderr)
    if not wait:
        return process
    out, err = process.communicate()
    return out.decode().rstrip('\n') if out else '', err.decode().rstrip('\n') if err else ''

def terminal_height():
    return int(shell('tput lines')[0])

def terminal_width():
    return int(shell('tput cols')[0])

def git_state(dir=None):
    cwd = os.getcwd()
    dir = dir or shell('git rev-parse --show-toplevel')[0]
    os.chdir(dir)
    status = shell('git status')[0]
    base_commit = shell('git rev-parse HEAD')[0]
    diff = shell('git diff %s' % base_commit)[0]
    os.chdir(cwd)
    return base_commit, diff, status

def attributes(obj, print=True):
    import inspect
    attrs = inspect.getmembers(obj, lambda a: not inspect.isroutine(a))
    if print:
        from pprint import pprint
        pprint(attrs)
    return attrs

def import_module(module_name, module_path):
    import imp
    module = imp.load_source(module_name, module_path)
    return module

def shorten(names, sep='_', new_sep='_'):
    names = [n.split(sep) for n in names]
    levels = [*itertools.zip_longest(*names, fillvalue='')]
    def rename(level):
        uniq = np.unique(level)
        keep_len = 1
        while True:
            new_uniq = np.unique([u[:keep_len] for u in uniq])
            if len(new_uniq) == len(uniq):
                mapping = dict(zip(uniq, new_uniq))
                return [mapping[x] for x in level]
            keep_len += 1
    new_levels = [*map(rename, levels)]
    return [new_sep.join(tup).rstrip(new_sep) for tup in zip(*new_levels)]

t_ = {}
def profile_start(*names):
    for name in names or ['anon']:
        t_[name] = [time()]

def profile(obj, name='anon'):
    t_[name].append(time())
    return obj

def profile_i(iterable, name='anon'):
    for obj in iterable:
        yield profile(obj, name=name)

def profile_end(name='anon'):
    times = t_.pop(name)
    t0 = times[0]
    times = [t - t0 for t in times[1:]]
    return np.array(times)

def debugger():
    import ptvsd
    ptvsd.enable_attach()
    ptvsd.wait_for_attach()

def get_time_log_path():
    return datetime.now().isoformat().replace(':', '_').rsplit('.')[0] + '.log'

_log_path = None
def logger(directory=None):
    global _log_path
    if directory and not _log_path:
        from datetime import datetime
        _log_path = Path(directory) / get_time_log_path()
    return log

def log(text):
    print(text)
    if _log_path:
        with open(_log_path, 'a') as f:
            f.write(text)
            f.write('\n')

def installed(pkg):
    out, err = shell('dpkg -l %s' % pkg)
    if err and err.startswith('dpkg-query: no packages found matching'):
        return False
    return True

def install(pkgs, root):
    root = Path(root)
    old_cwd = os.getcwd()
    self_installed = set()
    os.chdir(root)
    while pkgs:
        pkg = pkgs.pop()
        print('Processing %s' % pkg)
        if installed(pkg) or pkg in self_installed:
            continue
        out, err = shell('apt-cache depends %s' % pkg)
        deps = []
        for x in out.split('\n'):
            x = x.lstrip()
            if x.startswith('Depends:'):
                splits = x.split(' ')
                assert len(splits) == 2
                dep = splits[1]
                if not (dep in self_installed or installed(dep)):
                    deps.append(dep)
        print('Found needed dependencies %s for %s' % (deps, pkg))
        pkgs.extend(deps)
        tmp = Path('tmp')
        shell('mkdir tmp && cd tmp && apt download %s' % pkg)
        for deb in tmp.glob('*.deb'):
            shell('dpkg -x %s %s' % (deb, root))
            print('Installing %s with %s' % (pkg, deb))
            self_installed.add(pkg)
        tmp.rm()
    lib = Path('usr/lib')
    real_root = Path('/')
    for x in lib, lib / 'x86_64-linux-gnu':
        brokens = x.lslinks(exist=False)
        for broken in brokens:
            real = real_root / broken._up / os.readlink(broken)
            if real.exists():
                broken.link(real, force=True)
                print('Fixing broken link to be %s -> %s' % (broken, real))
            else:
                print('Could not fix broken link %s' % broken)
    os.chdir(old_cwd)

class Path(str):
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

    def lsdirs(self, show_hidden=True):
        return self.ls(show_hidden=show_hidden, dir_only=True)

    def lsfiles(self, show_hidden=True):
        return self.ls(show_hidden=show_hidden, file_only=True)

    def lslinks(self, show_hidden=True, exist=None):
        dirs, files = self.ls(show_hidden=show_hidden)
        return [x for x in dirs + files if x.islink() and (
            exist is None or not (exist ^ x.exists()))]

    def glob(self, glob_str):
        return [Path(p) for p in glob(self / glob_str, recursive=True)]

    def recurse(self, dir_fn=None, file_fn=None):
        if dir_fn is not None:
            dir_fn(self)
        dirs, files = self.ls()
        if file_fn is not None:
            list(map(file_fn, files))
        for dir in dirs:
            dir.recurse(dir_fn=dir_fn, file_fn=file_fn)

    def mk(self):
        os.makedirs(self, exist_ok=True)
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
        if self.lexists():
            if not force:
                return
            else:
                self.rm()
        os.symlink(target, self)

    def exists(self):
        return os.path.exists(self)

    def lexists(self):
        return os.path.lexists(self)

    def isfile(self):
        return os.path.isfile(self)

    def isdir(self):
        return os.path.isdir(self)

    def islink(self):
        return os.path.islink(self)

    def rel(self, start=None):
        return Path(os.path.relpath(self, start=start))

    def clone(self):
        name = self._name
        match = re.search('__([0-9]+)$', name)
        if match is None:
            base = self + '__'
            i = 1
        else:
            initial = match.group(1)
            base = self[:-len(initial)]
            i = int(initial) + 1
        while True:
            path = Path(base + str(i))
            if not path.exists():
                return path
            i += 1


    @property
    def _(self):
        return str(self)

    @property
    def _real(self):
        return Path(os.path.realpath(os.path.expanduser(self)))

    @property
    def _up(self):
        path = os.path.dirname(self.rstrip('/'))
        if path is '':
            path = os.path.dirname(self._real.rstrip('/'))
        return Path(path)

    @property
    def _name(self):
        return os.path.basename(self)

    @property
    def _ext(self):
        frags = self._name.rsplit('.', 1)
        if len(frags) == 1:
            return ''
        return frags[1]

    extract = extract
    load_json = load_json
    save_json = save_json
    load_txt = load_text
    save_txt = save_text
    load_p = load_pickle
    save_p = save_pickle

    def save_bytes(self, bytes):
        with open(self, 'wb') as f:
            f.write(bytes)

    def load_csv(self, index_col=0, **kwargs):
        return pd.read_csv(self, index_col=index_col, **kwargs)

    def save_csv(self, df, float_format='%.5g', **kwargs):
        df.to_csv(self, float_format=float_format, **kwargs)

    def load_npy(self):
        return np.load(self, allow_pickle=True)

    def save_npy(self, obj):
        np.save(self, obj)

    def load_yaml(self):
        with open(self, 'r') as f:
            return yaml.safe_load(f)

    def save_yaml(self, obj):
        obj = recurse(obj, lambda x: x._ if type(x) is Path else dict(x) if type(x) is Dict else x)
        with open(self, 'w') as f:
            yaml.dump(obj, f, default_flow_style=False, allow_unicode=True)

    def load_pth(self):
        return torch.load(self)

    def save_pth(self, obj):
        torch.save(obj, self)

    def load_pdf(self):
        """
        return: PdfReader object.
        Can use index and slice obj.pages for the pages, then call Path.save_pdf to save
        """
        from pdfrw import PdfReader
        return PdfReader(self)

    def save_pdf(self, pages):
        from pdfrw import PdfWriter
        writer = PdfWriter()
        writer.addpages(pages)
        writer.write(self)

    def load(self):
        return eval('self.load_%s' % self._ext)()

    def save(self, obj):
        return eval('self.save_%s' % self._ext)(obj)

    def wget(self, link):
        if self.isdir():
            return Path(wget(link, self))
        raise ValueError('Path %s needs to be a directory' % self)


class Namespace(Dict):
    def __init__(self, *args, **kwargs):
        self.var(*args, **kwargs)

    def var(self, *args, **kwargs):
        kvs = Dict()
        for a in args:
            if type(a) is str:
                kvs[a] = True
            else: # a is a dictionary
                kvs.update(a)
        kvs.update(kwargs)
        self.update(kvs)
        return self

    def unvar(self, *args):
        for a in args:
            self.pop(a)
        return self

    def setdefaults(self, *args, **kwargs):
        args = [a for a in args if a not in self]
        kwargs = {k: v for k, v in kwargs.items() if k not in self}
        return self.var(*args, **kwargs)

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as e:
            self.__getattribute__(key)

    def __setattr__(self, key, value):
        self[key] = value

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
    plt_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    arrayf = lambda *args, **kwargs: np.array(*args, **kwargs, dtype=np.float32)
    arrayl = lambda *args, **kwargs: np.array(*args, **kwargs, dtype=np.long)
    arrayb = lambda *args, **kwargs: np.array(*args, **kwargs, dtype=np.bool)
    arrayo = lambda *args, **kwargs: np.array(*args, **kwargs, dtype=object)

    def _sel(self, col, value):
        if type(value) == list:
            return self[self[col].isin(value)]
        return self[self[col] == value]
    pd.DataFrame.sel = _sel
except ImportError:
    pass
try:
    from sklearn.metrics import roc_auc_score as auroc, average_precision_score as auprc, roc_curve as roc, precision_recall_curve as prc, accuracy_score as accuracy
except ImportError:
    pass

def flatten(x):
    return [z for y in x for z in y]

def recurse(x, fn):
    T = type(x)
    if isinstance(x, dict):
        return T((k, recurse(v, fn)) for k, v in x.items())
    elif isinstance(x, (list, tuple)):
        return T(recurse(v, fn) for v in x)
    return fn(x)

def from_numpy(x):
    def helper(x):
        if type(x).__module__ == np.__name__:
            if isinstance(x, np.ndarray):
                return recurse(list(x), helper)
            return np.asscalar(x)
        return x
    return recurse(x, helper)

def smooth(y, box_pts):
    box = np.ones(box_pts) / box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

def normalize(x, eps=1e-8):
    return (x - x.mean()) / x.std()

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

def get_gpu_info(ssh_fn=lambda x: x):
    nvidia_str, _ = shell(ssh_fn('nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,nounits'))
    nvidia_str = nvidia_str.replace('[Not Supported]', '100').replace(', ', ',')
    nvidia_str_io = StringIO(nvidia_str)

    gpu_df = pd.read_csv(nvidia_str_io, index_col=0)
    devices_str = os.environ.get('CUDA_VISIBLE_DEVICES')
    if devices_str:
        devices = list(map(int, devices_str.split(',')))
        gpu_df = gpu_df.loc[devices]
        gpu_df.index = gpu_df.index.map({k: i for i, k in enumerate(devices)})

    out_df = pd.DataFrame(index=gpu_df.index)
    out_df['memory_total'] = gpu_df['memory.total [MiB]']
    out_df['memory_used'] = gpu_df['memory.used [MiB]']
    out_df['memory_free'] = out_df['memory_total'] - out_df['memory_used']
    out_df['utilization'] = gpu_df['utilization.gpu [%]'] / 100
    out_df['utilization_free'] = 1 - out_df['utilization']
    return out_df

def get_process_gpu_info(pid=None, ssh_fn=lambda x: x):
    nvidia_str, _ = shell(ssh_fn('nvidia-smi --query-compute-apps=pid,gpu_name,used_gpu_memory --format=csv,nounits'))
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
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader

    def to_torch(x, device='cuda' if torch.cuda.is_available() else 'cpu', **kwargs):
        def helper(x):
            if x is None:
                return None
            elif isinstance(x, torch.Tensor):
                return x.to(device=device, **kwargs)
            elif np.isscalar(x):
                return x
            return torch.from_numpy(x).to(device=device, **kwargs)
        return recurse(x, helper)

    def from_torch(t, force_scalar=False):
        def helper(t):
            if not isinstance(t, torch.Tensor):
                return t
            x = t.detach().cpu().numpy()
            if force_scalar and (x.size == 1 or np.isscalar(x)):
                return np.asscalar(x)
            return x
        return recurse(t, helper)

    def count_params(network, requires_grad=False):
        return sum(p.numel() for p in network.parameters() if not requires_grad or p.requires_grad)

    def report_memory(device=None, max=False):
        if device:
            device = torch.device(device)
            if max:
                alloc = torch.cuda.max_memory_allocated(device=device)
            else:
                alloc = torch.cuda.memory_allocated(device=device)
            alloc /=  1024 ** 2
            print('%.3f MBs' % alloc)
            return alloc

        numels = Counter()
        for obj in gc.get_objects():
            if torch.is_tensor(obj):
                print(type(obj), obj.size())
                numels[obj.device] += obj.numel()
        print()
        for device, numel in sorted(numels.items()):
            print('%s: %s elements, %.3f MBs' % (str(device), numel, numel * 4 / 1024 ** 2))

    def clear_gpu_memory():
        gc.collect()
        torch.cuda.empty_cache()
        for obj in gc.get_objects():
            if torch.is_tensor(obj):
                obj.cpu()
        gc.collect()
        torch.cuda.empty_cache()

    class Reshape(nn.Module):
        def __init__(self, *shape, split=None, merge=None):
            super(Reshape, self).__init__()
            self.shape = shape
            self.split = split
            self.merge = merge

        def forward(self, input):
            if self.split is None and self.merge is None:
                return input.reshape(*self.shape)
            in_shape = input.shape

    class Transpose(nn.Module):
        def __init__(self, dim0, dim1):
            super(Transpose, self).__init__()
            self.dim0 = dim0
            self.dim1 = dim1

        def forward(self, input):
            return input.transpose(self.dim0, self.dim1)

    class Permute(nn.Module):
        def __init__(self, *dims):
            super(Permute, self).__init__()
            self.dims = dims

        def forward(self, input):
            return input.permute(*self.dims)

    class CausalConv1d(nn.Module):
        def __init__(self, in_depth, out_depth, kernel_size, dilation=1, stride=1, groups=1):
            super(CausalConv1d, self).__init__()
            self.padding = (kernel_size - 1) * dilation
            self.conv = nn.Conv1d(in_depth, out_depth, kernel_size, stride=stride, dilation=dilation, groups=groups)

        def forward(self, x, pad=True):
            if pad:
                x = F.pad(x, (self.padding, 0))
            return self.conv(x)

    class CausalMaxPool1d(nn.Module):
        def __init__(self, kernel_size, dilation=1, stride=1):
            super(CausalMaxPool1d, self).__init__()
            self.padding = (kernel_size - 1) * dilation
            self.pool = nn.MaxPool1d(kernel_size, stride=stride, dilation=dilation)

        def forward(self, x, pad=True):
            if pad:
                x = F.pad(x, (self.padding, 0))
            return self.pool(x)

except ImportError:
    pass

from .exp import Config

# deprecated

try:
    import visdom

    class Visdom(visdom.Visdom):
        def line(self, Y, X=None, win=None, env=None, opts={}, update='append', name=None):
            all_opts = Dict(title=win, showlegend=True).merge(opts)
            if update == 'remove':
                all_opts = None
            super(Visdom, self).line(Y=Y, X=X, win=win, env=env, opts=all_opts, update=update, name=name)

    _visdom_cache = {}
    def get_visdom(env='main', server=None, port=None, raise_exceptions=True, **kwargs):
        server = server or os.environ['VISDOM_SERVER']
        port = port or os.environ['VISDOM_PORT']
        key = (server, port, env or 'main')
        if key not in _visdom_cache:
            _visdom_cache[key] = Visdom(server=server, port=port, env=env, raise_exceptions=raise_exceptions, **kwargs)
        return _visdom_cache[key]

except ImportError:
    pass
