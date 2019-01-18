import subprocess, os, re, tempfile, zipfile, gzip, io, shutil, string, random, itertools, pickle, json
from time import time
from glob import glob
import pdb
d = d_ = pdb.set_trace
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

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
    
    def ls(self, hidden=False):
        subpaths = [Path(self / subpath) for subpath in os.listdir(self) if not hidden or not subpath.startswith('.')]
        isdirs = [os.path.isdir(subpath) for subpath in subpaths]
        subdirs = [subpath for subpath, isdir in zip(subpaths, isdirs) if isdir]
        files = [subpath for subpath, isdir in zip(subpaths, isdirs) if not isdir]
        return subdirs, files
        
    def mk(self):
        if not self.exists():
            os.makedirs(self)
        return self
    
    def rm(self):
        if not self.exists():
            pass
        elif self.isfile():
            os.remove(self)
        else:
            shutil.rmtree(self)
        return self
    
    def cp(self, dest):
        shutil.copy(self, dest)
    
    def link(self, target):
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

class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
    
    def update(self, **kwargs):
        self.__dict__.update(kwargs)
    
    def get(self, key, default=None):
        return self.__dict__.get(key, default)