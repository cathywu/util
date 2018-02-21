import numpy as np
import subprocess, os, re, time, zipfile, gzip, io, shutil, string, random, itertools, pickle, json, codecs

def get_name(path, ext=True):
    name = os.path.basename(path)
    if ext: return name
    else: return os.path.splitext(name)[0]

def list_dir(dir_, ext, return_name=False):
    if ext == '/':
        criteria = lambda x: os.path.isdir(os.path.join(dir_, x))
        strip_ext = lambda x: x
    else:
        if ext[0] != '.': ext = '.' + ext.lower()
        criteria = lambda x: x[-len(ext):].lower() == ext
        strip_ext = lambda x: x[:-len(ext)]
    files = (f for f in os.listdir(dir_) if criteria(f))
    files = sorted((strip_ext(file), os.path.join(dir_, file)) for file in files)
    if return_name: return files
    else: return [path for file, path in files]

def load_json(path):
    with open(path, 'rb') as f:
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

def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path

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

def get_temp_file(ext, N=10, parent_dir=None):
    if parent_dir is None: parent_dir = '/tmp/zxyan_temp'
    return parent_dir + ''.join(random.sample(string.digits, k=N)) + ext

def get_temp_dir(N=10):
    return get_temp_file('/', N)

def shell(cmd, wait=True, ignore_error=2):
    if type(cmd) != str:
        cmd = ' '.join(cmd)
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if not wait:
        return process
    out, err = process.communicate()
    if err:
        if ignore_error == 2:
            pass
        elif ignore_error:
            print(err.decode())
        else:
            print(err.decode())
            raise RuntimeError('Error in command line call')
    return out.decode()

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
     