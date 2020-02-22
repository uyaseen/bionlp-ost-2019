from pathlib import Path
import os
import glob
import codecs
import json
import shutil
import _pickle as cPickle


def get_filename(path):
    return os.path.basename(path).split('.')[0]


def join_path(path_a, path_b):
    return os.path.join(path_a, path_b)


def get_files(path, ext):
    return list(set(glob.glob(path + '*.' + ext) + glob.glob(path + '*.' + ext.upper())))


def get_nested_files(path, ext):
    files = []
    for f_name in Path(path).glob('**/*.{}'.format(ext)):
        files.append(f_name)
    return files


def filter_files(files, filter_str):
    filtered_files = []
    for file in files:
        if filter_str in file:
            filtered_files.append(file)
    return filtered_files


def file_exists(path):
    return os.path.exists(path)


def delete_file(path):
    os.remove(path)


def read_pickle(path):
    with open(path, 'rb') as f:
        return cPickle.load(f)


def write_pickle(data, path):
    with open(path, 'wb') as f:
        cPickle.dump(data, f)
    print('{} created'.format(path))


def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)


def delete_directory(path):
    shutil.rmtree(path)


def get_parent_directory(path):
    return os.path.dirname(path)


def write_json(data, path):
    with codecs.open(path, 'w', 'utf-8') as f:
        json.dump(data, f, indent=4, sort_keys=True)
        print('{} created'.format(path))
