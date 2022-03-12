#!/usr/bin/env python3

import os, pexpect, re, subprocess

PATH = os.path.dirname(os.path.realpath(__file__))
INDEX_EXE = os.path.join(PATH, 'index.py')
BTM_EXE = os.path.join(PATH, '../.build', 'btm')

def _find_k(vector_dir):
    return re.search(r'k(\d+)\.p(w_)?z$', os.listdir(vector_dir)[0]).group(1)

def _get_vocab(model_dir):
    return os.path.join(model_dir, 'vocab.txt')

def _get_vector_dir(model_dir):
    return os.path.join(model_dir, 'vectors')

class BTMInferrer:
    def __init__(self, model_dir):
        vocab_path = _get_vocab(model_dir)
        self.__indexer = pexpect.spawn(INDEX_EXE, [vocab_path], echo=False)

        vector_dir = _get_vector_dir(model_dir)
        k = _find_k(vector_dir)
        self.__inferrer = pexpect.spawn(BTM_EXE, ['inf-d', 'sum_b', k, vector_dir + '/'], echo=False)

    def __del__(self):
        self.__indexer.close()
        self.__inferrer.close()
        
    def infer(self, document):
        assert('\n' not in document)
        self.__indexer.sendline(document.strip())
        self.__inferrer.sendline(self.__indexer.readline().strip())
        return [float(a.strip()) for a in self.__inferrer.readline().split()]

def infer_file(model_dir, file_in, file_out, should_index=True):
    if should_index:
        word_file = file_in
        file_in += '.temp_indexed.txt'
        subprocess.run([INDEX_EXE, _get_vocab(model_dir), '-i', word_file, '-o', file_in])

    vector_dir = _get_vector_dir(model_dir)
    subprocess.run([BTM_EXE, 'inf', 'sum_b', _find_k(vector_dir), vector_dir + '/', file_in, file_out])

    if should_index: os.remove(file_in)

