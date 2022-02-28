#!/usr/bin/env python3

import os, subprocess, re

PATH = os.path.dirname(os.path.realpath(__file__))
INDEX_EXE = os.path.join(PATH, 'index.py')
BTM_EXE = os.path.join(PATH, '../.build', 'btm')

class BTMInferrer:
    def __init__(self, model_dir):
        vocab_path = os.path.join(model_dir, 'vocab.txt')
        self.__indexer = subprocess.Popen([INDEX_EXE, vocab_path], stdin=subprocess.PIPE, stdout=subprocess.PIPE)

        vector_dir = os.path.join(model_dir, 'vectors')
        k = re.search(r'k(\d+)\.p(w_)?z$', os.listdir(vector_dir)[0]).group(1)
        self.__inferrer = subprocess.Popen([BTM_EXE, 'inf-d', 'sum_b', k, vector_dir + '/'], stdin=subprocess.PIPE, stdout=subprocess.PIPE)

    def __del__(self):
        self.__indexer.communicate()
        self.__inferrer.communicate()
        
    def infer(self, document):
        assert('\n' not in document)
        self.__indexer.stdin.write(bytes(document.strip() + '\n', 'utf-8'))
        self.__indexer.stdin.flush()
        self.__inferrer.stdin.write(self.__indexer.stdout.readline())
        self.__inferrer.stdin.flush()
        return [float(a.strip()) for a in self.__inferrer.stdout.readline().decode('utf-8').split()]

