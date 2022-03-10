#!/usr/bin/env python3

import os, pexpect, re

PATH = os.path.dirname(os.path.realpath(__file__))
INDEX_EXE = os.path.join(PATH, 'index.py')
BTM_EXE = os.path.join(PATH, '../.build', 'btm')

class BTMInferrer:
    def __init__(self, model_dir):
        vocab_path = os.path.join(model_dir, 'vocab.txt')
        self.__indexer = pexpect.spawn(INDEX_EXE, [vocab_path], echo=False)

        vector_dir = os.path.join(model_dir, 'vectors')
        k = re.search(r'k(\d+)\.p(w_)?z$', os.listdir(vector_dir)[0]).group(1)
        self.__inferrer = pexpect.spawn(BTM_EXE, ['inf-d', 'sum_b', k, vector_dir + '/'], echo=False)

    def __del__(self):
        self.__indexer.close()
        self.__inferrer.close()
        
    def infer(self, document):
        assert('\n' not in document)
        self.__indexer.sendline(document.strip())
        self.__inferrer.sendline(self.__indexer.readline().strip())
        return [float(a.strip()) for a in self.__inferrer.readline().split()]

