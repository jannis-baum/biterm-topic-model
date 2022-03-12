#!/usr/bin/env python3

import os, shutil, sys, argparse, subprocess

PATH = os.path.dirname(os.path.realpath(__file__))
sys.path.append(PATH)

from index import index
from topics import output_topics

BTM_EXE = os.path.join(PATH,'..', '.build', 'btm')

def train(documents, model, k, alpha, beta, n_it, save_steps):
    model_dir = model
    shutil.rmtree(model_dir, ignore_errors=True)
    os.makedirs(model_dir)
    vocab_path = os.path.join(model_dir, 'vocab.txt')
    docIDs_path = os.path.join(model_dir, 'docs_ids.txt')

    print('INDEXING ...', end='')
    index(vocab_path, True, documents, docIDs_path)
    print(' done')

    vector_dir = os.path.join(model_dir, 'vectors')
    os.makedirs(vector_dir, exist_ok=True)

    print('TRAINING ...')
    with open(vocab_path, 'r') as vocab_fp: vocab_size = len(vocab_fp.readlines())
    subprocess.call([str(arg) for arg in [
        BTM_EXE, 'est', k, vocab_size, alpha, beta, n_it, save_steps, docIDs_path, vector_dir + '/'
    ]])
    print('done')

    print('WRITING TOPIC CSV ...', end='')
    output_topics(model_dir, out=os.path.join(model_dir, 'topics.csv'))
    print(' done')

    return docIDs_path
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('documents', help='file of documents, each line is a document')
    parser.add_argument('model', help='directory to create model in')
    parser.add_argument('--num-topics', '-k', dest='k', type=int, default=20, help='number of topics to generate; default 20')
    parser.add_argument('--alpha', '-a', dest='alpha', type=float, help='alpha; default 50/(num-topics)')
    parser.add_argument('--beta', '-b', dest='beta', type=float, default=0.005, help='beta; default 0.005')
    parser.add_argument('--num-iterations', '-n', dest='n_it', type=int, default=5, help='number of training iterations; default 5')
    parser.add_argument('--save-steps', '-s', dest='save_steps', type=int, default=500, help='number of iterations to save model after; default 500')

    args = parser.parse_args()
    if not args.alpha:
        args.alpha = float(50) / args.k

    train(args.documents, args.model, args.k, args.alpha, args.beta, args.n_it, args.save_steps)

