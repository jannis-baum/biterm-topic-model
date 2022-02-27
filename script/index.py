#!/usr/bin/env python3

import sys, argparse

def __index_new(file_in, file_out):
    vocab = dict()
    def id(token):
        if token not in vocab: vocab[token] = str(len(vocab))
        return vocab[token]

    for line in file_in:
        tokens = line.strip().split()
        file_out.write(' '.join([id(token) for token in tokens]) + '\n')
    
    return vocab

def __index_existing(vocab, file_in, file_out):
    for line in file_in:
        tokens = line.strip().split()
        file_out.write(' '.join([vocab[token] for token in tokens if token in vocab]) + '\n')

def index(filename_vocab, create_new, filename_in, filename_out):
    file_in = open(filename_in, 'r') if filename_in else sys.stdin
    file_out = open(filename_out, 'w') if filename_out else sys.stdout

    if create_new:
        vocab = __index_new(file_in, file_out)
        with open(filename_vocab, 'w') as vocab_fp:
            vocab_fp.write('\n'.join([
                f'{id}\t{token}' for token, id in sorted(vocab.items(), key=lambda v: int(v[1]))
            ]))
        
    else:
        with open(filename_vocab, 'r') as vocab_fp:
            vocab = dict([
                line.strip().split('\t')[::-1]
            for line in vocab_fp])
        __index_existing(vocab, file_in, file_out)
    
    file_in.close()
    file_out.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('vocab', help='word-id lookup file')
    parser.add_argument('-n', '--new-vocab', dest='create_new', action='store_true', help='create new vocab file instead of reading an existing one')
    parser.add_argument('-i', '--in', dest='filename_in', metavar='file', type=str, help='file to read documents from (stdin if omitted)')
    parser.add_argument('-o', '--out', dest='filename_out', metavar='file', type=str, help='file to outputs indeces to (stdout if omitted)')

    args = parser.parse_args()
    index(args.vocab, args.create_new, args.filename_in, args.filename_out)

