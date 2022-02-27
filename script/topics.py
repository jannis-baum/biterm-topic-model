#!/usr/bin/env python3

import os, sys, argparse

def __read_vocab(filepath):
    with open(filepath, 'r') as vocab_fp:
        return dict([
            line.strip().split('\t')
        for line in vocab_fp])

def __read_pz(filepath):
    with open(filepath, 'r') as pz_fp:
        return [float(p) for p in pz_fp.readline().split()]

def __topics(vocab, pz, pwz_filepath, n_words):
    with open(pwz_filepath, 'r') as pwz_fp:
        return [
            (pz[i], [vocab[str(id)] for id, _ in
                sorted(enumerate([float(v) for v in line.split()]), key=lambda t: t[1], reverse=True)[:n_words]
            ])
        for i, line in enumerate(pwz_fp.readlines())]

def output_topics(model, num_words=10, out=None):
    vector_dir = os.path.join(model, 'vectors')
    vector_paths = [os.path.join(vector_dir, f) for f in os.listdir(vector_dir)]
    pz_path = next(p for p in vector_paths if p.endswith('pz'))
    pwz_path = next(p for p in vector_paths if p.endswith('pw_z'))

    vocab = __read_vocab(os.path.join(model, 'vocab.txt'))
    pz = __read_pz(pz_path)

    topics = __topics(vocab, pz, pwz_path, num_words)

    fileout = open(out, 'w') if out else sys.stdout
    fileout.write('topic\tprob_topic\ttop_words\n')
    fileout.write('\n'.join([
        f'{i}\t{topic[0]}\t{", ".join(topic[1])}'
    for i, topic in enumerate(topics)]) + '\n')
    fileout.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model', help='directory model is in')
    parser.add_argument('-o', '--out', dest='filename_out', metavar='file', type=str, help='file to output topics to (stdout if omitted)')
    parser.add_argument('-n', '--num-words', dest='num_words', type=int, default=10, help='number of top words; default 10')
    args = parser.parse_args()

    output_topics(args.model, args.num_words, args.filename_out)

