"""Evaluation for arggen, wikigen, and absgen."""
import numpy as np
import json

import argparse

from eval_utils import evaluate_bleu_match
from eval_utils import evaluate_rouge_match, RougeScorer

def run_evaluation(ref_data, sys_data):
    # 1. run BLEU
    evaluate_bleu_match(ref_data, sys_data)

    # 2. run ROUGE
    evaluate_rouge_match(ref_data, sys_data)
    return

def print_output_statistics(sys_data):
    """Report the following:
    1. # samples
    2. # words per sample
    """
    word_cnt = []

    for ln in sys_data.values():
        if len(ln['pred']) == 0: continue
        if isinstance(ln['pred'][0], list):
            for ix, toks in enumerate(ln['pred']):
                cur_word_cnt = len(toks)
                word_cnt.append(cur_word_cnt)
        else:
            cur_word_cnt = len(ln['pred'])
            word_cnt.append(cur_word_cnt)

    print("{} samples evaluated".format(len(word_cnt)))
    print("{:.2f} words per sample".format(np.mean(word_cnt)))


def load_sys_output(fname):
    data = dict()
    path = 'data/system/{}'.format(fname)
    for ln in open(path):
        cur_obj = json.loads(ln)
        tid = cur_obj['tid']
        pred = cur_obj['pred']
        data[tid] = {'pred': pred}

    return data


def load_ref_data(task):
    data = dict()
    path = 'data/reference/{}_test.jsonl'.format(task)
    for ln in open(path):
        cur_obj = json.loads(ln)
        tid = cur_obj['tid']
        tgt = cur_obj['rr']
        tgt_words = []
        for sent in tgt:
            tgt_words.extend(sent)

        if not tid in data:
            data[tid] = []
        data[tid].append({'tgt_words': tgt_words,
                          'tgt_words_by_sent': tgt,})
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', choices=['arggen'], required=True)
    parser.add_argument('--model', type=str, required=True)
    args = parser.parse_args()

    sysname2path = {
        'arggen': {'ours': 'arggen_ours.jsonl'},
        #'wikigen': {'ours': ''},
        #'absgen': {'ours': ''},
    }

    sys_data = load_sys_output(sysname2path[args.task][args.model])
    ref_data = load_ref_data(args.task)

    print_output_statistics(sys_data)
    run_evaluation(ref_data, sys_data)


if __name__=='__main__':
    main()
