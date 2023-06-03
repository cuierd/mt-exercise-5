#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Preprocessing training data:
# Sub-sample parallel training data randomly to 100k sentence pairs to make
# training manageable on small machines.

import argparse
import random
import logging


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--sample_size", type=int, help="sentence count for sample size", required=True)
    parser.add_argument("--src_input", type=str, help="parallel data input file source language", required=True)
    parser.add_argument("--trg_input", type=str, help="parallel data input file target language", required=True)
    parser.add_argument("--src_output", type=str, help="sub-sample of parallel data source language output file", required=True)
    parser.add_argument("--trg_output", type=str, help="sub-sample of parallel data target language output file", required=True)

    args = parser.parse_args()

    return args


def write_sample_file(infilename: str, outfilename: str, sample: list):
    """
    Writes sub-sample of sentences from input file to output file
    parameter: filenames of input and output files, set of sample integer
    return: None
    """
    with open(outfilename, 'w', encoding='utf-8') as outfile:
        with open(infilename, 'r', encoding='utf-8') as infile:
            for index, sentence in enumerate(infile):
                if index in sample:
                    outfile.write(f'{sentence}')


def main():

    args = parse_args()

    logging.basicConfig(level=logging.DEBUG)
    logging.debug(args)

    src_inp_file = args.src_input
    src_out_file = args.src_output

    trg_inp_file = args.trg_input
    trg_out_file = args.trg_output

    num_lines = sum(1 for line in open(src_inp_file, 'r', encoding='utf-8'))
    sample = random.sample(range(0,num_lines-1), args.sample_size)

    write_sample_file(src_inp_file, src_out_file, sample)
    write_sample_file(trg_inp_file, trg_out_file, sample)


if __name__ == '__main__':
    main()
