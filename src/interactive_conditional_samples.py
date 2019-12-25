#!/usr/bin/env python3

import json
import os
import numpy as np
import tensorflow as tf

import model, sample, encoder
from utils import FLAGS, gen_batched_data, gen_batched_data_from_kg

def interact_model(
    sess,
    enc,
    PAD_ID,
    hparams,
    context,
    dataset=None,
    output_file_name=None,
    relation=None,
    temperature=1,
    top_k=0,
):
    """
    Interactively run the model
    :model_name=117M : String, which model to use
    :seed=None : Integer seed for random number generators, fix seed to reproduce
     results
    :length=None : Number of tokens in generated text, if None (default), is
     determined by model hyperparameters
    :temperature=1 : Float value controlling randomness in boltzmann
     distribution. Lower temperature results in less random completions. As the
     temperature approaches zero, the model will become deterministic and
     repetitive. Higher temperature results in more random completions.
    :top_k=0 : Integer value controlling diversity. 1 means only 1 word is
     considered for each step (token), resulting in deterministic completions,
     while 40 means 40 words are considered at each step. 0 (default) is a
     special setting meaning no restrictions. 40 generally is a good value.
     :models_dir : path to parent folder containing model subfolders
     (i.e. contains the <model_name> folder)     
    """

    if FLAGS.length > hparams.n_ctx:
        raise ValueError("Can't get samples longer than window size: %s" % hparams.n_ctx)
    batch_size = 1
    output = sample.sample_sequence(
        hparams=hparams, length=FLAGS.length,
        context=context,
        start_token=PAD_ID,
        batch_size=1,
        temperature=temperature,
        top_k=top_k,
    )
    if dataset is None:
        while True:
            raw_text = input("Model prompt >>> ")
            while not raw_text:
                print('Prompt should not be empty!')
                raw_text = input("Model prompt >>> ")
            context_tokens = enc.encode(raw_text)
            out = sess.run(output, feed_dict={
                context: [context_tokens for _ in range(batch_size)]
            })[:, len(context_tokens):]
            text = enc.decode(out[0], trunct=True).strip()
            print(text)
            print("=" * 80)
    else:
        fout = open(output_file_name, "w")
        st, ed = 0, 0
        while ed < len(dataset):
            st, ed = ed, ed + 1
            context_tokens = [[PAD_ID] + dataset[st]["st"][0]]
            out = sess.run(output, feed_dict={
                context: context_tokens,
            })
            for ipt, opt in zip(context_tokens, out):
                opt = enc.decode(opt[len(ipt):], trunct=True)
                ipt = enc.decode(ipt[1:], trunct=False)
                fout.write("ipt: " + ipt + "\n")
                fout.write("opt: " + opt + "\n")
                fout.write("-"*5+"\n")
        fout.close()