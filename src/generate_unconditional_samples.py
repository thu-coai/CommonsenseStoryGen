#!/usr/bin/env python3

import json
import os
import numpy as np
import tensorflow as tf

import model, sample, encoder
from utils import FLAGS

def sample_model(
    sess,
    enc,
    PAD_ID,
    hparams,
    temperature=1,
    top_k=0,
):
    """
    Run the sample_model
    :model_name=117M : String, which model to use
    :seed=None : Integer seed for random number generators, fix seed to
     reproduce results
    :nsamples=0 : Number of samples to return, if 0, continues to
     generate samples indefinately.
    :batch_size=1 : Number of batches (only affects speed/memory).
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

    length = hparams.n_ctx // 2 if FLAGS.length is None else FLAGS.length
    if length > hparams.n_ctx:
        raise ValueError("Can't get samples longer than window size: %s" % hparams.n_ctx)

    output = sample.sample_sequence(
        hparams=hparams, length=length,
        start_token=PAD_ID,
        batch_size=1,
        temperature=temperature, top_k=top_k
    )[:, 1:]

    generated = 0
    while generated < 1000:
        out = sess.run(output)
        generated += 1
        text = enc.decode(out[0], trunct=True)
        print("=" * 40 + " SAMPLE " + str(generated) + " " + "=" * 40)
        print(text)
