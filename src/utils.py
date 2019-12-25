import numpy as np
import tensorflow as tf
import sample, model, encoder
import json
import os
import random
tf.app.flags.DEFINE_integer("is_train", 1, "Set to 1/0 to train/inference.")
tf.app.flags.DEFINE_integer("cond", 1, "Set to 1/0 to generate stories unconditionally/conditionally on the beginning.")
tf.app.flags.DEFINE_string("model_dir", "./model/gpt2", "Model directory.")
tf.app.flags.DEFINE_string("gpu", "0", "Specify which gpu to use.")
tf.app.flags.DEFINE_integer("batch_size", 10, "Number of batches (only affects speed/memory).")
tf.app.flags.DEFINE_string("data_name", "roc", "Set `roc` to train the model on ROCStories corpus or \
                                                `kg` to train the model on the knowledge bases or\
                                                `multi_roc` to train the model on ROCStories with multi-task learning.")
tf.app.flags.DEFINE_integer("n_class", 4, "Number of classes for the auxiliary classification task.")
tf.app.flags.DEFINE_float("learning_rate", 1e-4, "Learning rate.")
tf.app.flags.DEFINE_string("data_dir", "./data", "Data directory.")
tf.app.flags.DEFINE_integer("length", 200, "Number of tokens in generated text.")
tf.app.flags.DEFINE_float("temperature", 0.7, "Float value controlling randomness in boltzmann distribution. Lower temperature results in less random completions. As the temperature approaches zero, the model will become deterministic and repetitive. Higher temperature results in more random completions.")
tf.app.flags.DEFINE_integer("top_k", 40, "Integer value controlling diversity.")
FLAGS = tf.app.flags.FLAGS
FLAGS.is_train = bool(FLAGS.is_train)
FLAGS.cond = bool(FLAGS.cond)
model_dir = os.path.expanduser(os.path.expandvars(FLAGS.model_dir))
enc = encoder.get_encoder(model_dir)
PAD_ID = enc.encoder['<|endoftext|>']
hparams = model.default_hparams()
with open(os.path.join(model_dir, 'hparams.json')) as f:
    hparams.override_from_dict(json.load(f))

def load_data(path, fname, enc, label):
    data = []
    print('loading %s/%s ......' % (path, fname))
    with open('%s/%s.txt' % (path, fname)) as f:
        tmp = []
        for k, line in enumerate(f):
            i = k + 1
            if i % 6 == 0:
                data.append({"st": tmp, "label": label})
                tmp = []
            else:
                tmp.append(enc.encode(line.strip().replace(" .", ". ")))
    return data

def load_data_kg(path, fname, enc):
    with open("%s/%s.txt" % (path, fname)) as fin:
        data = [enc.encode(line.strip()) for line in fin]
    return data

def padding(sent, l):
    return sent + [PAD_ID] * (l-len(sent))

def gen_batched_data(data):
    max_story_len = max([sum([len(item["st"][i]) for i in range(5)]) for item in data]) + 1
    max_input_story_len = max([len(item["st"][0]) for item in data]) + 1
    story, story_length, label = [], [], []
    input_story, input_story_length = [], []

    for item in data:
        input_story.append(padding(item["st"][0], max_input_story_len))
        input_story_length.append(len(item["st"][0]) + 1)    
        story.append([])
        for i in range(5):
            story[-1] += item["st"][i]
        story_length.append(len(story[-1]) + 1)
        story[-1] = padding(story[-1], max_story_len)
        label.append(item["label"])
    batched_data = {
        "story": np.array(story),
        "story_length": np.array(story_length),
        "input_story": np.array(input_story),
        "input_story_length": np.array(input_story_length),
        "label": np.array(label),
    }
    return batched_data

def gen_batched_data_from_kg(data):
    max_story_len = max([len(item) for item in data]) + 1
    story, story_length = [], []
    for item in data:
        story.append(padding(item, max_story_len))
        story_length.append(len(item) + 1)
    batched_data = {
        "story": np.array(story),
        "story_length": np.array(story_length),
    }
    return batched_data
