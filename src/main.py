import json
import os
import numpy as np
import tensorflow as tf
import time
import copy
import random
import model, sample, encoder
from utils import FLAGS, enc, PAD_ID, hparams, \
    load_data, load_data_kg, \
    gen_batched_data, gen_batched_data_from_kg
from interactive_conditional_samples import interact_model
from generate_unconditional_samples import sample_model
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu
print("Using %s-th gpu ..." % os.environ["CUDA_VISIBLE_DEVICES"])
train_dir = os.path.join(FLAGS.model_dir)
assert os.path.exists(train_dir)

def train(sess, dataset, is_train=True):
    def pro_acc(acc):
        final_acc = [[] for _ in range(FLAGS.n_class)]
        for ac in acc:
            for i in range(4):
                if np.sum(ac[i*4:(i+1)*4]) == 1:
                    final_acc[i].append(ac[i*4:(i+1)*4])
        for i in range(4):
            print("final classification confusion matrix (%d-th category):" % i, np.mean(final_acc[i], 0).tolist())
    st, ed = 0, 0
    loss, loss_lm, acc = [], [], []
    while ed < len(dataset):
        if is_train:
            output_feed = [
                model_loss,
                gradient_norm,
                update,
            ]
        else:
            output_feed = [
                model_loss,
            ]
        st, ed = ed, ed + FLAGS.batch_size if ed + \
            FLAGS.batch_size < len(dataset) else len(dataset)
        if FLAGS.data_name == "multi_roc" or FLAGS.data_name == "roc":
            batch_data = gen_batched_data(dataset[st:ed])
            input_feed = {
                context: batch_data["story"],
                context_length: batch_data["story_length"],
                label: batch_data["label"],
            }
            if FLAGS.data_name == "multi_roc": 
                output_feed.append(model_loss_lm)
                output_feed.append(model_acc_list)
        elif FLAGS.data_name == "kg":
            batch_data = gen_batched_data_from_kg(dataset[st:ed])
            input_feed = {
                context: batch_data["story"],
                context_length: batch_data["story_length"],
            }
        else:
            print("DATANAME ERROR")
        outputs = sess.run(output_feed, input_feed)
        loss.append(outputs[0])
        if FLAGS.data_name == "multi_roc":
            loss_lm.append(outputs[-2])
            acc.append(outputs[-1])
            if (st+1) % 10000 == 0:
                print("current ppl:%.4f"%np.exp(np.mean(loss_lm)))
                pro_acc(acc)
                print("="*5)
    if is_train:
        sess.run(epoch_add_op)
    if FLAGS.data_name == "multi_roc":
        pro_acc(acc)
        return np.exp(np.mean(loss_lm))
    else:
        return np.exp(np.mean(loss))

#**********************************************************************************
#**********************************************************************************
#**********************************************************************************

print("begin loading dataset......")
if FLAGS.data_name == "roc":
    data = load_data(FLAGS.data_dir, FLAGS.data_name, enc, label=0)
    data_segnum = len(data) / 20
    data_train = data[:int(data_segnum*18)]
    data_dev = data[int(data_segnum*18):int(data_segnum*19)]
    data_test = data[int(data_segnum*19):]
elif FLAGS.data_name == "kg":
    data_train = load_data_kg(FLAGS.data_dir, 'train', enc)
    data_dev = load_data_kg(FLAGS.data_dir, 'valid', enc)
    data_test = load_data_kg(FLAGS.data_dir, 'test', enc)
elif FLAGS.data_name == "multi_roc":
    data_train, data_dev, data_test = [], [], []
    data_name = ["roc", "roc_shuffle", "roc_replace", "roc_repeat"]
    assert FLAGS.n_class == len(data_name)
    for _, name in enumerate(data_name):
        if "shuffle" in name:
            label = 1
        elif "replace" in name:
            label = 2
        elif "repeat" in name:
            label = 3
        else:
            label = 0
        data = load_data(FLAGS.data_dir, name, enc, label=label)
        data_segnum = len(data) / 20
        data_train.extend(data[:int(data_segnum*18)])
        data_dev.extend(data[int(data_segnum*18):int(data_segnum*19)])
        data_test.extend(data[int(data_segnum*19):])
else:
    print("DATANAME ERROR")
    exit()
random.shuffle(data_train)
random.shuffle(data_dev)
random.shuffle(data_test)
print("Number of data for training:%d"%len(data_train))
print("Number of data for validation:%d"%len(data_dev))
print("Number of data for testing:%d"%len(data_test))

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    context = tf.placeholder(tf.int32, [None, None])
    context_length = tf.placeholder(tf.int32, [None])
    label = tf.placeholder(tf.int32, [None])
    
    context_entity = tf.placeholder(tf.int32, [None, None, None, 3, None])
    context_entity_mask = tf.placeholder(tf.int32, [None, None, None, 3, None])

    epoch = tf.Variable(0, trainable=False, name='initialize/epoch')
    epoch_add_op = epoch.assign(epoch + 1)
    if FLAGS.data_name == "multi_roc":
        model_loss, model_loss_lm, model_acc_list = model.train_classify(
            hparams=hparams,
            context=context,
            context_length=context_length,
            label=label,
            enc=enc,
            n_class=FLAGS.n_class,
            PAD_ID=PAD_ID,
        )
    else:
        model_loss = model.train(
            hparams=hparams,
            context=context,
            context_length=context_length,
            enc=enc,
            PAD_ID=PAD_ID,
        )

    params = tf.trainable_variables()
    for item in params:
        print('%s: %s' % (item.name, item.get_shape()))

    # initialize the training process
    learning_rate = tf.Variable(float(FLAGS.learning_rate), trainable=False, 
            dtype=tf.float32, name="initialize/learning_rate")
    learning_rate_decay_op = learning_rate.assign(learning_rate * 0.95)
    learning_rate_assign_op = learning_rate.assign(FLAGS.learning_rate)
    global_step = tf.Variable(1, trainable=False, name="initialize/global_step")

    max_gradient_norm = 5
    clipped_gradients, gradient_norm = tf.clip_by_global_norm(tf.gradients(model_loss, params), 
            max_gradient_norm)
    update = tf.train.AdamOptimizer(learning_rate).apply_gradients(zip(clipped_gradients, params), global_step=global_step)

    try:
        saver = tf.train.Saver(tf.global_variables(), write_version=tf.train.SaverDef.V2, 
                max_to_keep=10, pad_step_number=True, keep_checkpoint_every_n_hours=1.0)
        print("Reading model parameters from %s" % (train_dir))
        saver.restore(sess, tf.train.latest_checkpoint(train_dir))
    except:
        sess.run(tf.global_variables_initializer())
        try:
            restore_tensor = []
            for tensor in tf.global_variables():
                if "fine_tuning" not in tensor.name:
                    restore_tensor.append(tensor)
                else:
                    print("to-be-initialized parameter:", tensor.name)
            print("="*5)
            saver = tf.train.Saver(restore_tensor, write_version=tf.train.SaverDef.V2, 
                    max_to_keep=10, pad_step_number=True, keep_checkpoint_every_n_hours=1.0)
            saver.restore(sess, tf.train.latest_checkpoint(train_dir))
            print("Initialize the classifier parameter.")
        except:
            restore_tensor = []
            for tensor in tf.global_variables():
                if "beta" not in tensor.name and "initialize" not in tensor.name and "fine_tuning" not in tensor.name and "Adam" not in tensor.name:
                    restore_tensor.append(tensor)
                else:
                    print("to-be-initialized parameter:", tensor.name)
            print("="*5)
            saver = tf.train.Saver(restore_tensor, write_version=tf.train.SaverDef.V2, 
                    max_to_keep=10, pad_step_number=True, keep_checkpoint_every_n_hours=1.0)
            saver.restore(sess, tf.train.latest_checkpoint(train_dir))
            print("Initialize all the fine-tuning parameter.")
        saver = tf.train.Saver(tf.global_variables(), write_version=tf.train.SaverDef.V2, 
                max_to_keep=10, pad_step_number=True, keep_checkpoint_every_n_hours=1.0)
        saver.save(sess, '%s/checkpoint' % train_dir, global_step=global_step.eval())
        print("Reading model parameters from %s and initialize the parameters for fine-tuning." % (train_dir))

    if FLAGS.is_train:
        best_loss = 1e10
        pre_losses = [1e18] * 3
        while True:
            random.shuffle(data_train)
            start_time = time.time()
            loss = train(sess, data_train, is_train=True)
            if loss > max(pre_losses):  # Learning rate decay
                sess.run(learning_rate_decay_op)
            pre_losses = pre_losses[1:] + [loss]
            print("Gen epoch %d learning rate %.4f epoch-time %.4f: " % (epoch.eval(), learning_rate.eval(), time.time() - start_time))
            print("PPL on training set:", loss)
            loss = train(sess, data_test, is_train=False)
            print("        PPL on validation set:", loss)
            if loss < best_loss:
                best_loss = loss
                loss = train(sess, data_test, is_train=False)
                print("        PPL on testing set:", loss)
                saver.save(sess, '%s/checkpoint' % train_dir, global_step=global_step.eval())
                print("saving parameters in %s" % train_dir)
    else:
        if FLAGS.cond:
            print("begin conditionally generating stories......")
            interact_model(sess=sess,
                            enc=enc,
                            PAD_ID=PAD_ID,
                            hparams=hparams,
                            context=context,
                            dataset=data_test, #  Accept console input if `dataset` is set to None 
                            output_file_name="./inference_gpt2.txt",
                            temperature=FLAGS.temperature,
                            top_k=FLAGS.top_k)
        else:
            print("begin unconditionally generating stories......")
            sample_model(sess=sess,
                            enc=enc,
                            PAD_ID=PAD_ID,
                            hparams=hparams,
                            temperature=FLAGS.temperature,
                            top_k=FLAGS.top_k)
        print("end generating stories......")