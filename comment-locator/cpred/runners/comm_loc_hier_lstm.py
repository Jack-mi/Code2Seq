"""
Run LSTM based predictor for comment locations 

Author: Annie Louis
Project: Comment prediction

"""
from __future__ import print_function

import os
import time
import sys
import logging
from datetime import timedelta

import numpy as np
import tensorflow as tf
from cpred.data import cloc_data_reader as data_helper
from cpred.nn import hier_lstm as predictor


flags = tf.flags

# paths
flags.DEFINE_string("data_path", None, "path to training")
flags.DEFINE_boolean("skip_empty_stmt", "False", "ignore all empty statements")
flags.DEFINE_string("train_dir", None, "output directory")
flags.DEFINE_boolean("test", False, "set to true for classifying a test set")

# data-related settings
flags.DEFINE_integer("code_vocab_size", 5000, "vocabulary size of input texts")
flags.DEFINE_integer("max_back_size", 1, "we dont care")
flags.DEFINE_integer("max_blks", 100, "number of blocks in a sequence")
flags.DEFINE_integer("max_len_blk", 200, "maximum number of steps in the lstm sequence")
flags.DEFINE_integer("max_len_stmt", 100, "maximum length of code string in tokens")
flags.DEFINE_integer("num_labels", 2, "number of labels in the classifier")
flags.DEFINE_boolean("is_bpe", False, "set to True if the input is bpe segmented")

# lstm-related parameters
flags.DEFINE_integer("lstm_num_layers", 1, "number of layers for encoder")
flags.DEFINE_integer("lstm_hidden_size", 300, "hidden size for encoder")
flags.DEFINE_integer("stmt_embed_size", 300, "size of the input word embeddings")
flags.DEFINE_string("stmt_embed_type", "avg_wembed", "type of sentence embedding to use (word_counts, avg_wembed, list_wids)")
flags.DEFINE_boolean("use_pretraining", False, "whether to use pretrained embeddings")

#other network parameters
flags.DEFINE_float("keep_prob", 1.0, "keep probability")
flags.DEFINE_integer("max_epoch", 200, "max epochs to run")
flags.DEFINE_integer("test_batch_size", 64, "batch size during test")
flags.DEFINE_integer("batch_size", 64, "batch size")
flags.DEFINE_float("init_scale", 0.05, "initializing scale")
flags.DEFINE_float("learning_rate", 0.05, "learning rate")
flags.DEFINE_float("max_grad_norm", 5.0, "clip gradients to this norm")
flags.DEFINE_float("lr_decay", 0.99, "learning rate decay")
flags.DEFINE_integer("valid_num_nonimproving", 5, "number of epochs wherein terminate if validation performance does not increase")
flags.DEFINE_integer("checkpoint_every_n_epochs", 20, "number of epochs after which to checkpoint")

FLAGS = flags.FLAGS


def get_gpu_config():
  gconfig = tf.ConfigProto()
  # gconfig.gpu_options.per_process_gpu_memory_fraction = 0.4
  # gconfig.allow_soft_placement = True
  return gconfig


def compute_PRF_2D_confmat(conf_mat, total_true_class_1):
  ncorr = conf_mat[0][0] + conf_mat[1][1]
  acc = ncorr / np.sum(conf_mat) * 100
  total_pred_class_1 = conf_mat[0][1] + conf_mat[1][1]
  if total_pred_class_1 == 0:
    precision = 100.0
  else:
    precision = conf_mat[1][1] * 1.0 / (total_pred_class_1) * 100
  if total_true_class_1 == 0:
    recall = 100.0
  else:
    recall = conf_mat[1][1] * 1.0 / (total_true_class_1) * 100
  fscore = 0.0
  if precision > 0.0 and recall > 0.0:
    fscore = 2 * precision * recall / (precision + recall)

  result_string = "Confusion matrix\n"
  result_string += "True\Pred\t0\t1\n"
  result_string += "\t0\t" + str(conf_mat[0][0]) + "\t" + str(conf_mat[0][1]) + "\n"
  result_string += "\t1\t" + str(conf_mat[1][0]) + "\t" + str(conf_mat[1][1]) + "\n"
  result_string += "total items = " + str(np.sum(conf_mat)) + ", ncorrect = " + str(ncorr) + "\n"
  result_string += "Accuracy = " + str(acc) + "%\n"
  result_string += "Precision = " + str(precision) + "%\n"
  result_string += "Recall = " + str(recall) + "%\n"
  result_string += "Fscore = " + str(fscore) + "%\n"

  return (acc, precision, recall, fscore, result_string)



def create_data(train_path, valid_path,
                skip_empty_stmt,
                max_len_stmt, max_back_size,
                code_vocab_size, 
                ignore_unknown, ignore_special,
                w2v_models, w2v_dims):


  print("Creating training dataset..")
  file_seqs, all_words, loc_back_terms, corpus_back_terms = data_helper.read_comment_location_file(train_path + "/data.txt", train_path + "/corpus_file_back.txt", max_len_stmt, skip_empty_stmt, None, None, None, None)


  (wordv, 
   loc_back_wordv, 
   corpus_back_wordv, 
   all_wordv, all_word_w2i, all_word_i2w) = data_helper.create_vocabularies(all_words,
                                                                            loc_back_terms,
                                                                            corpus_back_terms, 
                                                                            code_vocab_size,
                                                                            max_back_size)
    
  train_dataset = data_helper.prepare_loc_data(file_seqs, 
                                                 max_len_stmt,  
                                                 wordv, 
                                                 loc_back_wordv,
                                                 corpus_back_wordv,
                                                 all_wordv, all_word_w2i, all_word_i2w,
                                                 w2v_models, w2v_dims, 
                                                 ignore_unknown, ignore_special, 
                                                 False, False,
                                                 False)


  print("Creating validation dataset..")
  valid_file_seqs, valid_all_words, valid_loc_back_terms, valid_corpus_back_terms = \
    data_helper.read_comment_location_file(valid_path + "/data.txt", \
                                           valid_path + "/corpus_file_back.txt", \
                                           max_len_stmt, \
                                           skip_empty_stmt, None, None, None, None)

  valid_dataset = data_helper.prepare_loc_data(valid_file_seqs, 
                                                 max_len_stmt,
                                                 wordv, 
                                                 loc_back_wordv,
                                                 corpus_back_wordv,
                                                 all_wordv, all_word_w2i, all_word_i2w,
                                                 w2v_models, w2v_dims, 
                                                 ignore_unknown, ignore_special,
                                                 False, False,
                                                 False)
  return train_dataset, valid_dataset
  

def main(_):
    if not FLAGS.data_path:
      raise ValueError("Must set --data_path to directory with subdirectories train/valid/test")

    type_sent_embed = FLAGS.stmt_embed_type
    if not type_sent_embed in data_helper.sembed_options:
      raise ValueError("Must set --type_sent_embed to one of " + str(sembed_options.keys))
    
    train_path = FLAGS.data_path + "/train"
    valid_path = FLAGS.data_path + "/valid"
    test_path = FLAGS.data_path + "/test"
    output_dir = FLAGS.train_dir
    
    tuning = False
    ignore_unknown = True
    ignore_special = True
    stmt_embed_size = FLAGS.stmt_embed_size
    is_bpe = FLAGS.is_bpe
    
    w2v_models, w2v_dims = None, None
    if data_helper.sembed_options[type_sent_embed] == data_helper.AVG_WEMBED:
      w2v_models, w2v_dims = data_helper.read_w2v_models(is_bpe)                                

    if FLAGS.test:
      train_dataset, test_dataset = create_data(train_path, test_path, FLAGS.skip_empty_stmt, 
                                                 FLAGS.max_len_stmt, FLAGS.max_back_size,  
                                                 FLAGS.code_vocab_size, 
                                                 ignore_unknown, 
                                                 ignore_special,
                                                 w2v_models, w2v_dims)
      """
      IMPORTANT to do the following: we only know the size of the stmt after the training 
      vocab is computed
      """
      code_vocab_size = len(train_dataset.all_word_w2i)
      sent_embed_size = FLAGS.stmt_embed_size
      vocab_size = code_vocab_size

      config = predictor.Config(FLAGS.init_scale, FLAGS.learning_rate, FLAGS.max_grad_norm, 
                                FLAGS.max_blks, FLAGS.max_len_blk, FLAGS.max_len_stmt,
                                FLAGS.lstm_num_layers, FLAGS.lstm_hidden_size,
                                FLAGS.max_epoch, FLAGS.keep_prob, FLAGS.lr_decay, 
                                FLAGS.batch_size, FLAGS.test_batch_size, FLAGS.num_labels, 
                                vocab_size, 
                                data_helper.sembed_options[type_sent_embed],
                                sent_embed_size,
                                FLAGS.valid_num_nonimproving, FLAGS.checkpoint_every_n_epochs,
                                data_helper.HIER_LSTM_MODEL,
                                FLAGS.use_pretraining
      )

      results = test_model(test_dataset, output_dir, config)
      print("Done testing!")
      print("\n\n Done!")

    else:
      train_dataset, valid_dataset = create_data(train_path, valid_path, FLAGS.skip_empty_stmt, 
                                                 FLAGS.max_len_stmt, FLAGS.max_back_size,  
                                                 FLAGS.code_vocab_size, 
                                                 ignore_unknown, 
                                                 ignore_special,
                                                 w2v_models, w2v_dims)
      """
      IMPORTANT to do the following: we only know the size of the stmt after the training 
      vocab is computed
      """
      code_vocab_size = len(train_dataset.all_word_w2i)
      sent_embed_size = FLAGS.stmt_embed_size
      vocab_size = code_vocab_size

      config = predictor.Config(FLAGS.init_scale, FLAGS.learning_rate, FLAGS.max_grad_norm, 
                                FLAGS.max_blks, FLAGS.max_len_blk, FLAGS.max_len_stmt,
                                FLAGS.lstm_num_layers, FLAGS.lstm_hidden_size,
                                FLAGS.max_epoch, FLAGS.keep_prob, FLAGS.lr_decay, 
                                FLAGS.batch_size, FLAGS.test_batch_size, FLAGS.num_labels, 
                                vocab_size, 
                                data_helper.sembed_options[type_sent_embed],
                                sent_embed_size,
                                FLAGS.valid_num_nonimproving, FLAGS.checkpoint_every_n_epochs,
                                data_helper.HIER_LSTM_MODEL,
                                FLAGS.use_pretraining
      )

      (trained_model, train_epochs, time_taken,
       train_loss, valid_loss,
       valid_acc, valid_prec,
       valid_recall, valid_fscore,
       valid_gr_acc, valid_gr_prec,
       valid_gr_recall, valid_gr_fscore) = train_model(train_dataset, valid_dataset, 
                                                       output_dir, config, tuning)
      print("Done training!")
      print("\n\n Done!")
    

def test_model(test_dataset, output_dir, config):

    start_time = time.time()

    with tf.Graph().as_default():
      # if tuning:
      # tf.set_random_seed(10)
      # np.random.seed(30)
        
      with tf.Session(config=get_gpu_config()) as session:
        md = predictor.create_model(session, config, output_dir)
        print("Model parameters = " + str(md.get_parameter_count()))

        print("Creating test batched data")
        test_batched_data = test_dataset.create_batch_data(config.batch_size, 
                                                             config.max_blks_in_seq,
                                                             config.max_sents_in_blk, 
                                                             config.embed_type, 
                                                             config.model_type)
        print("Making predictions")
        truel, preds = md.predict(session, test_batched_data)
        pred_blks = len(preds)
        
    fp = open(output_dir + "/test.pred", "w")
    with open(output_dir + "/test.pred.header", "w") as fh:
      fh.write("pred\ttrue\tgpred\tprob1\n")
    fp_info = open(output_dir + "/test.info", "w")

    conf_mat = np.zeros((2, 2), dtype=np.int32)
    gr_conf_mat = np.zeros((2, 2), dtype=np.int32)
    test_examples = [loc for seq in test_dataset.file_seqs for loc in seq.get_all_locs()]

    labels = [e.label for e in test_examples]
    gr_labels = [e.gr_label for e in test_examples]
    true_num_blks = sum([1 for e in test_examples if e.blk_bd == 1])
    
    assert pred_blks == true_num_blks, "Number of true blocks %d does not match predicted number %d" % (true_num_blks, pred_blks)
    
    pred_num = -1
    for lnum in range(len(labels)):
      test_example_info = test_examples[lnum].__str__()
      true_label = labels[lnum]
      true_gr_label = labels[lnum]
      if test_examples[lnum].blk_bd == 1:
        pred_num += 1
        pred_label = preds[pred_num][0]
        pred_prob = preds[pred_num][1]
        conf_mat[true_label][pred_label] += 1
        if true_gr_label != -1:
          gr_conf_mat[true_gr_label][pred_label] += 1
      else:
        pred_label = -1
        pred_prob = 0.0
      fp.write(str(pred_label) + "\t" + str(true_label) + "\t" + str(true_gr_label) + "\t" + str(pred_prob) + "\n")
      fp_info.write(test_example_info + "\n")
    fp.close()
    fp_info.close()

    total_true_class_1 = conf_mat[1][0] + conf_mat[1][1]
    (test_acc, test_precision, test_recall, test_fscore, result_string) = compute_PRF_2D_confmat(conf_mat, total_true_class_1)
    (test_gr_acc, test_gr_precision, test_gr_recall, test_gr_fscore, gr_result_string) = compute_PRF_2D_confmat(gr_conf_mat, total_true_class_1)
    
    total_time = timedelta(seconds=time.time() - start_time)
    print("--------------------------------------------------------------------")
    print("Results")
    print(result_string)
    print("Results on grammar slots")
    print(gr_result_string)
    print("\n")
    print("Total time %s" % total_time)        
    print("Done training!")

    return (md, test_acc, test_precision, test_recall, test_fscore, test_gr_acc, test_gr_precision, test_gr_recall, test_gr_fscore)

  
def train_model(train_dataset, valid_dataset, output_dir, config, tuning):

    start_time = time.time()

    try:
      os.stat(output_dir)
    except:
      os.mkdir(output_dir)

    exit_criteria = predictor.ExitCriteria(config.max_epoch)

    with tf.Graph().as_default():
      # if tuning:
      # tf.set_random_seed(10)
      # np.random.seed(30)
        
      with tf.Session(config=get_gpu_config()) as session:
        md = predictor.create_model(session, config, output_dir)
        print("Model parameters = " + str(md.get_parameter_count()))
        md.write_model_parameters(output_dir)
        print("Creating train batched data")
        train_batched_data = train_dataset.create_batch_data(config.batch_size, 
                                                             config.max_blks_in_seq,
                                                             config.max_sents_in_blk,
                                                             config.embed_type, 
                                                             config.model_type)
        print("Creating valid batched data")
        valid_batched_data = valid_dataset.create_batch_data(config.batch_size, 
                                                             config.max_blks_in_seq,
                                                             config.max_sents_in_blk, 
                                                             config.embed_type, 
                                                             config.model_type)
        (train_epochs, train_loss, valid_loss) = md.train(session, train_batched_data, exit_criteria, valid_batched_data, output_dir, tuning)

        print("Making predictions")
        truel, preds = md.predict(session, valid_batched_data)
        pred_blks = len(preds)
        
    fp = open(output_dir + "/valid.pred", "w")
    with open(output_dir + "/valid.pred.header", "w") as fh:
      fh.write("pred\ttrue\tgpred\tprob1\n")
    fp_info = open(output_dir + "/valid.info", "w")

    conf_mat = np.zeros((2, 2), dtype=np.int32)
    gr_conf_mat = np.zeros((2, 2), dtype=np.int32)
    valid_examples = [loc for seq in valid_dataset.file_seqs for loc in seq.get_all_locs()]

    labels = [e.label for e in valid_examples]
    gr_labels = [e.gr_label for e in valid_examples]
    true_num_blks = sum([1 for e in valid_examples if e.blk_bd == 1])
    
    assert pred_blks == true_num_blks, "Number of true blocks %d does not match predicted number %d" % (true_num_blks, pred_blks)
    
    pred_num = -1
    for lnum in range(len(labels)):
      valid_example_info = valid_examples[lnum].__str__()
      true_label = labels[lnum]
      true_gr_label = labels[lnum]
      if valid_examples[lnum].blk_bd == 1:
        pred_num += 1
        pred_label = preds[pred_num][0]
        pred_prob = preds[pred_num][1]
        conf_mat[true_label][pred_label] += 1
        if true_gr_label != -1:
          gr_conf_mat[true_gr_label][pred_label] += 1
      else:
        pred_label = -1
        pred_prob = 0.0
      fp.write(str(pred_label) + "\t" + str(true_label) + "\t" + str(true_gr_label) + "\t" + str(pred_prob) + "\n")
      fp_info.write(valid_example_info + "\n")
    fp.close()
    fp_info.close()

    total_true_class_1 = conf_mat[1][0] + conf_mat[1][1]
    (valid_acc, valid_precision, valid_recall, valid_fscore, result_string) = compute_PRF_2D_confmat(conf_mat, total_true_class_1)
    (valid_gr_acc, valid_gr_precision, valid_gr_recall, valid_gr_fscore, gr_result_string) = compute_PRF_2D_confmat(gr_conf_mat, total_true_class_1)
    
    total_time = timedelta(seconds=time.time() - start_time)
    print("--------------------------------------------------------------------")
    print("Results")
    print(result_string)
    print("Results on grammar slots")
    print(gr_result_string)
    print("\n")
    print("Total time %s" % total_time)        
    print("Done training!")

    return (md, train_epochs, total_time, train_loss, valid_loss, valid_acc, valid_precision, valid_recall, valid_fscore, valid_gr_acc, valid_gr_precision, valid_gr_recall, valid_gr_fscore)
    

    
if __name__=="__main__":
    tf.app.run()
    
