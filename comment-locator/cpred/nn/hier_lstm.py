"""
LSTM based predictor for comment location. Each statement passes through
one LSTM cell. The output of the cell is used for a yes/no prediction. 
'Yes' means a comment appears before this statement. 

Author: Annie Louis
Project: Comment prediction

"""


from __future__ import print_function

import time
from datetime import timedelta
import logging

import inspect
import math
import json
import os
import sys

import tensorflow as tf
import numpy as np
import cpred.data.cloc_data_reader as data_helper

def data_type():
  return tf.float32
    

class HIER_LSTM(object):

  def __init__(self, config):

    self.config = config
    nlabels = config.nlabels
    nblks = config.max_blks_in_seq
    nsents = config.max_sents_in_blk
    nwords = config.max_words_in_sent
    vocab_size = config.vocab_size
    embed_type = config.embed_type
    embed_size = config.embed_size
    lstm_hidden_size = config.lstm_hidden_size
    lstm_num_layers = config.lstm_num_layers 

    self.global_step = tf.Variable(0, trainable=False)

    with tf.name_scope("Parameters"):
      self.learning_rate = tf.placeholder(tf.float32, name="learning_rate")
      self.keep_probability = tf.placeholder(tf.float32, name="keep_probability")
      
    with tf.name_scope("Input"):
      self.inputs = tf.placeholder(tf.int32, shape=(None, nblks, nsents, nwords), name="inputs")
      self.input_wts = tf.placeholder(tf.float32, shape=(None, nblks, nsents, nwords), name="input_wts") 
      batch_size = tf.shape(self.inputs, out_type=tf.int32)[0]

      self.targets = tf.placeholder(tf.int32, shape=(None, nblks), name="labels")
      self.target_weights = tf.placeholder(tf.float32, shape=(None, nblks), name="tgt_wts")
      self.pretrained_embeddings = tf.placeholder(tf.float32, shape=(vocab_size, embed_size), name="pre_embed")

    with tf.name_scope("Embeddings"):
      self.embedding_table = tf.Variable(tf.random_uniform((vocab_size, embed_size), -config.init_scale, config.init_scale), dtype=data_type(), name="embedding_table", trainable=True)
      self.embedding_init = self.embedding_table.assign(self.pretrained_embeddings)

      if embed_type == data_helper.AVG_WEMBED:
        self.input_wts_resh = tf.tile(tf.reshape(self.input_wts, [batch_size, nblks, nsents, nwords, 1]), [1, 1, 1, 1, embed_size])
        # print(self.input_wts_resh)
        self.input_embeddings = tf.reshape(\
                                tf.reduce_mean(\
                                tf.multiply(\
                                self.input_wts_resh, 
                                tf.nn.embedding_lookup(self.embedding_table, self.inputs)),\
                                               axis=3),\
                                           [batch_size, nblks, nsents, embed_size])
      else:
        self.input_embeddings = tf.cast(self.inputs, tf.float32)

        
    with tf.name_scope("RNN_STMT") and tf.variable_scope("RNN_STMT"):
      def stmt_lstm_cell():
        if 'reuse' in inspect.getargspec(tf.contrib.rnn.BasicLSTMCell.__init__).args:
          return tf.contrib.rnn.BasicLSTMCell(lstm_hidden_size, forget_bias=1.5, state_is_tuple=True, reuse=tf.get_variable_scope().reuse)
        else:
          return tf.contrib.rnn.BasicLSTMCell(lstm_hidden_size, forget_bias=1.5, state_is_tuple=True)

      def stmt_lstm_cell_with_dropout():
        return tf.contrib.rnn.DropoutWrapper(stmt_lstm_cell(), output_keep_prob=self.keep_probability)

      stmt_lstm_layers = tf.contrib.rnn.MultiRNNCell([stmt_lstm_cell_with_dropout() for _ in range(lstm_num_layers)], state_is_tuple=True)
      self.stmt_reset_state = stmt_lstm_layers.zero_state(batch_size * nblks, data_type())
      self.stmt_outputs, self.stmt_next_state = tf.nn.dynamic_rnn(stmt_lstm_layers, tf.reshape(self.input_embeddings, [batch_size * nblks, nsents, embed_size]), time_major=False, initial_state=self.stmt_reset_state)
      #max pool across the dimensions of states of all the statements in a block
      self.blk_rep = tf.reshape(tf.reduce_max(self.stmt_outputs, 1), [batch_size, nblks, lstm_hidden_size])
      #blk_rep is batch_size * nblks * lstm_hidden_size
      
    with tf.name_scope("RNN_BLK") and tf.variable_scope("RNN_BLK"):
      def blk_lstm_cell():
        if 'reuse' in inspect.getargspec(tf.contrib.rnn.BasicLSTMCell.__init__).args:
          return tf.contrib.rnn.BasicLSTMCell(lstm_hidden_size, forget_bias=1.5, state_is_tuple=True, reuse=tf.get_variable_scope().reuse)
        else:
          return tf.contrib.rnn.BasicLSTMCell(lstm_hidden_size, forget_bias=1.5, state_is_tuple=True)

      def blk_lstm_cell_with_dropout():
        return tf.contrib.rnn.DropoutWrapper(blk_lstm_cell(), output_keep_prob=self.keep_probability)
    
      blk_lstm_layers = tf.contrib.rnn.MultiRNNCell([blk_lstm_cell_with_dropout() for _ in range(lstm_num_layers)], state_is_tuple=True)
      self.blk_reset_state = blk_lstm_layers.zero_state(batch_size, data_type())

      self.blk_outputs, self.blk_next_state = tf.nn.dynamic_rnn(blk_lstm_layers, self.blk_rep, time_major=False, initial_state=self.blk_reset_state)

    with tf.name_scope("cost"):
      self.output = tf.reshape(self.blk_outputs, [batch_size * nblks, lstm_hidden_size])
      self.op_w = tf.get_variable("op_w", [lstm_hidden_size, nlabels], dtype=data_type(), initializer=tf.random_normal_initializer())
      self.op_b = tf.get_variable("op_b", [nlabels], dtype=data_type(), initializer=tf.random_normal_initializer())
      self.logits = tf.reshape(tf.nn.bias_add(tf.tensordot(self.output, self.op_w, axes=[[1], [0]]), self.op_b), [-1, nlabels])
      self.labels = tf.reshape(self.targets, [-1])
      cross_ent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labels, logits=self.logits)
      self.loss = tf.reduce_sum(tf.multiply(cross_ent, tf.reshape(self.target_weights, [-1])))
      self.cost = tf.div(self.loss, tf.cast(batch_size, tf.float32), name="cost")
      
    with tf.name_scope("train"):
        tvars = tf.trainable_variables()
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.gradients, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars), config.max_grad_norm, name="clip_gradients")
        self.train_step = optimizer.apply_gradients(zip(self.gradients, tvars), global_step=self.global_step)

    self.saver = tf.train.Saver(tf.global_variables())
    self.initialize = tf.global_variables_initializer()
    self.summary = tf.summary.merge_all()


  def get_parameter_count(self):
    params = tf.trainable_variables()
    total_parameters = 0
    for variable in params:
      shape = variable.get_shape()
      variable_parameters = 1
      for dim in shape:
        variable_parameters *= dim.value
      total_parameters += variable_parameters
    return total_parameters

    
  def train(self, session, train_data, exit_criteria, valid_data, train_dir, tuning):
    summary_writer = tf.summary.FileWriter(train_dir, session.graph)
    previous_valid_loss = []
    nglobal_steps = 0
    epoch = 1
    new_learning_rate = self.config.learning_rate

    if self.config.use_pretraining:
      print("Using pretrained embeddings")
      session.run(self.embedding_init, feed_dict={self.pretrained_embeddings: train_data.pretrained_embeddings})

    
    try:
      while True:
        train_epoch_loss = 0.0
        train_total_examples = 0.0
        print("Epoch %d Learning rate %0.3f" % (epoch, new_learning_rate))
        epoch_start_time = time.time()
        for step, (inp, inp_wts, target, tgt_wts) in enumerate(train_data.batch_producer(True)):
          # print(str(inp.shape))
          feed_dict={self.inputs: inp,
                     self.input_wts: inp_wts,
                     self.targets: target,
                     self.target_weights: tgt_wts, 
                     self.learning_rate: new_learning_rate,
                     self.keep_probability: self.config.keep_prob
          }
          _, cost, loss = session.run([self.train_step, self.cost, self.loss], feed_dict)
          nglobal_steps += 1
          train_epoch_loss += np.sum(loss)
          train_total_examples += np.sum(tgt_wts)
        train_loss = train_epoch_loss / train_total_examples
        valid_loss = self.test(session, valid_data)

        if not tuning:
          train_summary = tf.Summary() 
          train_summary.value.add(tag="train_loss", simple_value=train_loss)
          summary_writer.add_summary(train_summary, nglobal_steps)

          valid_summary = tf.Summary()
          valid_summary.value.add(tag="valid_loss", simple_value=valid_loss)
          summary_writer.add_summary(valid_summary, nglobal_steps)

        epoch_time = (time.time() - epoch_start_time) * 1.0 / 60 
        print ("END EPOCH %d global_steps %d learning_rate %.4f time(mins) %.4f train_loss %.4f valid_loss %.4f" % (epoch, nglobal_steps, new_learning_rate, epoch_time, train_loss, valid_loss))
        sys.stdout.flush()

        if epoch % self.config.checkpoint_every_n_epochs == 0 and not tuning:
          checkpoint_path = os.path.join(train_dir, "lstm.ckpt.epoch" + str(epoch))
          self.saver.save(session, checkpoint_path, global_step=self.global_step)

        # Decrease learning rate if valid loss increases 
        if len(previous_valid_loss) > 1 and valid_loss > previous_valid_loss[-1]:
          new_learning_rate = new_learning_rate * self.config.lr_decay

        # If validation perplexity has not improved over the last x epochs, stop training 
        if new_learning_rate == 0.0 or (len(previous_valid_loss) >= self.config.valid_nonimproving_epochs and valid_loss >= max(previous_valid_loss[-self.config.valid_nonimproving_epochs:])):
          raise StopTrainingException()
          
        previous_valid_loss.append(valid_loss)
        epoch += 1

        if epoch > exit_criteria.max_epochs:
          raise StopTrainingException()
        
    except (StopTrainingException, KeyboardInterrupt):
      checkpoint_path = os.path.join(train_dir, "lstm.ckpt.epoch" + str(epoch))
      self.saver.save(session, checkpoint_path, global_step=self.global_step)
      print("Finished training ........")
      return (epoch, train_loss, valid_loss)


  def test(self, session, test_data):
    total_loss, total_size = 0.0, 0.0
    batch_size = self.config.test_batch_size
    for step, (inp, inp_wts, target, tgt_wts) in enumerate(test_data.batch_producer(False)):
      feed_dict={self.inputs: inp,
                 self.input_wts: inp_wts, 
                 self.targets: target,
                 self.target_weights: tgt_wts, 
                 self.keep_probability: 1.0
      }
      loss, cost = session.run([self.loss, self.cost], feed_dict)
      total_loss += np.sum(loss)
      total_size += np.sum(tgt_wts)
    avg_loss = total_loss / total_size
    return avg_loss


  def predict(self, session, test_data):
    preds_for_test = []
    true_labels = []
    batch_size = self.config.test_batch_size
    nblks = self.config.max_blks_in_seq
    for step, (inp, inp_wts, target, tgt_wts) in enumerate(test_data.batch_producer(False)):
      feed_dict={self.inputs: inp,
                 self.input_wts: inp_wts,
                 self.targets: target,
                 self.target_weights: tgt_wts, 
                 self.keep_probability: 1.0
      }
      logits, cost = session.run([self.logits, self.cost], feed_dict)
      probs = tf.nn.softmax(logits)
      topk = tf.nn.top_k(probs, k=2, sorted=True)
      inds = session.run(topk.indices)
      vals = session.run(topk.values)
      pred_position = -1
      for i in range(batch_size):
        for j in range(nblks):
          pred_position += 1
          if tgt_wts[i][j] > 0:
            true_labels.append(target[i][j])
            pred_label = inds[pred_position][0]
            if pred_label == 1:
              prob_of_class1 = vals[pred_position][0]
            else:
              prob_of_class1 = vals[pred_position][1]
            preds_for_test.append((pred_label, prob_of_class1))
    return true_labels, preds_for_test


  def write_model_parameters(self, model_directory):
    parameters = {
      "lstm_num_layers": self.config.lstm_num_layers,
      "lstm_hidden_size": self.config.lstm_hidden_size, 
      "label_size": self.config.nlabels,
      "total_parameters": self.get_parameter_count()
    }
    with open(self.parameters_file(model_directory), "w") as f:
      json.dump(parameters, f, indent=4)

    with open(self.config_file(model_directory), "w") as f:
      json.dump(self.config.__dict__, f, indent=4)


  def config_file(self, model_directory):
    return os.path.join(model_directory, "config.json")

  
  def parameters_file(self, model_directory):
    return os.path.join(model_directory, "parameters.json")

  
def create_model(session, config, train_dir):
  model = HIER_LSTM(config)
  ckpt = tf.train.get_checkpoint_state(train_dir)
  if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
    print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
    model.saver.restore(session, ckpt.model_checkpoint_path)
  else:
    print("Created model with fresh parameters:")
    session.run(model.initialize)
  print("*Number of parameters* = " + str(model.get_parameter_count())) 
  return model


class ExitCriteria(object):
  def __init__(self, max_epochs):
    self.max_epochs = max_epochs

    
class StopTrainingException(Exception):
  pass


class Config(object):
  """those settings needed for model training"""

  def __init__(self, inits, lr, mgrad,
               nblks, nsents, nwords,
               lstm_layers, lstm_hsize,
               mepoch, kp, decay,
               bsize, tbsize, labsize, vocab_size,
               sembed_type, sembed_size,
               vnimp, ckpt_epochs,
               model_type, use_pretraining):
    self.init_scale = inits
    self.learning_rate = lr
    self.max_grad_norm = mgrad
    self.max_blks_in_seq = nblks
    self.max_sents_in_blk = nsents
    self.max_words_in_sent = nwords
    self.lstm_num_layers = lstm_layers
    self.lstm_hidden_size = lstm_hsize
    self.max_epoch = mepoch
    self.keep_prob = kp
    self.lr_decay = decay
    self.batch_size = bsize
    self.test_batch_size = tbsize
    self.nlabels = labsize
    self.vocab_size = vocab_size
    self.embed_type = sembed_type
    self.embed_size = sembed_size
    self.valid_nonimproving_epochs = vnimp
    self.checkpoint_every_n_epochs = ckpt_epochs
    self.model_type = model_type
    self.use_pretraining = use_pretraining


