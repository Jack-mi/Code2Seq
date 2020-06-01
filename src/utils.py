# -*- coding: utf-8 -*-
PAD = 0
BOS = 1
EOS = 2
UNK = 3

class Vocab(object):
    # 一个词典包含了word->idx 和 idx->word
    def __init__(self, word2id={}):
        self.word2id = dict(word2id)
        self.id2word = {v: k for k, v in self.word2id.items()}
    
    # 构建词典
    def build_vocab(self, sentences, min_count=1):
        # 统计sentences中所有词出现的频次
        word_counter = {}
        for word in sentences:
            word_counter[word] = word_counter.setdefault(word, 0) + 1

        # 用sentences中的word更新word2id和id2word
        for word, count in sorted(word_counter.items(), key=lambda x:-x[1]):
            # 此处count的唯一作用就是筛选出现词频小于min_count的word
            if count < min_count:
                break
            idx = len(self.word2id)
            # 如果word已经在词典中则idx不变，反之将其添加到字典中
            self.word2id.setdefault(word, idx)
            self.id2word[idx] = word

def sentence_to_ids(vocab, sentence):
    ids = [vocab.word2id.get(word, UNK) for word in sentence]
    ids.append(EOS)
    return ids

def ids_to_sentence(vocab, ids):
    return [vocab.id2wrod[id] for id in ids]

def calculate_results(refs, preds):
    #cal precision, F1 and recall
    filtered_refs = [ref[:ref.index(EOS)] for ref in refs]
    filtered_preds = [pred[:pred.index(EOS)] if EOS in pred else pred for pred in preds]

    true_positive, false_positive, false_negative = 0, 0, 0

    for f_ref, f_pred in zip(filtered_refs, filtered_preds):
        if f_ref == f_pred:
            true_positive += len(f_ref)
            continue

        for fp in f_pred:
            if fp in f_ref:
                true_positive += 1
            else :
                false_positive += 1

        for fr in f_ref:
            if fr not in f_pred:
                false_negative += 1

    if true_positive + false_positive > 0:
        precision = true_positive/(true_positive+false_positive)
    else:
        precision = 0

    if  true_positive + false_negative > 0:
        recall = true_positive/(true_positive+false_negative)
    else :
        recall = 0

    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0

    return precision, recall, f1

def calculate_results_set(refs, preds):
    filterd_refs = [ref[:ref.index(EOS)] for ref in refs]
    filterd_preds = [pred[:pred.index(EOS)] if EOS in pred else pred for pred in preds]

    filterd_refs = [list(set(ref)) for ref in filterd_refs]
    filterd_preds = [list(set(pred)) for pred in filterd_preds]

    true_positive, false_positive, false_negative = 0, 0, 0

    for filterd_pred, filterd_ref in zip(filterd_preds, filterd_refs):

        for fp in filterd_pred:
            if fp in filterd_ref:
                true_positive += 1
            else:
                false_positive += 1

        for fr in filterd_ref:
            if not fr in filterd_pred:
                false_negative += 1

    if true_positive + false_positive > 0:
        precision = true_positive / (true_positive + false_positive)
    else:
        precision = 0

    if true_positive + false_negative > 0:
        recall = true_positive / (true_positive + false_negative)
    else:
        recall = 0

    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0

    return precision, recall, f1

# 先把原seq最后一个EOS改为PAD，再进行PAD，最后再加上EOS
def pad_seq(seq, max_length):
    seq[-1] = PAD
    res = seq + [PAD for i in range(max_length - len(seq))]
    seq[-1] = EOS
    return res
