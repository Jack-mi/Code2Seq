
#-----------------------------------------------------------
# Author: Annie Louis
# Reader for comments location data, also produces batches for NN training
#
#------------------------------------------------------------


from numpy.random import seed
seed(0)
from tensorflow import set_random_seed
set_random_seed(0)
import random
random.seed(0)

import numpy as np
import tensorflow as tf
from collections import namedtuple
import gensim
import math
import sys
import os
import copy
import os.path
import getopt
import collections


# There are two ways in which a LOC can be empty. Either it was a blank line in 
# the source indicated by EMPTY, or it is a blank line because a comment at that 
# position was removed. The latter is referred to as COMMENTEMPTY.
EMPTY_LINE = "EMPTY" 
COMMENT_ONLY_LINE = "COMMENTEMPTY"


UNKNOWN_WORD = "-unk-"
special_symbols = [UNKNOWN_WORD]

# Types of data for the models. Currently there is only one average embeddings. 
UNKNOWN = 0
AVG_WEMBED = 1
sembed_options = {"unknown": UNKNOWN,
                  "avg_wembed": AVG_WEMBED,
}

LSTM_MODEL = 1
LSTM_BACK_MODEL = 2
LSTM_ONE_BACK_MODEL = 3
LSTM_MODEL_WITH_FEATURES = 4
SELF_ATTN_MODEL = 5
HIER_LSTM_MODEL = 6
SNIP_MULTI_MODEL = 7
MULTI_HIER_LSTM_MODEL = 8
model_types = [LSTM_MODEL, LSTM_BACK_MODEL, LSTM_ONE_BACK_MODEL, LSTM_MODEL_WITH_FEATURES, SELF_ATTN_MODEL, HIER_LSTM_MODEL, SNIP_MULTI_MODEL, MULTI_HIER_LSTM_MODEL]


CODE_EMBEDDINGS_DATA = "/disk/scratch1/alouis/projects/comm-pred/comment-location/data/code-big-vectors-negative300.bin"
BPE_CODE_EMBEDDINGS_DATA = "/disk/scratch1/alouis/projects/comm-pred/comment-location/data/code.10Mtoks.bpe.bin"
PRE_EMBED_DIM = 300


MIN_WORD_FREQ = 0



def read_w2v_models(is_bpe):
    """
    Reads pretrained w2v embeddings. 

    There are two types of embeddings -- trained on code or on English comments. The
    function reads both. 

    Parameters
    ----------
    is_bpe : boolean 
       If true, load embeddings trained on bpe. 

    Returns
    -------
    w2v_models : two gensim based w2v models, first code, the other comment
    w2v_dims : dimensions of the two models above. 

    """
    
    with tf.device('/cpu:0'):
        print("Loading pre-trained embeddings")
        print("\t ...")
        if is_bpe:
            print("Reading BPE based embeddings")
            print("\t sources: " + BPE_CODE_EMBEDDINGS_DATA)
            code_w2v_model = gensim.models.KeyedVectors.load_word2vec_format(BPE_CODE_EMBEDDINGS_DATA, binary = True)
            comm_w2v_model = None
        else:
            print("Reading word embeddings")
            print("\t sources: " + CODE_EMBEDDINGS_DATA)
            code_w2v_model = gensim.models.KeyedVectors.load_word2vec_format(CODE_EMBEDDINGS_DATA, binary = True)
            comm_w2v_model = None
        code_w2v_dim = code_w2v_model.vector_size
        comm_w2v_dim = None
        #comm_w2v_model.vector_size
        print("..done.")
        w2v_models = [code_w2v_model, comm_w2v_model]
        w2v_dims = [code_w2v_dim, comm_w2v_dim]
        return w2v_models, w2v_dims
  
    
def DEVICEread_comment_location_file(c_file, back_file, max_stmt_len, skip_empty_stmt, code_ppx_file=None, code_tkfeat_file=None, comm_ppx_file=None, comm_tkfeat_file=None):
    """
    This method reads a comment file. It creates the loc sequence for each file and the sequence
    of other info such as side features. The return value is a list of file sequences. 

    Parameters
    ----------
    c_file : string 
         a file containing comment location information. The format is 
         fileid\tloc\tlabel\tgramarlabel\tstmt\tmodulebackground\tlibrarybackground
         The label indicates whether a comment appears before this statement. The file 
         may also contain background knowledge. Side features are present in separate files.  
     
    skip_empty_stmt : boolean
         ignore lines where the stmt text matches exactly the empty line marker 

    code_ppx_file : string 
         file containing the perplexity values of each loc, each on a separate line

    comm_ppx_file : string 
         file containing perplexity values under the comment LM, each on a separate line

    code_tkfeat_file, comm_tkfeat_file : string, string
         files containing state features from the LM. The first tokens in each line are 
         the token level ppx values under an LM (Code, or comment) and then followed by 
         the LM state after reading each token. 

    Return
    ------
    file_seqs: list of file_seq. 
         Each file_seq is a list of line_of_codes
    
    all_words: counter 
         code and background tokens to their counts across the dataset 
    
    loc_background_terms : counter
         counts up the background terms associated with each LOC

    corpus_background_terms : counter
         counts up the background terms given for the whole corpus (in a separate file)
    
    """

    with tf.device('/cpu:0'):
        print("Reading comment locations...")

        curfile = ""
        cur_file_conts = file_sequence()
        cur_file_back_toks = []
        loc_back_terms = collections.Counter()
        corpus_back_terms = collections.Counter()
        all_words = collections.Counter() #vocab of code and background
        file_seqs = []

        # read the corpus level background terms
        with open(back_file, "r") as f_bk:
            for bl in f_bk:
                for tk in bl.strip().split(" "):
                    if not tk == "_":
                        corpus_back_terms[tk] += 1

        # read all the perplexity features
        code_ppx_all_lines = None
        code_tkfeats_all_lines = None
        if code_ppx_file != None:
            print("Reading code ppx file..")
            code_ppx_all_lines = (open(code_ppx_file, "r")).readlines()
            print("\t " + str(len(code_ppx_all_lines)) + " lines.")
        if code_tkfeat_file != None:
            print("Reading code tkfeat file..")
            code_tkfeats_all_lines = (open(code_tkfeat_file, "r")).readlines()
            print("\t " + str(len(code_tkfeats_all_lines)) + " lines.")
        comm_ppx_all_lines = None
        comm_tkfeats_all_lines = None
        if comm_ppx_file != None:
            print("Reading comment ppx file..")
            comm_ppx_all_lines = (open(comm_ppx_file, "r")).readlines()
            print("\t " + str(len(comm_ppx_all_lines)) + " lines.")
        if comm_tkfeat_file != None:
            print("Reading comment tkfeat file..")
            comm_tkfeats_all_lines = (open(comm_tkfeat_file, "r")).readlines()
            print("\t " + str(len(comm_tkfeats_all_lines)) + " lines.") 
            
            
        f_ch = open(c_file, "r")
        line_id = -1
        for line in f_ch:
            line_id += 1 
            fileid, loc, blk_bd, label, ll_label, gr_label, code, b_module, b_std = line.strip().split("\t")

            ##The blk_bds are used for the hierarchical LSTM model
            # 1 for begin, 2 for mid and 3 for end of block. If it is a case where
            #there is no system of blocks, then the bd value is always -1
            
            
            # skip lines which are blank because they had a comment
            if code == COMMENT_ONLY_LINE:
                continue

            # skip all empty lines if the marker is set
            if code == EMPTY_LINE and skip_empty_stmt:
                continue

            # store perplexities and LM states as side information
            code_side_info = {}
            if code_ppx_all_lines != None:
                code_side_info['code_ppx'] = float(code_ppx_all_lines[line_id].strip())
                code_tkfeats = code_tkfeats_all_lines[line_id].strip().split('\t')
                (code_num_ppx_toks, code_per_tok_loss, code_per_tok_states) = get_per_tok_feats(code_tkfeats)
                trunc_code_toks = min(code_num_ppx_toks, max_stmt_len)
                code_side_info['code_ntoks'] = trunc_code_toks
                code_side_info['code_tk_losses'] = code_per_tok_loss[0: trunc_code_toks]
                code_side_info['code_tk_states'] = code_per_tok_states[0: trunc_code_toks]
            if comm_ppx_all_lines != None:
                code_side_info['comm_ppx'] = float(comm_ppx_all_lines[line_id].strip())
                comm_tkfeats = comm_tkfeats_all_lines[line_id].strip().split('\t')
                (comm_num_ppx_toks, comm_per_tok_loss, comm_per_tok_states) = get_per_tok_feats(comm_tkfeats)
                trunc_code_toks = min(comm_num_ppx_toks, max_stmt_len)
                code_side_info['comm_ntoks'] = trunc_code_toks
                code_side_info['comm_tk_losses'] = comm_per_tok_loss[0: trunc_code_toks]
                code_side_info['comm_tk_states'] = comm_per_tok_states[0: trunc_code_toks]

            # clean and truncate the LOC
            code_toks = code.split()[0: min(len(code.split()), max_stmt_len)]
            for w in code_toks:
                all_words[w] += 1
            code = " ".join(code_toks).strip()
                
            # count and store background
            if b_module == EMPTY_LINE:
                b_module = ""
            if b_std == EMPTY_LINE:
                b_std = ""
            code_side_info['b_module'] = b_module
            code_side_info['b_std'] = b_std
            for k in b_module.split():
                if k != "_":
                    if not k in cur_file_back_toks:
                        cur_file_back_toks.append(k)
                    all_words[k] += 1
                    loc_back_terms[k] += 1
            for k in b_std.split():
                if k != "_":
                    if not k in cur_file_back_toks:
                        cur_file_back_toks.append(k)
                    all_words[k] += 1 
                    loc_back_terms[k] += 1

            cloc = line_of_code(fileid, loc, blk_bd, code, label, ll_label, gr_label, code_side_info)
            
            # store the current files contents before moving to next file
            if fileid != curfile:
                if cur_file_conts.num_locs() > 0:
                    cur_file_conts.add_background(cur_file_back_toks)
                    file_seqs.append(cur_file_conts) 
                    cur_file_conts = file_sequence()
                    cur_file_back_toks = []
                curfile = fileid

            cur_file_conts.add_loc(cloc)

            # set prev line counter so that we can keep only limited empty lines
            if code == EMPTY_LINE:
                prev_line_empty = True
            else:
                prev_line_empty = False

        if cur_file_conts.num_locs() > 0:
            cur_file_conts.add_background(cur_file_back_toks)
            file_seqs.append(cur_file_conts)        
        f_ch.close()


        # print(str(skip_empty_stmt))
        # print(str(file_seqs[0]))
        # print(str(file_seqs[1]))
        # sys.exit(1)

        print("\tThere were " + str(len(file_seqs)) + " file sequences")
        print("\tThere were a total of " + str(len(loc_back_terms)) + " loc background terms")
        print("\tThere were a total of " + str(len(corpus_back_terms)) + " corpus background terms")
        return file_seqs, all_words, loc_back_terms, corpus_back_terms



def get_per_tok_feats(tkfeats):
    """ 
    Parse a line of LM features and return the different fields

    Parameters
    ---------
    tkfeats : string
         a line from the tkfeat file. First token is number of tokens in line. Then 
         we have one float per token giving the loss of the token. Then we have the 
         states of the LM per token

    Return
    ------
    n_code_toks : int
         number of tokens in the LOC

    per_tok_loss : list of floats
         ppx for each token

    per_tok_states : list of list of floats
         LM states (a list) for each token
    
    """

    # if the line is empty, return 0 values
    if len(tkfeats) == 1 and tkfeats[0] == "0.0":
        per_tok_loss = [0.0]
        per_tok_states = [0.0]
        return (0, per_tok_loss, per_tok_states)
    
    n_code_toks = int(tkfeats[0])
    per_tok_loss = [float(tkfeats[x]) for x in range(1, n_code_toks + 1)]
    per_tok_states = []
    for x in range(n_code_toks + 1, 2 * n_code_toks + 1):
        one_tok_state = [float(y) for y in tkfeats[x].split(",")]
        per_tok_states.append(one_tok_state)
    retfeats = (n_code_toks, per_tok_loss, per_tok_states)
    return retfeats



def get_restricted_vocabulary(word_counts, vocab_size, min_freq, add_special):
    """ Create vocabulary of a certain max size
    
    Parameters
    ----------
    word_counts: Counter
    vocab_size: int
    add_special: boolean
         if true add special tokens into the vocab
    min_freq: int
         must be at least this frequent

    Return
    ------
    words: list of vocab words
    """
    
    non_special_size = min(vocab_size, len(word_counts))
    if add_special and non_special_size == vocab_size:
        non_special_size -= len(special_symbols)
    word_counts_sorted = sorted(word_counts.items(), key=lambda x: (-x[1], x[0]))
    words = [k for (k,v) in word_counts_sorted if k not in special_symbols and v >= min_freq][0:non_special_size]
    if add_special:
        for s in special_symbols:
            words.insert(0, s)
    return words


def assign_vocab_ids(word_list):
    """ 
    Given a word list, assign ids to it and return the id dictionaries
    """
    
    word_ids = [x for x in range(len(word_list))]
    word_to_id = dict(zip(word_list, word_ids))
    id_to_word = dict(zip(word_ids, word_list))
    return word_to_id, id_to_word


def write_vocab_to_file(id_to_vocab, filename):
    with tf.device('/cpu:0'):
        fout = open(filename, "w")
        for i in range(len(id_to_vocab)):
            fout.write(id_to_vocab[i] + "\n")
        fout.close()
        

def read_vocab_file(filename):
    with tf.device('/cpu:0'):
        words = []
        word_to_id, id_to_word = {}, {}
        with open(filename, "r") as fin:
            i = 0
            for l in fin:
                wd = l.strip()
                words.append(wd)
                word_to_id[wd] = i
                id_to_word[i] = wd
                i += 1
        return words, word_to_id, id_to_word



def code_each_word(wlist, vocab_to_id, ignore_unknown, ignore_special):
    """
    :return list of wordids
    """    
    ret = []
    for w in wlist:
        if ignore_special and w in special_symbols:
            continue
        if w in vocab_to_id:
            ret.append(vocab_to_id[w])
        elif not ignore_unknown:
            ret.append(vocab_to_id[UNKNOWN_WORD])
    return ret



def code_text(text, vocab_to_id, ignore_unknown, ignore_special):
    """ 
    Convert a text into list of ids 
    
    Parameters
    ----------
    ignore_unknown : boolean
         skip tokens which are not in the vocabulary (dont add UNK)
    ignore_special : boolean 
         skip special tokens
    """    

    ret = []
    return code_each_word(text.split(), vocab_to_id, ignore_unknown, ignore_special)


def code_to_text(coded_list, id_to_vocab):
    """
    Replace ids with words 
    """

    words = []
    for c in coded_list:
        words.append(id_to_vocab[int(c)])
    return " ".join(words).strip()


def get_each_word_embedding(widlist, id_to_vocab, w2v_model, w2v_dim):
    """ 
    Get a list of word embeddings corresponding to the wordid list that 
    is given. 
    """

    ret = []
    for wid in widlist:
        emb_for_word = get_avg_embedding([wid], id_to_vocab, w2v_model, w2v_dim)
        ret.append(emb_for_word)
    if len(ret) == 0:
        ret.append(np.zeros((w2v_dim), dtype=np.float32))
    return ret



def get_avg_embedding(wids, id_to_vocab, w2v_model, w2v_dim):
    """
    Get average embedding. If w2v_model is None, then the embedding is 
    all zeroes. Otherwise, average for the words that are in the model vocab. 
    """
    
    sum_embed = np.zeros(w2v_dim, dtype=np.float32)
    if w2v_model == None:
        return sum_embed
    found_wc = 0
    for wid in wids:
        tk = id_to_vocab[wid]
        if tk != UNKNOWN_WORD and tk in w2v_model.vocab:
            sum_embed += w2v_model.wv[tk]
            found_wc += 1
    if found_wc > 0: 
        return sum_embed / found_wc
    return sum_embed



def get_embedding_vectors(w2v_model, w2v_dim, vocab_to_id):
    """
    Get all embeddings in a [nwords * dimsize] matrix. If word not 
    found, have random embedding. 

    """
    
    emb = np.random.uniform(-0.001, 0.001, size=(len(vocab_to_id), w2v_dim))
    for w,i in vocab_to_id.items():
        if w in w2v_model.vocab:
            emb[i] = w2v_model.wv[w]
    # print(str(emb[0][0:25]))
    # print(str(emb[1][0:25]))
    # print(str(emb[2][0:25]))
    return emb



def create_grammar_features(gr_label):
    """
    Returns a one-hot encoding of the label
    """
    
    if gr_label == 0 or gr_label == 1: 
        return [1, 0] # is a grammar slot
    else:
        return [0, 1] 
    

    
def prepare_loc_data(file_seqs, 
                       max_lc_len, 
                       wordv, 
                       loc_backv, 
                       corpus_backv,
                       all_wordv, all_word_w2i, all_word_i2w, 
                       w2v_models, w2v_dims,
                       ignore_unknown, ignore_special, 
                       compute_code_ppx_features, compute_comm_ppx_features,
                       compute_grammar_features):
    """ 
    Represent the locs and create dataset object
   
    Parameters
    ----------

    file_seqs : list of file_sequence objects 
    max_lc_len : max length of a line of code
    loc_backv : list of str
        background terms vocab for module and standard library
    corpus_backv : list of str
        background terms vocab for the entire corpus
    
    Return
    ------
    A dataset object with all locs coded

    """

    print("Creating the dataset..")
    print("\tConverting to ids, creating code embeddings and targets..")
    cloc_id_to_features = {}
    num_side_features_1 = None
    num_side_features_2 = None

    corpus_back_coded = code_each_word(corpus_backv, all_word_w2i, ignore_unknown, ignore_special)
    corpus_back_embedded = get_each_word_embedding(corpus_back_coded, all_word_i2w, w2v_models[0], w2v_dims[0])
    corpus_back_features = {"coded": corpus_back_coded, 
                            "embedded": corpus_back_embedded, 
    }
    
    for file_num in range(len(file_seqs)):
        file_seq  = file_seqs[file_num]
        file_back = file_seq.get_background()
        file_back_valid = [fb for fb in file_back if fb in loc_backv]
        
        file_back_coded = code_each_word(file_back_valid, all_word_w2i, ignore_unknown, ignore_special)
        file_back_embedded = get_each_word_embedding(file_back_coded, all_word_i2w, w2v_models[0], w2v_dims[0])

        for lc_no in range(file_seq.num_locs()):
            lc = file_seq.get_loc(lc_no)
            b_module = lc.get_side_info('b_module')
            b_std = lc.get_side_info('b_std')
            lc_code = lc.code

            lc_coded = code_text(lc_code, all_word_w2i, ignore_unknown, ignore_special)
            lc_embedded = get_avg_embedding(lc_coded, all_word_i2w, w2v_models[0], w2v_dims[0])
            side_features_1 = []
            side_features_2 = []
            
            #--------------- grammar features ----------------------#
            #TODO check if grammar features should be in the same bin
            if compute_grammar_features:
                grammar_features = []
                cur_lc_gr_label = lc.gr_label
                prev_lc_gr_label = 0
                if lc_no > 0:
                    prev_lc_gr_label = file_seq.get_loc(lc_no - 1).gr_label
                next_lc_gr_label = 0
                if lc_no < file_seq.num_locs() - 1:
                    next_lc_gr_label = file_seq.get_loc(lc_no + 1).gr_label
                grammar_features.extend(create_grammar_features(cur_lc_gr_label))
                grammar_features.extend(create_grammar_features(prev_lc_gr_label))
                grammar_features.extend(create_grammar_features(next_lc_gr_label))
                side_features_2 += grammar_features

            #--------------- perplexity features --------------------#

            comm_embedding_features = []
            ppx_features = []
            ppx_context_features = []
            if compute_code_ppx_features:
                cur_ppx = lc.get_side_info('code_ppx')
                if cur_ppx < 500.0:
                    ppx_features.append(cur_ppx)
                else:
                    ppx_features.append(0.0)
                ppx_context = [file_seq.get_loc(bn).get_side_info('code_ppx') for bn in range(max(lc_no - 5, 0), lc_no)] + [file_seq.get_loc(bn).get_side_info('code_ppx') for bn in range(lc_no + 1, min(lc_no + 5, file_seq.num_locs()))]
                ppx_context_filtered = [pp for pp in ppx_context if pp < 500.0 and pp > 0.0]
                if len(ppx_context_filtered) > 0:
                    ppx_context_features.append(max(ppx_context_filtered) - min(ppx_context_filtered))
                else:
                    ppx_context_features = [0.0]
                    

            side_features_1 += ppx_features
            side_features_2 += ppx_context_features 

            
            if num_side_features_1 == None or num_side_features_2 == None:
                num_side_features_1 = len(side_features_1)
                num_side_features_2 = len(side_features_2)
                
            lc_features = line_of_code_features(lc_coded, lc_embedded, 
                                               file_back_coded, file_back_embedded, 
                                               side_features_1, side_features_2)
            cloc_id_to_features[str(lc.fileid) + "#" + str(lc.locid)] = lc_features
            
    print("\t \t done.")

    return loc_dataset(file_seqs,
                       cloc_id_to_features,
                       corpus_back_features, 
                       max_lc_len,
                       w2v_dims[0], 
                       wordv, 
                       loc_backv, 
                       corpus_backv,
                       all_wordv, all_word_w2i, all_word_i2w, 
                       w2v_models, w2v_dims,
                       num_side_features_1, num_side_features_2)



class line_of_code(object):
    """
    All information pertaining to a loc of code. The loc may be a
    loc, or a syntactically defined code loc
    """

    def __init__(self, fileid, locid, blk_bd, code, label, ll_label, gr_label, side_info):
        """
        Parameters
        ----------
        fileid : int
        locid : int
        blk_bd: int loc boundary (-1 for NA, 1 for start, 2 mid and 3 end of a block
        code : string rep of source code
        label : int 0/1, whether a label appears before this code loc
        side_info : dictionary containing name of side info and its value

        """
        self.fileid = int(fileid)
        self.locid = int(locid)
        self.blk_bd = int(blk_bd)
        self.code = code
        self.label = int(label)
        self.ll_label = int(ll_label)
        self.gr_label = int(gr_label)
        self.side_info = side_info
        
    def get_side_info(self, info_name):
        return self.side_info[info_name]

    def __str__(self):
        return str(self.fileid) + "\t" + str(self.locid) + "\t" + str(self.blk_bd) + "\t" + str(self.label) + "\t" + str(self.ll_label) + "\t" + str(self.gr_label) + "\t" + self.code + "\t" + self.side_info['b_module'] + "\t" + self.side_info['b_std']


class line_of_code_features(object):
    """
    Features and embeddings for a loc of code
    """

    def __init__(self,
                 coded,
                 lc_embedded,
                 file_back_coded,
                 file_back_embedded, 
                 side_features_1,
                 side_features_2):
        self.coded = coded
        self.embedded = lc_embedded

        self.file_back_coded = file_back_coded
        self.file_back_embedded = file_back_embedded
        
        self.side_features_1 = side_features_1
        self.side_features_2 = side_features_2

        
class file_sequence(object):
    """
    All the information attached to one file of source code. A source
    file has a list of line_of_codes
    """

    def __init__(self):
        self.clocs = []
        self.back = []
       
    def add_background(self, bk):
        self.back = bk
        
    def add_loc(self, loc):
        self.clocs.append(loc)
        
    def num_locs(self):
        return len(self.clocs)

    def get_background(self):
        return self.back
    
    def get_loc(self, bid):
        return self.clocs[bid]

    def get_all_locs(self):
        return self.clocs
    
    def __str__(self):
        return str([cb.__str__() for cb in self.clocs])

    
    
class loc_dataset(object):
    """
    A container for keeping all the LOCs, background, embeddings etc
    """
    
    def __init__(self, file_seqs, cloc_id_to_features, corpus_back_features, 
                 max_len_lc, sembed_size, 
                 wordv, 
                 loc_backv, 
                 corpus_backv,
                 all_wordv, all_word_w2i, all_word_i2w, 
                 w2v_models, w2v_dims,
                 num_side_feat_1, num_side_feat_2):

        
        self.file_seqs = file_seqs
        self.cloc_id_to_features = cloc_id_to_features
        self.num_side_feat_1 = num_side_feat_1
        self.num_side_feat_2 = num_side_feat_2
        self.corpus_back_features = corpus_back_features

        self.wordv = wordv
        self.loc_back_wordv = loc_backv
        self.corpus_back_wordv = corpus_backv

        self.all_wordv = all_wordv
        self.all_word_w2i = all_word_w2i
        self.all_word_i2w = all_word_i2w
        self.all_word_vocab_size = len(self.all_word_w2i)
        
        self.max_len_lc = max_len_lc
        self.lc_embed_size = sembed_size
        self.w2v_models = w2v_models
        self.w2v_dims = w2v_dims

        
    class batch_data(object):
        """
        This is a coded form of the dataset and its different attributes
        """
        
        def __init__(self, model_type, nbatches, batch_size, 
                     data, data_wts,
                     targets, target_weights,
                     ftargets, ftarget_weights,
                     back, back_weights, 
                     corpus_back, side_1, side_2, preembed):

            self.model_type = model_type
            self.nbatches = nbatches
            self.batch_size = batch_size
            self.data = data
            self.data_wts = data_wts
            self.targets = targets
            self.target_weights = target_weights
            self.ftargets = ftargets
            self.ftarget_weights = ftarget_weights
            self.back = back
            self.back_weights = back_weights
            self.corpus_back = corpus_back
            self.side_1 = side_1
            self.side_2 = side_2
            self.pretrained_embeddings = preembed

            
        def batch_producer(self, isTraining):
            """
            If training, the batch producer shuffles data before batching. 
            """

            if isTraining:
                shuffle_inds = np.random.permutation(np.arange(self.data.shape[0]))
            else:
                shuffle_inds = np.arange(self.data.shape[0])

            
            for b in range(self.nbatches):
                b_inds = shuffle_inds[b * self.batch_size: (b + 1) * self.batch_size]
                x = self.data[b_inds]
                x_wts = self.data_wts[b_inds]
                tar = self.targets[b_inds]
                tarwts = self.target_weights[b_inds]
                ftar = self.ftargets[b_inds]
                ftarwts = self.ftarget_weights[b_inds]

                if self.model_type == LSTM_MODEL:
                    yield (x, x_wts, tar, tarwts)
                elif self.model_type == HIER_LSTM_MODEL:
                    yield (x, x_wts, tar, tarwts)
                elif self.model_type == SNIP_MULTI_MODEL or self.model_type == MULTI_HIER_LSTM_MODEL:
                    yield (x, x_wts, ftar, ftarwts, tar, tarwts)
                elif self.model_type == LSTM_BACK_MODEL:
                    bk = self.back[b_inds]
                    bk_wts = self.back_weights[b_inds]
                    yield (x, tar, tarwts, bk, bk_wts)
                elif self.model_type == LSTM_ONE_BACK_MODEL:
                    cbk = self.corpus_back
                    yield (x, x_wts, tar, tarwts, cbk)
                elif self.model_type == SELF_ATTN_MODEL:
                    yield (x, x_wts, tar, tarwts)
                elif self.model_type == LSTM_MODEL_WITH_FEATURES:
                    x_side_1 = self.side_1[b_inds]
                    x_side_2 = self.side_2[b_inds]
                    yield (x, x_wts, tar, tarwts, x_side_1, x_side_2)
                else:
                    bk = self.back[b_inds]
                    bk_wts = self.back_weights[b_inds]
                    yield (x, tar, tarwts, bk, bk_wts)
                                    

                    
    def create_batch_data(self, batch_size, nblks, nsents, type_sent_embed, model_type):
        """ 
        Parameters
        ----------
        type_sent_embed: one of 
                AVG_WEMBED: average of pretrained word embeddings
        model_type: one of 
                LSTM_MODEL: lstm over the LOCs
                LSTM_MODEL_WITH_FEATURES: there are also side features in the LSTM model
                LSTM_ONE_BACK: lstm with single background at corpus level 
        
        Return
        ------
        x : matrix [batch_size * sequence_length * sentence_embed_size]. 
            Sentence_embed_size will vary depending on the value of type_sent_embed. 
        y : target values 0/1 for each sentence in the sequence [batch_size * sequence_length]
        z : weights 0/1 for each sentence in the sequence [batch_size * sequence_length]
        """
        
        data_by_seq_len = []
        cur_seq = []
        nblocks = 0

        hierarchical = False
        if model_type == HIER_LSTM_MODEL or model_type == SNIP_MULTI_MODEL or model_type == MULTI_HIER_LSTM_MODEL:
            hierarchical = True


        count_true_num_blks = 0
        Lc_and_Features = namedtuple('Lc_and_Features', ['lc', 'lc_feat'])
        for file_seq in self.file_seqs:
            if hierarchical:
                if len(data_by_seq_len) == 0 or data_by_seq_len[-1] != []:
                    data_by_seq_len.append([])
            for lcno in range(file_seq.num_locs()):
                lc = file_seq.get_loc(lcno)
                lc_features = self.cloc_id_to_features[str(lc.fileid) + "#" + str(lc.locid)]
                lc_bd = lc.blk_bd
                if hierarchical:
                    if lc_bd == 1:
                        if len(cur_seq) > 0:
                            data_by_seq_len[-1].append(cur_seq)
                            cur_seq = []
                        if len(data_by_seq_len[-1]) >= nblks:
                            data_by_seq_len.append([])
                        count_true_num_blks += 1
                        cur_seq.append(Lc_and_Features(lc, lc_features))
                    elif lc_bd == 2 or lc_bd == 3:
                        if len(cur_seq) < nsents:
                            cur_seq.append(Lc_and_Features(lc, lc_features))
                else:
                    cur_seq.append(Lc_and_Features(lc, lc_features))
                    if len(cur_seq) == nsents:
                        data_by_seq_len.append(cur_seq)  
                        cur_seq = []
            if len(cur_seq) > 0:
                if hierarchical:
                    data_by_seq_len[-1].append(cur_seq)
                else:
                    data_by_seq_len.append(cur_seq)
                cur_seq = []

                
        total_sequences = len(data_by_seq_len)

        if hierarchical:
            count_blks = 0
            for i in range(len(data_by_seq_len)):
                for j in range(len(data_by_seq_len[i])):
                    count_blks += 1
            # print("num_blocks read = " + str(count_blks))
            assert count_true_num_blks == count_blks, "Number of blks read into data %d does not match true number of blocks %d" % (count_blks, count_true_num_blks)
        # if hierarchical:
        #     for i in range(len(data_by_seq_len)):
        #         print("====================")
        #         for j in range(len(data_by_seq_len[i])):
        #             print("---------------")
        #             for k in range(len(data_by_seq_len[i][j])):
        #                 print(str(data_by_seq_len[i][j][k].lc))
        # else:
        #     for i in range(len(data_by_seq_len)):
        #         print("====================")
        #         for j in range(len(data_by_seq_len[i])):
        #             print(str(data_by_seq_len[i][j].lc))

        # sys.exit(1)
        
        nbatches = total_sequences // batch_size
        remaining_examples = total_sequences - nbatches * batch_size
        to_fill_sequences =  batch_size - remaining_examples

        if to_fill_sequences > 0:
            nbatches += 1
        
            
        if not hierarchical:
            if type_sent_embed == AVG_WEMBED:
                data = np.zeros((total_sequences + to_fill_sequences, nsents, self.max_len_lc), dtype=np.int32)
                data_wts = np.zeros((total_sequences + to_fill_sequences, nsents, self.max_len_lc), dtype=np.float32)

            back, back_weights = None, None
            if model_type == LSTM_ONE_BACK_MODEL:
                corpus_wid_back = np.array(self.corpus_back_features["coded"])

            side_1, side_2 = None, None
            if model_type == LSTM_MODEL_WITH_FEATURES:
                side_1 = np.zeros((total_sequences + to_fill_sequences, nsents, self.num_side_feat_1), dtype=np.float32)
                side_2 = np.zeros((total_sequences + to_fill_sequences, nsents, self.num_side_feat_2), dtype=np.float32)


            targets = np.zeros((total_sequences + to_fill_sequences, nsents), dtype=np.int32)
            target_weights = np.zeros((total_sequences + to_fill_sequences, nsents), dtype=np.float32)

            corpus_back = None
            if model_type == LSTM_ONE_BACK_MODEL:
                if type_sent_embed == AVG_WEMBED:
                    corpus_back = corpus_wid_back

            for i in range(total_sequences + to_fill_sequences):
                for j in range(nsents):
                    if i >= total_sequences or j >= len(data_by_seq_len[i]):
                        wid_datum = np.array([self.all_word_w2i[UNKNOWN_WORD]])
                        sembed_datum = np.zeros(self.w2v_dims[0], dtype=np.float32)
                        wid_back = np.array([self.all_word_w2i[UNKNOWN_WORD]])
                        sembed_back = np.zeros(self.w2v_dims[0], dtype=np.float32)
                        side_datum_1 = np.zeros(self.num_side_feat_1, dtype=np.float32)
                        side_datum_2 = np.zeros(self.num_side_feat_2, dtype=np.float32)
                    else:
                        wid_datum = np.array(data_by_seq_len[i][j].lc_feat.coded)
                        sembed_datum = data_by_seq_len[i][j].lc_feat.embedded
                        wid_back = np.array(data_by_seq_len[i][j].lc_feat.file_back_coded)
                        sembed_back = np.array(data_by_seq_len[i][j].lc_feat.file_back_embedded)
                        side_datum_1 = data_by_seq_len[i][j].lc_feat.side_features_1
                        side_datum_2 = data_by_seq_len[i][j].lc_feat.side_features_2
                        if data_by_seq_len[i][j].lc.label != -1:
                            targets[i][j] = data_by_seq_len[i][j].lc.label
                            target_weights[i][j] = 1.0
                        else:
                            targets[i][j] = 0
                            target_weights[i][j] = 0.0
                            
                    if type_sent_embed == AVG_WEMBED:
                        data[i][j][0:wid_datum.shape[0]] = wid_datum
                        data_wts[i][j][0:wid_datum.shape[0]] = np.ones(wid_datum.shape[0])

                    if model_type == LSTM_MODEL_WITH_FEATURES:
                        side_1[i][j] = side_datum_1
                        side_2[i][j] = side_datum_2

                    if model_type == LSTM_BACK_MODEL:
                        if type_sent_embed == AVG_WEMBED:
                            back[i][j][0:wid_back.shape[0]] = wid_back
                            back_weights[i][j][0:wid_back.shape[0]] = np.ones(wid_back.shape[0])
            ftargets = targets
            ftarget_weights = target_weights

        else:
            if type_sent_embed == AVG_WEMBED:
                data = np.zeros((total_sequences + to_fill_sequences, nblks, nsents, self.max_len_lc), dtype=np.int32)
                data_wts = np.zeros((total_sequences + to_fill_sequences, nblks, nsents, self.max_len_lc), dtype=np.float32)

            targets = np.zeros((total_sequences + to_fill_sequences, nblks), dtype=np.int32)
            target_weights = np.zeros((total_sequences + to_fill_sequences, nblks), dtype=np.float32)
            ftargets = np.zeros((total_sequences + to_fill_sequences, nblks, nsents), dtype=np.int32)
            ftarget_weights = np.zeros((total_sequences + to_fill_sequences, nblks, nsents), dtype=np.float32)
            back, back_weights, corpus_back = None, None, None

            side_1, side_2 = None, None
            if model_type == LSTM_MODEL_WITH_FEATURES:
                side_1 = np.zeros((total_sequences + to_fill_sequences, nblks, nsents, self.num_side_feat_1), dtype=np.float32)
                side_2 = np.zeros((total_sequences + to_fill_sequences, nblks, nsents, self.num_side_feat_2), dtype=np.float32)

            
            for i in range(total_sequences + to_fill_sequences):
                for k in range(nblks):
                    for j in range(nsents):
                        if i >= total_sequences or k >= len(data_by_seq_len[i]) or j >= len(data_by_seq_len[i][k]):
                            wid_datum = np.array([self.all_word_w2i[UNKNOWN_WORD]])
                            sembed_datum = np.zeros(self.w2v_dims[0], dtype=np.float32)
                            wid_back = np.array([self.all_word_w2i[UNKNOWN_WORD]])
                            sembed_back = np.zeros(self.w2v_dims[0], dtype=np.float32)
                            side_datum_1 = np.zeros(self.num_side_feat_1, dtype=np.float32)
                            side_datum_2 = np.zeros(self.num_side_feat_2, dtype=np.float32)
                        else:
                            wid_datum = np.array(data_by_seq_len[i][k][j].lc_feat.coded)
                            sembed_datum = data_by_seq_len[i][k][j].lc_feat.embedded
                            wid_back = np.array(data_by_seq_len[i][k][j].lc_feat.file_back_coded)
                            sembed_back = np.array(data_by_seq_len[i][k][j].lc_feat.file_back_embedded)
                            side_datum_1 = data_by_seq_len[i][k][j].lc_feat.side_features_1
                            side_datum_2 = data_by_seq_len[i][k][j].lc_feat.side_features_2
                            targets[i][k] = data_by_seq_len[i][k][0].lc.label
                            target_weights[i][k] = 1.0
                            ftargets[i][k][j] = data_by_seq_len[i][k][j].lc.ll_label
                            ftarget_weights[i][k][j] = 1.0
                            
                        if type_sent_embed == AVG_WEMBED:
                            data[i][k][j][0:wid_datum.shape[0]] = wid_datum
                            data_wts[i][k][j][0:wid_datum.shape[0]] = np.ones(wid_datum.shape[0])

                        if model_type == LSTM_MODEL_WITH_FEATURES:
                            side_1[i][k][j] = side_datum_1
                            side_2[i][k][j] = side_datum_2

        if type_sent_embed == AVG_WEMBED:
            pretrained_vectors = get_embedding_vectors(self.w2v_models[0], self.w2v_dims[0], self.all_word_w2i)

        return loc_dataset.batch_data(model_type, nbatches, batch_size, 
                                      data, data_wts,
                                      targets, target_weights,
                                      ftargets, ftarget_weights, 
                                      back, back_weights, 
                                      corpus_back, side_1, side_2,
                                      pretrained_vectors)


    class batch_data_for_classifier(object):

        def __init__(self, nbatches, batch_size, data, data_wts, targets, target_weights, preembed):
            self.nbatches = nbatches
            self.batch_size = batch_size
            self.data = data
            self.data_wts = data_wts
            self.targets = targets
            self.target_weights = target_weights
            self.pretrained_embeddings = preembed

        def batch_producer(self, isTraining):
            if isTraining:
                shuffle_inds = np.random.permutation(np.arange(self.data.shape[0]))
            else:
                shuffle_inds = np.arange(self.data.shape[0])

            for b in range(self.nbatches):
                b_inds = shuffle_inds[b * self.batch_size : (b + 1) * self.batch_size]
                x = self.data[b_inds]
                x_wts = self.data_wts[b_inds]
                tar = self.targets[b_inds]
                tarwts = self.target_weights[b_inds]
                yield (x, x_wts, tar, tarwts)

                
    def create_batch_data_for_classifier(self, batch_size, type_sent_embed):
        """ 
        :param type_sent_embed: one of 
                AVG_WEMBED: average of pretrained word embeddings
        :rtype x: matrix [batch_size * sentence_embed_size]. Sentence_embed_size will vary
        depending on the value of type_sent_embed. 
        :return y: target values 0/1 for each sentence in the sequence [batch_size]
        :return z: weights 0/1 for each sentence in the sequence [batch_size]
        """

        data_all = []
        total_lcs = 0
        Lc_and_Features = namedtuple('Lc_and_Features', ['lc', 'lc_feat'])
        for file_seq in self.file_seqs:
            for lcno in range(file_seq.num_locs()):
                lc = file_seq.get_loc(lcno)
                lc_features = self.cloc_id_to_features[str(lc.fileid) + "#" + str(lc.locid)]
                total_lcs += 1
                label = lc.label
                lc_and_feat = Lc_and_Features(lc, lc_features)
                data_all.append(lc_and_feat)

        nbatches = total_lcs // batch_size
        remaining_examples = total_lcs - nbatches * batch_size
        to_fill =  batch_size - remaining_examples
        if to_fill > 0:
            nbatches += 1

        if type_sent_embed == AVG_WEMBED:
            data = np.zeros((total_lcs + to_fill, self.max_len_lc), dtype=np.float32)
            data_wts = np.zeros((total_lcs + to_fill, self.max_len_lc), dtype=np.float32)
 
        targets = np.zeros((total_lcs + to_fill, 1), dtype=np.int32)
        target_weights = np.zeros((total_lcs + to_fill, 1), dtype=np.float32)
        pretrained_vectors = get_embedding_vectors(self.w2v_models[0], self.w2v_dims[0], self.all_word_w2i)
        
        for i in range(total_lcs):
            wid_datum = data_all[i].lc_feat.coded
            # sembed_datum = data_all[i].lc_feat.embedded
            # wc_datum = np.zeros((self.all_word_vocab_size))
            # for vid in data_all[i].lc_feat.coded:
            #     wc_datum[vid] += 1
    
            label = data_all[i].lc.label
            targets[i][0] = label
            target_weights[i][0] = 1.0
            
            if type_sent_embed == AVG_WEMBED:
                data[i][0:len(wid_datum)] = wid_datum
                data_wts[i][0:len(wid_datum)] = np.ones((len(wid_datum)), dtype=np.float32)

        return self.batch_data_for_classifier(nbatches, batch_size, data, data_wts, targets, target_weights, pretrained_vectors)





def create_vocabularies(all_words, loc_back_terms, corpus_back_terms, word_vsize, max_back_size):
    """
    Create a vocabulary that combines the loc terms and the background terms
    """
    
    print("\tCreating vocabularies for code and background..")
    wordv = get_restricted_vocabulary(all_words, word_vsize, MIN_WORD_FREQ, True)
    loc_back_wordv = get_restricted_vocabulary(loc_back_terms, max_back_size, MIN_WORD_FREQ, False)
    corpus_back_wordv = get_restricted_vocabulary(corpus_back_terms, max_back_size, 0, False)

    print("back vocab kept (loc) = " + str(len(loc_back_wordv)))
    print("back vocab kept (corpus) = " + str(len(corpus_back_wordv)))

    all_wordv = []
    all_wordv.extend(wordv)
    for bw in loc_back_wordv + corpus_back_wordv:
        if not bw in wordv:
            all_wordv.append(bw)
    
    all_word_w2i, all_word_i2w = assign_vocab_ids(all_wordv)

    return (wordv, 
            loc_back_wordv,
            corpus_back_wordv, 
            all_wordv, all_word_w2i, all_word_i2w)


def create_dataset(cfile,
                   back_file,
                   max_len_stmt,
                   skip_empty_line,
                   word_vsize,
                   max_back_size, 
                   code_ppx_file, code_tkfeat_file, 
                   comm_ppx_file, comm_tkfeat_file, 
                   compute_code_ppx_features, compute_comm_ppx_features,
                   compute_grammar_features,
                   is_bpe):

    file_seqs, all_words, loc_back_terms, corpus_back_terms = read_comment_location_file(cfile, back_file,
                                                                                         max_len_stmt, skip_empty_line,
                                                                                         code_ppx_file, code_tkfeat_file,
                                                                                         comm_ppx_file, comm_tkfeat_file)
    
    w2v_models = None
    w2v_dims = PRE_EMBED_DIM
    w2v_models, w2v_dims = read_w2v_models(is_bpe)

    (wordv,
     loc_back_wordv,
     corpus_back_wordv, 
     all_wordv, all_word_w2i, all_word_i2w) = create_vocabularies(all_words,
                                                                  loc_back_terms,
                                                                  corpus_back_terms, 
                                                                  word_vsize,
                                                                  max_back_size)
            
    loc_dataset = prepare_loc_data(file_seqs, 
                                     max_len_stmt,  
                                     wordv, 
                                     loc_back_wordv, 
                                     corpus_back_wordv,
                                     all_wordv, all_word_w2i, all_word_i2w, 
                                     w2v_models, w2v_dims,
                                     True, True, 
                                     compute_code_ppx_features, compute_comm_ppx_features, 
                                     compute_grammar_features)
    return loc_dataset

            
def main(argv):
    

    if len(argv) < 7:
        print("""cloc_data_reader.py -e <loc_file> 
        -z <background_terms_file> 
        -o <max_blks>
        -m <max_len_blk> 
        -n <max_len_stmt>
        -w <max_back_size> 
        -v <word_vocab_size> 
        -i <skip_empty_stmt?> 
        -a <code_ppx_file,code_tkfeat_file>  
        -b <comm_ppx_file,comm_tkfeat_file> 
        -g <no_arg:compute_grammar_features>
        -s <no arg:use_bpe>""")
        sys.exit(2)
                
    cfile, back_file, output_path = "", "", ""
    skip_empty_line = False
    word_vsize = 0
    max_blks, max_len_blk, max_len_stmt = 100, 100, 100
    max_back_size = 0
    use_bpe = False

    #side features files
    code_ppx_file = None
    code_tkfeat_file = None
    comm_ppx_file = None
    comm_tkfeat_file = None
    compute_code_ppx_features = False
    compute_comm_ppx_features = False
    compute_grammar_features = False

    try:
        opts, args = getopt.getopt(argv, "e:m:n:o:v:w:a:i:b:z:gs")
    except:
        print("""cloc_data_reader.py -e <loc_file> 
        -z <background_terms_file> 
        -o <max_blks>
        -m <max_len_blk> 
        -n <max_len_stmt> 
        -w <max_back_size> 
        -v <word_vocab_size> 
        -i <skip_empty_stmt?> 
        -a <code_ppx_file,code_tkfeat_file> 
        -b <comm_ppx_file,comm_tkfeat_file> 
        -g <no_arg:compute_grammar_features> 
        -s <no arg:use_bpe>""")
        print(str(argv))        
        sys.exit(2)
    
    
    for opt, arg in opts:
        if opt == "-e":
            cfile = arg
        elif opt == "-z":
            back_file = arg
        elif opt == "-o":
            max_blks = int(arg)
        elif opt == "-m":
            max_len_blk = int(arg)
        elif opt == "-n":
            max_len_stmt = int(arg)
        elif opt == "-w":
            max_back_size = int(arg)
        elif opt == "-v":
            word_vsize = int(arg)
        elif opt == "-i":
            if arg == "True":
                skip_empty_line = True
            else:
                skip_empty_line = False
        elif opt == "-a":
            if not arg.strip() == "" and not arg.strip() == ",":
                code_ppx_file = arg.split(',')[0]
                code_tkfeat_file = arg.split(',')[1]
                compute_code_ppx_features = True
        elif opt == "-b":
            if not arg.strip() == "" and not arg.strip() == ",":
                comm_ppx_file = arg.split(',')[0]
                comm_tkfeat_file = arg.split(',')[1]
                compute_comm_ppx_features = True
        elif opt == "-g":
            compute_grammar_features = True
        elif opt == "-s":
            use_bpe = True


    loc_dataset = create_dataset(cfile, back_file, max_len_stmt, skip_empty_line, 
                                 word_vsize, max_back_size, 
                                 code_ppx_file, code_tkfeat_file,
                                 comm_ppx_file, comm_tkfeat_file, 
                                 compute_code_ppx_features,
                                 compute_comm_ppx_features,
                                 compute_grammar_features,
                                 use_bpe)


    bdata = loc_dataset.create_batch_data(3, max_blks, max_len_blk, AVG_WEMBED, HIER_LSTM_MODEL)

    print(str(loc_dataset.all_word_w2i))
    
    with tf.Session() as session:
        print("\tBatch producer for HIERARCHICAL LSTM")
        for (x,xwts,tar,tarwts) in bdata.batch_producer(False):
            print(x.shape)
            print(x)
            print(xwts.shape)
            print(xwts)
            print(tar.shape)
            print(tar)
            print(tarwts.shape)
            print(tarwts)
            break

        
if __name__=="__main__":
    main(sys.argv[1:])
