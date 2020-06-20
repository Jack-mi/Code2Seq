# Inference from the hierarchical model 


data_path=/disk/scratch1/alouis/projects/comm-pred/comment-location/data/split30_word_data
model_path=/disk/scratch1/alouis/projects/comm-pred/comment-location/models/hier
code_vocab_size=5000
max_blks=30
max_len_blk=30
max_len_stmt=30
num_labels=2
lstm_num_layers=1
lstm_hidden_size=800
stmt_embed_size=300
stmt_embed_type="avg_wembed"
learning_rate=0.001
keep_prob=0.9
max_epoch=200
use_pretraining=True
is_bpe=False
skip_empty_stmt=False

train_dir=$model_path/L$lstm_num_layers-H$lstm_hidden_size-K$keep_prob-R$learning_rate
mkdir $train_dir

CUDA_VISIBLE_DEVICES=0 python3 runners/comm_loc_hier_lstm.py \
		    --data_path=$data_path \
		    --ignore_stmt_marker=$ignore_stmt_marker \
		    --train_dir=$train_dir \
		    --code_vocab_size=$code_vocab_size \
		    --max_blks=$max_blks \
		    --max_len_blk=$max_len_blk \
		    --max_len_stmt=$max_len_stmt \
		    --num_labels=$num_labels \
		    --lstm_num_layers=$lstm_num_layers \
		    --lstm_hidden_size=$lstm_hidden_size \
		    --stmt_embed_size=$stmt_embed_size \
		    --stmt_embed_type=$stmt_embed_type \
		    --learning_rate=$learning_rate \
		    --keep_prob=$keep_prob \
		    --max_epoch=$max_epoch \
		    --use_pretraining=$use_pretraining \
		    --is_bpe=$is_bpe \
		    --skip_empty_stmt=$skip_empty_stmt \
		    --test

