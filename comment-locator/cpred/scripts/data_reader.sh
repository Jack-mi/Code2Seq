# testing the data reader #

source_folder=/disk/scratch1/alouis/projects/comm-pred/comment-location/data/split30_word_data/train
source=$source_folder/data.txt
backf=$source_folder/corpus_back.txt

max_blks=2
max_len_blk=2
max_stmt_len=5
max_back_size=10
word_vsize=1000
skip_empty_stmt=True


CUDA_VISIBLE_DEVICES=0 python3 data/cloc_data_reader.py\
		    -e $source \
		    -z $backf \
		    -o $max_blks \
		    -m $max_len_blk \
		    -n $max_stmt_len \
		    -w $max_back_size \
		    -v $word_vsize \
		    -i $skip_empty_stmt 




