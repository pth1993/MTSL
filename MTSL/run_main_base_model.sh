#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0 python main_base_model.py --rnn_mode LSTM --num_epochs 200 --batch_size 16 \
 --hidden_size 256 --num_layers 1 --char_dim 30 --num_filters 30 --learning_rate 0.01 --decay_rate 0.05 --schedule 1 \
 --gamma 0.0 --p_in 0.33 --p_rnn 0.33 0.5 --p_out 0.5 --bigram --lm_loss 0.05 \
 --embedding_path "data/embedding/glove_embedding.txt" --option_path "data/embedding/elmo_option.json" \
 --weight_path "data/embedding/elmo_weight.hdf5" --word2index_path "output/base_model/word2index" \
 --out_path "output/base_model" --use_crf "False" --use_lm "False" --use_elmo "False" --label_type "ner" \
 --train "data/ner/train.ner" \
 --dev "data/ner/dev.ner" \
 --test "data/ner/test.ner"