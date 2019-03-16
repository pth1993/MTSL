#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0 python main_hierarchical_shared_model.py --rnn_mode LSTM --num_epochs 200 --batch_size 16 \
 --hidden_size 256 --num_layers 1 --char_dim 30 --num_filters 30 --learning_rate 0.01 --decay_rate 0.05 --schedule 1 \
 --gamma 0.0 --p_in 0.33 --p_rnn 0.33 0.5 --p_out 0.5 --bigram --lm_loss 0.05 \
 --embedding_path "data/embedding/glove_embedding.txt" --option_path "data/embedding/elmo_option.json" \
 --weight_path "data/embedding/elmo_weight.hdf5" --word2index_path "output/hierarchical_shared_model/chunk/word2index" \
 --out_path "output/hierarchical_shared_model/chunk" --use_crf "True" --use_lm "True" --use_elmo "True" \
 --label_type "chunk" "fgner" \
 --bucket_auxiliary 5 10 15 20 25 30 40 50 60 70 80 \
 --bucket_main 5 10 15 20 25 30 40 50 60 70 80 \
 --train "data/chunk/train.chunk" "data/fgner/train.fgner" \
 --dev "data/chunk/dev.chunk" "data/fgner/dev.fgner" \
 --test "data/chunk/test.chunk" "data/fgner/test.fgner"