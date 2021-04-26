#!/bin/bash


josep=/nfs/RESEARCH/crego/projects/PrimingNMT-2/

vocab=$PWD/minmt-vocab.py
setup=$PWD/minmt-setup.py
train=$PWD/minmt-train.py
avrge=$PWD/minmt-average.py
trans=$PWD/minmt-translate.py

tokenizer=$josep/MiNMT/tools/tokenizer.py
data=$josep/data
stovec=$josep/stovec

dir=$josep/minmt_base
dnet=$PWD/model_serie


python3 $setup -dnet $dnet -src_voc $dir/enfr.BPE.32k.voc -tgt_voc $dir/enfr.BPE.32k.voc
echo setup ok
echo debut du training

CUDA_VISIBLE_DEVICES=0 python3 $train -dnet $dnet -src_train data_corpus/trn_src_plein  -sim_train data_corpus/trn_sim_plein  -pre_train data_corpus/trn_pre_plein  -tgt_train data_corpus/trn_tgt_plein  -src_valid data_corpus/val_src_plein  -sim_valid data_corpus/val_sim_plein  -pre_valid data_corpus/val_pre_plein  -tgt_valid data_corpus/val_tgt_plein -max_steps 450000 -loss KLDiv -cuda -log_file $dnet/log &

echo fin du training
