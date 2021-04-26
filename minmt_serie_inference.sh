#!/bin/bash


josep=/nfs/RESEARCH/crego/projects/PrimingNMT-2/

vocab=$PWD/minmt-vocab.py
setup=$PWD/minmt-setup.py
train=$PWDT/minmt-train.py
avrge=$PWD/minmt-average.py
trans=$PWD/minmt-translate.py

tokenizer=$josep/MiNMT/tools/tokenizer.py
data=$josep/data
stovec=$josep/stovec

dir=$josep/minmt_base
dnet=$PWD/model_serie


fmod=$dnet/network.checkpoint_00015000.pt ####### A MODIF

	for corpus in Europarl; do
	    echo $corpus
	    echo test ...
	    fsrc=$stovec/clean.$corpus.en-fr.en.tst.bpe.vec.sim0.5_k5_n0_t0.8.src
	    ftgt=$stovec/clean.$corpus.en-fr.en.tst.bpe.vec.sim0.5_k5_n0_t0.8.tgt
	    fsim=$stovec/clean.$corpus.en-fr.en.tst.bpe.vec.sim0.5_k5_n0_t0.8.sim
	    fpre=$stovec/clean.$corpus.en-fr.en.tst.bpe.vec.sim0.5_k5_n0_t0.8.pre
      fout=$dnet/$corpus.out_k5_alpha0.7
      CUDA_VISIBLE_DEVICES=0 python3 $trans -dnet $dnet -m $fmod -batch_size 10 -beam_size 5 -alpha 0.7 -i_src $fsrc -i_sim $fsim -i_pre $fpre -o $fout -cuda -log_file $fout.log &

	    echo test ok

   echo corpus suivant :

  done

