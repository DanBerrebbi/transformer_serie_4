

josep=/nfs/RESEARCH/crego/projects/PrimingNMT-2/
data=$josep/data
tokenizer=tools/tokenizer.py
fref=$data/clean.Europarl.en-fr.fr.tst

sort -g model_serie/Europarl.out_k5_alpha0.7 | cut -f 2 | python3 $tokenizer -tok_config $data/BPE_config -detok | sacrebleu --force $fref > model_serie/Europarl.out_k5_alpha0.7.bleu

cat model_serie/Europarl.out_k5_alpha0.7.bleu
