import sys
import os
import time
import random
import logging
import torch
import numpy as np
from transformer.Dataset import Dataset, Vocab
from transformer.Model import Encoder_Decoder, load_checkpoint, numparameters
from transformer.Optimizer import OptScheduler, LabelSmoothing_NLL, LabelSmoothing_KLDiv
from transformer.Learning import Learning
from tools.Tools import create_logger, read_dnet

sys.stderr = open('error_serie_true.log', 'w')

######################################################################
### Options ##########################################################
######################################################################

class Options():
    def __init__(self, argv):
        self.prog = argv.pop(0)
        self.dnet = None
        self.src_train = None
        self.tgt_train = None
        self.src_valid = None
        self.tgt_valid = None
        ### learning
        self.max_steps = 0
        self.max_epochs = 0
        self.validate_every = 5000
        self.save_every = 5000
        self.report_every = 100
        self.keep_last_n = 5
        self.mask_prefix = False
        ### optim
        self.noam_scale = 2.0
        self.noam_warmup = 4000
        self.label_smoothing = 0.1
        self.loss = 'NLL'
        self.clip = 0.5
        ### data
        self.shard_size = 500000
        self.max_length = 100
        self.batch_size = 4096*2
        self.batch_type = 'tokens'

        self.cuda = False
        self.seed = 12345
        log_file = 'stderr'
        log_level = 'info'

        while len(argv):
            tok = argv.pop(0)
            if tok == "-h":
                self.usage()

            elif tok == '-dnet' and len(argv):
                self.dnet = argv.pop(0)
                self.dnet = self.dnet[:-1] if self.dnet[-1] == '/' else self.dnet  ### remove trailing '/'
            elif tok == '-max_steps':
                self.max_steps = int(argv.pop(0))
            elif tok == '-max_epochs':
                self.max_epochs = int(argv.pop(0))
            elif tok == '-validate_every':
                self.validate_every = int(argv.pop(0))
            elif tok == '-save_every':
                self.save_every = int(argv.pop(0))
            elif tok == '-report_every':
                self.report_every = int(argv.pop(0))
            elif tok == '-keep_last_n':
                self.keep_last_n = int(argv.pop(0))
            elif tok == '-mask_prefix':
                self.mask_prefix = True
            elif tok == '-noam_scale':
                self.noam_scale = float(argv.pop(0))
            elif tok == '-noam_warmup':
                self.noam_warmup = float(argv.pop(0))
            elif tok == '-label_smoothing':
                self.label_smoothing = float(argv.pop(0))
            elif tok == '-loss':
                self.loss = argv.pop(0)
            elif tok == '-clip':
                self.clip = float(argv.pop(0))

            elif tok == '-src_train':
                self.src_train = argv.pop(0)

            elif tok == '-pre_train':
                self.pre_train = argv.pop(0)
            elif tok == '-tgt_train':
                self.tgt_train = argv.pop(0)
            elif tok == '-src_valid':
                self.src_valid = argv.pop(0)

            elif tok == '-pre_valid':
                self.pre_valid = argv.pop(0)
            elif tok == '-tgt_valid':
                self.tgt_valid = argv.pop(0)
            elif tok == '-shard_size':
                self.shard_size = int(argv.pop(0))
            elif tok == '-max_length':
                self.max_length = int(argv.pop(0))
            elif tok == '-batch_size':
                self.batch_size = int(argv.pop(0))
            elif tok == '-batch_type':
                self.batch_type = argv.pop(0)

            elif tok == "-cuda":
                self.cuda = True
            elif tok == "-seed":
                self.seed = int(argv.pop(0))
            elif tok == "-log_file" and len(argv):
                log_file = argv.pop(0)
            elif tok == "-log_level" and len(argv):
                log_level = argv.pop(0)

            else:
                self.usage('Unrecognized {} option'.format(tok))

        if self.dnet is None:
            self.usage('missing -dnet option')

        if self.src_train is None or self.tgt_train is None:
            self.usage('missing -src_train/-tgt_train options')

        create_logger(log_file, log_level)
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        logging.info("Options = {}".format(self.__dict__))

    def usage(self, messg=None):
        if messg is not None:
            sys.stderr.write(messg + '\n')
        sys.stderr.write('''usage: {} -dnet DIR -src_train FILE -tgt_train FILE [-src_valid FILE] [-tgt_valid FILE] 
   -dnet              DIR : network directory [must exist]
   -src_train        FILE : source-side training file
   -tgt_train        FILE : target-side training file
   -src_valid        FILE : source-side validation file
   -tgt_valid        FILE : target-side validation file
   [Learning]
   -max_steps         INT : maximum number of training updates ({})
   -max_epochs        INT : maximum number of training epochs ({})
   -validate_every    INT : validation every INT model updates ({})
   -save_every        INT : save model every INT model updates ({})
   -report_every      INT : report every INT model updates ({})
   -keep_last_n       INT : save last INT checkpoints ({})
   -mask_prefix           : mask prefix tokens not appearing in target ({})
   [Optimization]
   -label_smoothing FLOAT : label smoothing probability ({})
   -loss           STRING : loss function: KLDiv, NLL ({})
   -clip            FLOAT : clips gradient norm of parameters ({})
   -noam_scale      FLOAT : scale of Noam decay for learning rate ({})
   -noam_warmup       INT : warmup steps of Noam decay for learning rate ({})
   [Data]
   -shard_size        INT : maximum shard size ({}) use 0 to consider all data in a single shard
   -max_length        INT : skip example if number of tokens exceeds this ({})
   -batch_size        INT : maximum batch size ({})
   -batch_type     STRING : sentences or tokens ({})
   -cuda                  : use cuda device instead of cpu ({})
   -seed              INT : seed for randomness ({})
   -log_file         FILE : log file  (stderr)
   -log_level      STRING : log level [debug, info, warning, critical, error] (info)
   -h                     : this help
'''.format(self.prog, self.max_steps, self.max_epochs, self.validate_every, self.save_every, self.report_every,
           self.keep_last_n, self.mask_prefix, self.label_smoothing, self.loss, self.clip, self.noam_scale,
           self.noam_warmup, self.shard_size, self.max_length, self.batch_size, self.batch_type, self.cuda, self.seed))
        sys.exit()


######################################################################
### MAIN #############################################################
######################################################################

if __name__ == '__main__':

    tic = time.time()
    o = Options(sys.argv)
    n, src_voc, tgt_voc = read_dnet(o.dnet)
    src_voc = Vocab(src_voc)
    tgt_voc = Vocab(tgt_voc)

    #####################################################
    ####     a verif
    #####################################################

    tgt_voc = tgt_voc


    ########################
    ### load model/optim ###
    ########################
    device = torch.device('cuda' if o.cuda and torch.cuda.is_available() else 'cpu')
    model = Encoder_Decoder(n['n_layers'], n['ff_dim'], n['n_heads'], n['emb_dim'], n['qk_dim'], n['v_dim'],
                            n['dropout'], n['share_embeddings'], len(src_voc), len(tgt_voc), src_voc.idx_pad).to(device)
    logging.info(
        'Built model (#params, size) = ({}) in device {}'.format(', '.join([str(f) for f in numparameters(model)]),
                                                                 next(model.parameters()).device))
    optim = torch.optim.Adam(model.parameters(), weight_decay=n['weight_decay'], betas=(n['beta1'], n['beta2']),
                             eps=n['eps'])
    last_step, model, optim = load_checkpoint(o.dnet + '/network', model, optim, device)

    ############################
    ### build scheduler/loss ###
    ############################
    optScheduler = OptScheduler(optim, n['emb_dim'], o.noam_scale, o.noam_warmup, last_step)
    if o.loss == 'KLDiv':
        criter = LabelSmoothing_KLDiv(len(tgt_voc), src_voc.idx_pad, o.label_smoothing).to(device)
    elif o.loss == 'NLL':
        criter = LabelSmoothing_NLL(len(tgt_voc), src_voc.idx_pad, o.label_smoothing).to(device)
    else:
        logging.error('bad -loss option')
        sys.exit()

    ##################
    ### load data ####
    ##################
    valid = None
    if o.src_valid is not None and o.tgt_valid is not None:
        # valid = Dataset([src_voc, tgt_voc], [o.src_valid, o.tgt_valid], o.shard_size, o.batch_size, o.batch_type, o.max_length)
        valid = Dataset([src_voc, tgt_voc, tgt_voc], [o.src_valid, o.tgt_valid,  o.pre_valid], o.shard_size, o.batch_size, o.batch_type,
                        o.max_length)
    #train = Dataset([src_voc, tgt_voc], [o.src_train, o.tgt_train], o.shard_size, o.batch_size, o.batch_type, o.max_length)

    train = Dataset([src_voc, tgt_voc,  tgt_voc], [o.src_train, o.tgt_train,  o.pre_train],
                    o.shard_size, o.batch_size, o.batch_type, o.max_length)

    #############
    ### learn ###
    #############
    learning = Learning(model, optScheduler, criter, o.dnet + '/network', src_voc.idx_pad, tgt_voc.idx_sep,
                        tgt_voc.idx_msk, o)
    learning.learn(train, valid, device)

    toc = time.time()
    logging.info('Done ({:.2f} seconds)'.format(toc - tic))
