# -*- coding: utf-8 -*-
# @Author       : William
# @Project      : TextGAN-william
# @FileName     : SeqGAN_G.py
# @Time         : Created at 2019-04-25
# @Blog         : http://zhiweil.ml/
# @Description  : 
# Copyrights (C) 2018. All Rights Reserved.

import torch
import torch.nn.functional as F

from models.transfomer_generator import TransformerGenerator


class TGAN_G(TransformerGenerator):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, max_seq_len, padding_idx, gpu=False):
        super(TGAN_G, self).__init__(embedding_dim, hidden_dim, vocab_size, max_seq_len, padding_idx, gpu)
        self.name = 'tgan'


    def batchPGLoss(self, inp, target, reward):
        """
        Returns a policy gradient loss

        :param inp: batch_size x seq_len, inp should be target with <s> (start letter) prepended
        :param target: batch_size x seq_len
        :param reward: batch_size (discriminator reward for each sentence, applied to each token of the corresponding sentence)
        :return loss: policy loss
        """

        batch_size, seq_len = inp.size()
        hidden = self.init_hidden(batch_size)

        out = self.forward(inp, hidden).view(batch_size, self.max_seq_len, self.vocab_size)
        target_onehot = F.one_hot(target, self.vocab_size).float()  # batch_size * seq_len * vocab_size
        pred = torch.sum(out * target_onehot, dim=-1)  # batch_size * seq_len
        loss = -torch.sum(pred * reward)

        return loss
