import math

import torch
import torch.nn as nn

import config as cfg
from utils.helpers import truncated_normal_

from transformers import GPT2Config, GPT2Tokenizer, GPT2LMHeadModel, AdamW
from transformers.optimization import get_linear_schedule_with_warmup


class TransformerGenerator(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, max_seq_len, padding_idx, gpu=False):
        super(TransformerGenerator, self).__init__()
        self.name = 'transformer'

        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.padding_idx = padding_idx
        self.gpu = gpu

        self.temperature = 1.0

        # loading pretrained tokenizer
        print("hey2")

        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.special_tokens = {'pad_token':'<|pad|>','sep_token':'<|sep|>'}
        self.num_add_toks = self.tokenizer.add_special_tokens(self.special_tokens)

        # loading pretrained model

        model = GPT2LMHeadModel.from_pretrained('gpt2')
        model.resize_token_embeddings(len(self.tokenizer))
        print("hey1")

        if self.gpu:
            model.to("cuda")


    def forward(self, inp, hidden, need_hidden=False):
        """
        Embeds input and applies LSTM
        :param inp: batch_size * seq_len
        :param hidden: (h, c)
        :param need_hidden: if return hidden, use for sampling
        """

        pred = self.model(inp)[0]

        return pred

    def sample(self, num_samples, batch_size, start_letter=cfg.start_letter):
        """
        Samples the network and returns num_samples samples of length max_seq_len.
        :return samples: num_samples * max_seq_length (a sampled sequence in each row)
        """
        num_batch = num_samples // batch_size + 1 if num_samples != batch_size else 1
        samples = torch.zeros(num_batch * batch_size, self.max_seq_len).long()

        # Generate sentences with multinomial sampling strategy
        for b in range(num_batch):
            hidden = self.init_hidden(batch_size)
            inp = torch.LongTensor([start_letter] * batch_size)
            if self.gpu:
                inp = inp.cuda()

            for i in range(self.max_seq_len):
                out, hidden = self.forward(inp, hidden, need_hidden=True)  # out: batch_size * vocab_size
                next_token = torch.multinomial(torch.exp(out), 1)  # batch_size * 1 (sampling from each row)
                samples[b * batch_size:(b + 1) * batch_size, i] = next_token.view(-1)
                inp = next_token.view(-1)
        samples = samples[:num_samples]

        return samples