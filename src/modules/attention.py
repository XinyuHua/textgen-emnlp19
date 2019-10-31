import torch
import torch.nn as nn
import torch.nn.functional as F
import utils.misc_utils as utils


class GlobalAttention(nn.Module):


    def __init__(self, query_dim, key_dim, stype_dim=0, type_conditional_lm=False):
        super(GlobalAttention, self).__init__()

        self.query_dim = query_dim
        self.key_dim = key_dim
        self.type_conditional_lm = type_conditional_lm

        self.linear_in = nn.Linear(query_dim, key_dim, bias=False)

        if self.type_conditional_lm:
            self.linear_out = nn.Linear(query_dim + key_dim + stype_dim, query_dim, False)
        else:
            self.linear_out = nn.Linear(query_dim + key_dim, query_dim, False)

        self.sigmoid = nn.Sigmoid()

    def score(self, h_t, h_s):
        """
        Args:
            h_t (FloatTensor): sequence of queries [batch x tgt_len x h_t_dim]
            h_s (FloatTensor): sequence of sources [batch x src_len x h_s_dim]

        Returns:
            raw attention scores for each src index [batch x tgt_len x src_len]
        """

        src_batch, src_len, src_dim = h_s.size()
        tgt_batch, tgt_len, tgt_dim = h_t.size()
        utils.aeq(src_batch, tgt_batch)
        # utils.aeq(src_dim, tgt_dim)

        h_t_ = h_t.view(tgt_batch * tgt_len, tgt_dim)
        h_t_ = self.linear_in(h_t_)
        h_t = h_t_.view(tgt_batch, tgt_len, src_dim)
        h_s_ = h_s.transpose(1, 2)
        return torch.bmm(h_t, h_s_)


    def forward(self, query, memory_bank, memory_lengths=None, use_softmax=True, stype_one_hot=None):
        """
        Args:
            query (FloatTensor): query vectors [batch x tgt_len x dim]
            memory_bank (FloatTensor): source vectors [batch x src_len x dim]
            memory_lengths (LongTensor): source context lengths [batch]
            use_softmax (bool): use softmax to produce alignment score,
                otherwise use sigmoid for each individual one
            stype_one_hot: [batch_size x max_sent_num x 4]: sentence type encoding used to learn conditional language model

        Returns:
            (FloatTensor, FloatTensor)

            computed attention weighted average: [batch x tgt_len x dim]
            attention distribution: [batch x tgt_len x src_len]
        """
        if query.dim == 2:
            one_step = True
            query = query.unsqueeze(1)
        else:
            one_step = False

        src_batch, src_len, src_dim = memory_bank.size()
        query_batch, query_len, query_dim = query.size()
        utils.aeq(src_batch, query_batch)

        align = self.score(query, memory_bank)

        if memory_lengths is not None:
            mask = utils.sequence_mask(memory_lengths, max_len=align.size(-1))
            mask = mask.unsqueeze(1)
            align.masked_fill_(1 - mask, -float('inf'))

        if use_softmax:
            align_vectors = F.softmax(align.view(src_batch * query_len, src_len), -1)
            align_vectors = align_vectors.view(src_batch, query_len, src_len)
        else:
            align_vectors = self.sigmoid(align)

        c = torch.bmm(align_vectors, memory_bank)

        if self.type_conditional_lm:
            # concat_c = torch.cat([c, query, stype_one_hot], 2).view(src_batch * query_len, src_dim + query_dim + 4)
            concat_c = torch.cat([c, query, stype_one_hot], 2).view(src_batch * query_len, -1)
        else:
            concat_c = torch.cat([c, query], 2).view(src_batch * query_len, src_dim + query_dim)

        attn_h = self.linear_out(concat_c).view(src_batch, query_len, query_dim)
        attn_h = torch.tanh(attn_h)

        if one_step:
            attn_h = attn_h.squeeze(1)
            align_vectors = align_vectors.squeeze(1)

            batch_, dim_ = attn_h.size()
            utils.aeq(src_batch, batch_)
            utils.aeq(src_dim, dim_)
            batch_, src_l_ = align_vectors.size()
            utils.aeq(src_batch, batch_)
            utils.aeq(src_len, src_l_)

        else:

            batch_, target_l_, dim_ = attn_h.size()
            utils.aeq(target_l_, query_len)
            utils.aeq(batch_, query_batch)
            utils.aeq(dim_, query_dim)

            batch_, target_l_, source_l_ = align_vectors.size()
            utils.aeq(target_l_, query_len)
            utils.aeq(batch_, query_batch)
            utils.aeq(source_l_, src_len)

        return attn_h, align_vectors, align, c