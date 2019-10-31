import torch
import torch.nn as nn
from modules import attention

EPS=1e-8

class WordDecoder(nn.Module):

    def __init__(self, vocab_size, opt):
        super(WordDecoder, self).__init__()
        self.emb_size = 300
        self.task = opt.task
        self.hidden_size = opt.hidden_size
        self.LSTM = nn.LSTM(input_size=self.emb_size,
                            hidden_size=opt.hidden_size,
                            num_layers=2,
                            dropout=opt.dropout,
                            batch_first=True,
                            bias=True)

        if opt.task == "arggen":
            self.stype_n = 4
            attn_key_dim = 512
        elif opt.task == "wikigen":
            self.stype_n = 5
            attn_key_dim = 300
        else:
            attn_key_dim = 300
            self.stype_n = 0

        self.pointer_generator = opt.pointer_generator
        self.readout = nn.Linear(opt.hidden_size, vocab_size, bias=True)
        self.word_transformation = nn.Linear(self.emb_size, self.emb_size, bias=True)
        self.planner_transformation = nn.Linear(self.hidden_size, self.emb_size, bias=True)
        self.enc_attn = attention.GlobalAttention(query_dim=self.hidden_size,
                                                  key_dim=attn_key_dim,
                                                  stype_dim=self.stype_n,
                                                  type_conditional_lm=opt.type_conditional_lm)
        self.state = {}
        self.softmax = nn.Softmax(dim=-1)
        if self.pointer_generator:
            self.p_gen_c = nn.Linear(in_features=self.emb_size, out_features=1, bias=True)
            self.p_gen_z = nn.Linear(in_features=opt.hidden_size, out_features=1, bias=False)
            self.p_gen_y = nn.Linear(in_features=self.emb_size, out_features=1, bias=False)
            self.sigmoid = nn.Sigmoid()
        return

    def init_state(self, encoder_final):
        """ Init decoder state with last state of the encoder """

        def _fix_enc_hidden(hidden):
            hidden = torch.cat([hidden[0:hidden.size(0):2],
                                hidden[1:hidden.size(0):2]], 2)
            return hidden
        if self.task == "wikigen":
            self.state["hidden"] = encoder_final
        elif self.task in ["arggen", "absgen"]:
            self.state["hidden"] = tuple([_fix_enc_hidden(enc_hid) for enc_hid in encoder_final])

    def map_state(self, fn):
        self.state["hidden"] = tuple(map(lambda x: fn(x, 1), self.state["hidden"]))

    def forward_onestep(self, word_inputs_emb, sent_planner_output, enc_memory_bank, enc_memory_len, stype_one_hot):
        merged_inputs = self.word_transformation(word_inputs_emb) + self.planner_transformation(sent_planner_output)
        rnn_input = torch.tanh(merged_inputs)
        rnn_output, dec_state = self.LSTM(rnn_input, self.state["hidden"])
        self.state["hidden"] = dec_state

        self.rnn_output = rnn_output
        dec_outs, enc_attn, _, _ = self.enc_attn.forward(
            rnn_output.contiguous(),
            enc_memory_bank.contiguous(),
            memory_lengths=enc_memory_len,
            stype_one_hot=stype_one_hot
        )
        readouts = self.readout(dec_outs)
        return enc_attn, readouts

    def forward(self, word_inputs_emb, sent_planner_output, sent_id_tensor, sent_mask_tensor,
                memory_bank, memory_len, ph_bank_word_ids, ph_bank_word_mask, stype_one_hot=None):

        max_tgt_len = word_inputs_emb.size(1)
        sent_planner_output_dim = sent_planner_output.size(-1)

        sent_id_template_expanded = sent_id_tensor.unsqueeze(dim=-1).expand(-1, max_tgt_len,
                                                                            sent_planner_output_dim)
        token_distributed_sent_planner_output = torch.gather(
            sent_planner_output, 1, sent_id_template_expanded)

        # at this point, the size of the following tensor should be:
        # [batch_size x max_rr_len x sp_out_dim]
        token_distributed_sent_planner_output_masked = sent_mask_tensor.unsqueeze(-1) \
                                                       * token_distributed_sent_planner_output

        if stype_one_hot is not None:
            sent_id_template_expanded_for_stype = sent_id_tensor.unsqueeze(dim=-1) \
                .expand(-1, max_tgt_len, self.stype_n)
            token_distributed_sent_type = torch.gather(
                stype_one_hot, 1, sent_id_template_expanded_for_stype
            )
        else:
            token_distributed_sent_type = None

        merged_inputs = self.word_transformation(word_inputs_emb) \
                        + self.planner_transformation(token_distributed_sent_planner_output_masked)
        rnn_input = torch.tanh(merged_inputs)
        rnn_output, dec_state = self.LSTM(rnn_input, self.state["hidden"])
        self.rnn_output = rnn_output
        dec_outs, enc_attn, _, c = self.enc_attn.forward(
            rnn_output.contiguous(),
            memory_bank.contiguous(),
            memory_lengths=memory_len,
            stype_one_hot=token_distributed_sent_type)
        readouts = self.readout(dec_outs)
        vocab_pred_dist = self.softmax(readouts)

        if self.pointer_generator:
            # p_gen: [batch_size]
            p_gen = self.sigmoid(self.p_gen_c(c) + self.p_gen_z(rnn_output) + self.p_gen_y(word_inputs_emb))
            vocab_dist = self.softmax(readouts)
            vocab_dist_ = p_gen * vocab_dist

            # attn_dist_: [batch_size x max_tgt_len x max_ph_bank]
            attn_dist_ = (1 - p_gen) * enc_attn

            # ph_bank_len: [batch_size x max_ph_bank]
            ph_bank_len = torch.sum(ph_bank_word_mask, dim=-1).float() + EPS

            # attn_avg: [batch_size x max_tgt_len x max_ph_bank]
            attn_avg = attn_dist_ / ph_bank_len.unsqueeze(1)

            max_ph_len = ph_bank_word_mask.size(-1)
            # attn_dist_rep: [batch_size x max_tgt_len x max_ph_bank x max_ph_len]
            attn_dist_rep = attn_avg.unsqueeze(-1).repeat(1, 1, 1, max_ph_len)

            # attn_masked: [batch_size x max_tgt_len x max_ph_bank
            max_tgt_len = attn_dist_.size(1)
            attn_masked = attn_dist_rep * ph_bank_word_mask.unsqueeze(1).repeat(1, max_tgt_len, 1, 1)

            # ph_bank_tensor_rep: [batch_size x max_tgt_len x max_ph_bank x max_ph_len]
            ph_bank_tensor_rep = ph_bank_word_ids.unsqueeze(1).repeat(1, max_tgt_len, 1, 1)

            batch_size = vocab_dist.size(0)
            vocab_dist_.scatter_add_(dim=2, index=ph_bank_tensor_rep.view(batch_size, max_tgt_len, -1),
                                     src=attn_masked.view(batch_size, max_tgt_len, -1))
            vocab_pred_dist = vocab_dist_

        return dec_state, enc_attn, vocab_pred_dist, readouts