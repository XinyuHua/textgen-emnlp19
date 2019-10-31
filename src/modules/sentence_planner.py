import torch
import torch.nn as nn

class SentencePlanner(nn.Module):

    def __init__(self, opt):

        super(SentencePlanner, self).__init__()
        self.hidden_size = opt.hidden_size
        self.batch_size = opt.batch_size

        planner_hidden_size = 300
        if opt.task == "arggen":
            self.sentence_type_n = 4
        elif opt.task == "wikigen":
            self.sentence_type_n = 5
            planner_hidden_size = 301  # global style extra bit
        else:
            self.sentence_type_n = 0

        self.opt = opt
        self.state = dict()

        self.planner = nn.LSTM(input_size=planner_hidden_size,
                               hidden_size=self.hidden_size,
                               num_layers=2,
                               dropout=opt.dropout,
                               batch_first=True,
                               bias=True)

        self.keyphrase_reader = nn.LSTM(input_size=300,
                                        hidden_size=150,
                                        num_layers=1,
                                        batch_first=True,
                                        bias=True,
                                        bidirectional=True)


        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()

        self.stype_inner = nn.Linear(self.hidden_size + planner_hidden_size, self.hidden_size, bias=True)
        self.stype_readout = nn.Linear(self.hidden_size, self.sentence_type_n, bias=True)
        self.keyphrase_sel_hidden_weights = nn.Linear(self.hidden_size, 1, bias=True)
        self.bilinear_layer = nn.Linear(300, 300, bias=False)
        return


    def init_state(self, encoder_final):
        def _fix_enc_hidden(hidden):
            hidden = torch.cat([hidden[0:hidden.size(0):2],
                                hidden[1:hidden.size(0):2]], 2)
            return hidden
        if self.opt.task == "wikigen":
            self.state["hidden"] = encoder_final
        elif self.opt.task in ["arggen", "absgen"]:
            self.state["hidden"] = tuple([_fix_enc_hidden(enc_hid) for enc_hid in encoder_final])

    def forward_onestep(self, kp_ph_bank_vec, ph_bank_sel_ind_inputs, ph_bank_sel_ind_history):
        """
        run forward pass on text planning decoder for one step only
        Args:
            kp_ph_bank_vec [batch_size x max_ph_bank_size x ph_vec_dim]: keyphrase memory representations
            ph_bank_sel_ind_inputs: [batch_size x max_ph_bank_size]: keyphrase selection for current step
            ph_bank_sel_ind_history: [batch_size x max_ph_bank_size x step-1]: selection history so far

        """
        ph_bank_sel_ind_inputs_tensor_sq = ph_bank_sel_ind_inputs.unsqueeze(-1).float()
        ph_sel_raw = ph_bank_sel_ind_inputs_tensor_sq * kp_ph_bank_vec
        # ph_sum_emb:
        ph_sum_emb = torch.sum(ph_sel_raw, -2).unsqueeze(1)

        self.rnn_output, dec_state = self.planner(ph_sum_emb, self.state["hidden"])
        self.state["hidden"] = dec_state

        stype_pred_logits = self._predict_sentence_type(ph_sum_emb)
        stype_onehot = self.sigmoid(stype_pred_logits)

        next_sentence_pred_probs = self._predict_keyphrase_selection(stype_onehot,
                                                                     ph_bank_sel_ind_history,
                                                                     kp_ph_bank_vec)
        return self.rnn_output, stype_pred_logits, next_sentence_pred_probs

    def forward_with_ph_bank_vec(self, ph_bank_vec, style_emb, ph_bank_sel_ind_inputs, stype_one_hot_tensor,
                                 ph_sel_ind_mask):

        ph_bank_sel_ind_inputs_tensor_sq = ph_bank_sel_ind_inputs.unsqueeze(-1).float()
        kp_ph_bank_vec_sq = ph_bank_vec.unsqueeze(-3)
        ph_sel_raw = ph_bank_sel_ind_inputs_tensor_sq * kp_ph_bank_vec_sq

        ph_sum_emb = torch.sum(ph_sel_raw, -2)
        ph_batch_size, max_sent_len, _ = ph_sum_emb.size()

        rnn_input = torch.cat((ph_sum_emb, style_emb), -1)

        self.rnn_output, dec_state = self.planner(rnn_input, self.state["hidden"])

        stype_pred_logits = self._predict_sentence_type(ph_sum_emb)
        ph_bank_sel_cumulative = torch.cumsum(ph_bank_sel_ind_inputs, dim=1).float() * ph_sel_ind_mask
        next_sentence_sel_pred = self._predict_keyphrase_selection(stype_one_hot_tensor,
                                                                   ph_bank_sel_cumulative,
                                                                   ph_bank_vec)
        return dec_state, self.rnn_output, stype_pred_logits, next_sentence_sel_pred, ph_bank_vec

    def forward(self, ph_bank_emb, ph_bank_sel_ind_inputs, stype_one_hot_tensor, ph_sel_ind_mask, global_style_emb=None):
        """
        Args:
            ph_bank_emb: [batch_size x max_ph_bank x 300] word embedding based phrase vectors for ph_bank
            ph_bank_sel_ind_inputs: [batch_size x max_sent_num x max_ph_bank] 1-hot encoding of phrase selection
            style_emb: [batch_size x max_sent_num x 300]
            stype_one_hot_tensor: [batch_size x max_sent_num x 4] 1-hot encoding of sentence type
            ph_sel_ind_mask: [batch_size x max_sent_num x max_ph_bank] 1-hot encoding to indiacate paddings for ph_bank
            global_style_emb: [batch_size] global style for each instance, only applicable for wikigen
        Returns:
            dec_state:
            self.rnn_output:
            stype_pred_logits:
            next_sentence_sel_pred:
        """
        # run RNN over ph_bank to obtain context-aware keyphrase representation
        # kp_ph_bank_vec: [batch_size x max_ph_bank x 300]
        kp_ph_bank_vec, _ = self.keyphrase_reader(ph_bank_emb)
        ph_bank_sel_ind_inputs_tensor_sq = ph_bank_sel_ind_inputs.unsqueeze(-1).float()
        kp_ph_bank_vec_sq = kp_ph_bank_vec.unsqueeze(-3)
        ph_sel_raw = ph_bank_sel_ind_inputs_tensor_sq * kp_ph_bank_vec_sq

        ph_sum_emb = torch.sum(ph_sel_raw, -2)
        ph_batch_size, max_sent_len, _ = ph_sum_emb.size()

        if global_style_emb is not None:
            global_style_emb = global_style_emb.unsqueeze(1).unsqueeze(1)
            global_style_emb = global_style_emb.repeat((1, ph_sum_emb.shape[1], 1))
            ph_sum_emb = torch.cat((ph_sum_emb, global_style_emb), -1)

        self.rnn_output, dec_state = self.planner(ph_sum_emb, self.state["hidden"])

        stype_pred_logits = self._predict_sentence_type(ph_sum_emb)
        ph_bank_sel_cumulative = torch.cumsum(ph_bank_sel_ind_inputs, dim=1).float() * ph_sel_ind_mask
        next_sentence_sel_pred = self._predict_keyphrase_selection(stype_one_hot_tensor,
                                                                   ph_bank_sel_cumulative,
                                                                   kp_ph_bank_vec)
        return dec_state, self.rnn_output, stype_pred_logits, next_sentence_sel_pred, kp_ph_bank_vec

    def _predict_sentence_type(self, ph_sum_emb):
        """
        predict sentence type based on hidden state s_j and phrase sum embedding m_j:
        t_j = softmax( self.readout( tanh(W[m_j; s_j]) ))
        """
        concat_kp_hidden = torch.cat((ph_sum_emb, self.rnn_output), dim=-1)
        stype_logits = self.stype_readout(self.tanh(self.stype_inner(concat_kp_hidden)))
        return stype_logits

    def _predict_keyphrase_selection(self, stype_one_hot, ph_bank_sel_cum, ph_bank_vec):
        """
        using history selection weights together with decoder RNN, keyphrase are represented as RNN states
        Args:
            stype_one_hot:
            ph_bank_sel_cum:
            ph_bank_vec:
        """

        sentence_level_features = self.keyphrase_sel_hidden_weights(self.rnn_output)

        max_ph_size = ph_bank_sel_cum.size(-1)
        sentence_level_features_broadcast = sentence_level_features.repeat((1, 1, max_ph_size))

        ph_bank_weighted_sum = torch.bmm(ph_bank_sel_cum.float(), ph_bank_vec)  # [batch_size x max_sent_num x 300]
        ph_bank_weighted_sum_repeat = ph_bank_weighted_sum.unsqueeze(-2).repeat(1, 1, ph_bank_vec.size(1), 1)
        ph_bank_emb_unsqueeze = ph_bank_vec.unsqueeze(-3)
        partial_prods = self.bilinear_layer(ph_bank_emb_unsqueeze)
        prods = partial_prods * ph_bank_weighted_sum_repeat
        word_level_features = torch.sum(prods, dim=-1)

        # word_level_features = self.keyphrase_sel_cov_weights(ph_bank_weighted_sum) # [batch_size x max_sent_num x 1]

        # word_level_features = self.keyphrase_sel_cov_weights(ph_bank_sel_cum.unsqueeze(dim=-1)).squeeze() * ph_bank_mask
        content_sel_logits = sentence_level_features_broadcast + word_level_features
        content_sel_pred = self.sigmoid(content_sel_logits)

        return content_sel_pred