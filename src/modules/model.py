import torch
import torch.nn as nn

from modules import content_decoder
from modules import sentence_planner

from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack

class EncoderRNN(nn.Module):

    def __init__(self, opt):
        super(EncoderRNN, self).__init__()
        self.hidden_size = opt.hidden_size // 2 # use bidirectional RNN
        self.LSTM = nn.LSTM(input_size=300,
                            hidden_size=self.hidden_size,
                            num_layers=2,
                            batch_first=True,
                            dropout=opt.dropout,
                            bidirectional=True)

        return

    def forward(self, input_embedded, input_lengths):
        """forward path, note that inputs are batch first"""


        lengths_list = input_lengths.view(-1).tolist()
        packed_emb = pack(input_embedded, lengths_list, True)

        memory_bank, encoder_final = self.LSTM(packed_emb)
        memory_bank = unpack(memory_bank)[0].view(input_embedded.size(0),input_embedded.size(1), -1)

        return memory_bank, encoder_final


class Model(nn.Module):
    def __init__(self, word_emb, vocab_size, opt):
        super(Model, self).__init__()
        self.word_emb = word_emb
        self.vocab_size = vocab_size

        self.sp_dec = sentence_planner.SentencePlanner(opt)
        self.wd_dec = content_decoder.WordDecoder(vocab_size, opt)
        self.ce_loss = nn.CrossEntropyLoss(reduction="sum", ignore_index=-1)
        self.nll_loss = nn.NLLLoss(reduction="sum", ignore_index=-1)
        self.bce_loss = nn.BCELoss(reduction="none")

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def compute_word_loss_probs(self, word_prob, word_targets):
        """
        Calculate cross-entropy loss on words.
        Args:
            word_prob: [batch_size, ]
            word_targets: [batch_size, ]
        """
        word_loss = self.nll_loss(torch.log(word_prob).view(-1, self.vocab_size), word_targets.view(-1))
        ppl = torch.exp(word_loss / torch.sum(word_targets >= 0))
        word_loss /= word_targets.size(0)
        return word_loss, ppl

    def compute_stype_loss(self, stype_pred, stype_labels):
        """
        Calculate cross-entropy loss on sentence type prediction.
        Args:
            stype_pred: [batch_size, max_sent_num, 4]: logits for type prediction
            stype_labels: [batch_size, max_sent_num]: gold-standard sentence type indices
        Returns:
            st_loss: scalar loss value averaged over all samples in the batch
        """
        st_loss = self.ce_loss(stype_pred.view(-1, self.sp_dec.sentence_type_n),
                               stype_labels.view(-1)) / stype_labels.size(0)
        return st_loss

    def compute_content_selection_loss(self, cs_pred, cs_labels, ph_bank_mask):
        """
        Calculate binary cross-entropy loss on keyphrase selection.
        Args:
            cs_pred: [batch_size, max_sent_num, max_ph_bank_size]
            cs_labels: [batch_size, max_sent_num, max_ph_bank_size]
            ph_bank_mask: [batch_size, max_sent_num, max_ph_bank_size]
        Returns:
            cs_loss: scalar loss value averaged over all samples in the batch.
        """
        cs_loss_flat = self.bce_loss(cs_pred.view(-1), cs_labels.view(-1))
        cs_loss_masked = ph_bank_mask.view(-1) * cs_loss_flat
        cs_loss = torch.sum(cs_loss_masked) / torch.sum(ph_bank_mask)
        return cs_loss


class ArgGenModel(Model):

    def __init__(self, word_emb, vocab_size, opt):
        super(ArgGenModel, self).__init__(word_emb, vocab_size, opt)
        self.encoder = EncoderRNN(opt)

    def forward_enc(self, src_inputs_tensor, src_len_tensor):
        src_emb = self.word_emb(src_inputs_tensor)
        enc_outs, enc_final = self.encoder.forward(input_embedded=src_emb, input_lengths=src_len_tensor)

        self.sp_dec.init_state(enc_final)
        self.wd_dec.init_state(enc_final)
        return enc_outs, enc_final

    def forward(self, tensor_dict, device=None):

        batch_size, sent_num, _ = tensor_dict["phrase_bank_selection_index"].size()
        enc_outs, _ = self.forward_enc(src_inputs_tensor=tensor_dict["src_inputs"],
                         src_len_tensor=tensor_dict["src_lens"])
        ph_bank_emb_raw = self.word_emb(tensor_dict["phrase_bank"])

        ph_bank_emb = torch.sum(ph_bank_emb_raw, -2)

        _, sp_dec_outs, stype_pred_logits, next_sent_sel_pred_probs, kp_mem_outs = \
            self.sp_dec.forward(
                ph_bank_emb=ph_bank_emb,
                ph_bank_sel_ind_inputs=tensor_dict["phrase_bank_selection_index"],
                stype_one_hot_tensor=tensor_dict["tgt_sent_type_onehot"],
                ph_sel_ind_mask=tensor_dict["phrase_bank_selection_index_mask"],
            )

        wd_dec_state, enc_attn, wd_pred_prob, wd_logits = self.wd_dec.forward(
            word_inputs_emb=self.word_emb(tensor_dict["tgt_word_ids_input"]),
            sent_planner_output=sp_dec_outs,
            sent_id_tensor=tensor_dict["tgt_sent_ids"],
            sent_mask_tensor=tensor_dict["tgt_word_ids_input_mask"],
            memory_bank=enc_outs,
            memory_len=tensor_dict["phrase_bank_len"],
            ph_bank_word_ids=tensor_dict["phrase_bank"],
            ph_bank_word_mask=tensor_dict["phrase_bank_word_mask"],
            stype_one_hot=tensor_dict["tgt_sent_type_onehot"].float(),
        )

        return stype_pred_logits, next_sent_sel_pred_probs, wd_pred_prob, wd_logits, enc_attn, kp_mem_outs




class AbsGenModel(Model):

    def __init__(self, word_emb, vocab_size, opt):
        super(AbsGenModel, self).__init__(word_emb, vocab_size, opt)
        self.encoder = EncoderRNN(opt)

    def forward_enc(self, src_inputs_tensor, src_len_tensor):
        src_emb = self.word_emb(src_inputs_tensor)
        enc_outs, enc_final = self.encoder.forward(src_emb, src_len_tensor)
        self.sp_dec.init_state(enc_final)
        self.wd_dec.init_state(enc_final)
        return enc_outs, enc_final


    def forward(self, tensor_dict, device=None):
        enc_outs, _ = self.forward_enc(src_inputs_tensor=tensor_dict["src_inputs"],
                                       src_len_tensor=tensor_dict["src_lens"])


        ph_bank_emb_raw = self.word_emb(tensor_dict["phrase_bank"])
        ph_bank_emb = torch.sum(ph_bank_emb_raw, -2)

        _, sp_dec_outs, _, next_sent_sel_pred_probs, kp_mem_outs = \
            self.sp_dec.forward(
                ph_bank_emb=ph_bank_emb,
                ph_bank_sel_ind_inputs=tensor_dict["phrase_bank_selection_index"],
                stype_one_hot_tensor=None,
                ph_sel_ind_mask=tensor_dict["phrase_bank_selection_index_mask"],
            )

        wd_dec_state, enc_attn, wd_pred_prob, wd_logits = self.wd_dec.forward(
            word_inputs_emb=self.word_emb(tensor_dict["tgt_word_ids_input"]),
            sent_planner_output=sp_dec_outs,
            sent_id_tensor=tensor_dict["tgt_sent_ids"],
            sent_mask_tensor=tensor_dict["tgt_word_ids_input_mask"],
            memory_bank=kp_mem_outs,
            memory_len=tensor_dict["phrase_bank_len"],
            ph_bank_word_ids=tensor_dict["phrase_bank"],
            ph_bank_word_mask=tensor_dict["phrase_bank_word_mask"],
            stype_one_hot=None,
        )

        return None, next_sent_sel_pred_probs, wd_pred_prob, wd_logits, enc_attn, kp_mem_outs


class WikiGenModel(Model):

    def __init__(self, word_emb, vocab_size, opt):
        super(WikiGenModel, self).__init__(word_emb, vocab_size, opt)

        # For Wikipedia generation, encoder is simply a linear layer for the title
        # self.encoder = (nn.Linear(300, 2 * 512, bias=True).cuda(), nn.Linear(300, 2 * 512, bias=True).cuda())
        self.encoder = nn.ModuleList([nn.Linear(300, 2 * 512, bias=True), nn.Linear(300, 2 * 512, bias=True)])

    def forward_enc(self, src_inputs_tensor):
        """
        Run feedforward encoder layer, where the input is the sum of word embeddings
        in the Wikipedia article title.

        Args:
            src_inputs_tensor: [batch_size, num_words] input title word ids
        """

        src_emb_word = self.word_emb(src_inputs_tensor)
        src_emb_instance = torch.sum(src_emb_word, dim=-2)
        enc_vec_h = torch.tanh(self.encoder[0](src_emb_instance))
        enc_vec_c = torch.tanh(self.encoder[1](src_emb_instance))

        self.sp_dec.init_state(encoder_final=(enc_vec_h.view(2, -1, 512),
                                              enc_vec_c.view(2, -1, 512)))
        self.wd_dec.init_state(encoder_final=(enc_vec_h.view(2, -1, 512),
                                              enc_vec_c.view(2, -1, 512)))


    def forward(self, tensor_dict, device=None):

        self.forward_enc(src_inputs_tensor=tensor_dict["src_inputs"])

        batch_size, sent_num, _ = tensor_dict["phrase_bank_selection_index"].size()


        if "style" in tensor_dict:
            style_embedded = tensor_dict["style"].float()
        else:
            style_embedded = torch.ones([batch_size, sent_num, 1], dtype=torch.float).to(device)


        # convert keyphrases into word embeddings
        # ph_bank_emb_raw: [batch_size, max_ph_bank, max_ph_words, 300]
        ph_bank_emb_raw = self.word_emb(tensor_dict["phrase_bank"])

        # sum up word embeddings for the keyphrase and create keyphrase embeddings
        # ph_bank_emb: [batch_size, max_ph_bank, 300]
        ph_bank_emb = torch.sum(ph_bank_emb_raw, -2)
        _, sp_dec_outs, stype_pred_logits, next_sent_sel_pred_probs, kp_mem_outs = \
            self.sp_dec.forward(
                ph_bank_emb=ph_bank_emb,
                ph_bank_sel_ind_inputs=tensor_dict["phrase_bank_selection_index"],
                stype_one_hot_tensor=tensor_dict["tgt_sent_type_onehot"],
                ph_sel_ind_mask=tensor_dict["phrase_bank_selection_index_mask"],
                global_style_emb=style_embedded
            )


        wd_dec_state, enc_attn, wd_pred_prob, wd_logits = self.wd_dec.forward(
            word_inputs_emb=self.word_emb(tensor_dict["tgt_word_ids_input"]),
            sent_planner_output=sp_dec_outs,
            sent_id_tensor=tensor_dict["tgt_sent_ids"],
            sent_mask_tensor=tensor_dict["tgt_word_ids_input_mask"],
            memory_bank=kp_mem_outs,
            memory_len=tensor_dict["phrase_bank_len"],
            ph_bank_word_ids=tensor_dict["phrase_bank"],
            ph_bank_word_mask=tensor_dict["phrase_bank_word_mask"],
            stype_one_hot=tensor_dict["tgt_sent_type_onehot"].float(),
        )


        return stype_pred_logits, next_sent_sel_pred_probs, wd_pred_prob, wd_logits, enc_attn, kp_mem_outs

