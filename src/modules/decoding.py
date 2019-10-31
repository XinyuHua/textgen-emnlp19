import torch
import torch.nn as nn
import numpy as np
from modules.beam import Beam, GNMTGlobalScorer
import utils.misc_utils as utils


def infer_batch(model, batch, opt, eos_symbols):
    """
    Run inference over a batch
    """
    batch_size = len(batch["src_lens"])
    enc_outs, enc_final = model.forward_enc(src_inputs_tensor=batch["src_inputs"],
                                src_len_tensor=batch["src_lens"])
    enc_memory_length = utils.tile(batch["src_lens"], opt.beam_size, 0)
    enc_memory_bank = utils.tile(enc_outs, opt.beam_size, 0)

    if opt.use_true_kp:
        sp_results = run_planner_on_true_kp(model, batch, batch_size)
    else:
        sp_results = infer_planner(model, batch, batch_size, opt)
    wd_results = infer_wd(model, sp_results, enc_memory_bank, enc_memory_length, batch_size, opt, eos_symbols)

    return sp_results, wd_results


def infer_wd(model, sp_results, enc_memory_bank, enc_memory_lengths, batch_size, opt, eos_symbols):
    """
    Args:
        tensor_data_dict:
            "ph_bank_tensor": [batch_size x max_ph_size x max_ph_len] tensor of phrase word ids
            "ph_bank_word_mask_tensor": [batch_size x max_ph_size x max_ph_len] tensor of phrase word mask
        sp_results: a list of planner decoding results, each consists of 5 fields:
            1) "sent_num": number of sentences in this sample
            2) "stype_id": a list of tensors, each is a LongTensor indicating sentence type
            3) "stype_onehot": a list of tensors, each is a onehot decoding indicating sentence type
            4) "dec_outs": a list of tensors, each is of dimension [1, 512], indicating planner's hidden states
            5) "content_selection_preds": a list of tensors, each is a binary vector encoding phrase selection
        enc_memory_bank: [(batch_size * beam_size) x max_src_len x 512] size of encoder hidden states, tiled
        enc_memory_lengths: [(batch_size * beam_size)] size of source sequence, tiled
        batch_size
        opt
        eos_symbols: a list of word id for symbols that end sentences, used to change sentence id
    Returns:
        wd_results:
    """

    # do tiling on states
    # ph_bank_vec: [batch_size x max_ph_size x 300]
    model.wd_dec.map_state(lambda state, dim: utils.tile(state, opt.beam_size, dim=dim))

    scorer = GNMTGlobalScorer(alpha=0., beta=0., length_penalty='none',
                              cov_penalty="none")

    beam = [Beam(size=opt.beam_size, pad=utils.PAD_id, bos=utils.SOS_id,
                 eos_lst=eos_symbols,
                 eos=utils.EOS_id,
                 n_best=1, cuda=True,
                 global_scorer=scorer,
                 min_length=opt.min_target_words,
                 max_sent_num=item["sent_num"],
                 stepwise_penalty=False,
                 block_ngram_repeat=opt.block_ngram_repeat,
                 exclusion_tokens=set()) for item in sp_results]

    sent_ids = torch.zeros(batch_size * opt.beam_size, dtype=torch.long).cuda()

    wd_results = [{"word_preds": [], "sent_ids": [], "sent_type": [], "scores": [], "attention": []}
                  for _ in range(batch_size)]

    # sp_dec_outs: [batch_size x max_sent_num x 512]
    sp_dec_outs_tmp = [torch.cat(s["dec_outs"]) for s in sp_results]
    sp_dec_outs = torch.cat([k.unsqueeze(0) for k in sp_dec_outs_tmp])
    sp_dec_outs = utils.tile(sp_dec_outs, opt.beam_size, dim=0)

    # stype_preds: [batch_size x max_sent_num x 4]
    stype_preds_tmp = [torch.cat(k["stype_onehot"]) for k in sp_results]
    stype_preds = torch.cat([k.unsqueeze(0) for k in stype_preds_tmp])
    stype_preds = utils.tile(stype_preds, opt.beam_size, dim=0)

    def pick_sentence_states(sent_ids):
        # cur_stype_onehot: [(batch_size * beam_size) x 1 x 4]
        sent_id_expanded_stype = sent_ids.clone().cuda().unsqueeze(-1).unsqueeze(-1).expand(-1, 1, 4)
        cur_stype_onehot = torch.gather(stype_preds, 1, sent_id_expanded_stype)

        # cur_dec_outs: [(batch_size * beam_size) x 1 x 512]
        sent_id_expanded_dec_outs = sent_ids.clone().cuda().unsqueeze(-1).unsqueeze(-1).expand(-1, 1, 512)
        cur_dec_outs = torch.gather(sp_dec_outs, 1, sent_id_expanded_dec_outs)
        return cur_stype_onehot, cur_dec_outs

    cur_stype_onehot, cur_dec_outs = pick_sentence_states(sent_ids)
    steps_executed = 0

    for word_t in range(opt.max_tgt_words):
        # print("word_t=%d" % word_t)
        if all(b.done() for b in beam): break
        steps_executed += 1

        # word_input: [(batch_size * beam_size) x 1 x 1]
        # word_input_emb: [(batch_size * beam_size) x 1 x 300]
        word_input = torch.stack([b.get_current_state() for b in beam])
        word_input = word_input.view(-1, 1)
        word_input_emb = model.word_emb(word_input)

        enc_attn, wd_logits = model.wd_dec.forward_onestep(word_inputs_emb=word_input_emb,
                                                           sent_planner_output=cur_dec_outs,
                                                           enc_memory_bank=enc_memory_bank,
                                                           enc_memory_len=enc_memory_lengths,
                                                           stype_one_hot=cur_stype_onehot)

        # wd_probs: [(batch_size * beam_size) x vocab_size]
        # beam_attn:[(batch_size * beam_size) x max_src_len]
        wd_probs = model.wd_dec.softmax(wd_logits).view(batch_size, opt.beam_size, -1)
        beam_attn = enc_attn.view(batch_size, opt.beam_size, -1)

        select_indices_array = []
        sid_changed = []
        for sample_id, b in enumerate(beam):
            cur_sid_changed = b.advance(wd_probs[sample_id,:],
                      beam_attn.data[sample_id, :, :enc_memory_lengths[sample_id]])
            select_indices_array.append(b.get_current_origin() + sample_id * opt.beam_size)
            sid_changed.extend(cur_sid_changed)

        select_indices = torch.cat(select_indices_array)

        model.wd_dec.map_state(lambda state, dim: state.index_select(dim, select_indices))
        if sum(sid_changed) > 0 or word_t == 0:
            new_sent_ids_array = []
            for sample_id, b in enumerate(beam):
                new_sent_ids_array.append(b.get_current_sent_id())
            new_sent_ids = torch.cat(new_sent_ids_array)
            cur_stype_onehot, cur_dec_outs = pick_sentence_states(new_sent_ids)

    for sample_id, b in enumerate(beam):
        scores, ks = b.sort_finished(minimum=1)
        hyps, attn, sent_ids = [], [], []
        for i, (times, k) in enumerate(ks[:1]):
            hyp, att, sent_id = b.get_hyp(times, k)
            hyps.append(hyp)
            attn.append(att)
            sent_ids.append(sent_id)
        wd_results[sample_id]["word_preds"] = [wid.cpu().tolist() for wid in hyps[0]]
        wd_results[sample_id]["scores"] = scores
        wd_results[sample_id]["attention"] = attn
        wd_results[sample_id]["sent_ids"] = [sid.cpu().tolist() for sid in sent_ids[0]]
    print("decoding finished for batch, steps executed=%d" % steps_executed)
    return wd_results



def infer_planner(model, batch, batch_size, opt):
    """
    Run greedy decoding on sentence planning decoder.
    """

    sp_results = [{"content_selection_preds": [],
                   "stype_id": [],
                   "stype_onehot": [],
                   "dec_outs": [],
                   "sent_num": 0} for _ in range(batch_size)]

    sp_inputs = dict()
    if opt.task == "arggen":
        stype_num = 4
    elif opt.task == "wikigen":
        stype_num = 5
    else:
        stype_num = 0

    max_ph_bank_size = batch["phrase_bank"].size(1)
    ph_bank_emb_raw = model.word_emb(batch["phrase_bank"])
    ph_bank_emb = torch.sum(ph_bank_emb_raw, -2)
    ph_bank_vec, _ = model.sp_dec.keyphrase_reader(ph_bank_emb)

    # in decoding time, the first step selection is always <BOS> token
    initial_step_ph_sel = torch.zeros((batch_size, max_ph_bank_size), dtype=torch.long).cuda()
    for i in range(batch_size):
        initial_step_ph_sel[i][0] = 1 # <BOS> token is always the first element in ph_bank

    sp_inputs["ph_bank_sel_ind_inputs_tensor"] = initial_step_ph_sel
    sp_inputs["ph_bank_sel_ind_history"] = initial_step_ph_sel.unsqueeze(1)
    finished_sample_ids = torch.zeros(batch_size, dtype=torch.uint8).cuda()

    for sent_t in range(opt.max_sent_num):
        # ph_bank_vec: [batch_size x max_ph_bank x 300]
        # ph_bank_sel_ind_inputs_tensor: [batch_size x max_ph_bank]
        # ph_bank_sel_ind_history: [batch_size x 1 x max_ph_bank]
        sp_dec_outs, stype_pred_logits, next_sentence_sel_pred_probs = model.sp_dec.forward_onestep(
            kp_ph_bank_vec=ph_bank_vec,
            ph_bank_sel_ind_inputs=sp_inputs["ph_bank_sel_ind_inputs_tensor"],
            ph_bank_sel_ind_history=sp_inputs["ph_bank_sel_ind_history"]
        )

        next_sent_pred_sel = (next_sentence_sel_pred_probs.squeeze() > 0.5).long()
        stype_pred_logits_max, stype_pred_id = torch.max(stype_pred_logits, dim=-1)
        stype_pred_onehot = (stype_pred_logits == stype_pred_logits_max.unsqueeze(-1)).float()

        sp_inputs["ph_bank_sel_ind_inputs_tensor"] = next_sent_pred_sel
        # TODO: make sure the value is correct here!!
        sp_inputs["ph_bank_sel_ind_history"] = sp_inputs["ph_bank_sel_ind_history"] + next_sent_pred_sel.unsqueeze(1)

        masked_next_sent_pred_sel = (next_sent_pred_sel * batch["phrase_bank_mask"]).byte()
        cur_fin = torch.eq(masked_next_sent_pred_sel, batch["phrase_bank_eos_template"]).all(dim=-1)

        for sample_id in range(batch_size):
            cur_sample_fin = finished_sample_ids[sample_id]
            if cur_sample_fin:
                sp_results[sample_id]["dec_outs"].append(torch.zeros((1, 512), dtype=torch.float).cuda())
                if opt.task != "absgen":
                    sp_results[sample_id]["stype_onehot"].append(torch.zeros((1, stype_num), dtype=torch.float).cuda())
                continue

            sp_results[sample_id]["sent_num"] += 1
            sp_results[sample_id]["content_selection_preds"].append(next_sent_pred_sel[sample_id])
            sp_results[sample_id]["dec_outs"].append(sp_dec_outs[sample_id])
            if opt.task != "absgen":
                sp_results[sample_id]["stype_onehot"].append(stype_pred_onehot[sample_id])
                sp_results[sample_id]["stype_id"].append(stype_pred_id[sample_id])

        finished_sample_ids = (cur_fin + finished_sample_ids) > 0
        if finished_sample_ids.all():
            break

    return sp_results


def run_planner_on_true_kp(model, data_dict, batch_size):
    """
    Teacher forced inference on planner decoder
    Args:
        data_dict:
            ph_bank_tensor: [batch_size x max_ph_size x max_ph_len]
            ph_bank_sel_ind_inputs_tensor: [batch_size x max_sent_num x max_ph_size]
            ph_sel_ind_mask: [batch_size x max_sent_num x max_ph_size] 0/1 mask for both phrase bank size and
                sentence number
            stype_one_hot_tensor: [batch_size x max_sent_num x sent_type_n] one hot encoding for sentence types
        batch_size:
    Returns:
        sp_results: a list of batch_size elements
            content_selection_preds: a list of tensors, each stands for content selection for a sentence
            dec_outs: a list of [1 x 512] tensors, each is the hidden states of planner decoder
            sent_num: number of sentences
            stype_id: a list of sentence type
            stype_onehot: a list of sentence onehot encoding
    """
    sp_results = [{"content_selection_preds": [],
                   "stype_id": [],
                   "stype_onehot": [],
                   "dec_outs": [],
                   "sent_num": 0} for _ in range(batch_size)]
    # sp_dec_outs: [batch_size x max_sent_num x 512]
    # stype_pred_logits: [batch_size x max_sent_num x sent_type_n]
    # next_sentence_sel_pred_probs: [batch_size x max_sent_num x max_ph_bank_size]
    sp_dec_outs, stype_pred_logits, next_sentence_sel_pred_probs = model.forward_sp_teacher(data_dict)
    max_sent_num = max(data_dict["sent_num"])
    for sample_id in range(batch_size):
        cur_sent_num = data_dict["sent_num"][sample_id]
        for i in range(max_sent_num):
            if i < cur_sent_num:
                sp_results[sample_id]["dec_outs"].append(sp_dec_outs[sample_id][i].unsqueeze(0))
                sp_results[sample_id]["stype_id"].append(data_dict["stype_array"][sample_id][i])
                sp_results[sample_id]["stype_onehot"].append(data_dict["stype_one_hot_tensor"][sample_id][i].float().unsqueeze(0))
                sp_results[sample_id]["content_selection_preds"].append(data_dict["ph_bank_sel_ind_targets_tensor"][sample_id][i])
            else:
                sp_results[sample_id]["dec_outs"].append(torch.zeros((1, 512), dtype=torch.float).cuda())
                sp_results[sample_id]["stype_onehot"].append(torch.zeros((1, 4), dtype=torch.float).cuda())

        sp_results[sample_id]["sent_num"] = cur_sent_num

    return sp_results

