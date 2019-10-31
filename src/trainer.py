import os
import time
import logging
import json
import numpy as np
import torch
import tqdm
import torch.nn as nn

from modules.decoding import infer_batch
from sklearn.metrics import f1_score
logging.getLogger().setLevel(logging.INFO)


def train_epoch(model, train_data_sampler, opt, optimizer, device):
    start_time = time.time()
    total_losses = {"total": [], "content_selection": [], "word_xent": []}
    if opt.task in ["wikigen", "arggen"]:
        total_losses["sentence_type"] = []

    softmax = nn.Softmax(dim=-1)
    for batch in train_data_sampler:
        model.zero_grad()
        sp_type_pred_logits, sp_next_sentence_sel_pred, wd_readouts, enc_attn, _, _ = model(batch, device)
        wd_loss, ppl = model.compute_word_loss_probs(wd_readouts, batch["tgt_word_ids_output"])
        if not opt.task == "absgen":
            stype_pred_probs = softmax(sp_type_pred_logits)
            stype_pred = torch.argmax(stype_pred_probs, dim=-1)
            st_loss = model.compute_stype_loss(sp_type_pred_logits, batch["tgt_sent_type"])
        cs_loss = model.compute_content_selection_loss(sp_next_sentence_sel_pred,
                                                       batch["phrase_bank_selection_index_target"],
                                                       batch["phrase_bank_selection_index_mask"])
        cs_results = (sp_next_sentence_sel_pred > 0.5).cpu().numpy()

        if not opt.task == "absgen":
            model_loss = wd_loss + opt.loss_gamma * st_loss + opt.loss_eta * cs_loss
            total_losses["sentence_type"].append(st_loss.cpu().item())
        else:
            model_loss = wd_loss + opt.loss_eta * cs_loss

        total_losses["total"].append(model_loss.cpu().item())
        total_losses["content_selection"].append(cs_loss.cpu().item())
        total_losses["word_xent"].append(wd_loss.cpu().item())

        model_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), opt.max_grad_norm)
        optimizer.step()

    logging.info("Training time: {:.2f} secs".format(time.time() - start_time))
    mean_losses = {k: np.mean(cur_loss) for k, cur_loss in total_losses.items()}
    return mean_losses


def valid_epoch(model, dev_data_sampler, opt, device):
    start_time = time.time()
    total_losses = {"total": [], "content_selection": [], "word_xent": []}
    if opt.task in ["wikigen", "arggen"]:
        total_losses["sentence_type"] = []
    total_ppl = []
    softmax = nn.Softmax(dim=-1)
    stype_predictions = []
    stype_truth = []
    cs_pred = []
    cs_truth = []

    for batch in tqdm.tqdm(dev_data_sampler):

        model.zero_grad()
        sp_type_pred_logits, sp_next_sentence_sel_pred, wd_readouts, enc_attn, _, _ = model(batch, device)

        wd_loss, ppl = model.compute_word_loss_probs(wd_readouts, batch["tgt_word_ids_output"])
        if not opt.task == "absgen":
            st_loss = model.compute_stype_loss(sp_type_pred_logits, batch["tgt_sent_type"])
        cs_loss = model.compute_content_selection_loss(sp_next_sentence_sel_pred,
                                                       batch["phrase_bank_selection_index_target"],
                                                       batch["phrase_bank_selection_index_mask"])
        if opt.task == "absgen":
            model_loss = wd_loss + opt.loss_eta * cs_loss
        else:
            model_loss = wd_loss + opt.loss_gamma * st_loss + opt.loss_eta * cs_loss
            total_losses["sentence_type"].append(st_loss.cpu().item())

        total_losses["total"].append(model_loss.cpu().item())
        total_losses["content_selection"].append(cs_loss.cpu().item())
        total_losses["word_xent"].append(wd_loss.cpu().item())

        total_ppl.append(ppl.cpu().item())

        ph_pred_hard_sel = sp_next_sentence_sel_pred > 0.5
        ph_pred_hard_sel_np = ph_pred_hard_sel.cpu().data.numpy()
        ph_sel_truth = batch["phrase_bank_selection_index_target_array"]

        if not opt.task == "absgen":
            stype_pred_probs = softmax(sp_type_pred_logits)
            stype_pred = torch.argmax(stype_pred_probs, dim=-1)
            stype_mask = batch["tgt_sent_type_mask"].cpu().data.numpy()

        for sample_id in range(len(batch["src_lens"])):
            ph_pred_masked = []
            ph_truth_masked = []
            ph_bank_mask = batch["phrase_bank_selection_index_mask_array"][sample_id]
            for sent_id in range(len(ph_bank_mask)):
                cur_ln_mask = ph_bank_mask[sent_id]
                if sum(cur_ln_mask) == 0.0: break
                length = int(sum(cur_ln_mask))
                ph_pred_masked.append(ph_pred_hard_sel_np[sample_id][sent_id][:length].tolist())
                ph_truth_masked.append(ph_sel_truth[sample_id][sent_id][:length].tolist())
                cs_pred.extend(ph_pred_masked[-1])
                cs_truth.extend(ph_truth_masked[-1])

            if not opt.task == "absgen":
                for sent_id in range(len(ph_pred_hard_sel_np[sample_id])):
                    if stype_mask[sample_id][sent_id] == 0.0:
                        break

                    stype_truth.append(batch["tgt_sent_type_array"][sample_id][sent_id])
                    stype_predictions.append(int(stype_pred[sample_id][sent_id]))

    logging.info("Validation time: {:.2f} sec".format(time.time() - start_time))
    mean_losses = {k: np.mean(cur_loss) for k, cur_loss in total_losses.items()}
    mean_ppl = np.mean(total_ppl)
    cs_f1 = f1_score(cs_truth, cs_pred, average="binary")
    cs_corr_cnt = sum([x == y for x, y in zip(cs_pred, cs_truth)])
    cs_acc = cs_corr_cnt / len(cs_pred)
    logging.info("Content selection accuracy: %.2f" % cs_acc)
    if opt.task == "absgen":
        stype_f1 = None
    else:
        stype_f1 = f1_score(stype_truth, stype_predictions, average="macro")
    logging.info("Validation perplexity: %.2f\tContent selection F1: %.2f" % (mean_ppl, cs_f1))
    if opt.task == "wikigen":
        f1_0 = f1_score(stype_truth, stype_predictions, average="macro", labels=[0])
        f1_1 = f1_score(stype_truth, stype_predictions, average="macro", labels=[1])
        f1_2 = f1_score(stype_truth, stype_predictions, average="macro", labels=[2])
        f1_3 = f1_score(stype_truth, stype_predictions, average="macro", labels=[3])
        f1_4 = f1_score(stype_truth, stype_predictions, average="macro", labels=[4])
        logging.info("     one-vs-rest F1 for sentence type:")
        logging.info("         sentence type SOS f1: %.3f" % f1_0)
        logging.info("         sentence type (0, 10] f1: %.3f" % f1_1)
        logging.info("         sentence type (10, 20] f1: %.3f" % f1_2)
        logging.info("         sentence type (20, 30] f1: %.3f" % f1_3)
        logging.info("         sentence type (30, inf) f1: %.3f" % f1_4)
    return mean_losses, mean_ppl, cs_f1, stype_f1


def infer_epoch(model, data_sampler, vocab, opt, fout):
    """
    Run decoding algorithm on a given model.
    """
    start_time = time.time()
    eos_symbols = [".", "!", "?", "???", "!!!", "..."]
    eos_wids = [vocab.word2id(eos) for eos in eos_symbols if eos in vocab._word2id]

    for batch in tqdm.tqdm(data_sampler):

        sp_results, wd_results = infer_batch(model, batch, opt, eos_wids)

        for ix in range(len(wd_results)):
            cur_para_preds = sp_results[ix]
            cur_stypes = [k[0].cpu().tolist() for k in cur_para_preds['stype_id']]
            cur_pred_text = [vocab.id2word(wid) for wid in wd_results[ix]["word_preds"]]
            cur_src_str = batch["src_str"][ix]
            if opt.replace_unk and (wd_results[ix]["attention"] is not None) and (cur_src_str is not None):


                unk_free = []
                for wid, w in enumerate(cur_pred_text):
                    if w in ["SEP", "EOS", "SOS", "PAD"]: continue
                    if w == "UNK":
                        _, max_index = wd_results[ix]["attention"][0][wid].max(0)
                        replaced = cur_src_str[max_index.item()]
                        unk_free.append(replaced)
                    else:
                        unk_free.append(w)
                pred_cleaned = unk_free
            else:
                pred_cleaned = [w for w in cur_pred_text if not w in ["SEP", "PAD", "SOS", "EOS"]]

            towrite_obj = {"pred": pred_cleaned, "stype": cur_stypes, "tid": batch["tid"][ix]}
            fout.write(json.dumps(towrite_obj) + "\n")
    logging.info("Decoding finished in %.2f seconds" % (time.time() - start_time))
    return