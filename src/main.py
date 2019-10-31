# Author: Xinyu Hua
"""Program entry for training and test"""

import argparse
import json
import logging
import os
import glob

import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from torch import optim

import utils.misc_utils as misc_utils
import utils.data_utils as data_utils
from utils.misc_utils import Vocabulary
from utils.datasets import WikiDataset, ArgDataset, AbsDataset, DataSampler
from modules.model import WikiGenModel, ArgGenModel, AbsGenModel
from trainer import train_epoch, valid_epoch, infer_epoch


TASK_CONFIG = {
    "wikigen": (WikiGenModel, WikiDataset),
    "arggen": (ArgGenModel, ArgDataset),
    "absgen": (AbsGenModel, AbsDataset),
}

logging.getLogger().setLevel(logging.INFO)

torch.manual_seed(1)

parser = argparse.ArgumentParser(description="main.py")

parser.add_argument("--exp_name", type=str, required=False,
                    help="Name for experiment directory.")
parser.add_argument("--mode", type=str, required=False,
                    choices=["train", "predict"],
                    help="Whether to run training or prediction.")
parser.add_argument("--task", type=str, required=False,
                    choices=["wikigen", "arggen", "absgen"],
                    help="Which task to run, select from `wikigen`, "
                         "`arggen`, `absgen`.")
parser.add_argument("--debug", action="store_true",
                    help="Whether to run in debug mode, in which only a "
                         "fraction of train/val data will be loaded.")

parser.add_argument("--batch_size", type=int, default=32,
                    help="Batch size for training and prediction.")
parser.add_argument("--num_train_epochs", type=int, default=30,
                    help="Maximum number of epochs for training.")
parser.add_argument("--logging_freq", type=int, default=2,
                    help="Logging this many times per epoch.")
parser.add_argument("--hidden_size", type=int, default=512,
                    help="Hidden state dimension for LSTM.")
parser.add_argument("--pointer_generator", action="store_true",
                    help="Whether to use pointer generator for decoder.")
parser.add_argument("--type_conditional_lm", action="store_true",
                    help="Whether directly feed sentence type one-hot encoding to word decoder.")
parser.add_argument("--encode_passage", action="store_true",
                    help="Whether to encode passage for arggen case.")

parser.add_argument("--max_src_words", type=int, default=200,
                    help="Maximum allowed number of words in the source.")
parser.add_argument("--max_passage_words", type=int, default=200,
                    help="Maximum allowed number of words in the appended passage.")
parser.add_argument("--max_sent_num", type=int, default=10,
                    help="Maximum allowed number of sentences.")
parser.add_argument("--max_phrase_in_sentence", type=int, default=30,
                    help="Maximum allowed number of phrases per sentence.")
parser.add_argument("--max_bank_size", type=int, default=30,
                    help="Maximum allowed number of phrases in phrase bank.")
parser.add_argument("--max_tgt_words", type=int, default=100,
                    help="Maximum allowed number of words in the target.")

parser.add_argument("--learning_rate", type=float, default=0.15,
                    help="Initial learning rate for optimizer.")
parser.add_argument("--init_accum", type=float, default=0.1,
                    help="Initial accumulator.")
parser.add_argument("--dropout", type=float, default=0.2,
                    help="Dropout probability between LSTM layers.")
parser.add_argument("--max_grad_norm", type=float, default=2.0,
                    help="Maximum gradient norm, do clip if exceeded.")
parser.add_argument("--loss_gamma", type=float, default=1.0,
                    help="Weight for sentence type loss.")
parser.add_argument("--loss_eta", type=float, default=1.0,
                    help="Weight for keyphrase selection loss.")

parser.add_argument("--beam_size", type=int, default=3,
                    help="Beam size for decoding.")
parser.add_argument("--min_target_words", type=int, default=10,
                    help="Minimum words for output.")
parser.add_argument("--block_ngram_repeat", type=int, default=3,
                    help="Disallow the repetition of n-gram.")
parser.add_argument("--load_model_path", type=str,
                    help="Path to saved model for evaluation.")
parser.add_argument("--test_output_name", type=str, default="demo",
                    help="Path to saved model for evaluation.")
parser.add_argument("--replace_unk", action="store_true",
                    help="Whether to replace <UNK> with the highest attended "
                         "word in the source.")
parser.add_argument("--use_true_kp", action="store_true",
                    help="Whether to use gold-standard KP for planning decoder, "
                         "this corresponds to the oracle setup in the paper.")

opt = parser.parse_args()


def run_training(model, train_dev_data_raw, optimizer, vocab, opt, device):
    ckpt_path = misc_utils.EXP_DIR + opt.exp_name + "/"
    if not os.path.exists(ckpt_path):
        os.mkdir(ckpt_path)
    elif os.listdir(ckpt_path) and not opt.debug:
        raise ValueError("Output directory ({}) already exists and is not empty!".format(ckpt_path))

    with open(ckpt_path + "config.json", 'w') as f:
        json.dump(vars(opt), f)

    fout_log = open(ckpt_path + "training.log", 'w')
    tb_writer = SummaryWriter(os.path.join(ckpt_path + "tensorboard"))

    train_data = TASK_CONFIG[opt.task][1](set_type="train")
    train_data.load_data(raw_data=train_dev_data_raw["train"], opt=opt, vocab=vocab)
    train_data_sampler = DataSampler(dataset=train_data, sequential=False, opt=opt, device=device)

    dev_data = TASK_CONFIG[opt.task][1](set_type="dev")
    dev_data.load_data(raw_data=train_dev_data_raw["train"], opt=opt, vocab=vocab)
    dev_data_sampler = DataSampler(dataset=dev_data, sequential=True, opt=opt, device=device)

    model.eval()
    with torch.no_grad():

        avg_losses, avg_val_ppl, cs_acc, st_acc = valid_epoch(model, dev_data_sampler, opt, device)
        logging.info("--------------- BEFORE TRAINING ---------------")
        logging.info("Validation Loss: {:.3f}\tValidation Perplexity: {:.3f}"
                     .format(avg_losses["total"], avg_val_ppl))
        if opt.task == "absgen":
            logging.info("Keyphrase selection accuracy: {:.2f}"
                         .format(cs_acc * 100))
            fout_log.write(
                "epoch: -1\ttrain_loss: --\tval_loss: {:.3f}\tval_ppl: {:.3f}"
                "\tkp_selection_acc: {:.4f}\n".
                    format(avg_losses["total"], avg_val_ppl, cs_acc))
        else:
            logging.info("Keyphrase selection accuracy: {:.2f}\tSentence type accuracy: {:.2f}"
                         .format(cs_acc * 100, st_acc * 100))
            fout_log.write(
                "epoch: -1\ttrain_loss: --\tval_loss: {:.3f}\tval_ppl: {:.3f}"
                "\tkp_selection_acc: {:.4f}\tstype_acc: {:.4f}\n".
            format(avg_losses["total"], avg_val_ppl, cs_acc, st_acc))
        fout_log.flush()


    for n_epoch in range(1, opt.num_train_epochs + 1):

        logging.info("--------------- STARTING EPOCH %d ---------------" % n_epoch)
        model.train()

        avg_train_losses = train_epoch(model, train_data_sampler, opt, optimizer, device)
        with torch.no_grad():
            model.eval()
            avg_losses, avg_val_ppl, cs_acc, st_acc = valid_epoch(model, dev_data_sampler, opt, device)

        ckpt_name = ckpt_path + "epoch_%d_train_%.4f_val_%.4f_ppl_%.4f.tar" % \
                                (n_epoch, avg_train_losses["total"], avg_losses["total"], avg_val_ppl)
        ckpt_dict = {"embedding": model.word_emb.state_dict(),
                     "encoder": model.encoder.state_dict(),
                     "word_decoder": model.wd_dec.state_dict(),
                     "planning_decoder": model.sp_dec.state_dict(),
                     "optimizer": optimizer.state_dict,
                     "epoch": n_epoch,}

        torch.save(ckpt_dict, ckpt_name)
        if opt.task == "absgen":
            fout_log.write("epoch: {:3d}\ttrain_loss: {:.3f}\ttrain_kp_sel_loss: {:.3f}"
                           "\tval_loss: {:.3f}\tval_ppl: {:.3f}\tkp_sel_acc: {:.4f}\n".
                           format(n_epoch, avg_train_losses["total"],
                                  avg_train_losses["content_selection"],
                                  avg_losses["total"], avg_val_ppl, cs_acc))
        else:
            fout_log.write("epoch: {:3d}\ttrain_loss: {:.3f}\ttrain_sent_type_loss: {:.3f}\ttrain_kp_sel_loss: {:.3f}"
                           "\tval_loss: {:.3f}\tval_ppl: {:.3f}\tkp_sel_acc: {:.4f}\tsent_type_acc: {:.4f}\n".
                           format(n_epoch, avg_train_losses["total"],
                                  avg_train_losses["sentence_type"],
                                  avg_train_losses["content_selection"],
                                  avg_losses["total"], avg_val_ppl, cs_acc, st_acc))



        fout_log.flush()
        for k in avg_train_losses:
            tb_writer.add_scalars("%s_loss" % k, {"train": avg_train_losses[k],
                                                  "valid": avg_losses[k]}, n_epoch)


        tb_writer.add_scalar("valid_perplexity", avg_val_ppl, n_epoch)
        tb_writer.add_scalar("learning_rate", optimizer.param_groups[0]['lr'], n_epoch)
        tb_writer.flush()
    fout_log.close()
    tb_writer.close()


def run_inference(model, test_data_raw, vocab, opt, device):
    """
    load existing checkpoint and run greedy decoding
    """
    opt.load_model_path = misc_utils.EXP_DIR + opt.exp_name + "/" + opt.load_model_path
    ckpt_name_lst = glob.glob(opt.load_model_path)

    assert len(ckpt_name_lst) == 1, "cannot find specified checkpoint in %s" % opt.load_model_path

    ckpt_fpath = ckpt_name_lst[0]
    misc_utils.load_prev_checkpoint(model, ckpt_fpath, None)

    test_data = TASK_CONFIG[opt.task][1](set_type="test")
    test_data.load_test_data(raw_data=test_data_raw, opt=opt, vocab=vocab)

    test_data_sampler = DataSampler(dataset=test_data,
                                    sequential=True,
                                    opt=opt,
                                    device=device)

    # store examples
    fout_log = open("infer_logs/%s_output.jsonlist" \
                        % (opt.test_output_name), "w")

    with torch.no_grad():
        model.eval()
        infer_epoch(model, test_data_sampler, vocab, opt, fout_log)
    fout_log.close()
    return



def main():
    logging.info("Loading vocabulary and embedding...")
    vocab = Vocabulary(opt.task)
    glove_emb = misc_utils.load_glove_emb(vocab)
    word_emb = nn.Embedding.from_pretrained(torch.tensor(glove_emb, dtype=torch.float))

    logging.info("Building generation model...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    model = TASK_CONFIG[opt.task][0](word_emb=word_emb, vocab_size=len(vocab), opt=opt).to(device)

    if opt.mode == "train":

        logging.info("Start running in training mode...")
        optimizer = optim.Adagrad(filter(lambda p: p.requires_grad, model.parameters()),
                                  lr=opt.learning_rate,
                                  initial_accumulator_value=opt.init_accum)
        train_dev_data_raw = data_utils.load_train_data(demo=opt.debug, task=opt.task)
        run_training(model, train_dev_data_raw, optimizer, vocab, opt, device)

    elif opt.mode == "predict":
        logging.info("Start beam search decoding...")
        test_data = data_utils.load_test_data(demo=opt.debug, task=opt.task)
        run_inference(model, test_data, vocab, opt, device)


if __name__ == "__main__":
    main()
