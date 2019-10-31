import random
import torch
import torch.nn as nn
import numpy as np

import utils.misc_utils as utils


class Dataset(object):
    """A Dataset base class.

       Usage:
           dataset = Dataset(set_type="train")
           raw_data = load_train_data()
           dataset.load_data(raw_data, opt, vocab)
           print(len(dataset)) # print number of data samples
           dataset[0] # integer index to access data samples
       """
    def __init__(self, set_type):
        self.set_type = set_type

    def load_source(self, *args, **kwargs):
        raise NotImplementedError

    def load_target(self, *args, **kwargs):
        raise NotImplementedError

    def load_data(self, *args, **kwargs):
        raise NotImplementedError

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        raise NotImplementedError


class WikiDataset(Dataset):
    """
    A Wikipedia dataset class, inherited from Dataset
    """
    def __init__(self, set_type):
        super(WikiDataset, self).__init__(set_type)

    @classmethod
    def sentence_type_func(cls, sentence_wids):
        """
        Method to assign type labels for sentences. In Wikipedia case, assign label based on length.
        """
        if len(sentence_wids) <= 10:
            label = 1
        elif len(sentence_wids) <= 20:
            label = 2
        elif len(sentence_wids) <= 30:
            label = 3
        else:
            label = 4
        return label

    def load_source(self, raw_data_src, opt, vocab):
        self.src_inputs, self.src_lens, self.src_strs = utils.encode_title_to_id_lists(title_list=raw_data_src,
                                                                        vocab=vocab,
                                                                        max_title_words=opt.max_src_words)
        self.size = len(self.src_lens)


    def load_target(self, raw_data_tgt, opt, vocab):
        self.tgt_word_ids_input, self.tgt_word_ids_output, self.tgt_sent_ids, self.tgt_sent_type \
            = utils.encode_sentence_and_type_to_list(target_list=raw_data_tgt, vocab=vocab,
                                                     max_sent_num=opt.max_sent_num,
                                                     sentence_type_func=WikiDataset.sentence_type_func)


    def load_phrase_selection(self, raw_phrase_selection, raw_ph_bank, opt, vocab):
        self.phrase_bank, _, self.phrase_selection_inputs, self.phrase_bank_selection_index \
            = utils.encode_ph_sel_to_word_ids(ph_sel_list=raw_phrase_selection, ph_bank_list=raw_ph_bank,
                                                   vocab=vocab, max_sent_num=opt.max_sent_num,
                                                   max_ph_bank=opt.max_bank_size)



    def load_data(self, raw_data, opt, vocab):
        self.load_source(raw_data_src=raw_data["title"], opt=opt, vocab=vocab)
        self.load_target(raw_data_tgt=raw_data["tgt"], opt=opt, vocab=vocab)
        self.load_phrase_selection(raw_phrase_selection=raw_data["kp_sel"], raw_ph_bank=raw_data["ph_bank"], opt=opt, vocab=vocab)
        self.style = raw_data["style"]
        self.title = [' '.join(title) + "_" + str(style) for title, style in zip(raw_data["title"], raw_data["style"])]

        # check size information
        assert len(self.src_inputs) == self.size
        assert len(self.src_lens) == self.size
        assert len(self.tgt_word_ids_input) == self.size
        assert len(self.tgt_sent_ids) == self.size
        assert len(self.tgt_sent_type) == self.size
        assert len(self.phrase_selection_inputs) == self.size
        assert len(self.phrase_bank) == self.size
        assert len(self.phrase_bank_selection_index) == self.size
        assert len(self.style) == self.size


    def __getitem__(self, index):

        if self.set_type in ["train", "dev"]:
            return {"src_inputs": self.src_inputs[index],
                    "src_lens": self.src_lens[index],
                    "tgt_word_ids_input": self.tgt_word_ids_input[index],
                    "tgt_word_ids_output": self.tgt_word_ids_output[index],
                    "tgt_sent_ids": self.tgt_sent_ids[index],
                    "tgt_sent_type": self.tgt_sent_type[index],
                    "src_strs": self.src_strs[index],
                    "phrase_bank": self.phrase_bank[index],
                    "phrase_selection_inputs": self.phrase_selection_inputs[index],
                    "phrase_bank_selection_index": self.phrase_bank_selection_index[index],
                    "title": self.title[index],
                    "style": self.style[index]}
        else:
            return {"src_inputs": self.src_inputs[index],
                    "src_lens": self.src_lens[index],
                    "src_strs": self.src_strs[index],
                    "phrase_bank": self.phrase_bank[index],
                    "title": self.title[index],
                    "style": self.style[index]}


class ArgDataset(Dataset):
    """Dataset class for argument generation task.

    Usage:
        dataset = Dataset(set_type="train")
        raw_data = load_train_data()
        dataset.load_data(raw_data, opt, vocab)
        print(len(dataset)) # print number of data samples
        dataset[0] # integer index to access data samples
    """
    def __init__(self, set_type):
        super(ArgDataset, self).__init__(set_type)

    @classmethod
    def sentence_type_func(cls, sent_type):
        """
        Method to assign type labels for sentences. In Wikipedia case, assign label based on length.
        """
        if sent_type == "functional":
            label = 1
        elif sent_type == "claim":
            label  = 2
        else:
            label = 3
        return label

    def load_source(self, raw_data_src, opt, vocab):
        """Load OP and passages."""
        self.src_inputs, self.src_lens, self.src_strs = utils.encode_text_to_id_lists(op_list=raw_data_src["op"],
                                                                       passage_list=raw_data_src["passages"],
                                                                       vocab=vocab,
                                                                       max_op_words=opt.max_src_words,
                                                                       max_passage_words=opt.max_passage_words,
                                                                       encode_passage=opt.encode_passage)

        self.size = len(self.src_lens)


    def load_phrase_bank(self, raw_data_ph, opt, vocab):
        self.phrase_bank_wids, self.phrase_bank_words = utils.encode_phrase_bank_to_id_lists(
                                                                   phrase_bank_list=raw_data_ph["passage_kp"],
                                                                   vocab=vocab,
                                                                   max_bank_size=opt.max_bank_size)

    def load_target(self, raw_data_tgt, opt, vocab):
        tgt_info = utils.encode_sentence_and_type_to_list(target_list=raw_data_tgt, vocab=vocab,
                                                          max_sent_num=opt.max_sent_num,
                                                          sentence_type_func=self.sentence_type_func)
        self.tgt_word_ids_input = tgt_info[0]
        self.tgt_word_ids_output = tgt_info[1]
        self.tgt_sent_ids = tgt_info[2]
        self.tgt_sent_type = tgt_info[3]


    def load_phrase_selection(self, raw_data_tgt, opt, vocab):
        self.phrase_selection_inputs, self.phrase_bank, self.phrase_bank_selection_index \
            = utils.encode_phrase_sel_to_word_ids(target_list=raw_data_tgt, vocab=vocab,
                                                  max_phrase_in_sentence=opt.max_phrase_in_sentence,
                                                  max_rr_sent_num=opt.max_sent_num)

    def load_data(self, raw_data, opt, vocab):
        self.load_source(raw_data_src=raw_data["src"], opt=opt, vocab=vocab)
        self.load_target(raw_data_tgt=raw_data["tgt"], opt=opt, vocab=vocab)
        self.load_phrase_selection(raw_data_tgt=raw_data["tgt"], opt=opt, vocab=vocab)
        self.tids = raw_data["id"]
        # check size information
        assert len(self.src_inputs) == self.size
        assert len(self.src_lens) == self.size
        assert len(self.tgt_word_ids_input) == self.size
        assert len(self.tgt_sent_ids) == self.size
        assert len(self.tgt_sent_type) == self.size
        assert len(self.phrase_selection_inputs) == self.size
        assert len(self.phrase_bank) == self.size
        assert len(self.phrase_bank_selection_index) == self.size

    def load_test_data(self, raw_data, opt, vocab):
        self.load_source(raw_data, opt=opt, vocab=vocab)
        self.load_phrase_bank(raw_data, opt=opt, vocab=vocab)
        self.tids = raw_data["id"]



    def __getitem__(self, index):
        if self.set_type in ["train", "dev"]:
            return {"src_inputs": self.src_inputs[index],
                    "src_lens": self.src_lens[index],
                    "tgt_word_ids_input": self.tgt_word_ids_input[index],
                    "tgt_word_ids_output": self.tgt_word_ids_output[index],
                    "tgt_sent_ids": self.tgt_sent_ids[index],
                    "tgt_sent_type": self.tgt_sent_type[index],
                    "phrase_selection_inputs": self.phrase_selection_inputs[index],
                    "phrase_bank" : self.phrase_bank[index],
                    "tid": self.tids,
                    "phrase_bank_selection_index": self.phrase_bank_selection_index[index]}
        else:
            return {"src_inputs": self.src_inputs[index],
                    "src_lens": self.src_lens[index],
                    "phrase_bank_words": self.phrase_bank_words[index],
                    "tid": self.tids[index],
                    "src_strs": self.src_strs[index],
                    "phrase_bank": self.phrase_bank_wids[index]}


class AbsDataset(Dataset):
    """
    dataset class for abstract generation data.
    """
    def __init__(self, set_type):
        super(AbsDataset, self).__init__(set_type)

    def load_source(self, raw_data_src, opt, vocab):
        self.src_inputs, self.src_lens, self.src_strs = utils.encode_title_to_id_lists(title_list=raw_data_src,
                                                                        vocab=vocab,
                                                                        max_title_words=opt.max_src_words)
        self.size = len(self.src_lens)

    def load_target(self, raw_data_tgt, opt, vocab):
        self.tgt_word_ids_input, self.tgt_word_ids_output, self.tgt_sent_ids, self.tgt_sent_type \
            = utils.encode_sentence_and_type_to_list(target_list=raw_data_tgt, vocab=vocab,
                                                     max_sent_num=opt.max_sent_num)

    def load_phrase_selection(self, raw_phrase_selection, raw_ph_bank, opt, vocab):
        self.phrase_bank, _, self.phrase_selection_inputs, self.phrase_bank_selection_index \
            = utils.encode_ph_sel_to_word_ids(ph_sel_list=raw_phrase_selection, ph_bank_list=raw_ph_bank,
                                                   vocab=vocab, max_sent_num=opt.max_sent_num,
                                                   max_ph_bank=opt.max_bank_size)

    def load_data(self, raw_data, opt, vocab):
        self.load_source(raw_data_src=raw_data["title"], opt=opt, vocab=vocab)
        self.load_target(raw_data_tgt=raw_data["tgt"], opt=opt, vocab=vocab)
        self.load_phrase_selection(raw_phrase_selection=raw_data["kp_sel"], raw_ph_bank=raw_data["ph_bank"], opt=opt,
                                   vocab=vocab)

    def __getitem__(self, index):
        if self.set_type in ["train", "dev"]:
            return {"src_inputs": self.src_inputs[index],
                    "src_lens": self.src_lens[index],
                    "src_strs": self.src_strs[index],
                    "tgt_word_ids_input": self.tgt_word_ids_input[index],
                    "tgt_word_ids_output": self.tgt_word_ids_output[index],
                    "tgt_sent_ids": self.tgt_sent_ids[index],
                    "tgt_sent_type": self.tgt_sent_type[index],
                    "phrase_bank": self.phrase_bank[index],
                    "phrase_selection_inputs": self.phrase_selection_inputs[index],
                    "phrase_bank_selection_index": self.phrase_bank_selection_index[index],
                    }
        else:
            return {"src_inputs": self.src_inputs[index],
                    "src_lens": self.src_lens[index],
                    "src_strs": self.src_strs[index],
                    "phrase_bank": self.phrase_bank[index],}

class DataSampler(object):
    """A dataset sampler that go over the entire Dataset object.

    Usage: (has to use with Dataset object)
        dataset = Dataset(set_type="train")
        dataset.load_train_data()
        data_sampler = DataSampler(dataset=dataset, batch_size=10, sequential)
        for tensor_item in data_sampler:
            src_inputs = tensor_item["src_inputs"]
            src_lens = tensor_item["src_lens"]
    """

    def __init__(self, dataset, sequential=True, opt=None, device=None):
        """
        Args:
             dataset: Dataset object which implements __len__ and __getitem__
             batch_size: int
             sequential: bool. Whether to iterate over the dataset in its original order.
        """
        self.dataset = dataset
        self.batch_size = opt.batch_size
        self.start_idx = 0
        self.sequential = sequential
        self.device = device
        self.task = opt.task
        if sequential:
            self.indices = range(len(self.dataset))
        else:
            self.indices = random.sample(range(len(self.dataset)), len(self.dataset))

        if opt.task == "arggen":
            self.stype_n = 4
        elif opt.task == "wikigen":
            self.stype_n = 5
        else:
            self.stype_n = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.start_idx == len(self.indices):
            self.start_idx = 0
            if not self.sequential:
                self.indices = random.sample(range(len(self.dataset)), len(self.dataset))
            raise StopIteration

        end_idx = min(len(self.indices), self.start_idx + self.batch_size)
        batch_data = {k: [] for k in self.dataset[0]}

        # need to sort idx by source length because of RNN packing constraint
        cur_indices = [(idx, self.dataset[idx]["src_lens"]) for idx in self.indices[self.start_idx: end_idx]]
        sorted_indices = sorted(cur_indices, key=lambda x: x[1], reverse=True)

        for idx, _ in sorted_indices:
            for k, v in self.dataset[idx].items():
                batch_data[k].append(v)
        self.start_idx = end_idx

        # padding and convert to tensors
        tensor_items = dict()

        for k, v in batch_data.items():

            # currently the dimensionality check is by hand
            if k in ["src_inputs", "tgt_word_ids_input", "tgt_word_ids_output", "tgt_sent_ids"]:
                padded, mask = utils.pad_2d_sequence(v)

            elif k in ["phrase_bank"]:
                padded, padded_lens, mask_2d, mask_3d = utils.pad_3d_sequence(v)
                phrase_bank_eos_template = utils.generate_eos_template(v)
                tensor_items["phrase_bank_eos_template"] = torch.tensor(phrase_bank_eos_template, dtype=torch.uint8).to(self.device)
                tensor_items["phrase_bank_word_mask"] = torch.tensor(mask_3d, dtype=torch.long).to(self.device)
                tensor_items["phrase_bank_len"] = torch.tensor(padded_lens, dtype=torch.long).to(self.device)
                tensor_items["phrase_bank_mask"] = torch.tensor(mask_2d, dtype=torch.long).to(self.device)
                mask = None

            elif k in ["phrase_selection_inputs"]:
                padded = utils.pad_4d_sequence(v)
                mask = None

            elif k == "phrase_bank_selection_index":
                padded, padded_target, mask = utils.pad_3d_sequence_with_target(v)
                tensor_items[k + "_target_array"] = padded_target
                tensor_items[k + "_target"] = torch.tensor(padded_target, dtype=torch.float).to(self.device)

            elif k == "tgt_sent_type" and not self.task == "absgen":
                padded, mask = utils.pad_2d_sequence(v, pad_token=-1)
                onehot_array = utils.create_onehot_for_categorical(v, k=self.stype_n)
                tensor_items["tgt_sent_type_onehot"] = torch.tensor(onehot_array, dtype=torch.long).to(self.device)
                tensor_items["tgt_sent_type_onehot_array"] = onehot_array

            elif k in ["tid", "src_strs", "title_words"]:
                tensor_items[k] = v
                mask = None
            else:
                padded = v
                mask = None

            if k in ["title", "phrase_bank_words"]: # non-tensor items
                tensor_items[k] = padded
            elif k not in ["tid", "src_strs", "title_words"]:
                if self.task == "absgen" and k in ["tgt_sent_type"]: continue
                tensor_items[k] = torch.tensor(padded, dtype=torch.long).to(self.device)
                tensor_items[k + "_array"] = padded

            if mask is not None:
                tensor_items[k + "_mask"] = torch.tensor(mask, dtype=torch.float32).to(self.device)
                tensor_items[k + "_mask_array"] = mask

        return tensor_items
