# Author: Xinyu Hua
"""utility functions for model training and testing"""

import torch
import logging
import numpy as np

PAD_id = 0
SOS_id = 1
SEP_id = 2
EOS_id = 3
UNK_id = 4
PH_PAD_id = 0
PH_SOS_id = 1
PH_EOS_id = 2

setup_configs = {
    "oracle": {"src": ["op", "rr_psg_kp", "rr_psg", "tid"], "tgt": ["rr"]},
    "system": {"src": ["op", "op_psg_kp", "op_psg"], "tgt": ["rr"]},
}

# DATA_DIR = os.environ["DATA_DIR_PREFIX"]
# EXP_DIR = os.environ["EXP_DIR_PREFIX"]
# WEMB_DIR = os.environ["WEMB_DIR"]
DATA_DIR = "/data/model/xinyu/emnlp2019_code_release/data/"
EXP_DIR = "/data/model/xinyu/emnlp2019_code_release/exp/"
WEMB_DIR = "/data/model/embeddings/glove.6B.300d.txt"

PRETRAINED_ENCODER_PATH = DATA_DIR + "pretrained_encoder_weights.tar"
PRETRAINED_DECODER_PATH = DATA_DIR + "pretrained_decoder_weights.tar"

class Vocabulary(object):
    """Vocabulary class"""

    def __init__(self, task=None):
        """Constructs the vocabulary by loading from disk."""
        self._word2id = dict()
        self._id2word = list()

        for ln in open(DATA_DIR + task + "/vocab.txt"):
            _, word, freq = ln.strip().split("\t")
            self._word2id[word] = len(self._word2id)
            self._id2word.append(word)

            if len(self._id2word) == 50000:
                break

        self.bos_token = self._word2id["SOS"]
        self.unk_token = self._word2id["UNK"]
        self.eos_token = self._word2id["EOS"]
        if "SEP" in self._word2id:
            self.sep_token = self._word2id["SEP"]

        assert self._word2id["SOS"] == SOS_id
        assert self._word2id["UNK"] == UNK_id, \
            "self._word2id['UNK']=%d\tUNK_id=%d" % (self._word2id["UNK"], UNK_id)
        assert self._word2id["EOS"] == EOS_id
        assert self._word2id["PAD"] == PAD_id
        assert self._word2id["SEP"] == SEP_id

    def __len__(self):
        return len(self._id2word)

    def word2id(self, word):
        if word in self._word2id:
            return self._word2id[word]
        else:
            return self.unk_token

    def id2word(self, id):
        assert id >= 0 and id < len(self._id2word)
        return self._id2word[id]


def encode_title_to_id_lists(title_list, vocab, max_title_words):
    """
    Convert title words into word ids for Wikipedia and Abstract dataset.
    Args:
        title_list: a list of titles, each title is a list of lowercased string.
        vocab: vocabulary object to convert word into word ids
        max_title_words: int. If the length of title is greater than this limit, it will be truncated.
    Returns:
        title_inputs: a list of word ids, unpadded.
        title_lens: a list of int, indicating the length for each title.
    """
    title_inputs = []
    title_lens = []
    title_words = []

    for title in title_list:
        wids = []
        words = []
        for w in title:
            wid = vocab.word2id(w)
            if wid == vocab.unk_token:continue
            wids.append(wid)
            words.append(w)

        if len(wids) > max_title_words:
            wids = wids[:max_title_words]
            words = words[:max_title_words]
        title_inputs.append(wids)
        title_lens.append(len(wids))
        title_words.append(words)
    return title_inputs, title_lens, title_words

def encode_ph_sel_to_word_ids(ph_sel_list, ph_bank_list, vocab, max_sent_num, max_ph_bank):
    """
    Encode phrase bank selection into word ids and selection indicators.
    Args:
        ph_sel_list: a list of paragraph phrase selection, where each instance is a list of sentence level selection
            e.g. [[["an", "activist"], ["south", "korea"]], # sentence 1
                  [["grandparatens"], ["south", "korea"], ["family"]], # sentence 2
                  ...]
        ph_bank_list: a list of phrase banks
        vocab: Vocabulary object to convert word into ids
        max_phrase_in_sentence: int. maximum allowed number of phrases per sentence
        max_sent_num: int. maximum number of sentences
    Returns:
        ph_bank_ids: unpadded list of phrase banks
        ph_sel_inputs
        ph_sel_ind_array:
    """
    ph_bank_str = []
    ph_bank_ids = []
    ph_sel_inputs = []
    ph_sel_ind_array = []

    for sample_id, sample in enumerate(ph_sel_list):
        cur_sample_sel = [[[vocab.bos_token]]] # first sentence is always [SOS_token]
        cur_sample_ph_bank_str = [("BOS",)]
        cur_sample_ph_bank = [[vocab.bos_token]]

        for sent_id, sent in enumerate(sample):
            if sent_id == max_sent_num: break
            cur_sent_sel = []
            for ph in sent: # ph is like ["united", "states"]
                ph_ids = [vocab.word2id(w) for w in ph]
                ph_ids_tuple = tuple(ph_ids)
                if not ph_ids_tuple in cur_sample_ph_bank:
                    cur_sample_ph_bank.append(ph_ids_tuple)
                    cur_sample_ph_bank_str.append(tuple(ph))
                if len(cur_sample_ph_bank) == max_ph_bank:continue
                cur_sent_sel.append(ph_ids_tuple)
            cur_sample_sel.append(cur_sent_sel)

        if ph_bank_list is not None:
            cur_ph_bank = ph_bank_list[sample_id]
            for ph in cur_ph_bank:
                cur_ph_ids = tuple([vocab.word2id(w) for w in ph])
                if not cur_ph_ids in cur_sample_ph_bank:
                    cur_sample_ph_bank.append(cur_ph_ids)
                    cur_sample_ph_bank_str.append(tuple(ph))

        cur_sample_ph_bank.append([vocab.eos_token])
        cur_sample_sel.append([[vocab.eos_token]])
        cur_sample_ph_bank_str.append(("EOS",))

        ph_bank_ids.append(cur_sample_ph_bank)
        ph_bank_str.append(cur_sample_ph_bank_str)
        ph_sel_inputs.append(cur_sample_sel)

        cur_ph_bank_sel_ind = []
        for sent in cur_sample_sel:
            cur_sel = [1 if ph_tuple in sent else 0 for ph_tuple in cur_sample_ph_bank]
            cur_ph_bank_sel_ind.append(cur_sel)
        ph_sel_ind_array.append(cur_ph_bank_sel_ind)

    return ph_bank_ids, ph_bank_str, ph_sel_inputs, ph_sel_ind_array


def pad_2d_sequence(raw_input, pad_token=0):
    """Pad 2d sequence with `pad_token` and return mask"""
    max_len = max([len(x) for x in raw_input])
    padded = pad_token * np.ones([len(raw_input), max_len], dtype=np.long)
    mask = np.zeros([len(raw_input), max_len], dtype=np.long)

    for sample_id, sample in enumerate(raw_input):
        padded[sample_id][:len(sample)] = sample
        mask[sample_id][:len(sample)] = 1

    return padded, mask

def pad_3d_sequence_with_target(raw_input):
    """Pad 3d sequence with target and source sequence to facilitate forward/backward pass.

    Args:
        raw_input: a 3D list, e.g. [batch_size x phrase_bank-size x phrase_word_num]
    Return:
        padded_source: numpy.array. The padded 3D array without EOS at the end
        padded_target: numpy.array. The padded 3D array without BOS at the begining
    """
    max_2nd_dim = max([len(x) for x in raw_input])
    max_3rd_dim = 0
    for sample in raw_input:
        sample_max = max([len(ph) for ph in sample])
        max_3rd_dim = max(max_3rd_dim, sample_max)

    padded_source = np.zeros([len(raw_input), max_2nd_dim - 1, max_3rd_dim], dtype=np.int)
    padded_target = np.zeros([len(raw_input), max_2nd_dim - 1, max_3rd_dim], dtype=np.int)
    padded_mask = np.zeros([len(raw_input), max_2nd_dim - 1, max_3rd_dim], dtype=np.int)

    for sample_id, sample in enumerate(raw_input):
        for sent_id in range(len(sample)):
            cur_sent_sel = sample[sent_id]
            if sent_id != 0:
                padded_target[sample_id][sent_id - 1][:len(cur_sent_sel)] = cur_sent_sel
                padded_mask[sample_id][sent_id - 1][:len(sample[-1])] = 1
            if sent_id != len(sample) - 1:
                padded_source[sample_id][sent_id][:len(sample[-1])] = cur_sent_sel

    return padded_source, padded_target, padded_mask

def pad_3d_sequence(raw_input, pad_token=0):
    """Pad 3d sequence with `pad_token` and return mask

    Args:
        raw_input: a 3D list, e.g. [batch_size x phrase_bank_size x phrase_word_num]
        pad_token: int. the word id for padding token
    Return:
        padded_3d_array: numpy.array The padded 3D array from raw_input.
        padded_lens: numpy.array The 2D array to indicate the number of second dimension (phrase_bank_size).
        padded_2d_mask: numpy.array. 0/1 mask on the first 2 dimensions
        padded_3d_mask: numpy.array. 0/1 mask on all dimensions
    """

    max_2nd_dim = max([len(x) for x in raw_input])
    max_3rd_dim = 0
    for sample in raw_input:
        sample_max = max([len(ph) for ph in sample])
        max_3rd_dim = max(max_3rd_dim, sample_max)

    padded_3d_array = pad_token * np.ones([len(raw_input), max_2nd_dim, max_3rd_dim])
    padded_lens = np.zeros(len(raw_input), dtype=int)
    padded_3d_mask = np.zeros([len(raw_input), max_2nd_dim, max_3rd_dim])
    padded_2d_mask = np.zeros([len(raw_input), max_2nd_dim])

    for ix, sample in enumerate(raw_input):
        for ph_ix, ph in enumerate(sample):
            padded_3d_array[ix][ph_ix][:len(ph)] = ph
            padded_3d_mask[ix][ph_ix][:len(ph)] = 1

        padded_lens[ix] = len(sample)
        padded_2d_mask[ix][:len(sample)] = 1

    return padded_3d_array, padded_lens, padded_2d_mask, padded_3d_mask


def pad_4d_sequence(raw_input, pad_token=0):
    """Pad 4d sequence with `pad_token` and return mask

    Args:
        raw_input: a 4D list, e.g. [batch_size x max_sent_num x max_phrase_num x phrase_word_num]
        pad_token: int. token id for padding token.
    Returns:
        padded_4d_array
        padded_mask
    """

    # e.g. max_sent_num
    max_2nd_dim = max([len(x) for x in raw_input])

    # e.g. max_phrase_num
    max_3rd_dim = 0

    # e.g. max_word_num
    max_4th_dim = 0
    for sample in raw_input:
        for sent in sample:
            max_3rd_dim = max(max_3rd_dim, len(sent))
            for ph in sent:
                max_4th_dim = max(max_4th_dim, len(ph))

    padded_4d_array = pad_token * np.ones([len(raw_input), max_2nd_dim, max_3rd_dim, max_4th_dim])
    for ix, sample in enumerate(raw_input):
        for sent_id, sent in enumerate(sample):
            for ph_id, ph in enumerate(sent):
                padded_4d_array[ix][sent_id][ph_id][:len(ph)] = ph

    return padded_4d_array


def create_onehot_for_categorical(input, k):
    """Create one-hot encoding for categorical inputs.

    Args:
         input: a list of lists, the terminal list consists of 0-K indicating one of the K+1 categories
         k: int. Number of categories (-1)
    Returns:
        onehot_array: padded one hot encoding of input.
    """
    batch_size = len(input)
    max_sent_num = max([len(x) for x in input])
    onehot_array = np.zeros([batch_size, max_sent_num, k])

    for sample_id, sample in enumerate(input):
        for sent_id, sent_label in enumerate(sample):
            onehot_array[sample_id][sent_id][sent_label] = 1
    return onehot_array

def load_prev_checkpoint(model, ckpt_path, optimizer=None):
    """
    Load available checkpoint to either continue training or do inference.
    """
    ckpt_loaded = torch.load(ckpt_path)
    done_epochs = ckpt_loaded["epoch"]

    logging.info("Loading checkpoint epoch=%d" % done_epochs)
    logging.info(ckpt_path)

    model.word_emb.load_state_dict(ckpt_loaded["embedding"])
    model.encoder.load_state_dict(ckpt_loaded["encoder"])
    model.wd_dec.load_state_dict(ckpt_loaded["word_decoder"])
    model.sp_dec.load_state_dict(ckpt_loaded["planning_decoder"])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt_loaded["optimizer"])
    return done_epochs




def load_glove_emb(vocab):
    """
    Load Glove embedding for words in the vocabulary, if no embedding exist, initialize randomly
     Params:
      `word2id`: a dictionary mapping word to id
    """

    random_init = np.random.uniform(-.25, .25, (len(vocab), 300))
    random_init[0] = np.zeros(300)

    for ln in open(WEMB_DIR):
        lsplit = ln.strip().split("\t")
        if len(lsplit) < 300:
            continue

        word = lsplit[0]

        if vocab.word2id(word) == vocab.unk_token:
            continue

        wid = vocab.word2id[word]
        vec = np.array([float(x) for x in lsplit[1:]])
        random_init[wid] = vec
    return random_init



def encode_text_to_id_lists(op_list, passage_list, vocab, max_op_words, max_passage_words, encode_passage):
    """ Convert source input into word ids, when passage is provided, append to the OP.

    Args:
        op_list: the list of tokenized OP text.
        passage_list: the list of tokenized passages.
        vocab: vocabulary object.
        opt: options.

    Return:
        src_inputs: a list of word ids for each source input instance.
        src_lens: the sizes of source input instances.
    """
    assert len(op_list) == len(passage_list)

    src_inputs = []
    src_lens = []
    src_strs = []

    for op_instance, passage_set_instance in zip(op_list, passage_list):
        input_ids = [vocab.word2id(w) for w in op_instance]
        input_strs = [w for w in op_instance]
        if len(input_ids) > max_op_words:
            input_ids = input_ids[:max_op_words]
            input_strs = input_strs[:max_op_words]

        if not encode_passage:
            src_inputs.append(input_ids)
            src_lens.append(len(input_ids))
            src_strs.append(input_strs)
            continue

        input_ids.append(vocab.sep_token)
        input_strs.append("[SEP]")
        passage_ids = []
        passage_strs = []
        for psg in passage_set_instance:
            for sent in psg:
                passage_ids.extend([vocab.word2id(w) for w in sent])
                passage_strs.extend([w for w in sent])
            passage_ids.append(vocab.sep_token)
            passage_strs.append("[SEP]")

        if len(passage_ids) > max_passage_words:
            passage_ids = passage_ids[:max_passage_words]
            passage_strs = passage_strs[:max_passage_words]

        input_ids = input_ids + passage_ids
        input_strs = input_strs + passage_strs
        assert len(input_ids) == len(input_strs)
        src_inputs.append(input_ids)
        src_lens.append(len(input_ids))
        src_strs.append(input_strs)

    return src_inputs, src_lens, src_strs

def encode_phrase_sel_to_word_ids(target_list, vocab, max_phrase_in_sentence, max_rr_sent_num):
    """Encode sentence level phrase selection into word ids and phrase bank.

    Args:
        target_list: a list of target counterarguments, where each sentence has `selected_keyphrases`, which
            is a list of keyphrases selected for that sentence.
        vocab: the Vocabulary object
        max_phrase_in_sentence: the maximum number of keyphrases allowed for one sentence
        max_rr_sent_num: the maximum amount of sentences allowed in each instance

    Return:
        phrase_selection_word_ids: a 4d array of size [num_samples, num_sentences, num_phrases, num_words], which
            indicates the phrases in each sentence as a list of word ids.
        phrase_bank: a 3d array of size [num_samples, num_phrases, num_words], which is the collection of used keyphrase
            in each training sample.
        phrase_bank_ids: a 3d array of size [num_samples, num_sentences, num_phrases], which denotes the selection
            vector on phrase_bank for each training sample.
    """
    phrase_selection_word_ids = []
    phrase_bank = []
    phrase_bank_ids = []


    for sample in target_list:
        # _cur_ph_bank: used to store the initial set of phrases
        # cur_ph_input_ids: stores phrase word ids for each sentence
        _cur_ph_bank = set()

        # always has bos token as the first selection
        # so that train/test are consistent
        cur_ph_input_ids = [[[vocab.bos_token]]]

        for sent_id, sent in enumerate(sample):
            if sent_id == max_rr_sent_num: break
            cur_sent_ph = list()
            for ph in sent["selected_keyphrases"]:
                ph = ph.lower()
                ph_ids = [vocab.word2id(w) for w in ph.split()]
                if len(cur_sent_ph) <= max_phrase_in_sentence:
                    cur_sent_ph.append(ph_ids)
                    _cur_ph_bank.add(ph)
                else:
                    break

            cur_ph_input_ids.append(cur_sent_ph)
        cur_ph_input_ids.append([[vocab.eos_token]])

        # convert the set to a list of phrase word ids
        cur_ph_bank_lst = [[vocab.bos_token]]
        for ph in _cur_ph_bank:
            ph_ids = [vocab.word2id(w) for w in ph.split()]
            cur_ph_bank_lst.append(ph_ids)
        cur_ph_bank_lst.append([vocab.eos_token])

        # create phrase selection vector
        cur_ph_bank_sel_ind = list()
        for sent in cur_ph_input_ids:
            cur_sel = [1 if ph in sent else 0 for ph in cur_ph_bank_lst]
            cur_ph_bank_sel_ind.append(cur_sel)

        phrase_bank.append(cur_ph_bank_lst)
        phrase_bank_ids.append(cur_ph_bank_sel_ind)
        phrase_selection_word_ids.append(cur_ph_input_ids)

    return phrase_selection_word_ids, phrase_bank, phrase_bank_ids

def encode_phrase_bank_to_id_lists(phrase_bank_list, vocab, max_bank_size):
    """Encode keyphrase bank into word ids.

    Args:
         phrase_bank_list (list): the list of keyphrase bank for each trianing data, each list
            is a list of keyphrase (already deduplicated), each keyphrase is a string.
         vocab (Vocab): vocabulary to convert words into ids
         max_bank_size (int): maximum allowed number of keyphrase per instance.
    Returns:
        phrase_bank_word_ids (list): the list of phrase bank word ids
        phrase_bank_words (list): the list of tokenized phrase words, with the same
            dimension as phrase_bank_word_ids
    """
    phrase_bank_word_ids = []
    phrase_bank_words = []

    for sample in phrase_bank_list:
        cur_ph_bank_wids = []
        cur_ph_bank_words = []
        for ph in sample:
            ph = ph.lower()
            cur_ph = []
            cur_ph_ids = []
            for w in ph.split():
                if w in vocab._word2id:
                    cur_ph.append(w)
                    cur_ph_ids.append(vocab.word2id(w))
            if len(cur_ph_ids) == 0: continue
            cur_ph_bank_wids.append(cur_ph_ids)
            cur_ph_bank_words.append(cur_ph)
            if len(cur_ph_bank_wids) == max_bank_size:
                break

        phrase_bank_word_ids.append(cur_ph_bank_wids)
        phrase_bank_words.append(cur_ph_bank_words)

    return phrase_bank_word_ids, phrase_bank_words


def encode_sentence_and_type_to_list(target_list, vocab, max_sent_num, sentence_type_func=None):
    """Encode target sentence and its type into word ids.

     Args:
        target_list: a list of target counterarguments, where each sentence has `selected_keyphrases`, which
            is a list of keyphrases selected for that sentence.
        vocab: the Vocabulary object
        opt: options
        max_sent_num: int. If an instance has more than this many sentences, drop the excedding ones.
        sentence_type_func: the method to assign type label for each sentence, if not found in the dataset.
    Return:
        word_ids: word ids for target counterargument
        sent_ids: sentence ids for each word in word_ids
        sent_type: whether each sentence is MC or FILL, 0 for SOS, 1 for FILL, 2 for MC
    """
    word_ids_input = list()
    word_ids_output = list()
    sent_ids = list()
    sent_type = list()

    for sample in target_list:
        cur_sample_word_ids = list()
        cur_sample_sent_ids = list()
        cur_sample_sent_types = list()

        cur_sample_sent_ids.append(0)
        cur_sample_sent_types.append(0)

        for sent_id, sent in enumerate(sample):
            if sent_id == max_sent_num: break
            if isinstance(sent, list):
                toks = [vocab.word2id(w) for w in sent]
            else:
                toks = [vocab.word2id(w) for w in sent["tokens"]]

            s_ids = [sent_id + 1 for _ in toks]
            cur_sample_word_ids.extend(toks)
            cur_sample_sent_ids.extend(s_ids)
            if sentence_type_func is not None:
                cur_sample_sent_types.append(sentence_type_func(toks))
            elif "style" in sent:
                cur_sample_sent_types.append(sentence_type_func(sent["style"]))
            else:
                cur_sample_sent_types.append(None)

        word_ids_input.append([vocab.bos_token] + cur_sample_word_ids)
        word_ids_output.append(cur_sample_word_ids + [vocab.eos_token])
        sent_ids.append(cur_sample_sent_ids)
        sent_type.append(cur_sample_sent_types)

    return word_ids_input, word_ids_output, sent_ids, sent_type


def pad_text_id_list_into_array(batch_text_lists, max_len=500, add_start=True, sos_id=SOS_id, eos_id=EOS_id):
    """
    Pad text id list into array.
     Params:
      `batch_text_lists`: a list of word ids without adding SOS or EOS
      `max_len`: maximum allowed length for words (including SOS and EOS)
      `add_start`: boolean, denotes whether to add "SOS" at the beginning of the sequence, used for decoder
      `sos_id`: integer word id for SOS token
      `eos_id`: integer word id for EOS token
    """

    batch_size = len(batch_text_lists)
    max_word_num_in_batch = max([len(x) + 1 for x in batch_text_lists])
    if add_start:
        max_word_num_in_batch += 1

    max_word_num_in_batch = min(max_word_num_in_batch, max_len)

    word_inputs = np.zeros([batch_size, max_word_num_in_batch]).astype("float32")
    word_targets = np.zeros([batch_size, max_word_num_in_batch]).astype("float32")
    word_count = np.zeros(batch_size)

    for sample_id, sample in enumerate(batch_text_lists):
        if add_start:
            truncated_sample = sample[:max_word_num_in_batch - 1]
            input_sample = [sos_id] + truncated_sample
            target_sample = truncated_sample + [eos_id]
            word_count[sample_id] = len(truncated_sample + 1)
        else:
            truncated_sample = sample[:max_word_num_in_batch]
            input_sample = truncated_sample
            target_sample = truncated_sample
            word_count[sample_id] = len(truncated_sample)

        word_inputs[sample_id][:len(input_sample)] = input_sample
        word_targets[sample_id][:len(target_sample)] = target_sample
    return word_inputs, word_targets, word_count


def generate_eos_template(batch_ph_bank):
    """
    Args:
        batch_ph_bank (batch_size x max_ph_bank_size x max_word_per_ph) list of integers
    Returns:
        ph_bank_eos_template (batch_size x max_ph_bank_size) one-hot encoding of EOS phrases
    """
    max_ph_num = max([len(x) for x in batch_ph_bank])
    batch_size = len(batch_ph_bank)
    ph_bank_eos_template = np.zeros([batch_size, max_ph_num])
    for ix, sample in enumerate(batch_ph_bank):
        ph_bank_eos_template[ix][len(sample) - 1] = 1
    return ph_bank_eos_template


def aeq(*args):
    """
    Assert all arguments have the same value
    """
    arguments = (arg for arg in args)
    first = next(arguments)
    assert all(arg == first for arg in arguments), \
        "Not all arguments have the same value: " + str(args)


def sequence_mask(lengths, max_len=None):
    batch_size = lengths.numel()
    max_len = max_len or lengths.max()
    return (torch.arange(0, max_len)
            .type_as(lengths)
            .repeat(batch_size, 1)
            .lt(lengths.unsqueeze(1)))


def tile(x, count, dim=0):
    """
    Tiles x on dimension dim count times.
    """
    perm = list(range(len(x.size())))
    if dim != 0:
        perm[0], perm[dim] = perm[dim], perm[0]
        x = x.permute(perm).contiguous()
    out_size = list(x.size())
    out_size[0] *= count
    batch = x.size(0)
    x = x.view(batch, -1) \
        .transpose(0, 1) \
        .repeat(count, 1) \
        .transpose(0, 1) \
        .contiguous() \
        .view(*out_size)
    if dim != 0:
        x = x.permute(perm).contiguous()
    return x