import json
import tqdm
import logging
from utils.misc_utils import DATA_DIR

def load_test_data(task="arggen", demo=False):
    if task == "arggen":
        return _load_arggen_test_data(demo=demo)
    elif task == "wikigen":
        return _load_wikigen_test_data(demo=demo)
    else:
        return _load_absgen_test_data(demo=demo)

def load_train_data(demo=False, task=None):
    assert task is not None, "Task has to be specified!"
    if task == "wikigen":
        return _load_wiki_train_data(demo=demo)
    elif task == "arggen":
        return _load_arggen_train_data(demo=demo)
    elif task == "absgen":
        return _load_absgen_train_data(demo=demo)
    else:
        raise ValueError("Specified task {} does not exist!".format(task))


def _load_arggen_test_data(demo=False):
    """
    Load test data for argument generation task.
    """
    path = DATA_DIR + "arggen/test.jsonl"
    dataset = {"op": [], "passages": [], "passage_kp": [], "id": []}

    logging.info("Loading test data for arggen...")
    raw_lns = open(path).readlines()
    if demo:
        raw_lns = raw_lns[:100]

    for ln in tqdm(raw_lns):
        cur_obj = json.loads(ln)
        dataset["op"].append(cur_obj["op"])
        dataset["id"].append(cur_obj["id"])
        cur_passage_sent_lst = []
        cur_passage_kp_set = set()

        for psg in cur_obj["op_retrieved_passages"]:
            cur_passage_sent_lst.append(psg["sentences"])
            for kp in psg["keyphrases"]:
                cur_passage_kp_set.add(kp)


        dataset["passages"].append(cur_passage_sent_lst)
        dataset["passage_kp"].append(cur_passage_kp_set)

    logging.info("Arggen test data loaded. %d samples in total." % (len(dataset["id"])))
    return dataset


def _load_wikigen_test_data(demo=False):
    """
    Load test data for wikipedia paragraph generation task.
    """
    path = DATA_DIR + "wikigen/test.jsonl"
    dataset = {"title": [], "style": [], "ph_bank": []}

    logging.info("Loading test data for wikigen...")
    raw_lns = open(path).readlines()
    if demo:
        raw_lns = raw_lns[:100]

    for ln in tqdm(raw_lns):
        cur_obj = json.loads(ln)
        dataset["title"].append(cur_obj["title"])
        dataset["ph_bank"].append(cur_obj["ph_bank"])
        dataset["style"].append(1)

        dataset["title"].append(cur_obj["title"])
        dataset["ph_bank"].append(cur_obj["ph_bank"])
        dataset["style"].append(0)
    return dataset


def _load_absgen_test_data(demo=False):
    """
    Load test data for abstract generation task.
    """
    path = DATA_DIR + "absgen/test.jsonl"
    dataset = {"title": [], "ph_bank": []}
    logging.info("Loading test data for absgen...")
    raw_lns = open(path).readlines()
    if demo:
        raw_lns = raw_lns[:100]

    for ln in tqdm(raw_lns):
        cur_obj = json.loads(ln)
        dataset["title"].append(cur_obj["title"])
        dataset["ph_bank"].append(cur_obj["ph_bank"])

    return dataset


def _load_absgen_train_data(demo=False):
    """
    Load abstract generation training and dev data.
    Args:
        demo: bool. If set to True only load 100 samples.
    Returns:
        dataset:
    """
    dataset = {set_type: {"tgt": [], "title": [], "kp_sel": [], "ph_bank": []} \
               for set_type in ['train', 'dev']}

    for set_type in dataset:
        path = DATA_DIR + "absgen/%s.jsonl" % set_type

        for ln in open(path):
            cur_obj = json.loads(ln)

            dataset[set_type]["title"].append(cur_obj["title"])
            dataset[set_type]["ph_bank"].append(cur_obj["ph_bank"])
            dataset[set_type]["kp_sel"].append(cur_obj["ph_sel"])
            dataset[set_type]["tgt"].append(cur_obj["abstract_words"])

            if demo and len(dataset[set_type]["title"]) == 100:
                break
    print("Abstract data loaded. train/dev=%d/%d" % (len(dataset["train"]["title"]), len(dataset["dev"]["title"])))
    return dataset


def _load_wiki_train_data(demo=False):
    """
    Load Wikipedia generation training and dev data.
    Args:
        demo: bool. If set to True only load 100 samples.
    Returns:
        dataset:
    """
    dataset = {set_type: {"tgt": [], "title": [], "kp_sel": [], "style": [], "ph_bank": []} \
               for set_type in ["train", "dev"]}

    for set_type in dataset:
        if demo:
            set_type = "train"
        path = DATA_DIR + "wikigen/%s.jsonl" % set_type

        for ln in open(path):
            cur_obj = json.loads(ln)
            """ 
            "title": a list of string for the article title
                e.g. ["septic", "tank"]
            "sents": a list of sentences, each sentence is a list of words
                e.g. "sents": [["a", "septic", "tank", "is", "an", "underground", "chamber", "made", ...],
                               ["settling", "and", "anaerobic", "processes", "reduce", "solids", "and", ...],
                               ["septic", "tank", "systems", "are", "a", "type", "of", "simple",...]]
            "ph_sel": a list of phrases for each sentence, where each phrase is a list of words
                e.g. "ph_sel": [[["flows", "for", "basic", "treatment"], ["domestic", "wastewater"], ...],
                                [["moderate"], ["reduce", "solides"], ["anaerobic", "processes"], ...]]
            """
            dataset[set_type]["title"].append(cur_obj["title"])
            dataset[set_type]["tgt"].append(cur_obj["normal_sents"])
            dataset[set_type]["kp_sel"].append(cur_obj["normal_ph_sel"])
            dataset[set_type]["style"].append(1)
            dataset[set_type]["ph_bank"].append(cur_obj["ph_bank"])

            dataset[set_type]["title"].append(cur_obj["title"])
            dataset[set_type]["tgt"].append(cur_obj["simple_sents"])
            dataset[set_type]["kp_sel"].append(cur_obj["simple_ph_sel"])
            dataset[set_type]["style"].append(0)
            dataset[set_type]["ph_bank"].append(cur_obj["ph_bank"])

            if demo and len(dataset[set_type]["title"]) >= 100:
                break
    print("Wikipedia data loaded, train/dev=%d/%d" % (len(dataset["train"]["title"]), len(dataset["dev"]["title"])))
    return dataset


def _load_arggen_train_data(demo=False):
    """
    Load training and validation data. Data format is detailed below:
    `op` (list):  tokenized OP
    `target_counterarg` (list): a list of sentences in root reply (target argument)
    `target_retrieved_passages` (list): a list of retrieved passages, which contains sentences and keyphrases
    """
    dataset = dict()
    dataset["train"] = {"src": {"op": [], "passages": [], "passage_kp": []},
                        "tgt": [],
                        "id": []}

    dataset["dev"] = {"src": {"op": [], "passages": [], "passage_kp": []},
                      "tgt": [],
                      "id": []}

    for set_type in ["train", "dev"]:
        ln_cnt = 0
        logging.info("loading %s data..." % set_type)

        if demo:
            raw_lns = open(DATA_DIR + "arggen/train.jsonl").readlines()
            raw_lns = raw_lns[:10]
        else:
            raw_lns = open(DATA_DIR + "arggen/%s.jsonl" % set_type).readlines()

        for ln in tqdm(raw_lns):
            cur_obj = json.loads(ln)
            ln_cnt += 1

            dataset[set_type]["src"]["op"].append(cur_obj["op"])
            dataset[set_type]["id"].append(cur_obj["id"])
            dataset[set_type]["tgt"].append(cur_obj["target_counterarg"])

            cur_passage_set = list()
            cur_passage_kp_set = list()
            for psg in cur_obj["target_retrieved_passages"]:
                cur_passage_set.append(psg["sentences"])
                cur_passage_kp_set.append(psg["keyphrases"])
            dataset[set_type]["src"]["passages"].append(cur_passage_set)
            dataset[set_type]["src"]["passage_kp"].append(cur_passage_kp_set)

            if demo and ln_cnt >= 100:
                break

        logging.info("%s data loaded, %d samples in total" % (set_type, ln_cnt))

    return dataset