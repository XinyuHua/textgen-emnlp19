# for arggen, convert old format data to trainable for release

import json
from tqdm import tqdm

OLD_PATH = "/data/model/xinyu/cmv/processed/runnable_emnlp/"
NEW_PATH = "/data/model/xinyu/emnlp2019_code_release/data/arggen/"
URL_PREFIX = "https://www.reddit.com/r/changemyview/comments/"
TYPE2NAME = {"A": "functional",
             "B": "claim",
             "C": "premise",}

def build_train_dev():

    for set_type in ["dev", "train"]:
        raw_lns = open(OLD_PATH + "oracle-2_%s.jsonlist" % set_type).readlines()
        fout = open(NEW_PATH + "%s.jsonl" % set_type, 'w')
        for ln in tqdm(raw_lns):
            cur_obj = json.loads(ln)
            output_obj = {"op": cur_obj["op"],
                          "url": URL_PREFIX + cur_obj["tid"],
                          "id": cur_obj["tid"]}
            target_counterarg = []
            # add RR sentences
            for rr_toks, rr_type, rr_psg_kp_sel in zip(cur_obj["rr"], cur_obj["rr_stype"], cur_obj["rr_psg_kp_sel"]):
                target_counterarg.append({"tokens": rr_toks,
                                          "style": TYPE2NAME[rr_type],
                                          "selected_keyphrases": rr_psg_kp_sel})


            output_obj["target_counterarg"] = target_counterarg

            target_retrieved_passages = []
            # add passages
            for psg, psg_kp in zip(cur_obj["rr_psg"], cur_obj["rr_psg_kp"]):
                target_retrieved_passages.append({"sentences": psg, "keyphrases": psg_kp})

            output_obj["target_retrieved_passages"] = target_retrieved_passages
            fout.write(json.dumps(output_obj) + "\n")
        fout.close()

def build_test():
    # system setup
    fout = open(NEW_PATH + "test.jsonl", 'w')
    raw_lns = [json.loads(ln) for ln in open(OLD_PATH + "system_test.jsonlist")]
    for ln in tqdm(raw_lns):
        output_obj = {"op": ln["op"],
                      "id": ln["tid"],
                      "url": URL_PREFIX + ln["tid"]}
        op_retrieved_passages = []
        for psg, kp in zip(ln['passage'], ln['passage_kp']):
            op_retrieved_passages.append({"sentences": psg,
                                          "keyphrases": kp})
        output_obj["op_retrieved_passages"] = op_retrieved_passages
        fout.write(json.dumps(output_obj) + "\n")
    fout.close()

if __name__=='__main__':
    build_test()
