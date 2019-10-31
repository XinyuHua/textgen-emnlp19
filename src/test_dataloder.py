import argparse

from utils import Vocabulary, load_train_data
from utils.data_utils import DataSampler, WikiDataset

opt = argparse.Namespace()
opt.debug = True
opt.max_title_words = 20
opt.max_sent_num = 10
opt.max_bank_size = 50
opt.batch_size=3
opt.max_phrase_in_sentence = 10
opt.task = "wikigen"

vocab = Vocabulary(task=opt.task)
train_data = load_train_data(opt.debug, task=opt.task)
dataset = WikiDataset(set_type="train")
dataset.load_data(raw_data=train_data["train"], opt=opt, vocab=vocab)
data_sampler = DataSampler(dataset=dataset, batch_size=opt.batch_size, sequential=True)

for item in data_sampler:
    print(item["title"], data_sampler.start_idx)