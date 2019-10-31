## About

This repository contains code for the following paper:

Xinyu Hua and Lu Wang [*Sentence-Level Content Planning and Style Specification for Neural Text Generation*](https://arxiv.org/abs/1909.00734)

If you find our work useful, please cite:

```bibtex
@inproceedings{hua-wang-2019-sentence,
    title = "Sentence-Level Content Planning and Style Specification for Neural Text Generation",
    author = "Hua, Xinyu  and
              Wang, Lu",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    year = "2019",
    address = "Hong Kong, China",
    publisher = "Association for Computational Linguistics",
}
```


## Dataset

__download link:__ [link](https://drive.google.com/file/d/1oR5JmlsTihG8t_0FjYTGGijSgGsB9Js_/view)

| task      | \# tokens target | \# keyphrase | source          |
|-----------|------------------|--------------|-----------------|
|arggen     |      54.87       |   55.80      |[changemyview](https://www.reddit.com/r/changemyview/) |
|wikigen    |   70.57/48.60    |   23.56      |[Normal](https://www.wikipedia.org/)/[Simple](https://simple.wikipedia.org/) Wikipedia|
|absgen     |      141.34      |   12.23      |[AGENDA](https://github.com/rikdz/GraphWriter)      |


## Quickstart

__note__: all actions below assume `src/` to be the working directory.

To train an argument generation model:

```
python main.py --mode=train \
    --exp_name=arggen_exp \
    --encode_passage \
    --type_conditional_lm \
    --task=arggen \
    --batch_size=30 \
    --num_train_epochs=30 \
    --logging_freq=2
    --max_src_words=500 \
    --max_passage_words=400 \
    --max_sent_num=10 \
    --max_bank_size=70 \
```

To train an abstract generation model, which has no sentence level style labels:

```
python main.py --mode=train \
    --exp_name=absgen_exp \
    --task=absgen \
    --batch_size=30 \
    --num_train_epochs=30 \
    --max_src_words=1000 \
    --max_bank_size=30 \
    --logging_freq=2
```

To train a Wikipedia generation model:

```
python main.py --mode=train \
    --exp_name=wikigen_exp \
    --type_conditional_lm \
    --task=wikigen \
    --batch_size=30 \
    --max_bank_size=30 \
    --num_train_epochs=30 \
    --max_src_words=1000 \
    --logging_freq=2
```


## License

See the [LICENSE](LICENSE) file for details.


