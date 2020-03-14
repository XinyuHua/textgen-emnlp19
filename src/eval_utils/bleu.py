import collections
import math
from tqdm import tqdm

def evaluate_bleu_match(ref_data, pred_data, max_ngram=4):
    """
    run BLEU for arggen setup, where only the ref with highest bleu is considered
    Args:
        ref_data: a dictionary mapping tid to list of references, each reference is a list of words
        pred_data: a dictionary mapping tid to a system output
    Return:
    """
    reference_corpus = []
    translation_corpus = []

    samples_evaluated = 0
    miss_count = 0
    finished_count = 0
    for tid in tqdm(pred_data):
        cur_sys = pred_data[tid]["pred"]
        if not tid in ref_data:
            miss_count += 1
            continue
        cur_ref_lst = [x['tgt_words'] for x in ref_data[tid]]
        if len(cur_ref_lst) > 1:
            samples_evaluated += 1
            best_ref, highest_bleu = __bleu_pairs(cur_sys, cur_ref_lst, ngram=max_ngram)
            reference_corpus.append([best_ref])
            translation_corpus.append(cur_sys)

        elif len(cur_ref_lst) == 1:
            samples_evaluated += 1
            reference_corpus.append(cur_ref_lst)
            translation_corpus.append(cur_sys)

        finished_count += 1

    print("miss_count: %d" % miss_count)

    for max_order in [1, 2, 3, 4]:
        bleu_rst = compute_bleu(reference_corpus, translation_corpus, max_order=max_order, smooth=True)
        print("BLEU-%d: %.5f" % (max_order, bleu_rst[0]))
    print("%d sample evaluated" % samples_evaluated)
    return


def __bleu_pairs(system_output, refs, ngram=4):
    """
    Compute pair-wise BLEU for one system output to all possible references, pick the highest one
    Args:
        system_output: one single output, tokenized as a list of words, all lowercased
        refs: a list of references, each is a list of words, all lowercased
    Return:
        chosen_ref: the reference text with the highest bleu score
    """
    highest_bleu = 0
    picked_ref = []
    for ref in refs:
        bleu_rst = compute_bleu([[ref]], [system_output], max_order=ngram, smooth=True)
        if bleu_rst[0] > highest_bleu:
            highest_bleu = bleu_rst[0]
            picked_ref = ref
    return picked_ref, highest_bleu


def compute_bleu(reference_corpus, translation_corpus, max_order=4,
                 smooth=False):
  """Computes BLEU score of translated segments against one or more references.
  Args:
    reference_corpus: list of lists of references for each translation. Each
        reference should be tokenized into a list of tokens.
    translation_corpus: list of translations to score. Each translation
        should be tokenized into a list of tokens.
    max_order: Maximum n-gram order to use when computing BLEU score.
    smooth: Whether or not to apply Lin et al. 2004 smoothing.
  Returns:
    3-Tuple with the BLEU score, n-gram precisions, geometric mean of n-gram
    precisions and brevity penalty.
  """
  matches_by_order = [0] * max_order
  possible_matches_by_order = [0] * max_order
  reference_length = 0
  translation_length = 0
  for (references, translation) in zip(reference_corpus,
                                       translation_corpus):
    reference_length += min(len(r) for r in references)
    translation_length += len(translation)

    merged_ref_ngram_counts = collections.Counter()
    for reference in references:
      merged_ref_ngram_counts |= _get_ngrams(reference, max_order)
    translation_ngram_counts = _get_ngrams(translation, max_order)
    overlap = translation_ngram_counts & merged_ref_ngram_counts
    for ngram in overlap:
      matches_by_order[len(ngram)-1] += overlap[ngram]
    for order in range(1, max_order+1):
      possible_matches = len(translation) - order + 1
      if possible_matches > 0:
        possible_matches_by_order[order-1] += possible_matches

  precisions = [0] * max_order
  for i in range(0, max_order):
    if smooth:
      precisions[i] = ((matches_by_order[i] + 1.) /
                       (possible_matches_by_order[i] + 1.))
    else:
      if possible_matches_by_order[i] > 0:
        precisions[i] = (float(matches_by_order[i]) /
                         possible_matches_by_order[i])
      else:
        precisions[i] = 0.0

  if min(precisions) > 0:
    p_log_sum = sum((1. / max_order) * math.log(p) for p in precisions)
    geo_mean = math.exp(p_log_sum)
  else:
    geo_mean = 0

  ratio = float(translation_length) / reference_length

  if ratio > 1.0:
    bp = 1.
  elif ratio == 0.0:
    bp = 0.0
  else:
    bp = math.exp(1 - 1. / ratio)

  bleu = geo_mean * bp

  return (bleu, precisions, bp, ratio, translation_length, reference_length)


def _get_ngrams(segment, max_order):
  """Extracts all n-grams upto a given maximum order from an input segment.
  Args:
    segment: text segment from which n-grams will be extracted.
    max_order: maximum length in tokens of the n-grams returned by this
        methods.
  Returns:
    The Counter containing all n-grams upto max_order in segment
    with a count of how many times each n-gram occurred.
  """
  ngram_counts = collections.Counter()
  for order in range(1, max_order + 1):
    for i in range(0, len(segment) - order + 1):
      ngram = tuple(segment[i:i+order])
      ngram_counts[ngram] += 1
  return ngram_counts