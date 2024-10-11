import regex
import json
import string
import unicodedata
from typing import List
import numpy as np
from collections import Counter
import itertools


class SimpleTokenizer(object):
    ALPHA_NUM = r'[\p{L}\p{N}\p{M}]+'
    NON_WS = r'[^\p{Z}\p{C}]'

    def __init__(self):
        """
        Args:
            annotators: None or empty set (only tokenizes).
        """
        self._regexp = regex.compile(
            '(%s)|(%s)' % (self.ALPHA_NUM, self.NON_WS),
            flags=regex.IGNORECASE + regex.UNICODE + regex.MULTILINE
        )

    def tokenize(self, text, uncased=False):
        matches = [m for m in self._regexp.finditer(text)]
        if uncased:
            tokens = [m.group().lower() for m in matches]
        else:
            tokens = [m.group() for m in matches]
        return tokens


def check_answer(example, tokenizer) -> List[bool]:
    """Search through all the top docs to see if they have any of the answers."""
    answers = example['answers']
    ctxs = example['ctxs']

    hits = []

    for _, doc in enumerate(ctxs):
        text = doc['text']

        if text is None:  # cannot find the document for some reason
            hits.append(False)
            continue

        hits.append(has_answer(answers, text, tokenizer))

    return hits


def has_answer(answers, text, tokenizer=SimpleTokenizer()) -> bool:
    """Check if a document contains an answer string."""
    text = _normalize(text)
    text = tokenizer.tokenize(text, uncased=True)

    for answer in answers:
        answer = _normalize(answer)
        answer = tokenizer.tokenize(answer, uncased=True)
        for i in range(0, len(text) - len(answer) + 1):
            if answer == text[i: i + len(answer)]:
                return True
    return False


def _normalize(text):
    return unicodedata.normalize('NFD', text)


def normalize_answer(s):
    def remove_articles(text):
        return regex.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def exact_match_score(prediction, ground_truth):
    # print(prediction, ground_truth)
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def ems(prediction, ground_truths):
    return max([exact_match_score(prediction, gt) for gt in ground_truths])


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def f1(prediction, ground_truths):
    return max([f1_score(prediction, gt) for gt in ground_truths])


def rougel_score(prediction, ground_truth):
    rouge = Rouge()
    # no normalization
    try:
        scores = rouge.get_scores(prediction, ground_truth, avg=True)
    except ValueError:  # "Hypothesis is empty."
        return 0.0
    return scores["rouge-l"]["f"]


def rl(prediction, ground_truths):
    return max([rougel_score(prediction, gt) for gt in ground_truths])


## file-level evaluation ... ### 
def eval_recall(infile):

    tokenizer = SimpleTokenizer()
    lines = open(infile, 'r').readlines()[1:]

    has_answer_count = 0
    answer_lengths = []
    for line in lines:
        line = json.loads(line)
        answer = line['answer']
        output = ' || '.join(line['output'])

        if has_answer(answer, output, tokenizer):
            has_answer_count += 1

        answer_lengths.append(len(output.split()))

    recall = round(has_answer_count/len(lines), 4)
    lens = round(np.mean(answer_lengths), 4)

    return recall, lens


def eval_question_answering(infile):

    lines = open(infile, 'r').readlines()[1:]

    exact_match_count = 0
    answer_lengths = []
    for line in lines:
        line = json.loads(line)
        answer = line['answer']
        output = line['output'][0]

        if ems(output, answer): # EM evaluation
            exact_match_count += 1
        
        answer_lengths.append(len(output.split()))

    em = round(exact_match_count/len(lines), 4)
    lens = round(np.mean(answer_lengths), 4)

    return em, lens


def eval_fact_checking(infile):

    tokenizer = SimpleTokenizer()
    lines = open(infile, 'r').readlines()[1:]

    exact_match_count = 0
    answer_lengths = []
    for line in lines:
        line = json.loads(line)
        answer = line['answer']
        output = line['output'][0]

        if answer == ["refutes"]:
            answer = ["refutes", "no", "false"]
        if answer == ["supports"]:
            answer = ["supports", "yes", "true"]

        if has_answer(answer, output, tokenizer):
            exact_match_count += 1
        
        answer_lengths.append(len(output.split()))

    em = round(exact_match_count/len(lines), 4)
    lens = round(np.mean(answer_lengths), 4)

    return em, lens


def evaluate(dataset, retriever, k=None, round=5, topks=[1, 5, 10, 20, 30], k_emb=15):
    if retriever != 'knn':
        res = json.load(open('../result/{}/{}_{}.json'.format(dataset, retriever, k), 'rb'))
    else:
        res = json.load(open('../result/{}/{}_{}_{}.json'.format(dataset, retriever, k_emb, k), 'rb'))

    filter_res = [r for r in res if r['prediction'] != 'System mistake']

    f1s, emss, accs = [], [], []

    if retriever not in ['golden', 'no']:
        recall, precision, sp_em = [], [], []

    for r in filter_res:
        accs.append(('1' in r['grade']) * 1.0)

        if dataset in ['hotpotqa', 'wikimultihop', 'musique']:
            f1s.append(f1_score(r['prediction'], r['answer']))
            emss.append(exact_match_score(r['prediction'], r['answer']))

        elif dataset in ['iirc']:
            f1s.append(f1(r['prediction'], r['answer']))
            emss.append(ems(r['prediction'], r['answer']))

        r['corpus'] = list(itertools.chain(*[_.split('\n') for _ in r['corpus']]))
        if retriever not in ['golden', 'no']:
            evi = set([_[1] for _ in r['supports']])

            tmp_recall = []
            tmp_precision = []
            tmp_sp_em = []
            for kk in topks:
                if kk <= k:
                    tmp = set(r['corpus'][:kk])

                    tmp_recall.append(len(evi.intersection(tmp)) / len(evi))
                    tmp_precision.append(len(evi.intersection(tmp)) / kk)

                    if evi.issubset(tmp):
                        tmp_sp_em.append(1)
                    else:
                        tmp_sp_em.append(0)

            recall.append(tmp_recall)
            precision.append(tmp_precision)
            sp_em.append(tmp_sp_em)

    print('Acc:', np.mean(accs))
    print('F1:', np.mean(f1s))
    print('EM:', np.mean(emss))

    if retriever not in ['golden', 'no']:
        print('Recall:', np.mean(np.array(recall), axis=0))
        print('Precision:', np.mean(np.array(precision), axis=0))
        print('SP_EM:', np.mean(np.array(sp_em), axis=0))


def eval(prediction_file):
    with open(prediction_file) as f:
        prediction = json.load(f)
    f1s, emss = [], []
    strict_acc, fuzzy_acc = 0, 0
    for item in prediction:
        f1s.append(f1_score(item['prediction'], item['answer']))
        emss.append(exact_match_score(item['prediction'], item['answer']))
        if item['prediction'] == item['answer']:
            strict_acc += 1
    print('F1:', np.mean(f1s))
    print('EM:', np.mean(emss))
    print('Accuracy:', strict_acc/len(prediction))


if __name__ == '__main__':
    # predict_path = "./results/HotpotQA/only_qwen2-7b-instruct_dev_eval.json"
    # glod_path = "./results/HotpotQA/glod_dev.json"
    predict_path = './dev_prediction_hybrid_256.json'
    eval(predict_path)