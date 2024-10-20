from collections import defaultdict
import os
import requests
import json
import concurrent.futures
from tqdm import tqdm


MY_GCUBE_TOKEN = '1b134163-6474-4a56-8620-8a76070fbf27-843339462'

class WATAnnotation:
    # An entity annotated by WAT

    def __init__(self, d):

        # char offset (included)
        self.start = d['start']
        # char offset (not included)
        self.end = d['end']

        # annotation accuracy
        self.rho = d['rho']
        # spot-entity probability
        self.prior_prob = d['explanation']['prior_explanation']['entity_mention_probability']

        # annotated text
        self.spot = d['spot']

        # Wikpedia entity info
        self.wiki_id = d['id']
        self.wiki_title = d['title']


    def json_dict(self):
        # Simple dictionary representation
        return {'wiki_title': self.wiki_title,
                'wiki_id': self.wiki_id,
                'start': self.start,
                'end': self.end,
                'rho': self.rho,
                'prior_prob': self.prior_prob
                }


def wat_entity_linking(text):
    # Main method, text annotation with WAT entity linking system
    wat_url = 'https://wat.d4science.org/wat/tag/tag'
    payload = [("gcube-token", MY_GCUBE_TOKEN),
               ("text", text),
               ("lang", 'en'),
               ("tokenizer", "nlp4j"),
               ('debug', 9),
               ("method",
                "spotter:includeUserHint=true:includeNamedEntity=true:includeNounPhrase=true,prior:k=50,filter-valid,centroid:rescore=true,topk:k=5,voting:relatedness=lm,ranker:model=0046.model,confidence:model=pruner-wiki.linear")]

    proxies = {
        'http': 'http://127.0.0.1:7890',
        'https': 'http://127.0.0.1:7890'  # https -> http
    }
    # response = requests.get(wat_url, params=payload, proxies=proxies)
    response = requests.get(wat_url, params=payload)

    return [WATAnnotation(a) for a in response.json()['annotations']]


def wat_annotations(wat_annotations):
    json_list = [w.json_dict() for w in wat_annotations]
    
    return json_list


def wiki_kw_extract_chunk(chunk, prior_prob = 0.8):
    wat_annotations = wat_entity_linking(chunk)
    json_list = [w.json_dict() for w in wat_annotations]
    kw2chunk = defaultdict(set)
    chunk2kw = defaultdict(set)
    
    for wiki in json_list:
        if wiki['wiki_title'] != '' and wiki['prior_prob'] > prior_prob:
            kw2chunk[wiki['wiki_title']].add(chunk)
            chunk2kw[chunk].add(wiki['wiki_title'])
    
    # kw2chunk[title].add(chunk)
    # chunk2kw[chunk].add(title)

    return kw2chunk, chunk2kw


def tagme_extract(data, prior_prob):
    for d in tqdm(data):
        kw2chunk = defaultdict(set) #kw1 -> [chunk1, chunk2, ...]
        chunk2kw = defaultdict(set) #chunk -> [kw1, kw2, ...]

        with concurrent.futures.ProcessPoolExecutor() as executor:
            future_to_chunk = {executor.submit(wiki_kw_extract_chunk, chunk, prior_prob): chunk for chunk in d['text']}
            for future in concurrent.futures.as_completed(future_to_chunk):
                chunk, inv_chunk = future.result()
                for key, value in chunk.items():
                    kw2chunk[key].update(value)
                for key, value in inv_chunk.items():
                    chunk2kw[key].update(value)

        for key in kw2chunk:
            kw2chunk[key] = list(kw2chunk[key])

        for key in chunk2kw:
            chunk2kw[key] = list(chunk2kw[key])

        d['kw2chunk'] = kw2chunk
        d['chunk2kw'] = chunk2kw
    return data


def load_passages(fpath):
    if not os.path.exists(fpath):
        raise FileNotFoundError(f"{fpath} does not exist")

    passages = []
    with open(fpath) as fin:
        if fpath.endswith(".jsonl"):
            for k, line in enumerate(fin):
                ex = json.loads(line)
                passages.append(ex)
        else:
            reader = csv.reader(fin, delimiter="\t")
            for k, row in enumerate(reader):
                if not row[0] == "id":
                    ex = {"id": row[0], "title": row[2], "text": row[1]}
                    passages.append(ex)
    return passages


data = load_passages('/mnt/workspace/dataset/HotpotQA/corpus.jsonl')
wat_results = tagme_extract(data[:10], 0.8)
with open('/mnt/workspace/dataset/HotpotQA/tagme.json', 'w', encoding='utf-8') as f:
    json.dump(wat_results, f)