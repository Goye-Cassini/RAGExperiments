import os
import json
import csv
from tqdm import tqdm
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from llm.openai_llm import LLM_Model
import concurrent.futures


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


def extract(model, context):
    instruction = """Read the following text paragraph carefully and extract the core entities and the relationships between them. Each entity should represent a key concept, person, place, or object mentioned in the text. Relationships should capture how these entities are connected (e.g., 'is related to', 'works with', 'located in'). The output should be in JSON format, with two main fields: entities and relationships. Each entity should have a unique identifier, and relationships should specify the entities involved and the type of relationship.  Only return the JSON result, following this structure:
{
  'entities': [
    {'id': 'E1', 'name': 'Entity1'},
    {'id': 'E2', 'name': 'Entity2'}
  ],
  'relationships': [
    {'source': 'E1', 'target': 'E2', 'relation': 'works with'}
  ]
}
If no entities or relationships are found, return an empty list for both entities and relationships.
Context:
    """
    prompt = instruction + context
    response = model.request(prompt)
    return response


model = LLM_Model('llama3.1-8b')


def process_document(doc):
    response = extract(model, doc)
    try:
        ob = json.loads(response)
        return {'type': 'success', 'data': {'text': doc, 'triplet': ob}}
    except Exception as e:
        print(e)
        return {'type': 'error', 'data': {'text': doc, 'result': response}}


data = load_passages("/mnt/workspace/dataset/HotpotQA/corpus.jsonl")
documents = [Document(text=t['text'], metadata={'title': t['title']}) for t in data]
parser = SentenceSplitter(chunk_size=512, chunk_overlap=20)
nodes = parser.get_nodes_from_documents(documents, show_progress=True)
print("documents: ", len(documents))
print("node: ", len(nodes))
docs = [node.text for node in nodes]
print("doc: ", len(docs))

success = []
error = []

with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
    futures = {executor.submit(process_document, doc): doc for doc in docs}
    for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
        result = future.result()
        if result['type'] == 'success':
            success.append(result['data'])
        else:
            error.append(result['data'])

with open('./extract_triplet_success.json', 'w', encoding='utf-8') as f:
    json.dump(success, f)
with open('./extract_triplet_error.json', 'w', encoding='utf-8') as f:
    json.dump(error, f)