import os
import json
import csv
from tqdm import tqdm
from openai import OpenAI
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from pymilvus import (connections, utility, FieldSchema, CollectionSchema, DataType, Collection)
from milvus_model.hybrid import BGEM3EmbeddingFunction
from pymilvus import (
    AnnSearchRequest,
    WeightedRanker,
)
from concurrent.futures import ThreadPoolExecutor, as_completed
import concurrent.futures
from IPython.display import Markdown, display


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


class LLM_Model():
    def __init__(self, model_name, temperature=1.0, max_new_tokens=2000, top_p=1.0):
        self.modelName = model_name
        self.params = {
            "temperature": temperature,
            "max_new_tokens": max_new_tokens,
            "top_p": top_p
        }
    
    def request(self, query: str) -> str:
        client = OpenAI(
            base_url="http://localhost:8000/v1",
            api_key="token-abc123", # 随便填写，只是为了通过接口参数校验
        )

        completion = client.chat.completions.create(
          model=self.modelName,
          messages=[
            {"role": "user", "content": query}
          ],
          # temperature=self.params['temperature'],
          # max_tokens=self.params['max_new_tokens'],
          # top_p=self.params['top_p']
        )

        return completion.choices[0].message.content
    

def genrate_response(model, question, contexts):
    instruction = '''Read the provided context carefully and answer the following question in a simple and direct manner. If the context does not provide enough information to answer the question, respond only with 'No Answer.'.
    Context: {context}
    Question: {question}
    '''
    context_array = ",\n".join(contexts)
    prompt = instruction.format(context=f"[{context_array}]", question=question)
    try:
        response = model.request(prompt)
    except Exception as e:
        response = 'Error!'
    return response


data = load_passages("/mnt/workspace/dataset/HotpotQA/corpus.jsonl")
documents = [Document(text=t['text'], metadata={'title': t['title']}) for t in data]
parser = SentenceSplitter(chunk_size=512, chunk_overlap=20)
nodes = parser.get_nodes_from_documents(documents, show_progress=True)
print("documents: ", len(documents))
print("node: ", len(nodes))
docs = [node.text for node in nodes]
print("doc: ", len(docs))


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

# ef = BGEM3EmbeddingFunction(model_name="/mnt/workspace/huggingface_models/BAAI/bge-m3", use_fp16=False)
# ef = BGEM3EmbeddingFunction(model_name="/mnt/workspace/huggingface_models/BAAI/bge-m3", use_fp16=False, device="cpu")
# dense_dim = ef.dim["dense"]
# docs_embeddings = ef(docs)
# milvus_data = []
# for i in range(len(docs)):
#     milvus_data.append({
#         "id": i,
#         "dense_vector": docs_embeddings["dense"][i], 
#         "sparse_vector": docs_embeddings["sparse"][i:i + 1],
#         "text": docs[i],
#         "title": nodes[i].metadata['title']
#     })
# print("milvus data: ", len(milvus_data))

# connections.connect(uri="./db/milvus.db")
# fields = [
#     FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
#     FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=3072),
#     FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),
#     FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=dense_dim),
#     FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=256),
# ]
# schema = CollectionSchema(fields)
# col_name = "hotpotqa256_collection"
# if utility.has_collection(col_name):
#     Collection(col_name).drop()
# col = Collection(col_name, schema, consistency_level="Strong")
# sparse_index = {"index_type": "SPARSE_INVERTED_INDEX", "metric_type": "IP"}
# col.create_index("sparse_vector", sparse_index)
# dense_index = {"index_type": "AUTOINDEX", "metric_type": "COSINE"}
# col.create_index("dense_vector", dense_index)
# col.load()
# for idx in range(0, len(milvus_data), 50):
#     col.insert(milvus_data[idx : idx + 50])
# col = Collection(col_name, consistency_level="Strong")
# print("Number of entities inserted:", col.num_entities)

# 搜索测试

def dense_search(col, query_dense_embedding, limit=10):
    search_params = {"metric_type": "COSINE", "params": {}}
    res = col.search(
        [query_dense_embedding],
        anns_field="dense_vector",
        limit=limit,
        output_fields=["text"],
        param=search_params,
    )[0]
    return [hit.get("text") for hit in res]


def sparse_search(col, query_sparse_embedding, limit=10):
    search_params = {
        "metric_type": "IP",
        "params": {},
    }
    res = col.search(
        [query_sparse_embedding],
        anns_field="sparse_vector",
        limit=limit,
        output_fields=["text"],
        param=search_params,
    )[0]
    return [hit.get("text") for hit in res]


def hybrid_search(
    col,
    query_dense_embedding,
    query_sparse_embedding,
    sparse_weight=1.0,
    dense_weight=1.0,
    limit=10,
):
    dense_search_params = {"metric_type": "COSINE", "params": {}}
    dense_req = AnnSearchRequest(
        [query_dense_embedding], "dense_vector", dense_search_params, limit=limit
    )
    sparse_search_params = {"metric_type": "IP", "params": {}}
    sparse_req = AnnSearchRequest(
        [query_sparse_embedding], "sparse_vector", sparse_search_params, limit=limit
    )
    rerank = WeightedRanker(sparse_weight, dense_weight)
    res = col.hybrid_search(
        [sparse_req, dense_req], rerank=rerank, limit=limit, output_fields=["text"]
    )[0]
    return [hit.get("text") for hit in res]


question = "Who wrote the tragic play which also has a 1968 film with a soundtrack by Nino Rota?"
# query_embeddings = ef([question])
# dense_results = dense_search(col, query_embeddings["dense"][0])
# sparse_results = sparse_search(col, query_embeddings["sparse"][0:1])
# hybrid_results = hybrid_search(
#     col,
#     query_embeddings["dense"][0],
#     query_embeddings["sparse"][0:1],
#     sparse_weight=0.7,
#     dense_weight=1.0,
# )

# Dense search results
# print("====Dense Search Results====")
# for item in dense_results:
#     print(item)

# # Sparse search results
# print("====Sparse Search Results====")
# for item in sparse_results:
#     print(item)

# # Hybrid search results
# print("====Hybrid Search Results====")
# for item in hybrid_results:
#     print(item)


# model = LLM_Model('llama3.1-8b')

# response = genrate_response(model, question, dense_results)
# print(response)

# with open('/mnt/workspace/dataset/HotpotQA/dev.json', 'r', encoding='utf-8') as f:
#     data = json.load(f)

# results = []
# for item in tqdm(data):
#     query_embeddings = ef([item['question']])
#     contexts = hybrid_search(
#         col,
#         query_embeddings["dense"][0],
#         query_embeddings["sparse"][0:1],
#         sparse_weight=1.0,
#         dense_weight=0.7,
#     )
#     response = genrate_response(model, item['question'], contexts)
#     results.append({'question': item['question'], 'answer': item['answer'], 'prediction': response, 'level': item['level']})

# with open('./dev_prediction_hybrid_256.json', 'w', encoding='utf-8') as f:
#     json.dump(results, f)