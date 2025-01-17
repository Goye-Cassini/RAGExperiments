import os
import json
import csv
from tqdm import tqdm
from llm.openai_llm import LLM_Model
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


def genrate_response(model, question, contexts):
    # instruction = '''Read the provided context carefully and answer the following question in a simple and direct manner. If the context does not provide enough information to answer the question, respond only with 'No Answer.'.
    # Context: {context}
    # Question: {question}
    # '''
    instruction = '''Answer the following question based on the given context with one or few words.
    Question: {question}
    Context: {context}
    '''
    context_array = ",\n".join(contexts)
    prompt = instruction.format(context=f"[{context_array}]", question=question)
    try:
        response = model.request(prompt)
    except Exception as e:
        response = 'Error!'
    return response

model = LLM_Model('llama3.1-8b')
connections.connect(uri="./db/milvus.db")
# hotpotqa256_collection, hotpotqa_collection_COSINE, hotpotqa_collection
col_name = "hotpotqa256_collection"


# 构建索引
# data = load_passages("/mnt/workspace/dataset/HotpotQA/corpus.jsonl")
# documents = [Document(text=t['text'], metadata={'title': t['title']}) for t in data]
# parser = SentenceSplitter(chunk_size=512, chunk_overlap=20)
# nodes = parser.get_nodes_from_documents(documents, show_progress=True)
# print("documents: ", len(documents))
# print("node: ", len(nodes))
# docs = [node.text for node in nodes]
# print("doc: ", len(docs))

# ef = BGEM3EmbeddingFunction(model_name="/mnt/workspace/huggingface_models/BAAI/bge-m3", use_fp16=False)
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

# fields = [
#     FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
#     FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=3072),
#     FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),
#     FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=dense_dim),
#     FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=256),
# ]
# schema = CollectionSchema(fields)
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
# print("Number of entities inserted:", col.num_entities)


# answer
ef = BGEM3EmbeddingFunction(model_name="/mnt/workspace/huggingface_models/BAAI/bge-m3", use_fp16=False, device="cpu")
col = Collection(col_name, consistency_level="Strong")
with open('/mnt/workspace/dataset/HotpotQA/dev.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
results = []
for item in tqdm(data):
    query_embeddings = ef([item['question']])
    # contexts = dense_search(col, query_embeddings["dense"][0])
    # contexts = sparse_search(col, query_embeddings["sparse"][0:1])
    contexts = hybrid_search(
        col,
        query_embeddings["dense"][0],
        query_embeddings["sparse"][0:1],
        sparse_weight=1.0,
        dense_weight=0.7,
    )
    response = genrate_response(model, item['question'], contexts)
    results.append({'question': item['question'], 'answer': item['answer'], 'prediction': response, 'level': item['level']})

with open('./dev_prediction_hybrid_256.json', 'w', encoding='utf-8') as f:
    json.dump(results, f)