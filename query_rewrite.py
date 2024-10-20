import os
import json
import csv
import re
from tqdm import tqdm
from llm.openai_llm import LLM_Model
from pymilvus import (connections, utility, Collection)
from milvus_model.hybrid import BGEM3EmbeddingFunction
from pymilvus import (
    AnnSearchRequest,
    WeightedRanker,
)


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
    contextarry = '\n'.join(contexts)
    prompt = instruction.format(context=f"[{contextarry}]", question=question)
    
    try:
        response = model.request(prompt)
    except Exception as e:
        print(e)
        response = None
    return response


def rewrite_query(model, question):
    instruction = "Please write a passage to answer the question. Question: "
    try:
        response = model.request(instruction+question)
    except Exception as e:
        print(e)
        response = None
    return response


def triplets_extract(model, question, contexts):
    prompt = '''For following context, extract the relevant triplets (subject, predicate, object) related to the given question. 
    1.Focus on key entities (people, groups, albums, etc.) mentioned in the question. 
    2.Look for sentences in the context that mention these entities or provide relevant details about them. 
    3.For each related sentence, extract a triplet in the form (subject, predicate, object) that captures relationships involving the entities.
    4.If no relevant triplet involving the entities is found, return "No.". 
    5.Make sure each extracted triplet clearly shows a relationship between entities in the context.
    6.Returns results in the format of (subject, predicate, object)
    Question: {question}
    Context: {context}
    Output:'''
    contextstr = '\n'.join(contexts)
    query = prompt.format(question=question, context=contextstr)
    triplets = []
    try:
        response = model.request(query)
        pattern = r'\(([^()]*?(?:\([^()]*\)[^()]*)*?[^()]*?), ([^()]*?(?:\([^()]*\)[^()]*)*?[^()]*?), ([^()]*?(?:\([^()]*\)[^()]*)*?[^()]*?)\)'
        matches = re.findall(pattern, response)
        triplets.extend(matches)
    except Exception as e:
        print(e)
    return triplets


def triplets_reasoning(model, question, triplets):
    prompt = '''Use the provided triplets to infer and answer the following question with one or few words.
    Question: {question}
    Triplets: {triplets}
    '''
    unique_triplets = set(triplets)
    result_string = ', '.join(str(t) for t in unique_triplets)
    query = prompt.format(triplets=result_string, question=question)
    try:
        response = model.request(query)
    except Exception as e:
        print(e)
        response = ''
    return response


model = LLM_Model('llama3.1-8b')

ef = BGEM3EmbeddingFunction(model_name="/mnt/workspace/huggingface_models/BAAI/bge-m3", use_fp16=False, device="cpu")
dense_dim = ef.dim["dense"]

connections.connect(uri="./db/milvus.db")
print(utility.list_collections())

# hotpotqa256_collection, hotpotqa_collection_COSINE, hotpotqa_collection
col_name = "hotpotqa_collection_COSINE"
col = Collection(col_name, consistency_level="Strong")
col.load()
print("Number of entities inserted:", col.num_entities)

with open('/mnt/workspace/dataset/HotpotQA/dev.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

results = []
for index, item in tqdm(enumerate(data), total=len(data)):
    rewrite = rewrite_query(model, item['question'])
    # query = None
    query = str(rewrite) + item['question']
    if query is not None:
        query_embeddings = ef([query])
    else:
        query_embeddings = ef([item['question']])
    contexts = hybrid_search(
        col,
        query_embeddings["dense"][0],
        query_embeddings["sparse"][0:1],
        sparse_weight=1,
        dense_weight=0.7,
    )
    contexts.append(rewrite)
    triplets = triplets_extract(model, item['question'], contexts)
    # response = genrate_response(model, item['question'], contexts)
    response = triplets_reasoning(model, item['question'], triplets)
    results.append({'question': item['question'], 'answer': item['answer'], 'prediction': response, 'level': item['level']})
    if index < 10:
        print(f"Question: {item['question']}\nRewrite: {query}\nResponse:{response}")

with open('./dev_prediction_hybrid_triplets_d7s1.json', 'w', encoding='utf-8') as f:
    json.dump(results, f)