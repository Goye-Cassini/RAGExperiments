import os
import json
import re
from tqdm import tqdm
from llm.openai_llm import LLM_Model
from pymilvus import (connections, utility, Collection)
from milvus_model.hybrid import BGEM3EmbeddingFunction
from pymilvus import (
    AnnSearchRequest,
    WeightedRanker,
)

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


def generate_response(model, query):
    try:
        response = model.request(query)
    except Exception as e:
        print(e)
        response = None
    return response


model = LLM_Model('llama3.1-8b')

# For following context, extract the relevant triplets (subject, predicate, object) related to the given question. If no relevant triplets are found, return "No".
# =================================
# Example 1:
# Question: Coko and Justine Frischmann both enjoyed what music profession?
# Context: Arular is the debut studio album by English recording artist M.I.A.. It was released on 22 March 2005 in the United States, and one month later in the United Kingdom, with a slightly different track listing. In 2004, the album's release was preceded by two singles and a mixtape. M.I.A. wrote or co-wrote all the songs on the album and created the basic backing tracks using a Roland MC-505 sequencer/drum machine given to her by long-time friend Justine Frischmann. Collaborators included Switch, Diplo, Richard X, Ant Whiting and Greg \"Wizard\" Fleming. The album's title is the political code name used by her father, Arul Pragasam, during his involvement with Sri Lankan Tamil militant groups, and themes of conflict and revolution feature heavily in the lyrics and artwork. Musically, the album incorporates styles that range from hip hop and electroclash to funk carioca and punk rock.
# Output: No
# ==================================
# Example 2:
# Question: Who wrote the tragic play which also has a 1968 film with a soundtrack by Nino Rota?
# Context: Romeo and Juliet (1968 film soundtrack): The soundtrack for the 1968 film "Romeo and Juliet" was composed and conducted by Nino Rota.  It was originally released as a vinyl record, containing nine entries, most notably the song "What Is a Youth", composed by Nino Rota, written by Eugene Walter and performed by Glen Weston.  The music score won a Silver Ribbon award of the Italian National Syndicate of Film Journalists in 1968 and was nominated for two other awards (BAFTA Award for Best Film Music in 1968 and Golden Globe Award for Best Original Score in 1969).
# Output: (Nino Rota, composed soundtrack for, 1968 film "Romeo and Juliet")
# ================================

prompt = '''For following context, extract the relevant triplets (subject, predicate, object) related to the given question. 
1.Focus on key entities (people, groups, albums, etc.) mentioned in the question. 
2.Look for sentences in the context that mention these entities or provide relevant details about them. 
3.For each related sentence, extract a triplet in the form (subject, predicate, object) that captures relationships involving the entities.
4.If no relevant triplet involving the entities is found, return "No.". 
5.Make sure each extracted triplet clearly shows a relationship between entities in the context.
6.Returns results in the format of (subject, predicate, object)
Question: {question}
Context: {context}
Output:
'''

prompt2 = '''Use the provided triplets to infer and answer the following question based on the given context with one or few words.
Question: {question}
Triplets: {triplets}
'''

contexts = []


triplets = []
question = '2014 S/S is the debut album of a South Korean boy group that was formed by who?'

ef = BGEM3EmbeddingFunction(model_name="/mnt/workspace/huggingface_models/BAAI/bge-m3", use_fp16=False, device="cpu")
rewrite = rewrite_query(model, question)
print('Rewrite: ', rewrite)
query_embeddings = ef([str(rewrite)+question])
connections.connect(uri="./db/milvus.db")
print(utility.list_collections())
# hotpotqa256_collection, hotpotqa_collection_COSINE, hotpotqa_collection
col_name = "hotpotqa_collection_COSINE"
col = Collection(col_name, consistency_level="Strong")
print("Number of entities inserted:", col.num_entities)

contexts = hybrid_search(
    col,
    query_embeddings["dense"][0],
    query_embeddings["sparse"][0:1],
    sparse_weight=1,
    dense_weight=0.7,
)
contexts.append(rewrite)
triplets = []
contextstr = '\n'.join(contexts)
# for context in contexts:
query = prompt.format(question=question, context=contextstr)
print(query)
response = generate_response(model, query)
print('Response: ', response)
print("=======================================================")
pattern = r'\(([^()]*?(?:\([^()]*\)[^()]*)*?[^()]*?), ([^()]*?(?:\([^()]*\)[^()]*)*?[^()]*?), ([^()]*?(?:\([^()]*\)[^()]*)*?[^()]*?)\)'
matches = re.findall(pattern, response)
triplets.extend(matches)
    
unique_triplets = set(triplets)
result_string = ', '.join(str(t) for t in unique_triplets)
query = prompt2.format(triplets=result_string, question=question)
print(query)
response = generate_response(model, query)
print("Response: ", response)
