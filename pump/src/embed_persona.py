# args

# persona_val_path = "opinions_qa/persona_val/American_Trends_Panel_W34/date0828_personas_American_Trends_Panel_W34_testonly_haiku.json"
persona_val_path = "opinions_qa/persona_val/American_Trends_Panel_W34/date0831_personas_full_haiku_known_test.json"
user_mapping_filepath = "opinions_qa/similar_users/American_Trends_Panel_W34/date0831_personas_full_haiku_known_test_train200_top20.json"
top_k = 20

# import os
# os.chdir("/home/ubuntu/code/pump_post_midterm/pump")

from transformers import AutoTokenizer, AutoModel
import json


with open(persona_val_path, 'r') as f:
    p_vals = json.load(f)
print(len(list(p_vals.keys())))


import random
import pandas as pd
from src.utils import get_file_from_s3
random.seed(42)

file_key = "American_Trends_Panel_W34"
resp_df = pd.read_csv(get_file_from_s3(f"human_resp/{file_key}/responses.csv"))
test_user_idx = random.choices(range(len(resp_df)), k=int(len(resp_df)*0.1))
test_resp_df = resp_df.iloc[test_user_idx]
test_len = int(len(resp_df)*0.1)

test_user_list = list(p_vals.keys())[:test_len]
train_user_list = list(p_vals.keys())[test_len:][:200]

print('test:', len(test_user_list))
print('train:', len(train_user_list))


def get_user_pval_df(pvals, user_list):
    records = []
    for user in user_list:
        personas = pvals[user]
        entry = {'user': user}
        for p in personas:
            entry[p['name']] = p['inferred_value']
        records.append(entry)

    df = pd.DataFrame(records)
    return df


df = get_user_pval_df(p_vals, train_user_list)
df[:3]



from collections import Counter

skew_cnt = 0
skew_personas = []
for col in df.columns:
    if col == 'user': continue
    cnt = Counter(df[col].values)
    if len(cnt.keys()) < 2: 
        skew_cnt += 1
        skew_personas.append(col) 
        print(col)
        print(Counter(df[col].values))
        print()
        continue
    first, second = cnt.most_common(2)
    if first[1] / second[1] > 10:
        skew_cnt += 1
        skew_personas.append(col) 
        print(col)
        print(Counter(df[col].values))
        print()
print(skew_cnt / len(df.columns))


su = test_user_list[0]

def get_user_persona_repr(personas):
    return '; '.join(f"{p['inferred_value']}" for p in personas if p['name'] not in skew_personas)

get_user_persona_repr(p_vals[su])



import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import time

# Each query needs to be accompanied by an corresponding instruction describing the task.
task_name_to_instruct = {"retrieve_user_personas": "Given a user persona description, retrieve similar user presona descriptions",}

query_prefix = "Instruct: "+task_name_to_instruct["retrieve_user_personas"]+"\nQuery: "
queries = [
    get_user_persona_repr(p_vals[user]) for user in test_user_list
]

# No instruction needed for retrieval passages
passage_prefix = ""
passages = [
    get_user_persona_repr(p_vals[user]) for user in train_user_list[:200]
]

# load model with tokenizer
model = AutoModel.from_pretrained('nvidia/NV-Embed-v2', trust_remote_code=True, device_map='auto')

# get the embeddings
max_length = 4096
tic = time.time()
# query_embeddings = model.encode(queries, instruction=query_prefix, max_length=max_length)
# passage_embeddings = model.encode(passages, instruction=passage_prefix, max_length=max_length)
batch_size=2
query_embeddings = model._do_encode(queries, batch_size=batch_size, instruction=query_prefix, max_length=max_length, num_workers=32, return_numpy=True)
toc = time.time()
print('Test encoding time:', toc-tic)

passage_embeddings = model._do_encode(passages, batch_size=batch_size, instruction=passage_prefix, max_length=max_length, num_workers=32, return_numpy=True)
toc = time.time()
print('Total encoding time:', toc-tic)

# normalize embeddings
# query_embeddings = F.normalize(query_embeddings, p=2, dim=1)
# passage_embeddings = F.normalize(passage_embeddings, p=2, dim=1)
query_embeddings = F.normalize(torch.tensor(query_embeddings), p=2, dim=1)
passage_embeddings = F.normalize(torch.tensor(passage_embeddings), p=2, dim=1)

scores = (query_embeddings @ passage_embeddings.T) * 100
# print(scores.tolist())
# [[87.42692565917969, 0.462837278842926], [0.9652643203735352, 86.0372314453125]]


import numpy as np

def get_top_k_similar_users(scores_matrix, query_user_names, corpus_user_names, k):
    top_k_similar_users = {}
    scores_matrix = scores_matrix.detach().cpu().numpy()

    for i, curr_scores in enumerate(scores_matrix):
        # Get the indices of the top k scores
        top_k_indices = np.argsort(curr_scores)[-k:][::-1]
        
        # Map indices to corresponding user names in the corpus_user_names list
        top_k_users = [corpus_user_names[idx] for idx in top_k_indices]
        
        # Map the query user name to the list of top k similar users
        top_k_similar_users[query_user_names[i]] = top_k_users

    return top_k_similar_users


result = get_top_k_similar_users(scores, test_user_list, train_user_list, top_k)

with open(user_mapping_filepath, 'w') as f:
    json.dump(result, f, indent=4)
