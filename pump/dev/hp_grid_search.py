import os
os.chdir("/home/ubuntu/code/pump_post_midterm/pump")

import argparse
import json
import random
import pandas as pd
import torch
import torch.nn.functional as F
import time
import numpy as np
from transformers import AutoTokenizer, AutoModel
from src.utils import get_file_from_s3
from collections import Counter
from src.utils import TEST_KEY_MAPPING

random.seed(42)


def get_skew_personas_to_exclude(pvals, user_list, skew_thres):
    records = []
    for user in user_list:
        personas = pvals[user]
        entry = {'user': user}
        for p in personas:
            entry[p['name']] = p['inferred_value']

        records.append(entry)
    df = pd.DataFrame(records)

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
        if first[1] / second[1] > skew_thres:
            skew_cnt += 1
            skew_personas.append(col)
            print(col)
            print(Counter(df[col].values))
            print()

    print("Skewed Ratio: ", skew_cnt / len(df.columns))
    return skew_personas


def get_user_persona_repr(user, all_personas):
    return '; '.join(f"{p['name']}: {p['inferred_value']}" for p in all_personas[user])


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


def calc_gold_ratio(resp_df, similar_user_mapping, test_keys):
    res = []
    for key in test_keys:
        correct_cnt = 0
        total_cnt = 0
        for user in similar_user_mapping:
            similar_users = [int(_) for _ in similar_user_mapping[user]]
            row = resp_df.iloc[int(user)]
            gold = row[key]

            similar_df = resp_df[resp_df.index.isin(similar_users)]
            similar_answers = similar_df[key]
            most_freq = Counter(similar_answers).most_common(1)[0][0]
            if most_freq == gold:
                correct_cnt += 1
            total_cnt += 1
        res.append({
            "key": key,
            "correct_cnt": correct_cnt,
            "total_cnt": total_cnt,
            "ratio": correct_cnt / total_cnt
        })
    print(res)
    overall_ratio = sum([_['correct_cnt'] for _ in res]) / sum([_['total_cnt'] for _ in res])
    return overall_ratio


def calc_gold_ratio_personadb_surveys(resp_df, similar_user_mapping, user_test_q_key_mapping):
    res = []
    for user in similar_user_mapping:
        most_is_gold_cnt = 0
        correct_ratio = 0
        q_cnt = 0
        for key in user_test_q_key_mapping[user]:
            similar_users = [int(_) for _ in similar_user_mapping[user]]
            row = resp_df.iloc[int(user)]
            gold = row[key]

            similar_df = resp_df[resp_df.index.isin(similar_users)]
            similar_answers = similar_df[key]
            most_freq = Counter(similar_answers).most_common(1)[0][0]
            if most_freq == gold:
                most_is_gold_cnt += 1
            correct_ratio += Counter(similar_answers).most_common(1)[0][1] / len(similar_users)
            q_cnt += 1
        res.append({
            "key": key,
            "most_is_gold_cnt": most_is_gold_cnt,
            "q_cnt": q_cnt,
            "gold_concentration": correct_ratio
        })
    print(res)
    most_is_gold_ratio = sum([_['most_is_gold_cnt'] for _ in res]) / sum([_['q_cnt'] for _ in res])
    gold_concentration = sum([_['gold_concentration'] for _ in res]) / sum([_['q_cnt'] for _ in res])
    return most_is_gold_ratio, gold_concentration


def get_gold_ratio(args, setting, skew_thres, model, using_personadb_surveys=False):
    # 
    # Initialization 
    # 
    use_demo = setting == 'demo_only' or setting == 'demo_persona'
    use_persona = setting == 'persona_only' or setting == 'demo_persona'

    with open(args.persona_val_path, 'r') as f:
        p_vals = json.load(f)

    resp_df = pd.read_csv(get_file_from_s3(f"human_resp/{args.survey_name}/responses.csv"))
    
    if using_personadb_surveys:
        with open(f'experiment/data/human_resp/{args.survey_name}/user_test_q_key_mapping.json', 'r') as f:
            user_test_q_key_mapping = json.load(f)
        test_user_list = list(user_test_q_key_mapping.keys())
        train_user_list = [_ for _ in p_vals.keys() if _ not in test_user_list]
    else:
        test_user_idx = random.choices(range(len(resp_df)), k=int(len(resp_df)*0.1))
        test_resp_df = resp_df.iloc[test_user_idx]
        test_len = int(len(resp_df)*0.1)

        test_user_list = list(p_vals.keys())[:test_len]
        train_user_list = list(p_vals.keys())[test_len:]

    print("Number of users:", len(list(p_vals.keys())))
    print('test users:', len(test_user_list))
    print('train users:', len(train_user_list))

    #
    # Set up the personas for users
    # 

    all_personas = {}

    if use_persona:
        skew_personas_to_exclude = get_skew_personas_to_exclude(p_vals, test_user_list + train_user_list, skew_thres)
        for user in p_vals:
            if user not in all_personas:
                all_personas[user] = []
            for p in p_vals[user]:
                if p['name'] in skew_personas_to_exclude:
                    continue
                all_personas[user].append(p)

    if use_demo:
        meta_df = pd.read_csv(get_file_from_s3(f"human_resp/{args.survey_name}/metadata.csv"))
        meta_keys = list(meta_df['key'])
        for user in p_vals:
            if user not in all_personas:
                all_personas[user] = []
            row = resp_df.iloc[int(user)]
            demo = row[meta_keys].to_dict()
            key_mappings = {
                'CREGION': 'Region Where The Participant Lives',
                'AGE': 'Age Range Of The Participant',
                'SEX': 'Gender Of The Participant',
                'EDUCATION': 'Educational Attainment Of The Participant',
                'CITIZEN': 'Citizenship Status Of The Participant',
                'MARITAL': 'Marital Status Of The Participant',
                'RELIG': 'Religious Affiliation Of The Participant',
                'RELIGATTEND': 'Frequency Of Religious Service Attendance',
                'POLPARTY': 'Political Party Affiliation',
                'INCOME': 'Annual Income Range',
                'POLIDEOLOGY': 'Political Ideology',
                'RACE': 'Racial Background'
            }
            demo = {key_mappings[k]: v for k, v in demo.items()}
            for k, v in demo.items():
                all_personas[user].append({
                    'name': k,
                    'inferred_value': v
                })

    # 
    # Set up scoring infra
    # 

    task_name_to_instruct = {"retrieve_user_personas": "Given a user persona description, retrieve similar user presona descriptions",}
    query_prefix = "Instruct: "+task_name_to_instruct["retrieve_user_personas"]+"\nQuery: "
    queries = [get_user_persona_repr(user, all_personas) for user in test_user_list]
    passage_prefix = ""
    passages = [get_user_persona_repr(user, all_personas) for user in train_user_list]

    
    max_length = 4096
    batch_size=2

    tic = time.time()
    query_embeddings = model._do_encode(queries, batch_size=batch_size, instruction=query_prefix, max_length=max_length, num_workers=32, return_numpy=True)
    toc = time.time()
    print('Test encoding time:', toc-tic)

    passage_embeddings = model._do_encode(passages, batch_size=batch_size, instruction=passage_prefix, max_length=max_length, num_workers=32, return_numpy=True)
    toc = time.time()
    print('Total encoding time:', toc-tic)

    query_embeddings = F.normalize(torch.tensor(query_embeddings), p=2, dim=1)
    passage_embeddings = F.normalize(torch.tensor(passage_embeddings), p=2, dim=1)

    scores = (query_embeddings @ passage_embeddings.T) * 100

    out = []
    for top_k in list(range(40, 100, 20)) + list(range(100, 400, 50)):
        similar_user_mapping = get_top_k_similar_users(scores, test_user_list, train_user_list, top_k)
        # from IPython import embed; embed()
        if using_personadb_surveys:
            most_is_gold_ratio, gold_concentration = calc_gold_ratio_personadb_surveys(resp_df, similar_user_mapping, user_test_q_key_mapping)
            out.append({
                "setting": setting,
                "skew_thres": skew_thres,
                "top_k": top_k,
                "most_is_gold_ratio": most_is_gold_ratio,
                "gold_concentration": gold_concentration,
                "similar_user_mapping": similar_user_mapping
            })
        else:
            result = calc_gold_ratio(resp_df, similar_user_mapping, TEST_KEY_MAPPING[args.survey_name])
            out.append({
                "setting": setting,
                "skew_thres": skew_thres,
                "top_k": top_k,
                "result": result,
                "similar_user_mapping": similar_user_mapping
            })

    return out


def main(args):
    all_grids = []
    model = AutoModel.from_pretrained('nvidia/NV-Embed-v2', trust_remote_code=True, device_map='auto')
    for setting in ['demo_persona', 'persona_only', 'demo_only']:
        for skew_thres in [10, 5, 3]:
            out = get_gold_ratio(args, setting, skew_thres, model, using_personadb_surveys=args.using_personadb_surveys)
            all_grids += out

            for entry in out:
                print(entry['setting'])
                print(entry['skew_thres'])
                print(entry['top_k'])
                print(entry['result'])
                print()
            
            with open('hp_grid_search_personadb_surveys.json', 'w') as f:
                json.dump(all_grids, f, indent=4)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # args = parser.parse_args([])
    # args.persona_val_path = "opinions_qa/persona_val/American_Trends_Panel_W34/date0831_personas_full_haiku_known_test.json"
    # args.survey_name = "American_Trends_Panel_W34"

    parser.add_argument('--persona_val_path', type=str)
    parser.add_argument('--survey_name', type=str)
    parser.add_argument('--using_personadb_surveys', action='store_true')
    args = parser.parse_args()

    main(args)

