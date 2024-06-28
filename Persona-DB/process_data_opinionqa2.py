import chunk
import random
import time

import numpy as np
import requests
import os
import json

from tqdm import tqdm

from utils.utils import *
# from constants import *
from collections import defaultdict, Counter
# import networkx as nx
import preprocessor as p
from llm_api import chatbot
# from statsmodels.stats.inter_rater import fleiss_kappa
from copy import deepcopy
from transformers import GPT2Tokenizer
from copy import deepcopy
from sklearn.metrics import precision_recall_fscore_support
import pandas as pd
import json
import os
from utils.data_utils import process_news_profile_history, get_graph_dataset, GlobalIdMap, get_user_profile, get_data_slice, process_triplet, UserRetriever, get_user_profile3

import ast
from collections import defaultdict


random.seed(0)
np.random.seed(0)

parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default="")
parser.add_argument('--host', type=str, default="")
parser.add_argument('--port', type=int, default=1)
parser.add_argument("--num_nodes", type=int, default=1)
parser.add_argument("--node_id", type=int, default=0)
parser.add_argument("--cur_prompt_type", type=str, default="extraction_refine")
parser.add_argument("--save_to_userfile", type=int, default=0)
parser.add_argument("--entire_data", type=int, default=1)
parser.add_argument("--only_do_remaining", type=int, default=0)
parser.add_argument("--record_meta", type=int, default=0)
parser.add_argument('--dataset', type=str, default="34")
args = parser.parse_args()




data_dir="data/"
CNN_dir = f"{data_dir}OpinionQA/human_resp/American_Trends_Panel_W{args.dataset}/"
CNN_test_anno_filename = f"{CNN_dir}response_data_balanced_test_anno_1000.json"

CNN_post_filename = f"{CNN_dir}post_data.json"
CNN_users_info_rf_filename = f"{CNN_dir}users_info_dict_with_chatgptanno_rf.json"
CNN_users_info_sc_filename=CNN_users_info_rf_filename
CNN_users_history_filename = f"{CNN_dir}users_history_dict.json"
CNN_prediction_dir=f"{CNN_dir}predictions/"
CNN_results_dir=f"{CNN_dir}results/"
CNN_prediction_prompt_dir=f"{CNN_dir}prediction_prompts/"
CNN_api_embeddings_dir=f"{CNN_dir}api_embeddings/"
CNN_r_anno_json_dir=f""
CNN_rf_anno_json_dir=f""
CNN_non_empty_hist_uids_filename=f""
CNN_non_empty_hist_uids_processed_filename=f""


"""==========================================meta =============================================="""

# history_data = load_file(CNN_users_history_filename)
gpt_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
test_data_anno=load_file(CNN_test_anno_filename)
valid_uids_test = set()
for data in [test_data_anno]:
    for item in data:
        valid_uids_test.add(str(item["author_id"]))

history_data = load_file(CNN_users_history_filename)
# cur_extraction_prompt_type = "extraction_refine"
# cur_type = f"chatgptanno_r"  # r is updated version of structurization
# user_data = load_file(CNN_users_info_sc_filename)
# cur_extraction_prompt_type = args.cur_prompt_type
# cur_type = f"chatgptanno_rf" if cur_extraction_prompt_type == "refine" else f"chatgptanno_r"
# load_file(CNN_users_info_r_filename) if cur_extraction_prompt_type == "refine" else
user_data = load_file(CNN_users_info_sc_filename)
new_user_data = deepcopy(user_data)

# cur_dir = f"{CNN_dir}{cur_type}/"
# mkdir(cur_dir)
# cur_data=user_data
# top_100_longest_history_users = load_file(f"{CNN_dir}/case_study_dict.json")["longest_hist_100"]
# cur_data = top_100_longest_history_users



"""=====get user ids====="""
"""
# Here we use preprocessed neighbor retrieval to get user ids to reduce api cost
# This can be run after
# python llm_predict_opinionqa.py \
#      --prompt_types "prediction_with_prompt_hists_ret_collab" \
#      --cases "" \
#      --neighbor_topk <max neighbors in your task> \
#      --dataset $dataset \
#      --include_rf 0
"""

uid_to_excluded_post_ids=defaultdict(set)
for i, sample in enumerate(test_data_anno):
    uid_to_excluded_post_ids[str(sample["author_id"])].add(sample["conversation_id"])

cur_data=valid_uids_test
neib_dict={}
for file in os.listdir(f"{CNN_api_embeddings_dir}user_meta_key_retrieval_uids_top100_dp0"):
    data=load_file(f"{CNN_api_embeddings_dir}user_meta_key_retrieval_uids_top100_dp0/{file}")
    # cur_data.update(data[:4])
    # cur_data.add(file.split(".")[0])
    neib_dict[file.split(".")[0]]=data[:4]
for id in deepcopy(cur_data):
    if id in neib_dict:
        cur_data.update(neib_dict[id])
    else:
        print("id not in neib_dict",id)
cur_data = sorted(cur_data)
cur_data = batchify(cur_data, math.ceil(len(cur_data) / args.num_nodes))[args.node_id]

"""==========================================Get GPT annotation extraction =============================================="""

hist_size_file = "history_size_in_annotation.json"
history_size_in_annotation = load_file(f"{CNN_dir}{hist_size_file}", init={"chatgptanno_rf": {}, "chatgptanno_r": {}})

for cur_extraction_prompt_type in ["refine"]: #["extraction_refine", "refine"]
    cur_type = f"chatgptanno_rf" if cur_extraction_prompt_type == "refine" else f"chatgptanno_r"
    cur_dir = f"{CNN_dir}{cur_type}/"
    mkdir(cur_dir)

    print(f"\n\nStarting {cur_type}\n\n")

    for k in tqdm(cur_data):
        v = user_data[k]
        # preserve for caching
        """========Existing, then load========"""
        uanno_fname = f"{cur_dir}/{k}.txt"
        if path_exists(uanno_fname):
            new_user_data[k][cur_type] = load_file(uanno_fname)
            if not new_user_data[k][cur_type].strip():
                print(f"\nempty {cur_type} for", uanno_fname)
            continue

        cur_top_n_hist = 100

        # user_desc, history_text, history_posts, history_tweet_ids = get_user_profile2(user_id, user_data, history_data, top_n_hist=cur_top_n_hist, post_id=post_id)

        user_desc, history_text, history_posts, history_tweet_ids = get_user_profile3(k, user_data, history_data, top_n_hist=cur_top_n_hist, uid_to_excluded_post_ids=uid_to_excluded_post_ids)

        """========Loading for rf""========"""
        cur_key = ""
        # if cur_type == "chatgptanno_rf":
        #     uanno_fname_r = f"{CNN_dir}chatgptanno_r/{k}.txt"
        #     if not path_exists(uanno_fname_r):
        #         print(f"not exist uanno_fname_r when loading for rf", uanno_fname_r)
        #         print("[User Desc]\n", user_desc, "\n")
        #         print("\n[History]\n", history_text, "\n")
        #         continue
        #     cur_key = load_file(uanno_fname_r)

        cur_prompt = get_prompt(history=history_text, post=None, profile=user_desc, key=cur_key, nbs=None, sc=None, prompt_type=cur_extraction_prompt_type, prompts_dir="prompts/OpinionQA/")


        cur_length = len(gpt_tokenizer.encode(cur_prompt))
        while cur_length > 3600:
            print("\ncur_length", cur_length)

            cur_top_n_hist -= 1  # //=2 #
            # user_desc, history_text, history_posts, history_tweet_ids = get_user_profile(k, user_data=user_data, history_data=history_data, top_n_hist=cur_top_n_hist)

            user_desc, history_text, history_posts, history_tweet_ids = get_user_profile3(k, user_data, history_data, top_n_hist=cur_top_n_hist, uid_to_excluded_post_ids=uid_to_excluded_post_ids)
            cur_prompt = get_prompt(history=history_text, post=None, profile=user_desc, key=cur_key, nbs=None, sc=None, prompt_type=cur_extraction_prompt_type, prompts_dir="prompts/OpinionQA/")

            cur_length = len(gpt_tokenizer.encode(cur_prompt))
            # print("cur_top_n_hist", cur_top_n_hist)
        history_size_in_annotation[cur_type][k] = len(history_posts)

        # print("len(gpt_tokenizer.encode(cur_prompt))",len(gpt_tokenizer.encode(cur_prompt)))
        # uanno_fname = f"{cur_dir}/{k}.txt"
        # if not path_exists(uanno_fname):
        # print(f"not exist uanno_fname")
        # breakpoint()
        # print(f"starting {k}")
        cur_res = chatbot(cur_prompt) if history_text.strip() else ""
        if cur_res is not None and cur_res.strip():
            new_user_data[k][cur_type] = cur_res
            dump_file(cur_res, f"{cur_dir}/{k}.txt")
        else:
            print(f"user", k, "has empty user_desc or gptanno for", cur_type)
            print("[User Desc]\n", user_desc, "\n")
            print("\n[History]\n", history_text, "\n")
            continue
        print(f"Finished {k}")

    # if cur_extraction_prompt_type == "refine"
    #     dump_file(new_user_data, f"{CNN_dir}users_info_dict_with_chatgptanno_r.json")
    if args.save_to_userfile:
        dump_file(new_user_data, f"{CNN_dir}users_info_dict_with_chatgptanno_rf.json")
    dump_file(history_size_in_annotation, f"{CNN_dir}{hist_size_file}")
history_data = None
# exit()

"""====================================Clean GPT annotation using itself ===================================="""
print("\n\nCleaning\n\n")
target_uids = deepcopy(cur_data)  # set(load_file(f"{CNN_dir}/case_study_dict.json")["longest_hist_100"])
####### Below is: if uid not in longest_100_all: continue
##[f"{cur_uid}.json" for cur_uid in longest_100_all]
for cur_type in ["chatgptanno_rf"]:  # "chatgptanno_r", , "chatgptanno_rf"
    cur_dir, new_dir = f"{CNN_dir}{cur_type}/", f"{CNN_dir}{cur_type}_json/"
    mkdir(new_dir)
    print("cur_type", cur_type)

    # tmp = os.listdir(cur_dir)
    cur_data = [f"{cur_uid}.txt" for cur_uid in target_uids]
    print("len(cur_data)", len(cur_data))
    if args.only_do_remaining == 1: # for paraellization
        print("only_do_remaining")
        cur_data = [cur_file for cur_file in cur_data if not path_exists(f"{new_dir}/{cur_file.split('.')[0]}.json")]

    # cur_data = get_data_slice(tmp, args.num_nodes, args.node_id)
    # cur_data = tmp
    # print("len(cur_data)", len(cur_data))
    # loop through all users in cur_dir
    for k in tqdm(cur_data):
        uid = k.split(".")[0]
        new_res = {}
        # if uid not in target_uids: continue
        if not path_exists(f"{cur_dir}/{k}"):
            print("not existing", f"{cur_dir}/{k}")
        if path_exists(f"{new_dir}/{uid}.json") or not path_exists(f"{cur_dir}/{k}"):
            continue

        v = load_file(f"{cur_dir}/{k}")
        # print("\n=================\n")
        # print("uid", k)
        # print(v, "\n")
        resp = None
        res = {}
        cur_prompt = get_prompt(key=v, prompt_type="clean_anno", prompts_dir="prompts/OpinionQA/")

        try:
            resp = chatbot(cur_prompt)
            first_brace_position, last_brace_position = resp.find("{"), resp.rfind("}")
            res = resp[first_brace_position:last_brace_position + 1]
            res = json.loads(res)
        except:
            # print("cur_prompt\n", cur_prompt)
            try:
                resp = chatbot(cur_prompt, model="gpt-3.5-turbo-16k")  # sometimes 4k not responsive due to server
                first_brace_position, last_brace_position = resp.find("{"), resp.rfind("}")
                res = resp[first_brace_position:last_brace_position + 1]
                res = json.loads(res)
            except:
                try:
                    # print("try gpt-4-1106-preview")
                    resp = chatbot(cur_prompt, model="gpt-4-1106-preview")  # gpt-3.5-turbo-16k
                    first_brace_position, last_brace_position = resp.find("{"), resp.rfind("}")
                    res = resp[first_brace_position:last_brace_position + 1]
                    res = json.loads(res)
                except:
                    try:
                        # print("try gpt-4-turbo-2024-04-09")
                        resp = chatbot(cur_prompt, model="gpt-4-turbo-2024-04-09", seed=0)  # gpt-3.5-turbo-16k
                        first_brace_position, last_brace_position = resp.find("{"), resp.rfind("}")
                        res = resp[first_brace_position:last_brace_position + 1]
                        res = json.loads(res)
                    except:
                        print("\n=====e11 response=====")
                        print(resp)
                        continue
                        # breakpoint()
                        # exit()
        try:
            for key in res.keys():
                item = res[key]
                if (item is None or not item or isinstance(item, str) and not item.strip()) or item in [["None"], [None]]:
                    continue
                if ("interested issue" in key.lower() or "interested ent" in key.lower()) and isinstance(item, dict):
                    tmp = []
                    for subk, subv in item.items():
                        if subv is not None and not isinstance(subv, str) and not isinstance(subv, list):
                            print("subv1", subv)
                            print("uid", uid)
                            print("res", res)
                            continue
                        if isinstance(subv, list):
                            subv = "; ".join(subv)
                        if isinstance(subv, str) and not subv.strip(): subv = None
                        tmp.append(f"{subk}: {subv}".strip())
                        # tmp.append(f"{subk}: {subv}".strip() if not (subv is None and len(subk.split())>5) else subk.strip())
                    new_res[key] = tmp
                elif ("interested issue" in key.lower() or "interested ent" in key.lower()) and isinstance(item, list) and isinstance(item[0], dict):
                    tmp = []
                    for d in item:
                        for subk, subv in d.items():
                            if subv is not None and not isinstance(subv, str) and not isinstance(subv, list):
                                print("subv2", subv)
                                print("uid", uid)
                                print("res", res)
                                continue
                            if isinstance(subv, list):
                                subv = "; ".join(subv)
                            if isinstance(subv, str) and not subv.strip(): subv = None
                            tmp.append(f"{subk}: {subv}".strip())
                    new_res[key] = tmp

                elif "analys" in key.lower() and isinstance(res[key], list):
                    new_res[key] = ". ".join(item)

                else:
                    new_res[key] = item

            for key, val in new_res.items():
                if ("interested issue" in key.lower() or "interested ent" in key.lower()):
                    new_res[key] = [f"View toward {st}" if ":" in st else st for st in val]
                    # new_res[key]=[st for st in val if st.ends(": None")]

            # print(new_res)
            dump_file(new_res, f"{new_dir}/{uid}.json")
        except Exception as e:
            print("Exception e2")
            print(e)
            print("uid", uid)
            print("res\n", res)
            time.sleep(5)
            continue
            breakpoint()

history_data = None
exit()
