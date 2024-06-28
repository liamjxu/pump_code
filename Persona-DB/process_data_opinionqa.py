import chunk
import random
import time

import numpy as np
import requests
import os
import json

from tqdm import tqdm

from utils.utils import *
from constants import *
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
from utils.data_utils import process_news_profile_history, get_graph_dataset, GlobalIdMap, get_user_profile, get_data_slice, process_triplet, UserRetriever

import ast


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



# parse
args = parser.parse_args()



DATA_DIR = f"{data_dir}OpinionQA/human_resp/"
OUTPUT_DIR = f"{data_dir}OpinionQA/human_resp/"#os.path.dirname(os.path.abspath(__file__))
# survey_names = ["American_Trends_Panel_W34", "American_Trends_Panel_W41", "American_Trends_Panel_W82"]
survey_names = ["American_Trends_Panel_W26"]

def get_response_data(survey_name):
    key_df = pd.read_csv(os.path.join(DATA_DIR, survey_name, 'info.csv'))

    key_list = key_df['key'].tolist()

    df = pd.read_csv(os.path.join(DATA_DIR, survey_name, 'responses.csv'))

    response_list = []

    for element in key_list:
        for index, row in df.iterrows():
            data_dict = {"author_id": row['index'], "conversation_id": element, "label": row[element]}
        # for i, (index, row) in enumerate(df.iterrows()):
        #     data_dict = {"author_id": row['index'], "conversation_id": element, "label": row[element], "tweet_id":i}

            # do not add in nan values
            if not pd.isnull(data_dict["label"]):
                response_list.append(data_dict)

    with open(os.path.join(OUTPUT_DIR, survey_name, 'response_data.json'), 'w') as json_file:
        json.dump(response_list, json_file, indent=4)

    # randomly select 1000 samples for test
    with open(os.path.join(OUTPUT_DIR, survey_name, 'response_data_balanced_test_anno.json'), 'w') as json_file:
        json.dump(random.sample(response_list, 500), json_file, indent=4)
    with open(os.path.join(OUTPUT_DIR, survey_name, 'response_data_balanced_test_anno_1000.json'), 'w') as json_file:
        json.dump(random.sample(response_list, 1000), json_file, indent=4)

def get_user_info_dict(survey_name):
    info_df = pd.read_csv(os.path.join(DATA_DIR, survey_name, 'metadata.csv'))

    info_list = info_df['key'].tolist()

    df = pd.read_csv(os.path.join(DATA_DIR, survey_name, 'responses.csv'))

    user_info_dict = {}

    for index, row in df.iterrows():
        description = ["[{}]".format(info.replace("CITIZEN", "USCITIZEN")) + str(row[info]) for info in info_list]
        user_info_dict[row["index"]] = {"description": "".join(description)}

    with open(os.path.join(OUTPUT_DIR, survey_name, 'users_info_dict_with_chatgptanno_rf.json'), 'w') as json_file:
        json.dump(user_info_dict, json_file, indent=4)

def get_user_history_dict(survey_name):
    key_df = pd.read_csv(os.path.join(DATA_DIR, survey_name, 'info.csv'))

    key_question_dict = {}
    for index, row in key_df.iterrows():
        key = row['key']
        question = row['question']
        key_question_dict[key] = question

    df = pd.read_csv(os.path.join(DATA_DIR, survey_name, 'responses.csv'))

    user_history_dict = {}

    for index, row in df.iterrows():
        user_history_list = []
        for k in key_question_dict.keys():
            if str(row[k])!="nan":
                user_history_list.append([k,"[Question]{}[Answer]{}".format(key_question_dict[k], row[k])])
        user_history_dict[row["index"]] = user_history_list

    with open(os.path.join(OUTPUT_DIR, survey_name, 'users_history_dict.json'), 'w') as json_file:
        json.dump(user_history_dict, json_file, indent=4)


def get_post_data(survey_name):
    df = pd.read_csv(os.path.join(DATA_DIR, survey_name, 'info.csv'))

    key_question_dict = {}
    for index, row in df.iterrows():
        key = row['key']
        question = row['question']
        for k,v in ast.literal_eval(row['option_mapping']).items():
            assert int(k)== k, print(row['option_mapping'])
        key_question_dict[key] = {"question": question, "options": {v.lower().strip():int(k) for k,v in ast.literal_eval(row['option_mapping']).items()}}

    with open(os.path.join(OUTPUT_DIR, survey_name, 'post_data.json'), 'w') as json_file:
        json.dump(key_question_dict, json_file, indent=4)

# if __name__ == "__main__":
for survey_name in survey_names:
    os.makedirs(os.path.join(OUTPUT_DIR, survey_name), exist_ok=True)
    get_response_data(survey_name)
    get_user_info_dict(survey_name)
    get_user_history_dict(survey_name)
    get_post_data(survey_name)
