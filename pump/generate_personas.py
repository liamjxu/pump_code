import os
os.environ['HF_HOME'] = '/mnt/sagemaker-nvme/cache'
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

import torch
import pandas as pd
import json
import numpy as np
import argparse
import time
from tqdm import trange, tqdm
from sklearn.cluster import KMeans
from utils import last_token_pool, get_detailed_instruct, get_llm_response, PersonaDimension, list_s3_prefix, get_file_from_s3, get_topics
from transformers import AutoTokenizer, AutoModel
from dataclasses import asdict
import argparse


def extract_personas_from_survey(info_df, survey, extraction_prompt_type, output_dir, debug, model_id):
    if extraction_prompt_type == 'example':
        prompt_name = 'get_personas_from_questions'
    elif extraction_prompt_type == 'description':
        prompt_name = 'get_personas_from_questions_simple'
    else:
        raise ValueError(f'Invalid extraction_prompt_type: {extraction_prompt_type}')
    with open(f'prompts/{prompt_name}.txt') as f:
        prompt_template = f.read()

    res = []
    logs = {}
    valid_cnt = 0
    for idx, row in tqdm(info_df.iterrows(), total=len(info_df)):
        topics = get_topics(mapping, row['question'])
        input_dict = {
            "topic_fg": topics['fg'],
            "topic_cg": topics['cg'],
            "question": row['question'],
            "options": row['references'],
        }
        prompt = prompt_template.format(**input_dict)
        response = get_llm_response(prompt, model_id=model_id, prefill='[')
        response = '[' + response
        valid = None
        error_msg = None
        try:
            eval(response)
            valid = True
            valid_cnt += 1
            if debug:
                print(response)
        except Exception as e:
            print(e)
            valid = False
            error_msg = str(e)

        res.append({
            'valid': valid,
            'error_msg': error_msg,
            'input_dict': str(input_dict),
            'response': response,
        })

        logs['res_len'] = len(res)
        logs['valid_ratio'] = valid_cnt / len(res)

        os.makedirs(output_dir, exist_ok=True)

        with open(f'{output_dir}/personas_extracted_from_question_{survey}.json', 'w') as f:
            json.dump(res, f, indent=4)
        print(f"Saved to {output_dir}/personas_extracted_from_question_{survey}.json")
        with open(f'{output_dir}/logs_{survey}.json', 'w') as f:
            json.dump(logs, f, indent=4)
        print(f"Logs at: {output_dir}/logs_{survey}.json")


def get_personas_extracted_from_questions_df(personas_from_questions_filename):
    with open(personas_from_questions_filename, 'r') as f:
        data = json.load(f)
    res = []
    for entry in data:
        if not entry['valid']: continue
        for persona_dim in eval(entry['response']):
            entry_dict = asdict(persona_dim)
            input_dict = eval(entry['input_dict'])
            entry_dict['original_question'] = input_dict['question'] + ' ' + input_dict['options']
            res.append(entry_dict)
    df = pd.DataFrame(res)
    return df


def cluster_extracted_personas(survey, extraction_dir, output_dir, debug, tokenizer, model, clustering_algo, clustering_num_clusters):
    os.makedirs(output_dir, exist_ok=True)
    
    # Get data
    personas_from_questions_filename = f"{extraction_dir}/personas_extracted_from_question_{survey}.json"
    data = get_personas_extracted_from_questions_df(personas_from_questions_filename)

    # Get formatted string for clustering
    def get_formatted_persona_dim(row):
        task = 'Given a persona dimension description, retrieve semantically similar persona dimension descriptions.'
        persona = f"{row['name']}: {row['description']}. Candidate values: {row['candidate_values']}"
        return get_detailed_instruct(task, persona)
    data['formatted'] = data.apply(get_formatted_persona_dim, axis=1)

    for level in ['high', 'mid', 'low']:
        # Get subset and save artifacts
        level_df = data[data['level']==level]
        level_df.to_csv(f"{output_dir}/{level}-level_personas_{survey}.csv")
        
        # Get the embeddings
        max_length = 4096
        input_texts = level_df['formatted'].to_list()

        embeddings = []
        for text in input_texts:
            batch_dict = tokenizer([text], max_length=max_length, padding=True, truncation=True, return_tensors="pt")
            with torch.inference_mode():
                outputs = model(**batch_dict)
                embed = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])[0]
            embeddings.append(embed)
        embeddings = torch.stack(embeddings, axis=0)
        print(embeddings.shape, len(input_texts))
        
        # Clustering the embeddings and save artifacts
        if clustering_algo == 'kmeans':
            clustering_model = KMeans(n_clusters=clustering_num_clusters)
            clustering_model.fit(embeddings)
            level_df['cluster'] = clustering_model.labels_
        else:
            raise ValueError("unknown clustering algorithm")
        level_df = level_df.sort_values(by='cluster')
        level_df.to_csv(f'{output_dir}/clustered_{level}_level_personas_{survey}.csv')

        if debug:
            for idx in range(clustering_num_clusters):
                print(idx)
                for _, row in enumerate(level_df[level_df['cluster'] == idx]['formatted']):
                    print(row.split('\n')[1])
                print('\n\n')


def summarize_clustered_personas(prompt_name, survey, level, clustering_dir, output_dir, debug, clustering_num_clusters, model_id):
    os.makedirs(output_dir, exist_ok=True)

    # Get data
    clustered_persona_filename = f"{clustering_dir}/clustered_{level}_level_personas_{survey}.csv"
    data = pd.read_csv(clustered_persona_filename)
    with open(f'prompts/{prompt_name}.txt') as f:
        prompt_template = f.read()

    # summarize
    res = []
    logs = []
    for idx in trange(clustering_num_clusters):
        persona_cluster = []
        for _, row in data[data['cluster'].astype(str) == str(idx)].iterrows():
            persona = PersonaDimension(**row[['name', 'description', 'level', 'candidate_values']].to_dict())
            persona_cluster.append(persona)
        
        prompt = prompt_template.format(persona_dimensions='[\n' + ',\n'.join(repr(dim) for dim in persona_cluster) + '\n]')
        response = get_llm_response(prompt, model_id=model_id, prefill='[')
        response = '[' + response
        
        try:
            response = eval(response)
            for dim in response:
                res.append(dim)

            record_res = [str(dim) for dim in res]
            summarized_clustered_personas_filename = f"{output_dir}/summarized_{level}_level_personas_{survey}.json"
            with open(summarized_clustered_personas_filename, 'w') as f:
                json.dump(record_res, f, indent=4)
            if debug:
                print(response)

            logs.append({
                'survey': survey,
                'level': level,
                'cluster_idx': idx,
                'is_successful': True
            })
            logs_filename = f"{output_dir}/logs_summarized_{level}_level_personas_{survey}.json"
            with open(logs_filename, 'w') as f:
                json.dump(logs, f, indent=4)

        except:
            if debug:
                print(f"cluster {idx} failed")
                print(response)

            logs.append({
                'survey': survey,
                'level': level,
                'cluster_idx': idx,
                'is_successful': False
            })
            logs_filename = f"{output_dir}/logs_summarized_{level}_level_personas_{survey}.json"
            with open(logs_filename, 'w') as f:
                json.dump(logs, f, indent=4)
            continue


def clean_summarized_personas(prompt_name, survey, level, summarizing_dir, output_dir, debug, model_id):
    os.makedirs(output_dir, exist_ok=True)

    # Get data
    summarized_persona_filename = f"{summarizing_dir}/summarized_{level}_level_personas_{survey}.json"
    with open(summarized_persona_filename, 'r') as f:
        data = json.load(f)
    with open(f'prompts/{prompt_name}.txt') as f:
        prompt_template = f.read()

    # summarize
    prompt = prompt_template.format(persona_dimensions='[\n' + ',\n'.join(repr(dim) for dim in data) + '\n]')
    response = get_llm_response(prompt, prefill='[', max_tokens=4096, model_id=model_id)
    response = ''.join(['[', response])
    
    if debug:
        print(response)

    # validate
    try:
        eval(response)
        cleaned_summarized_personas_filename = f"{output_dir}/cleaned_{level}_level_personas_{survey}.json"
        with open(cleaned_summarized_personas_filename, 'w') as f:
            json.dump(response, f, indent=4)
        return True
    except:
        print(f'Cleaned result is not valid. Survey: {survey}, Level: {level}.')
        if debug:
            print(f"Response: {response}")
        return False


def main(args):
    loggings = []
    surveys = set()
    for path in list_s3_prefix("human_resp/"):
        if path.startswith("human_resp/American_Trends_Panel"):
            # Extract the folder name
            folder = path.split("/")[1]
            surveys.add(folder)
    surveys = sorted(list(surveys))
    mapping = np.load(get_file_from_s3('human_resp/topic_mapping.npy'), allow_pickle=True)
    mapping = mapping.item()

    # Extracting
    loggings_extraction = []
    print(f"Starting extraction for {len(surveys)} surveys.")
    for survey in surveys:
        print(f"Extracting from {survey}")
        file_key = f"human_resp/{survey}/info.csv"
        info_df = pd.read_csv(get_file_from_s3(file_key))
        tic = time.time()
        extract_personas_from_survey(info_df,
                                     survey,
                                     extraction_prompt_type=args.extraction_prompt_type,
                                     output_dir=f'{args.output_dir_root}/extraction',
                                     debug=args.debug,
                                     model_id=args.model_id)
        toc = time.time()
        loggings_extraction.append({
            'survey': survey,
            'extraction_time': toc - tic,
        })
    loggings['extraction'] = loggings_extraction
    with open(f"{args.output_dir_root}/loggings.json", 'w') as f:
        json.dump(loggings, f, indent=4)

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained('Salesforce/SFR-Embedding-2_R')
    model = AutoModel.from_pretrained('Salesforce/SFR-Embedding-2_R', device_map='auto')

    # Clustering
    for survey in tqdm(surveys):
        # print(f"Clustering {survey}")
        cluster_extracted_personas(survey,
                                   extraction_dir=f"{args.output_dir_root}/extraction",
                                   output_dir=f'{args.output_dir_root}/clustering',
                                   debug=args.debug,
                                   tokenizer=tokenizer,
                                   model=model,
                                   clustering_algo=args.clustering_algo,
                                   clustering_num_clusters=args.clustering_num_clusters)

    # Summarize
    for survey in surveys:
        for level in ['low', 'mid', 'high']:
            summarize_clustered_personas(prompt_name="summarize_clustered_personas",
                                         survey=survey,
                                         level=level,
                                         clustering_dir=f'{args.output_dir_root}/clustering',
                                         output_dir=f'{args.output_dir_root}/summarizing',
                                         clustering_num_clusters=args.clustering_num_clusters,
                                         debug=args.debug,
                                         model_id=args.model_id)

    # Cleaning
    logs = []
    for survey in surveys:
        print(f"Survey: {survey}")
        for level in tqdm(['low', 'mid', 'high']):
            status = clean_summarized_personas(prompt_name="clean_summarized_personas",
                                                         survey=survey,
                                                         level=level,
                                                         summarizing_dir=f'{args.output_dir_root}/summarizing',
                                                         output_dir=f'{args.output_dir_root}/cleaned',
                                                         clustering_num_clusters=args.clustering_num_clusters,
                                                         debug=args.debug,
                                                         model_id=args.model_id)
            if status:
                logs.append({
                    'survey': survey,
                    'level': level,
                    'is_successful': True
                })
            else:
                logs.append({
                    'survey': survey,
                    'level': level,
                    'is_successful': False
                })

    with open(f"{args.output_dir_root}/cleaned/logs.json", 'w') as f:
        json.dump(logs, f, indent=4)


if __name__ == '__main__':

    surveys = set()
    for path in list_s3_prefix("human_resp/"):
        if path.startswith("human_resp/American_Trends_Panel"):
            # Extract the folder name
            folder = path.split("/")[1]
            surveys.add(folder)
    surveys = sorted(list(surveys))
    mapping = np.load(get_file_from_s3('human_resp/topic_mapping.npy'), allow_pickle=True)
    mapping = mapping.item()

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--debug', action='store_true')
    argparser.add_argument('--output_dir_root', type=str, default="sm_local/outputs")
    argparser.add_argument('--model_id', type=str, choices=["anthropic.claude-3-haiku-20240307-v1:0", "anthropic.claude-3-sonnet-20240229-v1:0"])
    argparser.add_argument('--clustering_algo', type=str, choices=['kmeans'])
    argparser.add_argument('--extraction_prompt_type', type=str, choices=['description', 'example'])
    argparser.add_argument('--clustering_num_clusters', type=int)
    argparser.add_argument('--merging_personas_from_surveys', type=str, choices=['single', 'same_topic'])
    args = argparser.parse_args()
    main(args)
