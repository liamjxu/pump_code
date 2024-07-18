import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

import torch
import pandas as pd
import json
import numpy as np
import argparse
import time
from tqdm import trange, tqdm
from sklearn.cluster import KMeans
from utils import last_token_pool, get_detailed_instruct, get_llm_response, list_s3_prefix, get_file_from_s3, get_topics, PersonaDimension
from utils import persona_dim_object_list_to_dict_list, persona_dim_dict_to_object, persona_dim_dict_list_to_object_list
from transformers import AutoTokenizer, AutoModel
import argparse


def extract_personas_from_survey(info_df, survey, extraction_prompt_type, output_dir, debug, model_id):
    prompt_name_mapping = {
        'example': 'extract_personas_from_questions_example',
        'description': 'extract_personas_from_questions_description'
    }
    if extraction_prompt_type not in prompt_name_mapping:
        raise ValueError(f'Invalid extraction_prompt_type: {extraction_prompt_type}')
    else:
        prompt_name = prompt_name_mapping[extraction_prompt_type]

    with open(f'prompts/{prompt_name}.txt') as f:
        prompt_template = f.read()

    res = []
    logs = {
        'total_num_personas_extracted': 0
    }
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
            persona_json = persona_dim_object_list_to_dict_list(eval(response))
            num_persona = len(persona_json)
            valid = True
            valid_cnt += 1
        except Exception as e:
            print(f"Exception during extraction. Error Message:\n{e}")
            print(f"Response:\n{response}")
            valid = False
            error_msg = str(e)
            persona_json = None
            num_persona = 0

        res.append({
            'valid': valid,
            'error_msg': error_msg,
            'input_dict': str(input_dict),
            'response': persona_json,
        })

        logs['num_questions_used'] = len(res)
        logs['valid_ratio'] = valid_cnt / len(res)
        logs['total_num_personas_extracted'] = logs['total_num_personas_extracted'] + num_persona

        os.makedirs(output_dir, exist_ok=True)

        with open(f'{output_dir}/personas_extracted_from_question_{survey}.json', 'w') as f:
            json.dump(res, f, indent=4)
        with open(f'{output_dir}/logs_{survey}.json', 'w') as f:
            json.dump(logs, f, indent=4)

    print(f"Saved to {output_dir}/personas_extracted_from_question_{survey}.json")
    print(f"Logs at: {output_dir}/logs_{survey}.json")
    return logs


def get_personas_extracted_from_questions_df(personas_from_questions_filename):
    with open(personas_from_questions_filename, 'r') as f:
        data = json.load(f)
    res = []
    for entry in data:
        if not entry['valid']: continue
        for entry_dict in entry['response']:
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

    # summarizing
    logs = []
    res = []
    for idx in trange(clustering_num_clusters):
        persona_cluster = []
        for _, row in data[data['cluster'].astype(str) == str(idx)].iterrows():
            persona = persona_dim_dict_to_object(row[['name', 'description', 'level', 'candidate_values']].to_dict())
            persona_cluster.append(persona)
        
        prompt = prompt_template.format(persona_dimensions='[\n' + ',\n'.join(repr(dim) for dim in persona_cluster) + '\n]')
        response = get_llm_response(prompt, model_id=model_id, prefill='[')
        response = '[' + response
        
        try:
            record_res = persona_dim_object_list_to_dict_list(eval(response))
            res += record_res
            summarized_clustered_personas_filename = f"{output_dir}/summarized_{level}_level_personas_{survey}.json"
            with open(summarized_clustered_personas_filename, 'w') as f:
                json.dump(res, f, indent=4)

            logs.append({
                # 'survey': survey,
                # 'level': level,
                'cluster_idx': idx,
                'is_successful': True
            })
            logs_filename = f"{output_dir}/logs_summarized_{level}_level_personas_{survey}.json"
            with open(logs_filename, 'w') as f:
                json.dump(logs, f, indent=4)

        except:
            if debug:
                print(f"Summarizing failed. Survey: {survey}, level: {level}, cluster: {idx}")
                print(response)

            logs.append({
                # 'survey': survey,
                # 'level': level,
                'cluster_idx': idx,
                'is_successful': False
            })
            logs_filename = f"{output_dir}/logs_summarized_{level}_level_personas_{survey}.json"
            with open(logs_filename, 'w') as f:
                json.dump(logs, f, indent=4)
    
    return logs, len(res)


def clean_summarized_personas(prompt_name, survey, level, summarizing_dir, output_dir, debug, model_id):
    
    os.makedirs(output_dir, exist_ok=True)

    # Get data
    summarized_persona_filename = f"{summarizing_dir}/summarized_{level}_level_personas_{survey}.json"
    with open(summarized_persona_filename, 'r') as f:
        data = json.load(f)
        data = persona_dim_dict_list_to_object_list(data)
    with open(f'prompts/{prompt_name}.txt') as f:
        prompt_template = f.read()

    # summarize
    prompt = prompt_template.format(persona_dimensions='[\n' + ',\n'.join(repr(dim) for dim in data) + '\n]')
    response = get_llm_response(prompt, prefill='[', max_tokens=4096, model_id=model_id)
    response = ''.join(['[', response])

    # validate
    try:
        response = persona_dim_object_list_to_dict_list(eval(response))
        cleaned_summarized_personas_filename = f"{output_dir}/cleaned_{level}_level_personas_{survey}.json"
        with open(cleaned_summarized_personas_filename, 'w') as f:
            json.dump(response, f, indent=4)
        return True, response
    except Exception as e:
        print(f'Cleaned result is not valid. Survey: {survey}, Level: {level}.')
        print(f"Error Message:\n{e}")
        if debug:
            print(f"Response: {response}")
        return False, response


def main(args):
    if os.path.exists(f"{args.output_dir_root}/loggings.json"):
        with open(f"{args.output_dir_root}/loggings.json", 'r') as f:
            loggings = json.load(f)
    else:
        loggings = {}
    surveys = set()
    for path in list_s3_prefix("human_resp/"):
        if path.startswith("human_resp/American_Trends_Panel"):
            # Extract the folder name
            folder = path.split("/")[1]
            surveys.add(folder)
    surveys = sorted(list(surveys))[args.survey_starting:args.survey_ending]
    mapping = np.load(get_file_from_s3('human_resp/topic_mapping.npy'), allow_pickle=True)
    mapping = mapping.item()

    # Extracting
    if 'extraction' not in args.phases:
        print('Skipping extraction')
    else:
        loggings_extraction = []
        print(f"Starting extraction for {len(surveys)} surveys.")
        for survey in surveys:
            print(f"Extracting from {survey}")
            file_key = f"human_resp/{survey}/info.csv"
            info_df = pd.read_csv(get_file_from_s3(file_key))
            tic = time.time()
            logs = extract_personas_from_survey(info_df,
                                        survey,
                                        extraction_prompt_type=args.extraction_prompt_type,
                                        output_dir=f'{args.output_dir_root}/extraction',
                                        debug=args.debug,
                                        model_id=args.model_id)
            toc = time.time()
            loggings_extraction.append({
                'survey': survey,
                'extraction_time': toc - tic,
                **logs
            })
        loggings['extraction'] = loggings_extraction
        with open(f"{args.output_dir_root}/loggings.json", 'w') as f:
            json.dump(loggings, f, indent=4)

    # Clustering
    if 'clustering' not in args.phases:
        print('Skipping clustering')
    else:
        print('Starting clustering')
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained('Salesforce/SFR-Embedding-2_R')
        model = AutoModel.from_pretrained('Salesforce/SFR-Embedding-2_R', device_map='auto')

        # Clustering
        loggings_clustering = []
        for survey in tqdm(surveys):
            print(f"Clustering {survey}")
            tic = time.time()
            cluster_extracted_personas(survey,
                                    extraction_dir=f"{args.output_dir_root}/extraction",
                                    output_dir=f'{args.output_dir_root}/clustering',
                                    debug=args.debug,
                                    tokenizer=tokenizer,
                                    model=model,
                                    clustering_algo=args.clustering_algo,
                                    clustering_num_clusters=args.clustering_num_clusters)
            toc = time.time()
            loggings_clustering.append({
                'survey': survey,
                'clustering_time': toc - tic,
            })
        loggings['clustering'] = loggings_clustering
        with open(f"{args.output_dir_root}/loggings.json", 'w') as f:
            json.dump(loggings, f, indent=4)

    # Summarizing
    if 'summarizing' not in args.phases:
        print('Skipping summarizing')
    else:
        print('Starting summarizing')
        loggings_summarizing = []
        for survey in surveys:
            for level in ['low', 'mid', 'high']:
                print(f"Survey: {survey}, level: {level}")
                tic = time.time()
                clusters_logs, num_personas = summarize_clustered_personas(prompt_name="summarize_clustered_personas",
                                            survey=survey,
                                            level=level,
                                            clustering_dir=f'{args.output_dir_root}/clustering',
                                            output_dir=f'{args.output_dir_root}/summarizing',
                                            clustering_num_clusters=args.clustering_num_clusters,
                                            debug=args.debug,
                                            model_id=args.model_id)
                toc = time.time()
                loggings_summarizing.append({
                    'survey': survey,
                    'level': level,
                    'num_of_clusters': len(clusters_logs),
                    'num_of_personas': num_personas,
                    'summarizing_time': toc - tic,
                    'clusters': clusters_logs,
                    'valid_ratio': sum([entry['is_successful'] for entry in clusters_logs]) / len(clusters_logs)
                })
        loggings['summarizing'] = loggings_summarizing
        with open(f"{args.output_dir_root}/loggings.json", 'w') as f:
            json.dump(loggings, f, indent=4)

    # Cleaning
    if 'cleaning' not in args.phases:
        print('Skipping cleaning')
    else:
        print('Starting cleaning')
        logs = []
        for survey in surveys:
            print(f"Survey: {survey}")
            for level in tqdm(['low', 'mid', 'high']):
                tic = time.time()
                failure = 0
                response = None
                status = False
                while failure < 3:
                    try:
                        status, response = clean_summarized_personas(prompt_name="clean_summarized_personas",
                                                            survey=survey,
                                                            level=level,
                                                            summarizing_dir=f'{args.output_dir_root}/summarizing',
                                                            output_dir=f'{args.output_dir_root}/cleaned',
                                                            debug=args.debug,
                                                            model_id=args.model_id)
                        if status:
                            toc = time.time()
                            logs.append({
                                'survey': survey,
                                'level': level,
                                'cleaning_time': toc - tic,
                                'is_successful': True,
                                'num_final_personas': len(response),
                                'response (when failing)': None  # only recording when failing, successful ones are stored separately
                            })
                            break
                    except:
                        time.sleep(10)
                    failure += 1
                    print(f"Failed {failure}/3 times")
                
                if not status:
                    toc = time.time()
                    logs.append({
                        'survey': survey,
                        'level': level,
                        'cleaning_time': toc - tic,
                        'is_successful': False,
                        'num_final_personas': 0,
                        'response (when failing)': response
                    })

        loggings['cleaning'] = logs
        with open(f"{args.output_dir_root}/loggings.json", 'w') as f:
            json.dump(loggings, f, indent=4)


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
    argparser.add_argument('--survey_starting', type=int, default=None)
    argparser.add_argument('--survey_ending', type=int, default=None)
    argparser.add_argument('--phases', nargs='+',
                                       choices=['extraction', 'clustering', 'summarizing', 'cleaning'],  
                                       default=['extraction', 'clustering', 'summarizing', 'cleaning'])
    args = argparser.parse_args()
    main(args)
