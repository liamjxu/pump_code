import os
os.environ['HF_HOME'] = '/mnt/sagemaker-nvme/cache'
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

import torch
import pandas as pd
import json
from tqdm import trange, tqdm
from sklearn.cluster import KMeans
from utils import last_token_pool, get_detailed_instruct, get_llm_response, PersonaDimension, list_s3_prefix
from transformers import AutoTokenizer, AutoModel
from dataclasses import asdict

surveys = set()
for path in list_s3_prefix("human_resp/"):
    if path.startswith("human_resp/American_Trends_Panel"):
        # Extract the folder name
        folder = path.split("/")[1]
        surveys.add(folder)
surveys = sorted(list(surveys))[1:]


def get_personas_extracted_from_questions(personas_from_questions_filename):
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


def cluster_extracted_personas(survey, extraction_dir, output_dir, num_clusters, print_result, tokenizer, model):
    os.makedirs(output_dir, exist_ok=True)
    
    # Get data
    personas_from_questions_filename = f"{extraction_dir}/personas_extracted_from_question_{survey}.json"
    data = get_personas_extracted_from_questions(personas_from_questions_filename)

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
            with torch.no_grad():
                outputs = model(**batch_dict)
                embed = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])[0]
            embeddings.append(embed)
        embeddings = torch.stack(embeddings, axis=0)
        print(embeddings.shape, len(input_texts))
        
        # Clustering the embeddings and save artifacts
        clustering_model = KMeans(n_clusters=num_clusters)
        clustering_model.fit(embeddings)
        level_df['cluster'] = clustering_model.labels_
        level_df = level_df.sort_values(by='cluster')
        level_df.to_csv(f'{output_dir}/clustered_{level}_level_personas_{survey}.csv')

        if print_result:
            for idx in range(num_clusters):
                print(idx)
                for _, row in enumerate(level_df[level_df['cluster'] == idx]['formatted']):
                    print(row.split('\n')[1])
                print('\n\n')

# # Load tokenizer and model
# tokenizer = AutoTokenizer.from_pretrained('Salesforce/SFR-Embedding-2_R')
# model = AutoModel.from_pretrained('Salesforce/SFR-Embedding-2_R', device_map='auto')

# for survey in tqdm(surveys):
#     cluster_extracted_personas(survey,
#                                extraction_dir='sm_local/outputs/extraction',
#                                output_dir='sm_local/outputs/clustering',
#                                num_clusters=20,
#                                print_result=False,
#                                tokenizer=tokenizer,
#                                model=model)


def summarize_clustered_personas(prompt_name, survey, level, clustering_dir, output_dir, num_clusters, print_result):
    os.makedirs(output_dir, exist_ok=True)

    # Get data
    clustered_persona_filename = f"{clustering_dir}/clustered_{level}_level_personas_{survey}.csv"
    data = pd.read_csv(clustered_persona_filename)
    with open(f'prompts/{prompt_name}.txt') as f:
        prompt_template = f.read()

    # summarize
    res = []
    for idx in trange(num_clusters):
        persona_cluster = []
        for _, row in data[data['cluster'].astype(str) == str(idx)].iterrows():
            persona = PersonaDimension(**row[['name', 'description', 'level', 'candidate_values']].to_dict())
            persona_cluster.append(persona)
        
        prompt = prompt_template.format(persona_dimensions='[\n' + ',\n'.join(repr(dim) for dim in persona_cluster) + '\n]')
        response = get_llm_response(prompt, prefill='[')
        response = '[' + response
        
        try:
            response = eval(response)
            for dim in response:
                res.append(dim)

            record_res = [str(dim) for dim in res]
            summarized_clustered_personas_filename = f"{output_dir}/summarized_{level}_level_personas_{survey}.json"
            with open(summarized_clustered_personas_filename, 'w') as f:
                json.dump(record_res, f, indent=4)
    
            if print_result:
                print(response)
        except:
            if print_result:
                print(f"cluster {idx} failed")
                print(response)
            continue


# for survey in surveys:
#     for level in ['low', 'mid', 'high']:
#         summarize_clustered_personas(prompt_name="summarize_clustered_personas",
#                                      survey=survey,
#                                      level=level,
#                                      clustering_dir='sm_local/outputs/clustering',
#                                      output_dir='sm_local/outputs/summarizing',
#                                      num_clusters=20,
#                                      print_result=True)


def clean_summarized_personas(prompt_name, survey, level, summarizing_dir, output_dir, num_clusters, print_result):
    os.makedirs(output_dir, exist_ok=True)

    # Get data
    summarized_persona_filename = f"{summarizing_dir}/summarized_{level}_level_personas_{survey}.json"
    with open(summarized_persona_filename, 'r') as f:
        data = json.load(f)
    with open(f'prompts/{prompt_name}.txt') as f:
        prompt_template = f.read()

    # summarize
    prompt = prompt_template.format(persona_dimensions='[\n' + ',\n'.join(repr(dim) for dim in data) + '\n]')
    response = get_llm_response(prompt, prefill='[', max_tokens=4096)
    response = '[' + response

    # validate
    try:
        eval(response)
        cleaned_summarized_personas_filename = f"{output_dir}/cleaned_{level}_level_personas_{survey}.json"
        with open(cleaned_summarized_personas_filename, 'w') as f:
            json.dump(response, f, indent=4)
        return response
    except:
        print(f'Cleaned result is not valid. Survey: {survey}, Level: {level}. Response:')
        # print(response)
        return None


for survey in surveys:
    print(f"Survey: {survey}")
    for level in tqdm(['low', 'mid', 'high']):
        cleaned_personas = clean_summarized_personas(prompt_name="clean_summarized_personas",
                                                      survey=survey,
                                                      level=level,
                                                      summarizing_dir='sm_local/outputs/summarizing',
                                                      output_dir='sm_local/outputs/cleaned',
                                                      num_clusters=20,
                                                      print_result=True)
