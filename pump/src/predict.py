import re
import os
import argparse
import json
import random
import torch
import numpy as np
import pandas as pd
from pump.src.utils import list_s3_prefix, get_file_from_s3, get_formatted_persona_dim, last_token_pool
from pump.src.utils import get_llm_response, CLAUDE_NAME_MAPPING
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from scipy.spatial.distance import cdist


def get_persona_values(file_key, user_history, model_name="sonnet", persona_num=5):

    prompt_name = 'experiment/prompts/persona_inference/infer_persona_from_user_history.txt'
    with open(prompt_name) as f:
        prompt_template = f.read()

    ret_personas = []
    for level in ['high', 'mid', 'low']:
        persona_path = f"sm_local/outputs_sonnet_kmeans10_single_example/cleaned/cleaned_{level}_level_personas_{file_key}.json"
        with open(persona_path, 'r') as f:
            all_personas = json.load(f)

        for persona_dim in all_personas[:persona_num]:
            input_dict = {
                "user_history": user_history,
                "persona_dim": persona_dim,
            }
            prompt = prompt_template.format(**input_dict)
            response = get_llm_response(prompt, model_id=CLAUDE_NAME_MAPPING[model_name])
            persona_dim['inferred_value'] = response
            ret_personas.append(persona_dim)

    return ret_personas


def get_embeddings(input_texts, tokenizer, model, max_length=4096):
    embedding = []
    for text in input_texts:
        batch_dict = tokenizer([text], max_length=max_length, padding=True, truncation=True, return_tensors="pt")
        with torch.inference_mode():
            outputs = model(**batch_dict)
            embed = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])[0]
        embedding.append(embed)
    embedding = torch.stack(embedding, axis=0)
    return embedding


def get_query_to_persona_idx_mapping(test_q_keys, survey_df):
    tokenizer = AutoTokenizer.from_pretrained('Salesforce/SFR-Embedding-2_R')
    model = AutoModel.from_pretrained('Salesforce/SFR-Embedding-2_R', device_map='auto')

    q_indices = [survey_df.index[survey_df['key'] == key].tolist()[0] for key in test_q_keys]  # This has to be done this way because test_q_keys can be not in order
    with open("sm_local/outputs_sonnet_kmeans10_single_example/extraction/personas_extracted_from_question_American_Trends_Panel_W26.json", "r") as f:
        extraction = json.load(f)
    original_extraction = [extraction[idx]['response'] for idx in q_indices]
    original_extraction = [[get_formatted_persona_dim(persona_dim) for persona_dim in _] for _ in original_extraction]
    key_to_original_extraction = dict(zip(test_q_keys, original_extraction))

    clean = []
    for level in ['high', 'mid', 'low']:
        with open(f"sm_local/outputs_sonnet_kmeans10_single_example/cleaned/cleaned_{level}_level_personas_American_Trends_Panel_W26.json", 'r') as f:
            clean += json.load(f)
    clean = [get_formatted_persona_dim(_) for _ in clean]
    clean_embeddings = get_embeddings(clean, tokenizer, model)

    # Get the embeddings
    key_to_clean_idx = {}
    for key in test_q_keys:
        input_texts = key_to_original_extraction[key]
        left_embeddings = get_embeddings(input_texts, tokenizer, model)
        cosine_similarity = 1 - cdist(left_embeddings, clean_embeddings, 'cosine')
        most_similar_indices = np.argmax(cosine_similarity, axis=1).tolist()
        key_to_clean_idx[key] = most_similar_indices

    return key_to_clean_idx


def extract_cot_prediction(text):
    """
    Extracts text between <prediction> and </prediction> tags.
    
    Args:
    text (str): The input string containing the text with <prediction> tags.
    
    Returns:
    str: The extracted text between the tags. If no tags are found, returns an empty string.
    """
    pattern = re.compile(r'<prediction>(.*?)</prediction>', re.DOTALL)
    match = pattern.search(text)
    if match:
        return match.group(1)
    else:
        return ""


def main(args):
    random.seed(42)
    print(args)

    file_key = args.survey_name
    survey_df = pd.read_csv(get_file_from_s3(f"human_resp/{file_key}/info.csv"))
    resp_df = pd.read_csv(get_file_from_s3(f"human_resp/{file_key}/responses.csv"))
    meta_df = pd.read_csv(get_file_from_s3(f"human_resp/{file_key}/metadata.csv"))
    question_key_mapping = {row['key']: {'question': row['question'], 'references': row['references']} for _, row in survey_df.iterrows()}
    meta_keys = list(meta_df['key'])
    q_keys = list(survey_df['key'])
    assert len([_ for _ in meta_keys if _ in q_keys]) == 0

    # divide questions
    test_q_keys = random.choices(q_keys, k=5)  # monkey patch, ['GUNRESPNOKIDSB_W26', 'WORLDDANGER_W26', 'GUNIDENTITY_W26', 'REASONGUNC_W26', 'GUNRESPKIDSC_W26'] for random seed 42
    train_q_keys = [_ for _ in q_keys if _ not in test_q_keys]

    # divide users
    test_user_idx = random.choices(range(len(resp_df)), k=int(len(resp_df)*0.1))
    test_resp_df = resp_df.iloc[test_user_idx]
    # train_user_idx = [i for i in range(len(resp_df)) if i not in test_user_idx]
    # train_resp_df = resp_df.iloc[train_user_idx]

    # get prompt
    prompt_name_mapping = {
        "persona_infer": '_not_used',  # this won't be used
        "persona_infer_full": '_not_used',  # this won't be used
        "history": 'experiment/prompts/prediction/predict_history.txt',
        "history_demo": 'experiment/prompts/prediction/predict_history_demo.txt',
        "history_persona": 'experiment/prompts/prediction/predict_history_persona.txt',
        "history_demo_persona": 'experiment/prompts/prediction/predict_history_demo_persona.txt',
        "demo": 'experiment/prompts/prediction/predict_demo.txt',
        "persona": 'experiment/prompts/prediction/predict_persona.txt',
        "demo_persona": 'experiment/prompts/prediction/predict_demo_persona.txt',
    }
    pred_prompt_name = prompt_name_mapping[args.exp_setting]
    with open(pred_prompt_name) as f:
        pred_prompt_template = f.read()

    # get flags
    persona_infer = args.exp_setting in ["persona_infer", "persona_infer_full"]
    persona_infer_full = args.exp_setting == "persona_infer_full"
    use_demo = args.exp_setting in ["vanilla_demo", "vanilla_demo_persona", "vanilla_no_history_demo", "vanilla_no_history_demo_persona", "vanilla_demo_persona_cot"]
    use_persona = args.exp_setting in ["vanilla_persona", "vanilla_demo_persona", "vanilla_no_history_persona", "vanilla_no_history_demo_persona", "vanilla_demo_persona_cot"]

    # preparing
    cnt = 0
    correct = 0
    logs = []
    if persona_infer:
        persona_mapping = {}
        persona_num = 5
        if persona_infer_full:
            train_user_idx = [i for i in range(len(resp_df)) if i not in test_user_idx]
            train_resp_df = resp_df.iloc[train_user_idx]
            test_resp_df = pd.concat((test_resp_df, train_resp_df), axis=0)
            persona_num = None
    if not persona_infer and use_persona:
        with open(args.persona_filename, 'r') as f:
            persona_mapping = json.load(f)

    if args.use_only_relevant_persona:
        assert args.query_to_persona_idx_mapping_filename is not None, "You need to provide a filename for query_to_persona_idx_mapping_filename"
        if os.path.exists(args.query_to_persona_idx_mapping_filename):
            with open(args.query_to_persona_idx_mapping_filename, 'r') as f:
                query_to_persona_idx_mapping = json.load(f)
        else:
            query_to_persona_idx_mapping = get_query_to_persona_idx_mapping(test_q_keys, survey_df)
            with open(args.query_to_persona_idx_mapping_filename, 'w') as f:
                json.dump(query_to_persona_idx_mapping, f, indent=4)

    # for representing personas
    format_strings = {
        'desccandvalue': "{description} ({candidate_values}): {inferred_value}",
        'descvalue': "{description}: {inferred_value}",
        'value': "{inferred_value}",
        'namedescvalue': "{name} ({description}): {inferred_value}",
        'namedesccandvalue': "{name} ({description}) ({candidate_values}): {inferred_value}",
        'namevalue': "{name}: {inferred_value}",
    }

    # for reviewing
    review_path = f'opinions_qa/review/{args.log_name[:-5]}'
    os.makedirs(review_path, exist_ok=True)
    print(f"\n\nReview at path: {review_path}\n\n")

    # main loop
    for user_idx, row in tqdm(test_resp_df.iterrows(), total=len(test_resp_df)):
        # construct user history
        user_history = []
        for q_key in train_q_keys:
            question = question_key_mapping[q_key]['question']
            references = question_key_mapping[q_key]['references']
            # user_history[f"Question: {question}; Reference: {references}"] = test_resp_df.at[user_idx, q_key]
            user_answer = test_resp_df.at[user_idx, q_key]
            if isinstance(user_answer, pd.Series):
                user_answer = user_answer.iloc[0]

            # from IPython import embed; embed()
            if pd.isna(user_answer):
                continue
            user_history.append(
                f"Question: {question} ({'/'.join(eval(references))}): {user_answer}"
            )
        user_history = '\n'.join(user_history)

        if persona_infer:
            all_personas = get_persona_values(file_key, user_history, model_name=args.persona_inference_model_name, persona_num=persona_num)  # TODO: the personas neede to be re-generated
            persona_mapping[user_idx] = all_personas
            with open(args.persona_filename, 'w') as f:
                json.dump(persona_mapping, f, indent=4)
            continue

        if use_demo:
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
            demo = [f"{k}: {v}" for k, v in demo.items()]
            demo = '\n'.join(demo)


        # for each legal test question, predict
        for q_idx, q_key in enumerate(test_q_keys):
            gold_answer = row[q_key]
            if pd.notna(gold_answer):
                question = question_key_mapping[q_key]['question']
                options = eval(question_key_mapping[q_key]['references'])
                random.shuffle(options)
                references = "/".join(options)
                input_dict = {
                    "user_history": user_history,
                    "question": question,
                    "options": references
                }
                if use_demo:
                    input_dict["demo"] = demo
                if use_persona:
                    all_personas = persona_mapping[str(user_idx)]
                    format_string = format_strings[args.persona_repr]
                    if args.use_only_relevant_persona:
                        relevant_idx = query_to_persona_idx_mapping[q_key]
                        filtered_personas = [
                            format_string.format(**{
                                'description': _['description'],
                                'candidate_values': '/'.join(_['candidate_values']),
                                'inferred_value': _['inferred_value'],
                                'name': _['name']
                            }) for idx, _ in enumerate(all_personas)
                            if idx in relevant_idx and _['level'] in args.persona_levels
                        ]
                    else:
                        # filtered_personas = [f"{_['name']} ({_['description']}): {_['inferred_value']}" for _ in all_personas if _['level'] in args.persona_levels]
                        filtered_personas = [
                            format_string.format(**{
                                'description': _['description'],
                                'candidate_values': '/'.join(_['candidate_values']),
                                'inferred_value': _['inferred_value'],
                                'name': _['name']
                            }) for _ in all_personas
                            if _['level'] in args.persona_levels
                        ]
                    filtered_personas = '\n'.join(filtered_personas)
                    input_dict["personas"] = filtered_personas
                prompt = pred_prompt_template.format(**input_dict)
                # raise Exception(prompt)
                # print(prompt)
                response = get_llm_response(prompt, model_id="anthropic.claude-3-sonnet-20240229-v1:0")
                
                if args.use_cot:
                    pred = extract_cot_prediction(response)
                    is_correct = pred == gold_answer
                else:
                    is_correct = response == gold_answer
                if is_correct:
                    correct += 1
                cnt += 1
                logs.append({
                    "user_idx": user_idx,
                    "q_idx": q_idx,
                    "is_correct": is_correct,
                    "question": question,
                    "references": references,
                    "prediction": response,
                    "gold_answer": gold_answer
                })
                with open(f"opinions_qa/output/{args.log_name}", 'w') as f:
                    json.dump(logs, f, indent=4)

                record = prompt + f"\nPrediction: {response}\nGold: {gold_answer}"
                with open(f'{review_path}/user_{user_idx}_question_{q_idx}_{q_key}.txt', 'w') as f:
                    f.write(record)


if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--use_only_relevant_persona', action='store_true')
    argparser.add_argument('--use_cot', action='store_true')
    argparser.add_argument('--query_to_persona_idx_mapping_filename', type=str, default=None)
    argparser.add_argument('--persona_levels', nargs='+',
                                               choices=['low', 'mid', 'high'])
    argparser.add_argument('--persona_repr', choices = ['namedescvalue',
                                                        'descvalue',
                                                        'namevalue',
                                                        'value',
                                                        'namedesccandvalue',
                                                        'desccandvalue'
                                                        ])
    argparser.add_argument('--exp_setting', choices = ['persona_infer',
                                                       'persona_infer_full',
                                                       'history',
                                                       'history_demo',
                                                       'history_persona',
                                                       'history_demo_persona',
                                                       'demo',
                                                       'persona',
                                                       'demo_persona',
                                                       ])
    argparser.add_argument('--survey_name', default='American_Trends_Panel_W26')
    argparser.add_argument('--log_name', required=True)
    argparser.add_argument('--persona_filename', type=str, default=None)
    argparser.add_argument('--persona_inference_model_name', type=str, default="sonnet")
    args = argparser.parse_args()
    main(args)
