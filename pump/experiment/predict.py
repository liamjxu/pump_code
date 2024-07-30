import argparse
import pandas as pd
import json
import random
from utils import list_s3_prefix, get_file_from_s3
from utils import get_llm_response, CLAUDE_NAME_MAPPING
from tqdm import tqdm


def get_persona_values(file_key, user_history, model_name="sonnet", persona_num=5):

    prompt_name = 'prompts/infer_persona_from_user_history.txt'
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
    test_q_keys = random.choices(q_keys, k=5)
    train_q_keys = [_ for _ in q_keys if _ not in test_q_keys]

    # divide users
    test_user_idx = random.choices(range(len(resp_df)), k=int(len(resp_df)*0.1))
    test_resp_df = resp_df.iloc[test_user_idx]
    # train_user_idx = [i for i in range(len(resp_df)) if i not in test_user_idx]
    # train_resp_df = resp_df.iloc[train_user_idx]

    # get prompt
    prompt_name_mapping = {
        "persona_infer": 'prompts/vanilla_predict.txt',  # this won't be used
        "persona_infer_full": 'prompts/vanilla_predict.txt',  # this won't be used
        "vanilla": 'prompts/vanilla_predict.txt',
        "vanilla_demo": 'prompts/vanilla_demo_predict.txt',
        "vanilla_persona": 'prompts/vanilla_persona_predict.txt',
        "vanilla_demo_persona": 'prompts/vanilla_demo_persona_predict.txt',
        "vanilla_no_history_demo": 'prompts/vanilla_no_history_demo_predict.txt',
        "vanilla_no_history_persona": 'prompts/vanilla_no_history_persona_predict.txt',
        "vanilla_no_history_demo_persona": 'prompts/vanilla_no_history_demo_persona_predict.txt',
    }
    pred_prompt_name = prompt_name_mapping[args.exp_setting]
    with open(pred_prompt_name) as f:
        pred_prompt_template = f.read()

    # get flags
    persona_infer = args.exp_setting in ["persona_infer", "persona_infer_full"]
    persona_infer_full = args.exp_setting == "persona_infer_full"
    use_demo = args.exp_setting in ["vanilla_demo", "vanilla_demo_persona", "vanilla_no_history_demo", "vanilla_no_history_demo_persona"]
    use_persona = args.exp_setting in ["vanilla_persona", "vanilla_demo_persona", "vanilla_no_history_persona", "vanilla_no_history_demo_persona"]

    # main loop
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

    for user_idx, row in tqdm(test_resp_df.iterrows(), total=len(test_resp_df)):
        # construct user history
        user_history = {}
        for q_key in train_q_keys:
            question = question_key_mapping[q_key]['question']
            references = question_key_mapping[q_key]['references']
            user_history[f"Question: {question}; Reference: {references}"] = test_resp_df.at[user_idx, q_key]

        if persona_infer:
            all_personas = get_persona_values(file_key, user_history, model_name=args.persona_inference_model_name, persona_num=persona_num)
            persona_mapping[user_idx] = all_personas
            with open(args.persona_filename, 'w') as f:
                json.dump(persona_mapping, f, indent=4)
            continue

        if use_demo:
            demo = row[meta_keys].to_dict()
        if use_persona:
            all_personas = persona_mapping[str(user_idx)]
            all_personas = {_['name']: _['inferred_value'] for _ in all_personas if _['level'] in ['high']}

        # for each legal test question, predict
        for q_idx, q_key in enumerate(test_q_keys):
            gold_answer = row[q_key]
            if pd.notna(gold_answer):
                question = question_key_mapping[q_key]['question']
                references = question_key_mapping[q_key]['references']
                input_dict = {
                    "user_history": user_history,
                    "question": question,
                    "options": references
                }
                if use_demo:
                    input_dict["demo"] = demo
                if use_persona:
                    input_dict["personas"] = all_personas
                prompt = pred_prompt_template.format(**input_dict)
                # raise Exception(prompt)
                print(prompt)
                response = get_llm_response(prompt, model_id="anthropic.claude-3-sonnet-20240229-v1:0")
                
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



if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--exp_setting', choices = ['persona_infer',
                                                       'persona_infer_full',
                                                       'vanilla',
                                                       'vanilla_demo',
                                                       'vanilla_persona',
                                                       'vanilla_demo_persona',
                                                       'vanilla_no_history_demo',
                                                       'vanilla_no_history_persona',
                                                       'vanilla_no_history_demo_persona',
                                                       ])
    argparser.add_argument('--survey_name', default='American_Trends_Panel_W26')
    argparser.add_argument('--log_name', required=True)
    argparser.add_argument('--persona_filename', type=str, default=None)
    argparser.add_argument('--persona_inference_model_name', type=str, default="sonnet")
    # argparser.add_argument('--phases', nargs='+',
    #                                     choices=['relevance', 'diversity'],  
    #                                     default=['diversity'])
    args = argparser.parse_args()
    main(args)
