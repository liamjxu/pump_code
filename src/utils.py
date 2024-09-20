import boto3
import json
import torch
import os
import torch.nn.functional as F
import pandas as pd
import re
from io import StringIO, BytesIO
from torch import Tensor
from dataclasses import dataclass, asdict
from botocore.config import Config
from typing import List, Dict
from tqdm import tqdm
import numpy as np


brt = boto3.client(service_name='bedrock-runtime', region_name="us-east-1", config=Config(read_timeout=120))
s3_client = boto3.client('s3')
bucket_name = 'probabilistic-user-modeling'

TEST_KEY_MAPPING = {
    "American_Trends_Panel_W26": ["WORLDDANGER_W26", "WORRYE_W26", "GUN_W26", "GUNIDENTITY_W26", "GUNSOCIETY_W26", "GUNCOMMUNITY_W26", "GUNRESPKIDSC_W26"],
    "American_Trends_Panel_W27": ["WORK4A_W27", "WORK4B_W27", "WORK4C_W27", "ROBJOB2_W27", "ROBJOB3A_W27", "ROBJOB3B_W27", "ROBJOB4C_W27", "ROBJOB5B_W27", "ROBJOB5D_W27"],
    "American_Trends_Panel_W29": ["MASC2AF1_W29", "SEENMASC_W29", "SEENFEM_W29", "MAN1E_W29", "BOYSF1A_W29", "BOYSF1B_W29", "GIRLSF2A_W29"],
    "American_Trends_Panel_W32": ["SOCTRUST_W32", "COMATTACH_W32", "JOBSFUTURE_W32", "NEIGHKEYS_W32", "IMMIMPACT_W32", "ETHNCMAJ_W32", "WHADVANT_W32", "GAYMARR2_W32", "ABORTION_W32"],
    "American_Trends_Panel_W34": ["SCI1_W34", "EAT1_W34", "FUD35_W34", "MED2G_W34", "MED6D_W34", "MED6E_W34"],
}

CLAUDE_NAME_MAPPING = {
    "sonnet": "anthropic.claude-3-sonnet-20240229-v1:0",
    "haiku": "anthropic.claude-3-haiku-20240307-v1:0",
    "llama8b3": "meta.llama3-8b-instruct-v1:0",
    "llama70b3": "meta.llama3-70b-instruct-v1:0",
    "sonnet35": "anthropic.claude-3-5-sonnet-20240620-v1:0",
    "mistrallarge": "mistral.mistral-large-2402-v1:0",
}

@dataclass
class PersonaDimension:
    name: str  # a concise name of the persona aspect
    description: str  # a detailed description of the persona aspect
    level: str  # the abstractness level of this persona dimension, choose from ['low', 'mid', 'high']
    candidate_values: str  # the candidate values of this persona dimension
    
    def __repr__(self):
        return (f'    PersonaDimension(\n'
                f'        name="{self.name}",\n'
                f'        description="{self.description}",\n'
                f'        level="{self.level}",\n'
                f'        candidate_values={self.candidate_values}\n'
                f'    )')
    
    def to_dict(self):
        return {
            "name": self.name,
            "description": self.description,
            "level": self.level,
            "candidate_values": self.candidate_values
        }

    def linearize(self):
        return f"{self.name}: {self.description}. Candidate values: {self.candidate_values}"


def persona_dim_object_to_dict(persona_object: PersonaDimension) -> Dict:
    return asdict(persona_object)


def persona_dim_dict_to_object(persona_dict: Dict) -> PersonaDimension:
    return PersonaDimension(**persona_dict)


def persona_dim_object_list_to_dict_list(persona_object_list: List[PersonaDimension]) -> List[Dict]:
    return [persona_dim_object_to_dict(entry) for entry in persona_object_list]


def persona_dim_dict_list_to_object_list(persona_dict_list: List[Dict]) -> List[PersonaDimension]:
    return [persona_dim_dict_to_object(persona) for persona in persona_dict_list]


def get_llm_response(
    input_text,
    model_id = "anthropic.claude-3-haiku-20240307-v1:0",  # model_id: https://docs.aws.amazon.com/bedrock/latest/userguide/model-ids.html#model-ids-arns
    prefill = None,
    max_tokens = 1000,
    return_full = False
):
    if model_id.startswith("anthropic"):
        res = _get_llm_response_anthropic(input_text, model_id, prefill, max_tokens, return_full)
    if model_id.startswith("meta"):
        res = _get_llm_response_meta(input_text, model_id, max_tokens)
    if model_id.startswith("mistral"):
        res = _get_llm_response_mistral(input_text, model_id, max_tokens)
    
    return res


def _get_llm_response_mistral(
    input_text,
    model_id,
    max_tokens = 1000,
):

    conversation = [
        {
            "role": "user",
            "content": [{"text": input_text}],
        }
    ]

    inputs = {
        "modelId": model_id,
        "messages": conversation,
        "inferenceConfig": {"maxTokens": max_tokens, "temperature": 0},
        "additionalModelRequestFields": {}
    }

    failure = 0
    while failure < 3:
        try:
            response = brt.converse(**inputs)
            break
        except Exception as e:
            print(f"Failure {(failure/3)}: brt.invoke_model(**inputs) failed. Error Message: {e}.")
            print("Retrying.")
            failure += 1

    response_text = response["output"]["message"]["content"][0]["text"]

    return response_text
    

def _get_llm_response_meta(
    input_text,
    model_id,
    max_tokens = 1000,
):

    body = json.dumps({
        "temperature": 0,
        "prompt": input_text,
        "max_gen_len": max_tokens
    })

    inputs = {
        "modelId": model_id,
        "contentType": "application/json",
        "accept": "application/json",
        "body": body
    }
    failure = 0
    while failure < 3:
        try:
            response = brt.invoke_model(**inputs)
            break
        except Exception as e:
            print(f"Failure {(failure/3)}: brt.invoke_model(**inputs) failed. Error Message: {e}.")
            print("Retrying.")
            failure += 1

    response_body = json.loads(response.get('body').read())

    # text
    # from IPython import embed; embed()
    output = response_body['generation']
    return output
    

def _get_llm_response_anthropic(
    input_text,
    model_id = "anthropic.claude-3-haiku-20240307-v1:0",  # model_id: https://docs.aws.amazon.com/bedrock/latest/userguide/model-ids.html#model-ids-arns
    prefill = None,
    max_tokens = 1000,
    return_full = False
):
    if prefill is not None:
        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "temperature": 0,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": input_text,
                        }
                    ]
                },
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": prefill,
                        }
                    ]
                },
            ]
        })
    else:
        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "temperature": 0,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": input_text,
                        }
                    ]
                }
            ]
        })

    inputs = {
        "modelId": model_id,
        "contentType": "application/json",
        "accept": "application/json",
        "body": body
    }
    # from IPython import embed; embed()
    failure = 0
    while failure < 3:
        try:
            response = brt.invoke_model(**inputs)
            break
        except Exception as e:
            print(f"Failure {(failure/3)}: brt.invoke_model(**inputs) failed. Error Message: {e}.")
            print("Retrying.")
            failure += 1

    response_body = json.loads(response.get('body').read())

    if return_full:
        return response_body
    else:
        # text
        output = response_body.get('content')[0]['text']
        return output


def get_file_from_s3(file_key='info.csv'):
    if file_key.endswith('.npy'):
        s3_object = s3_client.get_object(Bucket=bucket_name, Key=file_key)
        s3_data = s3_object['Body'].read()
        return BytesIO(s3_data)

    elif file_key.endswith('.csv'):
        s3_object = s3_client.get_object(Bucket=bucket_name, Key=file_key)
        s3_data = s3_object['Body'].read().decode('utf-8')
        return StringIO(s3_data)

    else:
        raise ValueError('unknown file type')


def list_s3_prefix(prefix):
    response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
    if 'Contents' in response:
        return [obj['Key'] for obj in response['Contents']]
    else:
        raise ValueError("No objects found with the specified prefix.")


def get_topics(mapping, question):
    for key in mapping.keys():
        if key.startswith(question):
            return mapping[key]
    return None


def last_token_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


def get_detailed_instruct(task_description: str, query: str) -> str:
    return f'Instruct: {task_description}\n{query}'


def get_formatted_persona_dim(row):
    task = 'Given a persona dimension description, retrieve semantically similar persona dimension descriptions.'
    persona = f"{row['name']}: {row['description']}. Candidate values: {row['candidate_values']}"
    return get_detailed_instruct(task, persona)


def get_cosine_similarity_metrics(root_dir, tokenizer, model):
    root_dir = os.path.join(root_dir, 'cleaned')
    res = []
    for file_name in [_ for _ in os.listdir(root_dir) if not _.startswith('judged')]:
        with open(os.path.join(root_dir, file_name)) as f:
            data = json.load(f)
        data = [_.linearize() for _ in persona_dim_dict_list_to_object_list(data)]
        batch_dict = tokenizer(data, max_length=4096, padding=True, truncation=True, return_tensors="pt")
        with torch.inference_mode():
            outputs = model(**batch_dict)
            embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])

        normalized_embeddings = F.normalize(embeddings, p=2, dim=1)

        # Calculate pairwise cosine similarity
        cosine_similarity_matrix = torch.mm(normalized_embeddings, normalized_embeddings.t())

        # Extract upper triangle without diagonal
        triu_indices = torch.triu_indices(cosine_similarity_matrix.size(0), cosine_similarity_matrix.size(1), offset=1)
        pairwise_cosine_similarities = cosine_similarity_matrix[triu_indices[0], triu_indices[1]]

        # Define thresholds
        thresholds = [50, 70, 90]

        percentiles = np.percentile(pairwise_cosine_similarities.numpy(), thresholds)

        percentile_dict = {
            "low": percentiles[0],
            "mid": percentiles[1],
            "high": percentiles[2]
        }

        res.append(percentile_dict)

        df = pd.DataFrame(res)

        return df.mean().to_dict()
    

def get_relevant_ratio(root_dir, survey_topics_mapping):
    root_dir = os.path.join(root_dir, 'cleaned')

    with open(f'prompts/determine_relevance_zero.txt') as f:
        prompt_template = f.read()

    res = []
    for file_name in tqdm([_ for _ in os.listdir(root_dir) if not _.startswith('judged')]):
        with open(os.path.join(root_dir, file_name)) as f:
            data = json.load(f)
        persona_dimensions = ['[\n' + repr(dim) + '\n]' for dim in persona_dim_dict_list_to_object_list(data)]
        for dim_idx, dim in enumerate(persona_dimensions):
            pattern = re.compile(r'American_Trends_Panel_W\d{2}')
            survey_name = pattern.findall(file_name)[0]
            topics = survey_topics_mapping[survey_name]
            input_dict = {
                "persona_dimensions": dim,
                "survey_topics": topics
            }
            prompt = prompt_template.format(**input_dict)
            response = get_llm_response(prompt, max_tokens=2)
            if response.lower().startswith('yes'):
                res.append(1)
            else:
                res.append(0)
            data[dim_idx]['relevancy_judgement'] = response

        # print(os.path.join(root_dir, f"judged_{file_name}"))
        with open(os.path.join(root_dir, f"judged_{file_name}"), 'w') as f:
            json.dump(data, f, indent=4)
    
    return sum(res) / len(res)

