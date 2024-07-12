import boto3
import json
import torch
import torch.nn.functional as F
from io import StringIO, BytesIO
from torch import Tensor
from dataclasses import dataclass


brt = boto3.client(service_name='bedrock-runtime')
s3_client = boto3.client('s3')
bucket_name = 'probabilistic-user-modeling'


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

def get_llm_response(
    input_text,
    model_id = "anthropic.claude-3-sonnet-20240229-v1:0",
    prefill = None,
    max_tokens = 1000,
    return_full = False
):
    # model_id: https://docs.aws.amazon.com/bedrock/latest/userguide/model-ids.html#model-ids-arns
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

    response = brt.invoke_model(**inputs)

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


