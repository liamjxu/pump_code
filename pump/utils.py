import boto3
import json
from io import StringIO, BytesIO


brt = boto3.client(service_name='bedrock-runtime')
s3_client = boto3.client('s3')
bucket_name = 'probabilistic-user-modeling'


def get_llm_response(input_text, model_id="anthropic.claude-3-sonnet-20240229-v1:0"):
    # model_id: https://docs.aws.amazon.com/bedrock/latest/userguide/model-ids.html#model-ids-arns
    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1000,
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
                        "text": "[",
                    }
                ]
            },
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

    # text
    output = response_body.get('content')[0]['text']
    return output


def get_file(file_key='info.csv'):
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


def list_s3(prefix):
    response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
    if 'Contents' in response:
        # for obj in response['Contents']:
        #     print(obj['Key'])
        return [obj['Key'] for obj in response['Contents']]
    else:
        raise ValueError("No objects found with the specified prefix.")


def get_topics(mapping, question):
    for key in mapping.keys():
        if key.startswith(question):
            return mapping[key]
    return None
