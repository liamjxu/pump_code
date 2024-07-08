import boto3
import json

brt = boto3.client(service_name='bedrock-runtime')

def get_claude(input_text):
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
            }
        ]
    })

    inputs = {
        "modelId": "anthropic.claude-3-sonnet-20240229-v1:0",
        "contentType": "application/json",
        "accept": "application/json",
        "body": body
    }

    response = brt.invoke_model(**inputs)

    response_body = json.loads(response.get('body').read())

    # text
    output = response_body.get('content')[0]['text']
    return output
