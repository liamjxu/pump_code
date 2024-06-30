import time
import boto3
import json
from utils.utils import batchify

brt = boto3.client(service_name='bedrock-runtime')


def chatbot(input, model="bedrock-runtime", response_format="text", batch_size=1500, seed=42):

    output=None
    if "gpt" in model:
        while True:
            try:
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
                                    "text": input,
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
                break

            except Exception as e:  # wait 1 second and try again
                print(e)
                if e.type == "invalid_request_error":
                    break
                time.sleep(2)
                print("waiting 2 seconds")

    return output
