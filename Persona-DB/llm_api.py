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
                    "prompt": f"\n\nHuman: {input}\n\nAssistant:",
                    "max_tokens_to_sample": 300,
                    "temperature": 0,
                    "top_p": 0.9,
                })


                modelId = 'anthropic.claude-3-sonnet-20240229-v1:0'
                accept = 'application/json'
                contentType = 'application/json'

                response = brt.invoke_model(body=body, modelId=modelId, accept=accept, contentType=contentType)

                response_body = json.loads(response.get('body').read())

                # text
                output = response_body.get('completion')

                break
            except Exception as e:  # wait 1 second and try again
                print(e)
                if e.type == "invalid_request_error":
                    break
                time.sleep(2)
                print("waiting 2 seconds")

    return output
