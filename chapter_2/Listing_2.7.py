from litellm import completion
import os

os.environ['MISTRAL_API_KEY'] = ""
response = completion(
    model="mistral/mistral-tiny",
    messages=[
        {"role": "user", "content": "What is the capital of Russia?"}
    ],
)
print(response)
