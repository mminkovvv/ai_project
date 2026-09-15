from openai import OpenAI

client = OpenAI()

response = client.responses.create(
    model="gpt-5.6-luna",
    input="Explain what an API is in on simple sentence."
)
print(response.output_text)