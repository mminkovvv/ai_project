from openai import OpenAI
import json

client = OpenAI()


def calculate(a, b):
    return a + b


def get_weather(city):
    return f"The weather in {city} is sunny."


tools = [
    {
        "type": "function",
        "name": "calculate",
        "description": "Adds two numbers together.",
        "parameters": {
            "type": "object",
            "properties": {
                "a": {
                    "type": "number",
                    "description": "The first number"
                },
                "b": {
                    "type": "number",
                    "description": "The second number"
                }
            },
            "required": ["a", "b"]
        }
    },

    {
        "type": "function",
        "name": "get_weather",
        "description": "Gets the weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "The city name"
                }
            },
            "required": ["city"]
        }
    }
]

input = input("Ask something: ")
response = client.responses.create(
    model="gpt-5.6-luna",
    input=input,
    tools=tools
)


tool_outputs = []

for item in response.output:
    if item.type == "function_call":

        arguments = json.loads(item.arguments)

        if item.name == "calculate":
            result =calculate(
            arguments["a"],
            arguments["b"]
        )

        elif item.name =="get_weather":
            result = get_weather(
                arguments["city"]
            )
        tool_outputs.append({
            "type": "function_call_output",
            "call_id": item.call_id,
            "output": str(result)
        })


response = client.responses.create(
    model="gpt-5.6-luna",
    input=tool_outputs,
    previous_response_id=response.id
)


print(response.output_text)