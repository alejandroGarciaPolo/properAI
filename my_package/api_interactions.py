import openai 
import json
import requests

from tenacity import retry, stop_after_attempt, wait_random_exponential


class Config:
    _api_key = None

    @classmethod
    def set_api_key(cls, key):
        cls._api_key = key

    @classmethod
    def get_api_key(cls):
        if cls._api_key is None:
            raise ValueError("API key has not been set.")
        return cls._api_key


def create_function_json():
    function = {}
    
    function['name'] = input("Enter the function name: ")
    function['description'] = input("Enter the function description: ")

    # Defining parameter types that can be chosen
    available_types = ['string', 'integer', 'boolean']

    parameters = {
        "type": "object",
        "properties": {},
        "required": []
    }

    while True:
        param_name = input("Enter parameter name (or press enter to finish): ")
        if not param_name:
            break

        param_type = ""
        while param_type not in available_types:
            param_type = input(f"Enter parameter type ({'/'.join(available_types)}): ")

        param_desc = input("Enter parameter description: ")

        parameters['properties'][param_name] = {
            "type": param_type,
            "description": param_desc
        }

        if input("Is this parameter required? (yes/no): ").lower() == 'yes':
            parameters['required'].append(param_name)

    function['parameters'] = parameters

    # Printing the function JSON
    print("\nGenerated Function JSON:")
    print(json.dumps(function, indent=4))

def test_function(is_testing):
    print('it worked', is_testing)
# Add more functions as needed
# import my_package.api_interaction as api

# api.Config.set_api_key("your_api_key_here")

functions = [
{
    "name": "test_function",
    "description": "This is a testng function solely used to test your function calling ability, use it when requested",
    "parameters": {
        "type": "object",
        "properties": {
            "is_testing": {
                "type": "boolean",
                "description": "set to true when called since you would be testing"
            }
        },
        "required": [
            "is_testing"
        ]
    }
},
{
    "name": "create_function_json",
    "description": "This is a function that we can use to prompt the user to create a schema for a missing function",
    "parameters": {
        "type": "object",
        "properties": {},
        "required": []
    }
}
]
available_functions = {
    "test_function": test_function,
    "create_function_json": create_function_json}

@retry(wait=wait_random_exponential(multiplier=1, max=40), stop=stop_after_attempt(3))
def gpt_chat_and_execute(question, context=None, functions=functions, available_functions=available_functions, model="gpt-3.5-turbo-0613", function_call='auto'):
    # Send request to GPT
    api_key = Config.get_api_key()

    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer " + api_key,
    }
    
    messages = [{"role": "user", "content": question}]
    if context:
        for message in context:
            messages.append({"role": "system", "content": message})

    json_data = {"model": model, "messages": messages}
    
    if functions is not None:
        json_data.update({"functions": functions})
    if function_call is not None:
        json_data.update({"function_call": function_call})

    try:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=json_data,
        )
    except Exception as e:
        print("Unable to generate ChatCompletion response")
        print(f"Exception: {e}")
        return None

    # Execute function from response
    try:
        assistant_message = response.json()["choices"][0]["message"]
        # print(assistant_message)
        if 'function_call' in assistant_message:
            function_call = assistant_message['function_call']
            function_name = function_call['name']
            function_args = json.loads(function_call['arguments'])

            if function_name in available_functions:
                return available_functions[function_name](**function_args)
            else:
                raise ValueError(f"Function {function_name} not defined.")
        else:
            print(assistant_message['content'])
            return assistant_message['content']
    except Exception as e:
        print(f"Error executing function: {e}")
        return None

