from openai import OpenAI
import tiktoken
import pandas as pd
# import numpy as np
from numpy.linalg import norm 
from numpy import dot
import os
import pickle

from tenacity import retry, stop_after_attempt, wait_random_exponential
import json
import requests

client = OpenAI(api_key="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")

def num_tokens_from_string(string: str, encoding_name="cl100k_base") -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

def cosine_similarity(vec_a, vec_b):
    """Compute the cosine similarity between two vectors."""
    dot_product = dot(vec_a, vec_b)
    norm_a = norm(vec_a)
    norm_b = norm(vec_b)
    similarity = dot_product / (norm_a * norm_b)
    return similarity

def get_embedding(text, model="text-embedding-ada-002"):
   text = text.replace("\n", " ")
   return client.embeddings.create(input = [text], model=model).data[0].embedding

# def embedding_search(function_object, n=5, df=None, pickle_file='functionPickle.pkl'):
#     # Load the DataFrame from pickle if df is None
#     if df is None:
#         if os.path.exists(pickle_file):
#             with open(pickle_file, 'rb') as file:
#                 df = pickle.load(file)
#         else:
#             return None  # Return None if no pickled DataFrame is available

#     # Get the embedding of the query from the FunctionObject
#     query_embedding = function_object.embedding
#     if query_embedding is None:
#         function_object.generate_embedding()
#         query_embedding = function_object.embedding

#     # Compute cosine similarities
#     df['similarities'] = df['Embedding'].apply(lambda x: cosine_similarity(x, query_embedding))

#     # Sort the DataFrame by similarities and get the top n results
#     top_results = df.sort_values('similarities', ascending=False).head(n)

#     # Return a list of tuples (FunctionObject, similarity score)
#     return [(row['FunctionObject'], row['similarities']) for index, row in top_results.iterrows()]
# def embedding_search(query, n=5, df=None, pickle_file='functionPickle.pkl'):
#     # Load the DataFrame from pickle if df is None
#     if df is None:
#         if os.path.exists(pickle_file):
#             with open(pickle_file, 'rb') as file:
#                 df = pickle.load(file)
#         else:
#             return None  # Return None if no pickled DataFrame is available

#     # Get the embedding of the query
#     query_embedding = get_embedding(query)

#     # Compute cosine similarities
#     df['similarities'] = df['Embedding'].apply(lambda x: cosine_similarity(x, query_embedding))

#     # Sort the DataFrame by similarities and get the top n results
#     top_results = df.sort_values('similarities', ascending=False).head(n)

#     # Return a list of FunctionObjects
#     return [row['FunctionObject'] for index, row in top_results.iterrows()]
def embedding_search(query, n=5, df=None, pickle_file='functionPickle.pkl'):
    # Load the DataFrame from pickle if df is None
    if df is None:
        if os.path.exists(pickle_file):
            with open(pickle_file, 'rb') as file:
                df = pickle.load(file)
        else:
            return None  # Return None if no pickled DataFrame is available

    # Get the embedding of the query
    query_embedding = get_embedding(query)

    # Compute cosine similarities
    df['similarities'] = df['Embedding'].apply(lambda x: cosine_similarity(x, query_embedding))

    # Sort the DataFrame by similarities and get the top n results
    top_results = df.sort_values('similarities', ascending=False).head(n)

    # Return a list of FunctionObject's json_data
    return [row['FunctionObject'].json_data for index, row in top_results.iterrows()]






# def embed_and_add_to_df(func_obj=None, pickle_file='functionPickle.pkl'):
#     # Check if the DataFrame exists in pickle
#     if os.path.exists(pickle_file):
#         with open(pickle_file, 'rb') as file:
#             df = pickle.load(file)
#     else:
#         # Create a new DataFrame if pickle doesn't exist
#         df = pd.DataFrame(columns=['FunctionObject', 'Embedding'])

#     # If function object is provided
#     if func_obj is not None:
#         # Check if embedding exists, generate if not
#         if func_obj.embedding is None:
#             func_obj.generate_embedding()  # Assuming this method sets the embedding

#         # Add to DataFrame
#         df = df.append({'FunctionObject': func_obj, 'Embedding': func_obj.embedding}, ignore_index=True)

#         # Save the DataFrame to pickle
#         with open(pickle_file, 'wb') as file:
#             pickle.dump(df, file)

#     return df
def embed_and_add_to_df(func_obj=None, pickle_file='functionPickle.pkl'):
    # Check if the DataFrame exists in pickle
    if os.path.exists(pickle_file):
        with open(pickle_file, 'rb') as file:
            df = pickle.load(file)
    else:
        df = pd.DataFrame(columns=['FunctionObject', 'Embedding'])

    if func_obj is not None:
        if func_obj.embedding is None:
            func_obj.generate_embedding()

        new_row = pd.DataFrame([{'FunctionObject': func_obj, 'Embedding': func_obj.embedding}])
        df = pd.concat([df, new_row], ignore_index=True)

        with open(pickle_file, 'wb') as file:
            pickle.dump(df, file)

    return df


def update_function_descriptions(pickle_file='functionPickle.pkl'):
    if not os.path.exists(pickle_file):
        print("No pickle file found.")
        return

    with open(pickle_file, 'rb') as file:
        df = pickle.load(file)

    for index, row in df.iterrows():
        print("\nCurrent description:", row['FunctionObject'].query_description)
        new_desc = input("Enter new description (or press Enter to keep current//done to finish//): ").strip()

        if new_desc.lower() == 'done':
            break
        elif new_desc:
            row['FunctionObject'].give_human_description(new_desc)

    # Save changes back to the pickle file
    with open(pickle_file, 'wb') as file:
        pickle.dump(df, file)
    print("Changes saved.")

class FunctionObject:
    def __init__(self, json_data):
        self.json_data = json_data
        self.name = json_data['name']
        self.description = json_data['description']
        self.query_description = self.create_query_description_base()
        self.embedding = None

    def create_query_description_base(self):
        # Generate a human-readable description based on JSON data
        # This can be a combination of 'name', 'description', and key elements from 'parameters'
        return f"{self.description} with parameters like {', '.join(self.json_data['parameters']['properties'].keys())}"

    def generate_embedding(self):
        # Generate and store the embedding for the query_description
        self.embedding = get_embedding(self.query_description)

    def give_human_description(self, human_desc: str):
        """Example: Get current weather function which provides weather information in a given location like San Francisco, CA, supporting both Celsius and Fahrenheit units."""
        self.query_description = human_desc


####api
        
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

def initialize_or_load_functions(retrieve=True, pickle_file='functionsArray.pkl'):
    if retrieve and os.path.exists(pickle_file):
        with open(pickle_file, 'rb') as file:
            data = pickle.load(file)
            return data['functions'], data['available_functions']

    # Predefined functions (used when pickle file doesn't exist or retrieve is False)
    functionsJson = [
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

    functions = []
    for fun in functionsJson:
        functions.append(FunctionObject(fun))

    available_functions = {
        "test_function": test_function,
        "create_function_json": create_function_json}

    # Save to pickle file
    with open(pickle_file, 'wb') as file:
        pickle.dump({'functions': functions, 'available_functions': available_functions}, file)

    return functions, available_functions


def add_to_function_storage(retrieve=False, pickle_file='functionsArray.pkl'):
    if retrieve == False:
        new_function_json = create_function_json()  # Assume this returns a JSON structure
        new_function_name = new_function_json['name']
        new_function = None  # Replace with the actual implementation of the new function

        if not os.path.exists(pickle_file):
            functions, available_functions = initialize_or_load_functions(retrieve=False, pickle_file=pickle_file)
        else:
            with open(pickle_file, 'rb') as file:
                data = pickle.load(file)
                functions, available_functions = data['functions'], data['available_functions']

        # Append new function
        functions.append(new_function_json)
        available_functions[new_function_name] = new_function

        # Save updated data
        with open(pickle_file, 'wb') as file:
            pickle.dump({'functions': functions, 'available_functions': available_functions}, file)
    else:
        functions, available_functions = initialize_or_load_functions(retrieve=True, pickle_file=pickle_file)
        return functions, available_functions
    

####api
    
inputQ = input("\nWould you like to add a function? y/n")

if inputQ.lower() == 'y':
    add_to_function_storage()

functions, available_functions = initialize_or_load_functions(retrieve=True) #####yes or yes load the base, functions is not nessesary however, avalible is

print(f"current df {embed_and_add_to_df()}")
print(f'number of functions avalible {len(functions)}')


inputQ = input("\nWould you like to run the pre-setup? y/n")

if inputQ.lower() == 'y':
    
    for fun in functions:
        embed_and_add_to_df(fun)

inputQ = input("\n Would you like to give them human values? y/n")

if inputQ.lower() == 'y':
    update_function_descriptions()

print('fetching df')
df = embed_and_add_to_df() ##fetches the df
print('done.')
print('starting chat\n')


@retry(wait=wait_random_exponential(multiplier=1, max=40), stop=stop_after_attempt(3))
def gpt_chat_and_execute_function_bank(question, context=None, model="gpt-3.5-turbo-0613", function_call='auto'):
    # Send request to GPT
    api_key = "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"

    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer " + api_key,
    }
    
    messages = [{"role": "user", "content": question}]
    if context:
        for message in context:
            messages.append({"role": "system", "content": message})

    json_data = {"model": model, "messages": messages}
    
    functions = embedding_search(query=question,df=df)
    print('TESTINGGG', f'functions: {functions}')
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






while(True):
    message = input('Your question:')
    response = gpt_chat_and_execute_function_bank(message)
    print(response, "\n")







