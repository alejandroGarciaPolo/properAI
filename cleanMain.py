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

###FUNCTIONS
def extract_json_from_string(input_string):
    """
    Extracts a JSON string from the input string. The JSON string must be enclosed
    between triple backticks (```json and ```) in the input string.

    Parameters:
    - input_string (str): The string to search for the JSON content.

    Returns:
    - str or None: The extracted JSON string if found, otherwise None.
    """
    # Define the start and end markers for the JSON content
    start_marker = "```json"
    end_marker = "```"

    # Find the start and end positions of the JSON content
    start_pos = input_string.find(start_marker)
    end_pos = input_string.find(end_marker, start_pos + len(start_marker))

    # Check if both markers are found and extract the content if they are
    if start_pos != -1 and end_pos != -1:
        # Add the length of the start marker to start_pos to exclude it from the result
        json_content = input_string[start_pos + len(start_marker):end_pos].strip()
        return json_content

    return None

def create_json_agent(message):
    completion = client.chat.completions.create(
    model="gpt-4-1106-preview",
    messages=[
        {"role": "system", "content": """You are an expert at generating JSON schemas for functions. You have a deep understanding of function parameters and their types, and you're skilled in translating these into detailed JSON schemas. Whether given a function definition, like a Python function or a detailed description of its behavior and parameters, you can extract a JSON schema that accurately represents the function.

    Your task is to help users create valid JSON schemas that reflect the structure and requirements of the functions they describe. These schemas should detail the function's name, description, and parameters, including parameter types and whether they are required.

    For example, given a Python function like def test_function(is_testing):, you will generate a JSON schema that captures its essence. Here's an example of how you would represent the test_function in JSON schema:

    json
    Copy code
    {
        "name": "test_function",
        "description": "This is a testing function solely used to test your function calling ability, use it when requested",
        "parameters": {
            "type": "object",
            "properties": {
                "is_testing": {
                    "type": "boolean",
                    "description": "Set to true when called since you would be testing"
                }
            },
            "required": [
                "is_testing"
            ]
        }
    }

    Your role includes aiding in debugging issues with the schemas and providing guidance on modifying them as needed.

    Remember to adhere to the JSON schema standards and best practices, ensuring that the schemas are not only valid but also practical and useful for the users' needs."""},
        {"role": "user", "content": message}
    ]
    )
    responses = completion.choices[0].message.content
    cleaned_json = extract_json_from_string(responses)
    print(cleaned_json)
    return cleaned_json



###FUNCTIONS


def num_tokens_from_string(string: str, encoding_name="cl100k_base") -> int:
    """
    Calculate the number of tokens in a given text string using a specified tokenizer.

    Parameters:
    - string (str): The text string for which to calculate the token count.
    - encoding_name (str, optional): The name of the tokenizer encoding to use. Defaults to "cl100k_base".

    Returns:
    - int: The number of tokens in the provided string.
    """
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

def cosine_similarity(vec_a, vec_b):
    """
    Compute the cosine similarity between two vectors.

    Parameters:
    - vec_a (array-like): The first vector.
    - vec_b (array-like): The second vector.

    Returns:
    - float: The cosine similarity score between the two vectors.
    """
    dot_product = dot(vec_a, vec_b)
    norm_a = norm(vec_a)
    norm_b = norm(vec_b)
    similarity = dot_product / (norm_a * norm_b)
    return similarity

def get_embedding(text, model="text-embedding-ada-002"):
   """
    Retrieve the embedding for a given text string using the OpenAI API.

    Parameters:
    - text (str): The text string to be embedded.
    - model (str, optional): The embedding model to use. Defaults to "text-embedding-ada-002".

    Returns:
    - list: The embedding vector for the input text.
    """
   text = text.replace("\n", " ")
   return client.embeddings.create(input = [text], model=model).data[0].embedding

def embedding_search(query, n=5, df=None, pickle_file='functionPickle.pkl'):
    """
    Search for the top N function objects related to a query based on cosine similarity of embeddings.

    Parameters:
    - query (str): The query string to search for.
    - n (int, optional): The number of top results to return. Defaults to 5.
    - df (pd.DataFrame, optional): A DataFrame containing function objects and their embeddings. 
                                   If None, loads from a pickle file. Defaults to None.
    - pickle_file (str, optional): The path to the pickle file containing the DataFrame. 
                                   Defaults to 'functionPickle.pkl'.

    Returns:
    - list: A list of JSON data of the top N related function objects.
    """
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





def embed_and_add_to_df(func_obj=None, pickle_file='functionPickle.pkl'):
    """
    Embed a function object and add it to a DataFrame, then save the DataFrame to a pickle file.

    Parameters:
    - func_obj (FunctionObject, optional): The function object to embed and add. Defaults to None.
    - pickle_file (str, optional): Path to the pickle file to save the DataFrame. Defaults to 'functionPickle.pkl'.

    Returns:
    - pd.DataFrame: The updated DataFrame containing the function object and its embedding.
    """
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
    """
    Interactively update the human-readable descriptions of function objects stored in a DataFrame.

    Parameters:
    - pickle_file (str, optional): Path to the pickle file containing the DataFrame. Defaults to 'functionPickle.pkl'.

    Returns:
    - None
    """
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
    """
    A class representing a function object, including its JSON data, name, description, query description, and embedding.

    Attributes:
    - json_data (dict): The JSON data of the function.
    - name (str): The name of the function.
    - description (str): The description of the function.
    - query_description (str): A human-readable description used for querying.
    - embedding (list): The embedding vector of the function.

    Methods:
    - create_query_description_base: Generates a base human-readable description.
    - generate_embedding: Generates and stores the embedding for the query_description.
    - give_human_description: Updates the human-readable description of the function.
    """
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
    """
    Interactively prompts the user to create a JSON schema for a new function. 
    This includes the function's name, description, parameters, and parameter types.

    Returns:
    - dict: A dictionary representing the JSON schema of the newly created function.
    """
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
    """
    Example function to demonstrate the function calling ability. 
    It prints a message indicating that the function was successfully called.

    Parameters:
    - is_testing (bool): A flag to indicate whether the function is being called for testing purposes.
    """
    # print('it worked', is_testing)
    return f'it worked, param: {is_testing}'

def initialize_or_load_functions(retrieve=True, pickle_file='functionsArray.pkl'):
    """
    Initialize or load function objects and their availability from a pickle file. 
    If the file doesn't exist, initializes with predefined functions.

    Parameters:
    - retrieve (bool, optional): If True, tries to load from the pickle file. 
                                 If False, initializes with predefined functions. Defaults to True.
    - pickle_file (str, optional): The path to the pickle file. Defaults to 'functionsArray.pkl'.

    Returns:
    - tuple: A tuple containing two elements; the first is a list of `FunctionObject` instances, 
             and the second is a dictionary mapping function names to their respective callable entities.
    """
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
    },
    {
    "name": "cosine_similarity",
    "description": "Compute the cosine similarity between two vectors.",
    "parameters": {
        "type": "object",
        "properties": {
            "vec_a": {
                "type": "array",
                "items": {
                    "type": "number"
                },
                "description": "The first vector (array-like)."
            },
            "vec_b": {
                "type": "array",
                "items": {
                    "type": "number"
                },
                "description": "The second vector (array-like)."
            }
        },
        "required": ["vec_a", "vec_b"]
    },
    "returns": {
        "description": "The cosine similarity score between the two vectors.",
        "type": "number"
    }
}
    ]

    functions = []
    for fun in functionsJson:
        functions.append(FunctionObject(fun))

    available_functions = {
        "test_function": test_function,
        "create_function_json": create_function_json,
        "cosine_similarity": cosine_similarity}

    # Save to pickle file
    with open(pickle_file, 'wb') as file:
        pickle.dump({'functions': functions, 'available_functions': available_functions}, file)

    return functions, available_functions


def add_to_function_storage(retrieve=False, pickle_file='functionsArray.pkl'):
    """
    Add a new function to the function storage. If retrieve is False, it prompts the user to create a new function. 
    If the pickle file doesn't exist, initializes with predefined functions and adds the new function.

    Parameters:
    - retrieve (bool, optional): Flag to determine whether to retrieve existing functions or add a new one. Defaults to False.
    - pickle_file (str, optional): The path to the pickle file for storing functions. Defaults to 'functionsArray.pkl'.
    
    Returns:
    - None
    """
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
create_json_agent("""Hi please generate a schema for this "def cosine_similarity(vec_a, vec_b):
    \"\"\"
    Compute the cosine similarity between two vectors.

    Parameters:
    - vec_a (array-like): The first vector.
    - vec_b (array-like): The second vector.

    Returns:
    - float: The cosine similarity score between the two vectors.
    \"\"\"
    dot_product = dot(vec_a, vec_b)
    norm_a = norm(vec_a)
    norm_b = norm(vec_b)
    similarity = dot_product / (norm_a * norm_b)
    return similarity""")
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
def gpt_chat_and_execute_function_bankDIRECT(question, context=None, model="gpt-3.5-turbo-0613", function_call='auto'):
    """
    Sends a question to the GPT model and executes a function call based on the response. 
    The function call is determined by the model's response or by the most relevant function found in the available functions.

    Parameters:
    - question (str): The user's question or input to be sent to the GPT model.
    - context (list, optional): A list of previous messages for context. Defaults to None.
    - model (str, optional): The GPT model to be used. Defaults to "gpt-3.5-turbo-0613".
    - function_call (str, optional): The type of function call to execute. Defaults to 'auto'.

    Returns:
    - str or None: The response from the GPT model or the output of the executed function, or None in case of an error.
    """
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

@retry(wait=wait_random_exponential(multiplier=1, max=40), stop=stop_after_attempt(3))
def gpt_chat_and_execute_function_bank(question, context=None, model="gpt-3.5-turbo-0613", function_call='auto'):
    """
    Sends a question to the GPT model and executes a function call based on the response. 
    The function call is determined by the model's response or by the most relevant function found in the available functions.

    Parameters:
    - question (str): The user's question or input to be sent to the GPT model.
    - context (list, optional): A list of previous messages for context. Defaults to None.
    - model (str, optional): The GPT model to be used. Defaults to "gpt-3.5-turbo-0613".
    - function_call (str, optional): The type of function call to execute. Defaults to 'auto'.

    Returns:
    - str or None: The response from the GPT model or the output of the executed function, or None in case of an error.
    """
    # Prepare headers and initial message
    api_key = "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"
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer " + api_key,
    }
    messages = [{"role": "user", "content": question}]
    if context:
        for message in context:
            messages.append({"role": "system", "content": message})

    # Initial request data
    json_data = {"model": model, "messages": messages}
    
    # Get relevant functions
    functions = embedding_search(query=question, df=df)
    if functions is not None:
        json_data.update({"functions": functions})
    if function_call is not None:
        json_data.update({"function_call": function_call})

    # Send initial request to GPT
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

    # Process response and execute function calls
    function_responses = []
    try:
        assistant_message = response.json()["choices"][0]["message"]
        print('assistant')
        print(assistant_message)
        print('assistant')
        # print('PREDEBUG', assistant_message)
        if 'function_call' in assistant_message:
            # print('there is')

            # As 'function_call' is a dictionary, not a list, we don't need a loop here
            tool_call = assistant_message['function_call']
            # print('INN')
            function_name = tool_call['name']
            # print(f'name {function_name}')

            # Parse the 'arguments' JSON string into a Python dictionary
            function_args = json.loads(tool_call['arguments'])
            # print('arguments', function_args)
            # print('debug', tool_call)

            if function_name in available_functions:
                function_response = available_functions[function_name](**function_args)
                function_responses.append({
                   
                    "role": "user",
                    "content": f"This is a hidden system message that just shows you what the function returned, answer the previus user message given that this is what it evaluated too, only pay attention to the values not the prompt I am giving you now: {function_response}"
                })
            else:
                raise ValueError(f"Function {function_name} not defined.")
        else:
            # print(assistant_message['content'])
            return assistant_message['content']
    except Exception as e:
        print(f"Error executing function: {e}")
        return None


    # Send follow-up request with function responses
    if function_responses:
        # Start with the original user question
        follow_up_messages = [{"role": "user", "content": question}]

        # Add the AI's response that contains the 'tool_calls'
        if 'function_call' in assistant_message:
            follow_up_messages.append({
                "role": "assistant",
                "content": assistant_message.get('content'),
                "function_call": assistant_message['function_call']
            })

        # Append the tool responses
        follow_up_messages.extend(function_responses)
        print(f'follow up messages {follow_up_messages}')
        try:
            follow_up_response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json={"model": model, "messages": follow_up_messages}
            )

            print("DEBUG: Follow-up Response JSON:", follow_up_response.json())
            
            follow_up_message = follow_up_response.json()["choices"][0]["message"]
            print(follow_up_message['content'])
            return follow_up_message['content']
        except Exception as e:
            print("Unable to generate follow-up ChatCompletion response")
            print(f"Exception: {e}")
            return None



    return None  # Fallback return in case of no function calls or errors




while(True):
    message = input('Your question:')
    response = gpt_chat_and_execute_function_bank(message)
    print(response, "\n")







