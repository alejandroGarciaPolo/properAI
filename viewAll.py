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
import subprocess

def run_cmd_command(command):
    """
    Executes a given command in the Windows Command Prompt (CMD).

    :param command: The command to be executed.
    :return: The output and error (if any) as a string.
    """
    try:
        result = subprocess.run(command, check=True, text=True, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return result.stdout
    except subprocess.CalledProcessError as e:
        return e.stderr

def run_powershell_command(command):
    """
    Executes a given command in Windows PowerShell.

    :param command: The PowerShell command or script to be executed.
    :return: The output and error (if any) as a string.
    """
    try:
        ps_command = f"powershell -Command {command}"
        result = subprocess.run(ps_command, check=True, text=True, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return result.stdout
    except subprocess.CalledProcessError as e:
        return e.stderr

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

from pywinauto import Desktop, Application
import ctypes
import pyautogui
import time

def click_at(x, y, clicks=1, button='left'):
    """
    Click at the specified coordinates.
    :param x: X coordinate
    :param y: Y coordinate
    :param clicks: Number of clicks (2 for double-click)
    :param button: 'left' for left click, 'right' for right click
    """
    pyautogui.click(x, y, clicks=clicks, button=button)

def type_text(text, interval=0.05):
    """
    Type a string as if it's being typed on a keyboard.
    :param text: Text to type
    :param interval: Time interval between key presses
    """
    pyautogui.write(text, interval=interval)


def list_controls(window, indent=0):
    """
    Recursively lists controls and their information in a given window.
    
    :param window: The window whose controls are to be listed.
    :param indent: Indentation level (used for recursive calls to format output).
    :return: String containing information about all controls in the window.
    """
    children = window.children()
    controls_info = ''
    for child in children:
        control_type = child.element_info.control_type
        control_info = f"{' ' * indent}{child.window_text()} {control_type} {child.rectangle()}\n"
        controls_info += control_info
        controls_info += list_controls(child, indent + 4)
    return controls_info

def list_windows():
    """
    Lists all visible windows and their basic information.
    
    :return: List of window objects representing all visible windows.
    """
    windows = Desktop(backend="uia").windows()
    windows_list = []
    for i, w in enumerate(windows):
        if w.is_visible() and w.window_text():
            window_info = f"{i}: {w.window_text()}" + (" (current)" if w.handle == ctypes.windll.user32.GetForegroundWindow() else "")
            print(window_info)
            windows_list.append(w)
    return windows_list

def get_window_controls(index, windows):
    """
    Retrieves and prints UI elements of a specified window based on the given index.
    
    :param index: Index of the window in the provided windows list.
    :param windows: List of window objects.
    :return: String containing information about UI elements of the selected window.
    """
    if 0 <= index < len(windows):
        selected_window = windows[index]
        app = Application(backend="uia").connect(handle=selected_window.handle)
        window = app.window(handle=selected_window.handle)
        controls = list_controls(window)
        print(f"\nUI elements for: {selected_window.window_text()}\n")
        print(controls)
        return controls
    else:
        print("Invalid choice. Exiting.")
        return ""



from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
import time


driver = webdriver.Chrome()  # Or use another driver like Firefox ###UNCOMMENT ME FOR DIREVER $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
def open_website_or_query(query):
    
    """
    Opens a website in the browser controlled by the driver, or performs a Google search if a non-URL query is provided.
    
    :param driver: Selenium WebDriver instance.
    :param query: URL or search query.
    """
    if query.startswith("http://") or query.startswith("https://"):
        driver.get(query)
    else:
        driver.get("https://www.google.com")
        search_box = driver.find_element(By.NAME, "q")
        search_box.send_keys(query + Keys.RETURN)

def list_ui_elements(driver, truncate_length=30):
    """
    Lists all UI elements in the current page of the browser and returns their information as a string.
    
    :param driver: Selenium WebDriver instance.
    :param truncate_length: Length at which to truncate the text content of elements.
    :return: String containing information about all UI elements on the page.
    """
    elements = driver.find_elements(By.XPATH, "//*")
    elements_info = ""
    for i, element in enumerate(elements):
        text_content = element.text
        truncated_text = (text_content[:truncate_length] + '...') if len(text_content) > truncate_length else text_content
        element_info = f"{i}: {element.tag_name}, class: {element.get_attribute('class')}, id: {element.get_attribute('id')}, text: {truncated_text}\n"
        elements_info += element_info
    return elements_info

def click_on_ui_element(driver, index):
    """
    Clicks on a UI element in the current page of the browser based on its index.
    
    :param driver: Selenium WebDriver instance.
    :param index: Index of the element to click.
    """
    elements = driver.find_elements(By.XPATH, "//*")
    if 0 <= index < len(elements):
        elements[index].click()

def type_on_ui_element(driver, index, text):
    """
    Types text into a UI element in the current page of the browser based on its index.
    
    :param driver: Selenium WebDriver instance.
    :param index: Index of the element where text is to be typed.
    :param text: Text to type into the element.
    """
    elements = driver.find_elements(By.XPATH, "//*")
    if 0 <= index < len(elements):
        elements[index].send_keys(text)

# Example Usage



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
},
{
    "name": "click_at",
    "description": "Click at the specified coordinates.",
    "parameters": {
        "type": "object",
        "properties": {
            "x": {
                "type": "integer",
                "description": "X coordinate"
            },
            "y": {
                "type": "integer",
                "description": "Y coordinate"
            },
            "clicks": {
                "type": "integer",
                "description": "Number of clicks (2 for double-click)",
                "default": 1
            },
            "button": {
                "type": "string",
                "description": "'left' for left click, 'right' for right click",
                "enum": ["left", "right"],
                "default": "left"
            }
        },
        "required": ["x", "y"]
    }
},
{
    "name": "type_text",
    "description": "Type a string as if it's being typed on a keyboard.",
    "parameters": {
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "Text to type"
            },
            "interval": {
                "type": "number",
                "description": "Time interval between key presses",
                "default": 0.05
            }
        },
        "required": [
            "text"
        ]
    }
},
{
    "name": "list_controls",
    "description": "Recursively lists controls and their information in a given window, with a specified indentation level for formatting.",
    "parameters": {
        "type": "object",
        "properties": {
            "window": {
                "description": "The window whose controls are to be listed. This should be an object representing a window, compatible with the function's requirements.",
                "type": "object"
            },
            "indent": {
                "type": "integer",
                "description": "Indentation level (used for recursive calls to format output).",
                "default": 0
            }
        },
        "required": [
            "window"
        ]
    },
    "return": {
        "description": "String containing information about all controls in the window.",
        "type": "string"
    }
},
{
    "name": "list_windows",
    "description": "Lists all visible windows and their basic information.",
    "parameters": {
        "type": "object",
        "properties": {}
    },
    "return": {
        "description": "List of window objects representing all visible windows.",
        "type": "array",
        "items": {
            "type": "object",
            "description": "A window object with visible window information."
        }
    }
},
{
    "name": "get_window_controls",
    "description": "Retrieves and prints UI elements of a specified window based on the given index.",
    "parameters": {
        "type": "object",
        "properties": {
            "index": {
                "type": "integer",
                "description": "Index of the window in the provided windows list."
            },
            "windows": {
                "type": "array",
                "description": "List of window objects.",
                "items": {
                    "type": "object",
                    "description": "A window object representing a visible window."
                }
            }
        },
        "required": [
            "index",
            "windows"
        ]
    },
    "return": {
        "description": "String containing information about UI elements of the selected window, or an indication of an invalid index.",
        "type": "string"
    }
},
{
    "name": "open_website_or_query",
    "description": "Opens a website in the browser controlled by the driver, or performs a Google search if a non-URL query is provided.",
    "parameters": {
        "type": "object",
        "properties": {
            "driver": {
                "description": "Selenium WebDriver instance to control the browser.",
                "type": "object"
            },
            "query": {
                "type": "string",
                "description": "URL to open directly or a search query if it is not a valid URL."
            }
        },
        "required": [
            "driver",
            "query"
        ]
    }
},
{
    "name": "list_ui_elements",
    "description": "Lists all UI elements in the current page of the browser and returns their information as a string.",
    "parameters": {
        "type": "object",
        "properties": {
            "driver": {
                "description": "Selenium WebDriver instance to interact with the browser.",
                "type": "object"
            },
            "truncate_length": {
                "type": "integer",
                "description": "Length at which to truncate the text content of elements.",
                "default": 30
            }
        },
        "required": [
            "driver"
        ]
    },
    "return": {
        "description": "String containing information about all UI elements on the page.",
        "type": "string"
    }
},
{
    "name": "click_on_ui_element",
    "description": "Clicks on a UI element in the current page of the browser based on its index.",
    "parameters": {
        "type": "object",
        "properties": {
            "driver": {
                "description": "Selenium WebDriver instance to interact with the browser.",
                "type": "object"
            },
            "index": {
                "type": "integer",
                "description": "Index of the element to click."
            }
        },
        "required": [
            "driver",
            "index"
        ]
    }
},
{
    "name": "type_on_ui_element",
    "description": "Types text into a UI element in the current page of the browser based on its index.",
    "parameters": {
        "type": "object",
        "properties": {
            "driver": {
                "description": "Selenium WebDriver instance to interact with the browser.",
                "type": "object"
            },
            "index": {
                "type": "integer",
                "description": "Index of the element where text is to be typed."
            },
            "text": {
                "type": "string",
                "description": "Text to type into the element."
            }
        },
        "required": [
            "driver",
            "index",
            "text"
        ]
    }
},
{
    "name": "run_cmd_command",
    "description": "Executes a given command in the Windows Command Prompt (CMD) and returns the output and any errors as a string.",
    "parameters": {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "The command to be executed."
            }
        },
        "required": [
            "command"
        ]
    },
    "return": {
        "description": "The output of the command if it executes successfully, or error information if the command fails.",
        "type": "string"
    }
},
{
    "name": "run_powershell_command",
    "description": "Executes a given command or script in Windows PowerShell and returns the output and any errors as a string.",
    "parameters": {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "The PowerShell command or script to be executed."
            }
        },
        "required": [
            "command"
        ]
    },
    "return": {
        "description": "The output of the PowerShell command if it executes successfully, or error information if the command fails.",
        "type": "string"
    }
}
    ]

    functions = []
    available_functions = {}
    df = embed_and_add_to_df()
    for fun in functionsJson:
        function_name = fun['name']
        functionalObj= FunctionObject(fun)
        functions.append(functionalObj)
        embed_and_add_to_df(functionalObj)
        # Accessing the function from the global scope
        if function_name in globals():
            available_functions[function_name] = globals()[function_name]
        else:
            print(f"Warning: Function '{function_name}' is not defined.")
    

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


inputQ = input("\nWould you like to run the pre-setup? y/n")
if inputQ.lower() == 'y':
    initialize_or_load_functions(retrieve=False)
inputQ = input("\nWould you like to add a function? y/n")

if inputQ.lower() == 'y':
    add_to_function_storage()

functions, available_functions = initialize_or_load_functions(retrieve=True) #####yes or yes load the base, functions is not nessesary however, avalible is

# print(f"current df {embed_and_add_to_df()}")
# print(f'number of functions avalible {len(functions)}')


# inputQ = input("\nWould you like to run the pre-setup OLD? y/n")

# if inputQ.lower() == 'y':
    
#     for fun in functions:
#         embed_and_add_to_df(fun)

# inputQ = input("\n Would you like to give them human values? y/n")

# if inputQ.lower() == 'y':
#     update_function_descriptions()

print('fetching df')
df = embed_and_add_to_df() ##fetches the df
print('done.')
print('starting chat\n')
print()

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
def gpt_chat_and_execute_function_bank(question, context, model="gpt-3.5-turbo-0613", function_call='auto'):
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
    api_key = "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"
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer " + api_key,
    }
    
    messages = [{"role": "user", "content": question}]
    if context:
        temp = context + messages
        messages = temp
        context = messages

    json_data = {"model": model, "messages": messages}
    # print(df)
    functions = embedding_search(query=question, df=df) if 'embedding_search' in globals() else None
    if functions is not None and not []: ##if functions empty this gives error
        json_data.update({"functions": functions})
    if function_call is not None and functions != []:
        json_data.update({"function_call": function_call})
    # print('FUNCTIONS:', functions)
    try:
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=json_data)
        assistant_message = response.json()["choices"][0]["message"]
        # print('ASSISTANT', assistant_message['content'])
        if assistant_message['content']:
            # print('not none')
            messages.append({"role": "assistant", "content": assistant_message['content']})
        context = messages

        function_responses = []
        if 'function_call' in assistant_message:
            tool_call = assistant_message['function_call']
            function_name = tool_call['name']
            function_args = json.loads(tool_call['arguments'])
            messages.append({
                "role": "assistant",
                "content": assistant_message.get('content'),
                "function_call": assistant_message['function_call']
            })
            context = messages

            if function_name in available_functions:
                function_response = available_functions[function_name](**function_args)
                function_responses.append({
                    "role": "user",
                    "content": f"This is a hidden system message that just shows you what the function returned, answer the previous user message given that this is what it evaluated to, only pay attention to the values not the prompt I am giving you now: {function_response}"
                })
                messages.extend(function_responses)
                context = messages
            else:
                raise ValueError(f"Function {function_name} not defined.")

            if function_responses:
                
                
                follow_up_response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json={"model": model, "messages": messages})
                follow_up_message = follow_up_response.json()["choices"][0]["message"]
                messages.append({"role": "assistant", "content": follow_up_message['content']})
                context = messages
                return follow_up_message['content'], context
        else:
            return assistant_message['content'], context

    except Exception as e:
        print(f"Error during conversation: {e}")
        return None, messages

##this is like the chat area
conversation_history = []
while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":  # Exit condition
        break
    # print(conversation_history)
    response, conversation_history = gpt_chat_and_execute_function_bank(user_input, conversation_history)
    # print('HISTORY:',conversation_history)
    print("Assistant:", response)


# while(True):
#     message = input('Your question:')
#     response = gpt_chat_and_execute_function_bank(message)
#     print(response, "\n")







