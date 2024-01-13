from openai import OpenAI
import tiktoken
import pandas as pd
# import numpy as np
from numpy.linalg import norm 
from numpy import dot
import os
import pickle

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

def embedding_search(function_object, n=3, df=None, pickle_file='functionPickle.pkl'):
    # Load the DataFrame from pickle if df is None
    if df is None:
        if os.path.exists(pickle_file):
            with open(pickle_file, 'rb') as file:
                df = pickle.load(file)
        else:
            return None  # Return None if no pickled DataFrame is available

    # Get the embedding of the query from the FunctionObject
    query_embedding = function_object.embedding
    if query_embedding is None:
        function_object.generate_embedding()
        query_embedding = function_object.embedding

    # Compute cosine similarities
    df['similarities'] = df['Embedding'].apply(lambda x: cosine_similarity(x, query_embedding))

    # Sort the DataFrame by similarities and get the top n results
    top_results = df.sort_values('similarities', ascending=False).head(n)

    # Return a list of tuples (FunctionObject, similarity score)
    return [(row['FunctionObject'], row['similarities']) for index, row in top_results.iterrows()]

# Example usage:
# function_obj = FunctionObject(json_data)
# similar_functions = embedding_search(function_obj)




def embed_and_add_to_df(func_obj=None, pickle_file='functionPickle.pkl'):
    # Check if the DataFrame exists in pickle
    if os.path.exists(pickle_file):
        with open(pickle_file, 'rb') as file:
            df = pickle.load(file)
    else:
        # Create a new DataFrame if pickle doesn't exist
        df = pd.DataFrame(columns=['FunctionObject', 'Embedding'])

    # If function object is provided
    if func_obj is not None:
        # Check if embedding exists, generate if not
        if func_obj.embedding is None:
            func_obj.generate_embedding()  # Assuming this method sets the embedding

        # Add to DataFrame
        df = df.append({'FunctionObject': func_obj, 'Embedding': func_obj.embedding}, ignore_index=True)

        # Save the DataFrame to pickle
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


client = OpenAI(api_key="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")









