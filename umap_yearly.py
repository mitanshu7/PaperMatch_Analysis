import pandas as pd
import umap
from glob import glob
import os
from time import time
import numpy as np
from tqdm import tqdm

# Configuration
FLOAT = True
BINARY = True

def process_categories(categories:str) -> str:

    # Extract the first category. Get 'cs.LG' from 'cs.LG cs.AI cs.CL'.
    category = categories.split(' ')[0]

    # Extract the subject. Get 'math' from 'math.AC'.
    subject = category.split('.')[0]

    # Converge fragmented physic categories
    physics_tags = ['astro-ph', 'cond-mat', 'gr-qc', 'hep-ex', 'hep-lat', 'hep-ph', 
                    'hep-th', 'math-ph', 'nlin', 'nucl-ex', 'nucl-th', 'quant-ph']
    
    if subject in physics_tags:
        subject = 'physics'

    return subject

def unpack_binary(vector):

    # Read the stored bytes
    packed_bits = np.frombuffer(vector, dtype=np.uint8)

    # Unpack to binary vector
    unpacked_bits = np.unpackbits(packed_bits)

    return unpacked_bits


if FLOAT:

    print('Processing float vectors with Euclidean distance...')

    # Start the timer
    start = time()

    # Float32
    repo_id = "bluuebunny/arxiv_abstract_embedding_mxbai_large_v1_milvus"

    # Folder to store umap results in
    results_folder = repo_id + '/umap/euclidean'
    os.makedirs(results_folder, exist_ok=True)

    # Initiate the reducer
    reducer = umap.UMAP(metric='euclidean')

    # Gather all parquet files in the data folder
    parquets = glob(f'{repo_id}/data/*.parquet', recursive=True)
    parquets.sort()

    # Filter out the yearly files
    for parquet in tqdm(parquets):

        print(f'Processing {parquet}')

        # Read the parquet file into a pandas dataframe
        df = pd.read_parquet(parquet)

        # Process categories
        df['categories'] = df['categories'].apply(process_categories)

        # Fit and transform the data using UMAP
        reduced_data = reducer.fit_transform(df['vector'].to_list())

        # Add the columns in original df
        df['x'] = reduced_data[:, 0]
        df['y'] = reduced_data[:, 1]

        # Selecting id, vector and $meta to retain
        selected_columns = ['id', 'categories', 'year', 'x', 'y']

        # Save the data
        df[selected_columns].to_parquet(os.path.join(results_folder, os.path.basename(parquet)))


    end = time()

    print(f'Time taken: {end - start} seconds')
    print('Done!')

if BINARY:

    print('Processing binary vectors with Hamming distance...')

    # Start the timer
    start = time()

    # Binary
    repo_id = "bluuebunny/arxiv_abstract_embedding_mxbai_large_v1_milvus_binary"

    # Folder to store umap results in
    results_folder = repo_id + '/umap/hamming'
    os.makedirs(results_folder, exist_ok=True)

    # Initiate the reducer
    reducer = umap.UMAP(metric='hamming')

    # Gather all parquet files in the data folder
    parquets = glob(f'{repo_id}/data/*.parquet', recursive=True)
    parquets.sort()

    # Filter out the yearly files
    for parquet in tqdm(parquets):

        print(f'Processing {parquet}')

        # Read the parquet file into a pandas dataframe
        df = pd.read_parquet(parquet)

        # Process categories
        df['categories'] = df['categories'].apply(process_categories)

        # Convert bytes to binary vectors
        df['vector'] = df['vector'].apply(unpack_binary)

        # Fit and transform the data using UMAP
        reduced_data = reducer.fit_transform(df['vector'].to_list())

        # Add the columns in original df
        df['x'] = reduced_data[:, 0]
        df['y'] = reduced_data[:, 1]

        # Selecting id, vector and $meta to retain
        selected_columns = ['id', 'categories', 'year', 'x', 'y']

        # Save the data
        df[selected_columns].to_parquet(os.path.join(results_folder, os.path.basename(parquet)))


    end = time()

    print(f'Time taken: {end - start} seconds')
    print('Done!')