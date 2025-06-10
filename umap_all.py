import pandas as pd
import umap
from glob import glob
import os
from time import time
import numpy as np
import swifter

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


print('Processing binary vectors with Hamming distance...')

# Start the timer
start = time()

# Binary
repo_id = "bluuebunny/arxiv_abstract_embedding_mxbai_large_v1_milvus_binary"

# Folder to store umap results in
results_folder = repo_id + '/umap/hamming'
os.makedirs(results_folder, exist_ok=True)

# Initiate the reducer
reducer = umap.UMAP(metric='hamming', n_components=3, low_memory=True)

# Gather all parquet files in the data folder
parquets = glob(f'{repo_id}/data/*.parquet', recursive=True)
parquets.sort()

# Merge all parquet files
print(f'Found {len(parquets)} parquet files. Merging them...')
df = pd.concat([pd.read_parquet(parquet) for parquet in parquets], ignore_index=True)

print(f'Merged {len(df)} rows from {len(parquets)} files.')

# Reducing columns to save memory
df = df[['id', 'categories', 'year', 'vector']]

# Process categories
print('Processing categories...')
df['categories'] = df['categories'].swifter.apply(process_categories)

# Convert bytes to binary vectors
print('Unpacking binary vectors...')
df['vector'] = df['vector'].swifter.apply(unpack_binary)

# Fit and transform the data using UMAP
print('Fitting and transforming the data...')
reduced_data = reducer.fit_transform(df['vector'].to_list())

# Add the columns in original df
df['x'] = reduced_data[:, 0]
df['y'] = reduced_data[:, 1]
df['z'] = reduced_data[:, 2]

# Selecting id, categories, year, and the reduced dimensions
selected_columns = ['id', 'categories', 'year', 'x', 'y', 'z']

# Save the data
print('Saving the data...')
df[selected_columns].to_parquet(os.path.join(results_folder, 'all_years.parquet'), index=False)

end = time()

print(f'Time taken: {end - start} seconds')
print('Done!')
