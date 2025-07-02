import pandas as pd
import umap
import os
from time import time
import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA


# Function to process categories and extract the subject
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

########################################################################################
# Float32
repo_id = "bluuebunny/arxiv_abstract_embedding_mxbai_large_v1_milvus"

# Folder to store umap results in
results_folder = repo_id + '/umap/euclidean'
os.makedirs(results_folder, exist_ok=True)

# Initiate the reducer
reducer = umap.UMAP(metric='euclidean', n_components=3)

# Get the 2024 and 2025 files merged
parquet_2024 = "bluuebunny/arxiv_abstract_embedding_mxbai_large_v1_milvus/data/2024.parquet"
parquet_2025 = "bluuebunny/arxiv_abstract_embedding_mxbai_large_v1_milvus/data/2025.parquet"
print(f'Processing {parquet_2024} and {parquet_2025}')

df = pd.concat([pd.read_parquet(parquet) for parquet in [parquet_2024, parquet_2025]], ignore_index=True)

# Reducing columns to save memory
df = df[['id', 'categories', 'year', 'vector', 'title', 'month']]

# filter data from november 2024 to february 2025
df_nov_24 = df[ (df['year'] == 2024) & (df['month'] == 'November') ]
print(f'Found {len(df_nov_24)} rows in November 2024')
print(df_nov_24.sample(5))

df_dec_24 = df[ (df['year'] == 2024) & (df['month'] == 'December') ]
print(f'Found {len(df_dec_24)} rows in December 2024')
print(df_dec_24.sample(5))

df_jan_25 = df[ (df['year'] == 2025) & (df['month'] == 'January') ]
print(f'Found {len(df_jan_25)} rows in January 2025')
print(df_jan_25.sample(5))

df_feb_25 = df[ (df['year'] == 2025) & (df['month'] == 'February') ]
print(f'Found {len(df_feb_25)} rows in February 2025')
print(df_feb_25.sample(5))

df = pd.concat([df_nov_24, df_dec_24, df_jan_25, df_feb_25], ignore_index=True)
print(f'Found {len(df)} rows in total from November 2024 to February 2025')
# Print a sample of the data
print(df.sample(5))

# Process categories
df['categories'] = df['categories'].apply(process_categories)

# Fit and transform the data using UMAP
reduced_data = reducer.fit_transform(df['vector'].to_list())

# Add the columns in original df
df['x'] = reduced_data[:, 0]
df['y'] = reduced_data[:, 1]
df['z'] = reduced_data[:, 2]

# Selecting id, vector and $meta to retain
selected_columns = ['id', 'categories', 'year', 'x', 'y', 'z', 'title']

# Save the data
df[selected_columns].to_parquet(f'{results_folder}/Nov_2024_to_Feb_2025.parquet', index=False)

########################################################################################

