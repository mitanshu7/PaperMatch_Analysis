import pandas as pd # For data manipulation
from glob import glob # For file pattern matching
import os # For file path operations
from matplotlib import pyplot as plt # For plotting

# Folder to save histogram images
histogram_folder = 'histograms'
# Create the folder if it doesn't exist
os.makedirs(histogram_folder, exist_ok=True)


# Repository ID for the dataset
repo_id = "bluuebunny/arxiv_abstract_embedding_mxbai_large_v1_milvus_binary"

# Folder where umap results are
results_folder = repo_id + '/' + 'umap'

# Gather all parquet files in the data folder
parquets = glob(f'{results_folder}/*.parquet', recursive=True)
parquets.sort()

for parquet in parquets:

    print(f"Processing {parquet}")

    # Read the parquet file
    df = pd.read_parquet(parquet)

    # Sort the DataFrame by 'categories'
    df = df.sort_values(by='categories')

    # histogram of category 
    df['categories'].hist(bins=16, alpha=0.5, color='blue', edgecolor='black')
    plt.xlabel('Category')
    plt.ylabel('Frequency')
    plt.title(f'Category Distribution for {os.path.basename(parquet)}')
    plt.grid(axis='y', alpha=0.75)
    plt.savefig(f'{histogram_folder+"/"+os.path.basename(parquet).replace(".parquet","")}_category_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    