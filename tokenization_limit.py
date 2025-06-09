import kagglehub # To download the dataset from Kaggle
from transformers import AutoTokenizer # To tokenize the text data
from datasets import load_dataset # To load dataset without breaking ram
from multiprocessing import cpu_count # To get the number of CPU cores
import matplotlib.pyplot as plt # For plotting the histogram
import os # For file path operations

# Folder to save histogram images
histogram_folder = 'histograms'
# Create the folder if it doesn't exist
os.makedirs(histogram_folder, exist_ok=True)

# Download the dataset
# Dataset name
dataset_path = 'Cornell-University/arxiv'

# Download folder
download_folder = kagglehub.dataset_download(dataset_path)

# Data file path
download_file = f'{download_folder}/arxiv-metadata-oai-snapshot.json'

print(f"Loading json metadata")
dataset = load_dataset("json", data_files= str(f"{download_file}"))

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained('mixedbread-ai/mxbai-embed-large-v1')

# Function to tokenize and count tokens in the abstract
def tokenize_and_count(example):

    text = example['abstract']
    tokens = tokenizer.tokenize(text)
    token_count = len(tokens)
    example['token_count'] = token_count

    return example

# Apply the tokenization and counting function to the dataset
dataset = dataset.map(tokenize_and_count, num_proc=cpu_count())

df = dataset['train'].to_pandas()

# Draw a straight line at 512
plt.plot([512, 512], [0, 185000], color='red', linestyle='--')
# Plot the token counts
df['token_count'].hist(bins=100, alpha=0.5, color='blue', edgecolor='black')
plt.xlabel('Token Count')
plt.ylabel('Frequency')
plt.title('Token Count Distribution')
plt.xlim(0, 1600)
plt.ylim(0, 185000)
plt.grid(axis='y', alpha=0.75)
plt.savefig(f'{histogram_folder}/token_count_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

# Count the number of papers with token count > 512
count_above_512 = (df['token_count'] > 512).sum()
print(f"Number of papers with token count > 512: {count_above_512}")

# Count the number of papers with token count <= 512
count_below_512 = (df['token_count'] <= 512).sum()
print(f"Number of papers with token count <= 512: {count_below_512}")

# Calculate the percentage of papers with token count > 512
percentage_above_512 = (count_above_512 / len(df)) * 100
print(f"Percentage of papers with token count > 512: {percentage_above_512:.2f}%")

# Calculate the percentage of papers with token count <= 512
percentage_below_512 = (count_below_512 / len(df)) * 100
print(f"Percentage of papers with token count <= 512: {percentage_below_512:.2f}%")

# Calculate the average token count
average_token_count = df['token_count'].mean()
print(f"Average token count: {average_token_count:.2f}")

# Calculate the median token count
median_token_count = df['token_count'].median()
print(f"Median token count: {median_token_count:.2f}")

# Calculate the standard deviation of token counts
std_dev_token_count = df['token_count'].std()
print(f"Standard deviation of token counts: {std_dev_token_count:.2f}")

# Calculate the maximum token count
max_token_count = df['token_count'].max()
print(f"Maximum token count: {max_token_count}")

# Calculate the minimum token count
min_token_count = df['token_count'].min()
print(f"Minimum token count: {min_token_count}")

# Calculate the 25th percentile of token counts
percentile_25_token_count = df['token_count'].quantile(0.25)
print(f"25th percentile of token counts: {percentile_25_token_count:.2f}")

# Calculate the 75th percentile of token counts
percentile_75_token_count = df['token_count'].quantile(0.75)
print(f"75th percentile of token counts: {percentile_75_token_count:.2f}")

# Calculate the interquartile range (IQR) of token counts
iqr_token_count = percentile_75_token_count - percentile_25_token_count
print(f"Interquartile range (IQR) of token counts: {iqr_token_count:.2f}")

# Calculate the skewness of token counts
skewness_token_count = df['token_count'].skew()
print(f"Skewness of token counts: {skewness_token_count:.2f}")

# Calculate the kurtosis of token counts
kurtosis_token_count = df['token_count'].kurtosis()
print(f"Kurtosis of token counts: {kurtosis_token_count:.2f}")
