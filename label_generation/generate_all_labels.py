import os
# WARNING: uncomment if needed
os.environ['HF_HOME'] = '/mnt/afs/250010218/hf_cache'
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from keybert import KeyBERT
from transformers import pipeline
import pickle
import torch

# tmp path (saving path)
# User can modify this path to save outputs to a different location
# Default: './tmp' for local storage
tmp_path = './data/zuco_preprocessed_dataframe'
# Create tmp directory if it doesn't exist
os.makedirs(tmp_path, exist_ok=True)

# Use CUDA if available, otherwise fallback to CPU
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# 1. Load raw data
df_input_path = tmp_path + '/zuco_label_input_text.df'
data = pd.read_pickle(df_input_path)

# 2. Generate sentence embeddings (Shape: N, 768)
sbert_model = SentenceTransformer('all-mpnet-base-v2')
sentence_embeddings = sbert_model.encode(data['input text'].tolist(), show_progress_bar=True)

# 3.1 Generate topic labels
print("Generating topic labels...")
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=device)
candidate_labels = ["Movie Reviews and Sentiment", "Biographies and Factual Knowledge"]
batch_results = classifier(data['input text'].tolist(), candidate_labels)
topic_labels = [result['labels'][0] for result in batch_results]

# 3.2 Generate sentiment labels
print("Generating sentiment labels...")
sentiment_pipeline = pipeline("sentiment-analysis", 
                              model="cardiffnlp/twitter-roberta-base-sentiment-latest", 
                              device=device)
def generate(text:str) -> str:
    """
    Generate sentiment label for the given text.
    
    Args:
        text: Input text to analyze
        
    Returns:
        Sentiment label as string: 'negative', 'neutral', or 'positive'
        
    Raises:
        ValueError: If the sentiment pipeline returns an unexpected label
    """
    try:
        result = sentiment_pipeline(text)
        label = result[0]['label']
        # Validate the label is one of the expected values
        if label not in ['negative', 'neutral', 'positive']:
            raise ValueError(f"Unexpected sentiment label: {label}")
        return label
    except Exception as e:
        print(f"Error analyzing sentiment for text: {text[:50]}...")
        print(f"Error: {e}")
        # Return neutral as default for failed cases
        return 'neutral'

sentiment_labels = []
for text in data['input text'].tolist():
    sentiment_labels.append(generate(text))

sentiment_labels = [ lbl if lbl == 'neutral' else 'non_neutral' for lbl in sentiment_labels ]

# 4. Extract keyword text (Top 3)
print("Extracting keywords...")
kw_model = KeyBERT(model=sbert_model) # Reuse model to save memory
keywords_list = []
for text in data['input text']:
    keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 1), stop_words='english', top_n=3)
    # Pad with empty strings if less than 3 keywords found
    keywords_list.append([k[0] for k in keywords] + [''] * (3 - len(keywords)))

# 5. Generate keyword embeddings
# Flatten for batch encoding efficiency
flat_keywords = [kw for sublist in keywords_list for kw in sublist]

print("Encoding keywords...")
kw_embeddings_flat = sbert_model.encode(flat_keywords, show_progress_bar=True)

# Reshape from (N*3, 768) to (N, 3, 768)
# Index [i, 0, :] corresponds to Top 1 keyword
num_sentences = len(data)
keyword_embeddings = kw_embeddings_flat.reshape(num_sentences, 3, 768)

# 6. Prepare metadata DataFrame (text only)
keywords_df = pd.DataFrame(keywords_list, columns=['keyword_1', 'keyword_2', 'keyword_3'])
topic_df = pd.DataFrame({'topic_label': topic_labels})
sentiment_df = pd.DataFrame({'sentiment label' : sentiment_labels})
data_new = pd.concat([data, keywords_df, topic_df, sentiment_df], axis=1)

#######################################
""" Assign embeddings to Unique IDs """
#######################################
assert (sentence_embeddings.shape[0] == len(data_new)) and (keyword_embeddings.shape[0] == len(data_new)), "[ERROR] Expected number of sentences to be the same"
embedding_dict = { }
for row, uid in enumerate(data_new['text uid']):
    embedding_dict[uid] = { 'sentence' : sentence_embeddings[row, : ], 'keyword' : keyword_embeddings[row, : , : ] }

# 7. Save outputs (override source)
# Save text metadata
df_output_path = tmp_path + '/zuco_label_input_text.df'
csv_output_path = tmp_path + '/zuco_label_input_text.csv'
pd.to_pickle(data_new, df_output_path)
data_new.to_csv(csv_output_path, index=False)

# Save sentence embeddings (N, 768)
sent_npy_path = tmp_path + '/zuco_sentence_embeddings.npy'
np.save(sent_npy_path, sentence_embeddings)

# Save keyword embeddings (N, 3, 768)
kw_npy_path = tmp_path + '/zuco_keyword_embeddings.npy'
np.save(kw_npy_path, keyword_embeddings)

###############################
""" Save Packed Embeddings """
###############################
with open(tmp_path + "/embeddings.pickle", "wb") as f:
    pickle.dump(embedding_dict, f, protocol = pickle.HIGHEST_PROTOCOL)

# debug verbose
print("-" * 30)
print(f"Processing Complete!")
print(f"1. CSV metadata saved: {csv_output_path}\n   Dataframe saved: {df_output_path}")
print(f"2. Sentence embeddings shape: {sentence_embeddings.shape}")
print(f"3. Keyword embeddings shape: {keyword_embeddings.shape}")
print("-" * 30)