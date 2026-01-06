import pickle
import numpy as np
from transformers import pipeline
import pandas as pd
# Load task_2 pickle file
csv_input_path = 'data/zuco_preprocessed_dataframe/zuco_label_input_text.csv'
data = pd.read_csv(csv_input_path)

sentence = data['raw text']


# 1. 加载 Zero-Shot 分类器
classifier = pipeline("zero-shot-classification", 
                      model="facebook/bart-large-mnli", 
                      device=0) # device=0 使用 GPU

# 2. 定义

candidate_labels = [
    "Movie Reviews and Sentiment", 
    "Biographies and Factual Knowledge"
]

# Batch process all sentences at once for GPU efficiency
batch_results = classifier(sentence.tolist(), candidate_labels)
results = [result['labels'][0] for result in batch_results]

print(len(results))

#统计结果
from collections import Counter
counter = Counter(results)
print(counter)

