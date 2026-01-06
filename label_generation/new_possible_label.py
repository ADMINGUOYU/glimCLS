import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from transformers import pipeline, GPT2LMHeadModel, GPT2Tokenizer
import torch
from tqdm import tqdm
import os

# ==========================================
# 0. 配置：指定模型缓存路径
# ==========================================
# 请修改为你想要保存模型的具体文件夹路径
model_cache_dir = "../hf_cache"  
# 比如: '/mnt/afs/250010218/models_cache'

# 确保目录存在（可选，transformers通常会自动创建，但手动创建更保险）
if not os.path.exists(model_cache_dir):
    os.makedirs(model_cache_dir)
    print(f"已创建缓存目录: {model_cache_dir}")

# ---------------------------------------------------------
# 1. 数据读取
# ---------------------------------------------------------
csv_input_path = 'data/zuco_preprocessed_dataframe/zuco_label_input_text.csv'

try:
    data = pd.read_csv(csv_input_path)
    sentence = data['raw text']
except FileNotFoundError:
    print("Warning: 文件未找到，正在生成模拟数据...")
    data = pd.DataFrame({
        'raw text': [
            "What is the capital of France", 
            "The sky is blue today",
            "How are you doing", 
            "This is a statement",
            "Why is the output not showing"
        ] * 20
    })
    sentence = data['raw text']

# 检查设备
device = 0 if torch.cuda.is_available() else -1

# ---------------------------------------------------------
# 2. 特征提取
# ---------------------------------------------------------

# === 特征 1: 句法类型 (陈述句 vs 疑问句) - Zero-Shot ===

print(f"正在加载 Zero-Shot 分类器 (缓存至 {model_cache_dir})...")

# 修改点 1: 在 pipeline 中通过 model_kwargs 传递 cache_dir
classifier = pipeline("zero-shot-classification", 
                      model="facebook/bart-large-mnli", 
                      device=device,
                      model_kwargs={"cache_dir": model_cache_dir}) 
                      # 注意：pipeline 不直接接受 cache_dir，需放入 model_kwargs

candidate_labels = ["question", "statement"]

print("正在进行句子类型分类...")
batch_results = classifier(sentence.tolist(), candidate_labels, batch_size=8)
labels = [result['labels'][0] for result in batch_results]

# === 特征 2: 句子长度 ===
lengths = sentence.apply(lambda x: len(str(x).split()))

# === 特征 3: 惊奇度 (Surprisal) - GPT-2 ===

print(f"正在加载 GPT-2 模型 (缓存至 {model_cache_dir})...")

# 修改点 2: 在 from_pretrained 中直接传递 cache_dir
gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2', cache_dir=model_cache_dir)
gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2', cache_dir=model_cache_dir)

gpt2_model = gpt2_model.to("cuda" if torch.cuda.is_available() else "cpu")

def calculate_surprisal(text):
    inputs = gpt2_tokenizer(text, return_tensors='pt').to(gpt2_model.device)
    with torch.no_grad():
        outputs = gpt2_model(**inputs, labels=inputs['input_ids'])
    return outputs.loss.item()

tqdm.pandas(desc="Calculating Surprisal")
surprisals = sentence.progress_apply(calculate_surprisal)

# 整合特征
df_features = pd.DataFrame({
    'raw_text': sentence,
    'label': labels,
    'length': lengths,
    'surprisal': surprisals
})

# ---------------------------------------------------------
# 3. 数据可视化
# ---------------------------------------------------------
plt.figure(figsize=(18, 6))

# 图 1: 句子类型
plt.subplot(1, 3, 1)
label_counts = df_features['label'].value_counts()
plt.pie(label_counts, labels=label_counts.index, autopct='%1.1f%%', 
        startangle=140, colors=['#66b3ff', '#ff9999'])
plt.title('Sentence Types (Model-based)')

# 图 2: 长度
plt.subplot(1, 3, 2)
plt.hist(df_features['length'], bins=15, color='salmon', edgecolor='black', alpha=0.7)
plt.title('Length Distribution')

# 图 3: 惊奇度
plt.subplot(1, 3, 3)
plt.hist(df_features['surprisal'], bins=15, color='violet', edgecolor='black', alpha=0.7)
plt.title('Surprisal Distribution')

plt.tight_layout()
plt.savefig('sentence_features_analysis_custom_cache.png')
print("分析完成，图片已保存。")

# ---------------------------------------------------------
# 4. 输出样本句子
# ---------------------------------------------------------
questions = df_features[df_features['label'] == 'question'].head(10)
statements = df_features[df_features['label'] == 'statement'].head(10)

print("\n" + "="*60)
print("10 QUESTION SENTENCES:")
print("="*60)
for idx, row in questions.iterrows():
    print(f"{idx+1}. {row['raw_text']}")

print("\n" + "="*60)
print("10 STATEMENT SENTENCES:")
print("="*60)
for idx, row in statements.iterrows():
    print(f"{idx+1}. {row['raw_text']}")