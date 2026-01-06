import numpy as np
import pandas as pd
import sys
import os

# tmp path (saving path)
# User can modify this path to save outputs to a different location
# Default: './tmp' for local storage
tmp_path = '/nfs/usrhome2/yguoco/glim_cls/tmp'
# Create tmp directory if it doesn't exist
os.makedirs(tmp_path, exist_ok=True)

"""
Please make sure you have zuco_merged.df and topic_label.csv ready
!!!!!! Important !!!!!!
WARNING: this works on zuco_merged.df, if you need the MTV version, run that after this script
"""

TOPIC_LABEL = 'topic_label'

# load dataframes
merged_df: pd.DataFrame = pd.read_pickle(tmp_path + '/zuco_merged.df')
topics: pd.DataFrame = pd.read_csv(tmp_path + '/topic_label.csv')

# add that column
merged_df[TOPIC_LABEL] = ""

# iterate all rows of zuco_merged, check alignment
for idx in range(len(merged_df)):
    # the text is:
    text: str = merged_df.iloc[idx]['input text']
    # get row index of topic labels
    index = topics.index[topics['input text'] == text].to_list()
    assert len(index) >= 1, "[ERROR] cannot locate input text, merge error"
    index = index[0]
    # copy topics
    merged_df.loc[idx, TOPIC_LABEL] = topics.iloc[index][TOPIC_LABEL]

# Save the merged dataframe
save_location = tmp_path + '/zuco_merged_with_topics.df'
pd.to_pickle(merged_df, save_location)
print(f"Saved merged data to {save_location}")

# visualize
print(f"[first row]:\n{merged_df.iloc[0]}")