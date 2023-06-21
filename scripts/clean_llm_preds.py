"""
Author: Joshua Ashkinaze
Date: 2023-06-20

Description: This script cleans the LLM predictions, merges them with the Metaculus events, drops rows with NaNs in key columns.
"""
import numpy as np
import pandas as pd
def main():
    events = pd.read_csv("../data/metaculus_events.csv")
    preds = pd.read_csv("../data/raw_llm_preds.csv")
    events2 = pd.merge(events, preds, on='id')
    print(events2.columns)
    events2['gpt_pred'] = events2['answer'].apply(lambda x: 1 if x == 'YES' else 0 if x == "NO" else np.NaN)
    events2['gpt_pred_conf'] = np.where(events2['gpt_pred'] == 1, events2['confidence'], 1 - events2['confidence'])
    events2['resolution'] = events2['resolution'].apply(lambda x: 1 if x == 1 else 0 if x == 0 else np.NaN)
    events2['init_pred'] = events2['init_pred_conf'].apply(lambda x: 1 if x >= 0.5 else 0)
    events2['final_pred'] = events2['final_pred_conf'].apply(lambda x: 1 if x >= 0.5 else 0)
    events2 = events2.dropna(
        subset=['gpt_pred', 'resolution', 'init_pred', 'final_pred', 'gpt_pred_conf', 'init_pred_conf', 'final_pred_conf'])
    events2.to_csv("../data/clean_llm_preds_and_events.csv", index=False)

if __name__ == '__main__':
    main()
