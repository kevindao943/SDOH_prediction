import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

ground_truth = pd.read_csv('/MIMIC-SBDH/NIHMS1767978-supplement-MIMIC_SBDH.csv')
annotated = pd.read_csv('gpt4o-mini_annotated_results.csv')
merged_df = pd.merge(ground_truth, annotated, on='row_id', suffixes=('_true', '_pred'))

columns = ['sdoh_community_present', 'sdoh_community_absent', 'sdoh_education',
           'sdoh_economics', 'sdoh_environment', 'behavior_alcohol',
           'behavior_tobacco', 'behavior_drug']

accuracy_dict = {}
f1_dict = {}

for col in columns:
    y_true = merged_df[f'{col}_true']
    y_pred = merged_df[f'{col}_pred']
    acc = accuracy_score(y_true, y_pred)
    accuracy_dict[col] = acc
    f1 = f1_score(y_true, y_pred, average='weighted')
    f1_dict[col] = f1

overall_accuracy = sum(accuracy_dict.values()) / len(accuracy_dict)
overall_f1 = sum(f1_dict.values()) / len(f1_dict)

with open('evaluation_results.txt', 'w') as f:
    f.write('SDOH Annotation Evaluation Report\n')
    f.write('================================\n\n')
    
    f.write('SDOH Evaluation Results:\n')
    for col in columns:
        f.write(f"{col} - Accuracy: {accuracy_dict[col]:.4f}, F1-score: {f1_dict[col]:.4f}\n")
    
    f.write('\nOverall Results:\n')
    f.write(f"Overall Accuracy: {overall_accuracy:.4f}\n")
    f.write(f"Overall F1-score: {overall_f1:.4f}\n")

print("Evaluation completed successfully. Results are saved in 'evaluation_results.txt'.")