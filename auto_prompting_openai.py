from openai import OpenAI, OpenAIError
import pandas as pd
import time
import random
from tqdm import tqdm
import os

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise EnvironmentError("OPENAI_API_KEY not found in environment variables.")
client = OpenAI(api_key = api_key)

def annotate_sdoH(text, retries=3, backoff_factor=1.5):
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": """Annotate the following clinical note based on provided guidelines:
             
The notes should be annotated for the patient’s status of the following Social and Behavioral Determinants of Health (SBDHs):
             
1) Community Present (sdoh_community_present) (0: False, 1: True)
2) Community Absent (sdoh_community_absent) (0: False, 1: True)
3) Education (sdoh_education) (0: False, 1: True)
4) Economics (sdoh_economics) (0: None, 1: True, 2: False)
5) Environment (sdoh_environment) (0: None, 1: True, 2: False)
6) Alcohol Use (behavior_alcohol) (0: None, 1: Present, 2: Past, 3: Never, 4: Unsure)
7) Tobacco Use (behavior_tobacco) (0: None, 1: Present, 2: Past, 3: Never, 4: Unsure)
8) Drug Use (behavior_drug) (0: None, 1: Present, 2: Past, 3: Never, 4: Unsure)

For each of the above category, associating tags are given and a value is expected to be provided for each tag, with each value corresponding to a status.
For each tag, the following areas should be considered before assigning a numerical value:
             
1) sdoh_community_present / sdoh_community_absent : social integration, support systems, community engagement, discrimination, stress
2) sdoh_education : literacy, language, early childhood education, vocational training, higher education
3) sdoh_economics : employment, income, expenses, debt, medical bills, support
4) sdoh_environment : housing, transportation, safety, parks, playgrounds, walkability, zip code/geography
5) behavior_alcohol : consumes alcohol
6) behavior_tobacco : uses tobacco 
7) behavior_drug : uses a controlled substance, including marijuana, for which the patient does not have a prescription

Provide EXACTLY and ONLY 8 CSV values in this order WITHOUT ANY EXPLANATION OR JUSTIFICATION:
sdoh_community_present,sdoh_community_absent,sdoh_education,sdoh_economics,sdoh_environment,behavior_alcohol,behavior_tobacco,behavior_drug
                    """},
                    {"role": "user", "content": text}
                ],
                temperature=0.0,
            )
            content = response.choices[0].message.content.strip()
            fields = content.split(',')
            
            if len(fields) != 8:
                raise ValueError(f"Expected 8 fields but got {len(fields)}: '{content}'")
            
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }
            return fields, usage

        except (OpenAIError, ValueError) as e:
            print(f"API/Error on attempt {attempt+1}: {e}")
            if attempt < retries - 1:
                sleep_time = backoff_factor * (2 ** attempt) + random.uniform(0, 1)
                print(f"Retrying after {sleep_time:.2f} seconds...")
                time.sleep(sleep_time)
            else:
                # Default safe values if all retries fail
                return ["0","0","0","0","0","0","0","0"], {"prompt_tokens":0,"completion_tokens":0,"total_tokens":0}

df = pd.read_csv('MIMIC-SBDH_w_notes.csv')

annotations = []
usage_data = []

df_subset = df.head(100)

for idx, row in tqdm(df_subset.iterrows(), total=len(df_subset)):
    annotation_fields, usage = annotate_sdoH(row['TEXT'])
    annotations.append([row['ROW_ID']] + annotation_fields)
    usage_data.append(usage)

columns = [
    'row_id', 'sdoh_community_present', 'sdoh_community_absent', 'sdoh_education',
    'sdoh_economics', 'sdoh_environment', 'behavior_alcohol', 'behavior_tobacco', 'behavior_drug'
]

annotations_df = pd.DataFrame(annotations, columns=columns)
annotations_df.to_csv('gpt4o-mini_annotated_results.csv', index=False)

df_usage = pd.DataFrame(usage_data)
df_usage.to_csv('token_usage.csv', index=False)
