"""
Extract cholesterol-lowering medication data from the UK Biobank dataset.
Categorize medication use by sex and age
"""

import pandas as pd

df = pd.read_csv("/lustre03/project/6008063/neurohub/UKB/Tabular/current.csv", nrows=0)

# 6177 records medication use in males and 6153 records
# medication use in females:
medication_cols = [c for c in df.columns if '6177-0' in c or '6153-0' in c]

df = pd.read_csv("/lustre03/project/6008063/neurohub/UKB/Tabular/current.csv",
                 usecols=['eid', '22001-0.0', '21022-0.0', '30780-0.0', '30760-0.0', '30690-0.0', '30870-0.0'] + medication_cols)

# Rename columns for clarity
df.rename(columns={
    'eid': 'IID',
    '22001-0.0': 'Sex',
    '21022-0.0': 'Age',
    '30780-0.0': 'LDL',
    '30760-0.0': 'HDL',
    '30690-0.0': 'TC',
    '30870-0.0': 'TG'
}, inplace=True)

# Remove rows with missing values for sex or age or LDL:
df.dropna(subset=['Age', 'Sex', 'LDL'], inplace=True)
df['Sex'] = df['Sex'].astype(int).map({
    0: 'Female',
    1: 'Male'
})

# Collapse the medication use columns into a single binary column:
df['chol_med'] = (df[medication_cols] == 1).any(axis=1)

# Create age bins:

bins = [0, 50, 60, float('inf')]
labels = ['Age<50', 'Age 50–60', 'Age>60']

df['AgeGroup'] = pd.cut(df['Age'], bins=bins, labels=labels, right=False).astype(str)

res = []

res.append(
    df.groupby('Sex')['chol_med'].mean().reset_index().rename(
        columns={
            'Sex': 'Group',
            'chol_med': 'Proportion_Using_Medication'
        }
    )
)

res.append(
    df.groupby('AgeGroup')['chol_med'].mean().reset_index().rename(
        columns={
            'AgeGroup': 'Group',
            'chol_med': 'Proportion_Using_Medication'
        }
    )
)

group_df = pd.concat(res, ignore_index=True)
group_df.to_csv("data/misc/cholesterol_medication_prevalence.csv", index=False)
