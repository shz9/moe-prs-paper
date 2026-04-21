import numpy as np
import pandas as pd

cad_relevant_fields = [
    "131296-0.0",
    "131298-0.0",
    "131300-0.0",
    "131302-0.0",
    "131304-0.0",
    "131306-0.0",
]

death_date_cols = ["40000-0.0", "40000-1.0"]
birth_date_cols = ["34-0.0", "52-0.0"]

df = pd.read_csv(
    "/project/rpp-aevans-ab/neurohub/UKB/Tabular/current.csv",
    usecols=["eid"] + birth_date_cols + death_date_cols + cad_relevant_fields,
)

# Extract date of birth from year / month:
df["birth_date"] = pd.to_datetime(
    df["34-0.0"].astype(str) + "-" + df["52-0.0"].astype(str)
)

# Turn all dates to datetime objects:
for c in cad_relevant_fields + death_date_cols:
    df[c] = pd.to_datetime(df[c])
    # Remove fields that are obviously erroneous:
    df[c] = df[c].mask(df[c] < df["birth_date"])

# Determine the end date for the available records:
end_of_record = df[cad_relevant_fields + death_date_cols].max().max()

# Determine date of death:
df["death_date"] = df[death_date_cols].max(axis=1)

# Remove records that happen after death date:
for c in cad_relevant_fields:
    df[c] = df[c].mask(df[c] > df["death_date"])

# Compute age at diagnosis:
for c in cad_relevant_fields:
    df[c] = (df[c] - df["birth_date"]).dt.days / 365

df["age_at_diagnosis"] = np.round(np.nanmin(df[cad_relevant_fields], axis=1))

# Compute age at death:
df["age_at_death"] = np.round((df["death_date"] - df["birth_date"]).dt.days / 365)

# Compute age at end of records:
age_end = (end_of_record - df["birth_date"]).dt.days / 365

df["age_at_record_end"] = np.round(
    np.minimum(age_end, df["age_at_death"].fillna(np.inf))
)

clean_df = df[["eid", "age_at_diagnosis", "age_at_death", "age_at_record_end"]].rename(
    columns={"eid": "IID"}
)

# For robustness, exclude samples where age at diagnosis < 40
clean_df.loc[clean_df["age_at_diagnosis"] < 40, "age_at_diagnosis"] = np.nan

clean_df.to_csv(
    "data/covariates/ukbb/age_at_first_diagnosis_cad.txt", sep="\t", index=False
)
