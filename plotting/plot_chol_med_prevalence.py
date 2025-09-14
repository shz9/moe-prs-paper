import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv("data/misc/cholesterol_medication_prevalence.csv")
plt.figure(figsize=(5, 5))

sns.barplot(data=df, x='Group', y='Proportion_Using_Medication', color='skyblue',
            order=['Female', 'Male', 'Age<50', 'Age 50–60', 'Age>60'])
plt.xlabel("Evaluation Group")
plt.ylabel("Proportion")
plt.title("Proportion of samples using\ncholesterol-lowering medication (UKB)")

plt.savefig("figures/section_3/cholesterol_medication_prevalence.eps",
            bbox_inches='tight')
plt.close()
