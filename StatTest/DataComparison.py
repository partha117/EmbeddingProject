import pandas as pd
from scipy.stats import wilcoxon
import numpy as np


if __name__ == "__main__":
    df = pd.read_csv("../Data/Embedding Results - For CSV.csv")
    dataset_performance = ['AspectJ', 'Birt', 'Eclipse', 'JDT', 'SWT', 'Tomcat', 'Overall']
    curated_data = {item: [] for item in df["Training\nData (CNN Model)"].unique().tolist()}
    for model in sorted(df['Model\nName'].unique().tolist()):
        for training_method in sorted(df['Embedding Training\nStrategy'].unique().tolist()):
            for data in sorted(curated_data.keys()):
                for dataset in dataset_performance:
                    temp = df[(df['Model\nName'] == model) & (df['Embedding Training\nStrategy'] == training_method) & (df["Training\nData (CNN Model)"] == data)][dataset].tolist()
                    curated_data[data].append(temp[0] if len(temp) > 0 else None)
    curated_df = pd.DataFrame(curated_data)
    curated_df = curated_df.dropna()
    # Null Hypothesis: The first one is greater than the second one
    print(wilcoxon(curated_df['BLDS'], curated_df['Bench-BLDS'], alternative='less'))