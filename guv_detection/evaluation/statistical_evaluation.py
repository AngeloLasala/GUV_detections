"""
Compute the statistical evaluation between different model 
tanking as single 'item' one acquisition 
- old dataset: 1 item = 4 images
- new dataset: 1 item = 16 image
"""
import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.stats import wilcoxon, friedmanchisquare
from scipy.stats import ttest_rel
import json

def main(args):
    """
    Compute the statistical evaluation between different model
    """

    grey_path = os.path.join(args.folder, f'DATA_training_grey_txt', 'test', 'statistical_analysis')
    rgb_path = os.path.join(args.folder, f'DATA_training_rgb_txt', 'test', 'statistical_analysis')

    grey_metrics_dict = {i: None for i in os.listdir(grey_path)}
    rgb_metrics_dict = {i: None for i in os.listdir(grey_path)}

    for i in os.listdir(grey_path):
    
        if i in os.listdir(rgb_path):
            grey_metrics_path = os.path.join(grey_path, i, 'metrics.txt')
            rgb_metrics_path = os.path.join(rgb_path, i, 'metrics.txt')

            if os.path.exists(grey_metrics_path) and os.path.exists(rgb_metrics_path):
                with open(grey_metrics_path, 'r') as f:
                    grey_metrics = f.readlines()
                with open(rgb_metrics_path, 'r') as f:
                    rgb_metrics = f.readlines()

                grey_metrics_dict[i] = {line.split(': ')[0]: float(line.split(': ')[1]) for line in grey_metrics}
                rgb_metrics_dict[i] = {line.split(': ')[0]: float(line.split(': ')[1]) for line in rgb_metrics}

    precision_grey, recall_grey, mAP50_grey, mAP50_95_grey = [], [], [], []
    precision_rgb, recall_rgb, mAP50_rgb, mAP50_95_rgb = [], [], [], []
    for key in grey_metrics_dict.keys():
        precision_grey.append(grey_metrics_dict[key]['metrics/precision(B)'])
        recall_grey.append(grey_metrics_dict[key]['metrics/recall(B)'])
        mAP50_grey.append(grey_metrics_dict[key]['metrics/mAP50(B)'])
        mAP50_95_grey.append(grey_metrics_dict[key]['metrics/mAP50-95(B)'])
        precision_rgb.append(rgb_metrics_dict[key]['metrics/precision(B)'])
        recall_rgb.append(rgb_metrics_dict[key]['metrics/recall(B)'])
        mAP50_rgb.append(rgb_metrics_dict[key]['metrics/mAP50(B)'])
        mAP50_95_rgb.append(rgb_metrics_dict[key]['metrics/mAP50-95(B)'])

    ## Print the mean and std of the metrics
    print(f"Precision) Grey: {np.mean(precision_grey):.4f} ± {np.std(precision_grey):.4f} - RGB: {np.mean(precision_rgb):.4f} ± {np.std(precision_rgb):.4f}")
    print(f"Recall) Grey: {np.mean(recall_grey):.4f} ± {np.std(recall_grey):.4f} - RGB: {np.mean(recall_rgb):.4f} ± {np.std(recall_rgb):.4f}")
    print(f"mAP50) Grey: {np.mean(mAP50_grey):.4f} ± {np.std(mAP50_grey):.4f} - RGB: {np.mean(mAP50_rgb):.4f} ± {np.std(mAP50_rgb):.4f}")
    print(f"mAP50-95) Grey: {np.mean(mAP50_95_grey):.4f} ± {np.std(mAP50_95_grey):.4f} - RGB: {np.mean(mAP50_95_rgb):.4f} ± {np.std(mAP50_95_rgb):.4f}")

    # Example data
    metrics = ['Precision', 'Recall', 'mAP50', 'mAP50-95']
    grey_metrics = [precision_grey, recall_grey, mAP50_grey, mAP50_95_grey]
    rgb_metrics  = [precision_rgb, recall_rgb, mAP50_rgb, mAP50_95_rgb]

    ## Wilcoxon test
    # Statistical tests
    for i, metric in enumerate(metrics):
        stat, p = ttest_rel(grey_metrics[i], rgb_metrics[i])
        print(f"Wilcoxon test for {metric}: stat={stat:.4f}, p-value={p:.4f}")
        if p < 0.05:
            print(f"  -> Significant difference in {metric} between Grey and RGB models (p < 0.05)")
        else:
            print(f"  -> No significant difference in {metric} between Grey and RGB models (p >= 0.05)")


    # Create a single DataFrame
    df_list = []
    for i, metric in enumerate(metrics):
        df = pd.DataFrame({
            'Metric': metric,
            'Grey': grey_metrics[i],
            'RGB': rgb_metrics[i]
        })
        df_melted = df.melt(id_vars=['Metric'], var_name='Model', value_name='Value')
        df_list.append(df_melted)

    # Concatenate all metrics
    data_all = pd.concat(df_list, ignore_index=True)

    # Plot all in one figure
    plt.figure(figsize=(10, 6))
    sns.violinplot(
        x='Metric', y='Value', hue='Model', data=data_all,
        palette='Set2', split=False
    )
    sns.stripplot(
        x='Metric', y='Value', hue='Model', data=data_all,
        dodge=True, color='k', alpha=0.6, size=4
)
    plt.title('Violin Plots of Metrics for Grey and RGB Models')
    plt.ylabel('Metric Value')
    plt.show()

  

                

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Statistical evaluation of different models")
    parser.add_argument("--model_size", type=str, default="n", help="size of YOLO model, e.g., n, s, m, l, x")
    parser.add_argument("--folder", type=str, default="/media/angelo/OS/Users/lasal/OneDrive - Scuola Superiore Sant'Anna/PhD_notes/Liposomes detection", 
                        help="Path to the Liposomes detection folder")
    args = parser.parse_args()

    main(args)