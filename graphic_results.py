import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

# Name of csv with metrics
metric_filename = "model_metrics.csv"

# Path where is the csv with metrics
root_metrics_path = "models_metrics"

# List all models that has results
models_metrics = os.listdir(root_metrics_path)

dict_metrics = {}

# Loop to all folders to get the model metrics
pbar = tqdm(models_metrics)
for model_name in pbar:
    pbar.set_description(f"Processing model: {model_name}")
    
    df_metrics = pd.read_csv(f"{root_metrics_path}/{model_name}/{metric_filename}")

    ssim_mean = df_metrics["SSIM"].mean()
    psnr_mean = df_metrics["PSNR"].mean()
    lpisp_mean = df_metrics["LPISP"].mean()

    # save the means in a dict
    dict_metrics[model_name] = [float(ssim_mean), float(psnr_mean), float(lpisp_mean)]

# Create df with all means and set the columns names
df_model_metrics = pd.DataFrame.from_dict(dict_metrics)
df_model_metrics = df_model_metrics.T
df_model_metrics.columns = ["SSIM", "PSNR", "LPISP"]

# create the fold to save the csv results
path_save_results = "results/"
os.makedirs(path_save_results, exist_ok=True)

# save the csv metrics
df_model_metrics.to_csv(f"{path_save_results}models_metrics_means.csv")