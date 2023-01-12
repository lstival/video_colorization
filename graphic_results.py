import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

# Dataset
# dataset = "DAVIS"
dataset = "videvo"

# Name of csv with metrics
metric_filename = "model_metrics.csv"

# Path where is the csv with metrics
root_metrics_path = f"models_metrics/{dataset}"

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
path_save_results = f"results/{dataset}"
os.makedirs(path_save_results, exist_ok=True)

# Read the map (losses for each model)
df_model_map = pd.read_csv("models_map_losses.csv")
df_model_metrics = pd.concat([df_model_metrics, df_model_map.set_index("time stap")], axis=1, join="inner")

# fill the model with "+++" to do not consider in the metrics
df_model_metrics.fillna("+++", inplace=True)

# save the csv metrics
df_model_metrics.to_csv(f"{path_save_results}/models_metrics_means.csv")
df_model_metrics = df_model_metrics.iloc[:, [3,0,1,2]]

######### Generating the df to plot

# One metric
df_one_metric = df_model_metrics[df_model_metrics.apply(lambda x: x["loss"].count("+"), axis=1) == 0]
df_one_metric.round(3).to_csv(f"{path_save_results}/models_one_means.csv")

df_two_metric = df_model_metrics[df_model_metrics.apply(lambda x: x["loss"].count("+"), axis=1) == 1]
df_two_metric.round(3).to_csv(f"{path_save_results}/models_two_means.csv")

df_three_metric = df_model_metrics[df_model_metrics.apply(lambda x: x["loss"].count("+"), axis=1) == 2]
df_three_metric.round(3).to_csv(f"{path_save_results}/models_three_means.csv")