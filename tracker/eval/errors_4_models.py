import argparse
import json
import os

import numpy as np
from tracker.coma_model.data import data
from matplotlib import pyplot as plt
from util.log_util import date_print
from tracker.config import config as cfg

parser = argparse.ArgumentParser()
parser.add_argument("--first-prediction", default="results/spiral-l1_result.npy",
                    help="The path to the predictions.npy")
parser.add_argument("--second-prediction", default="results/cheb-l1_result.npy",
                    help="The path to the predictions.npy")

parser.add_argument("--third-prediction", default="results/spiral-l1_result.npy",
                    help="The path to the predictions.npy")
parser.add_argument("--fourth-prediction", default="results/cheb-l1_result.npy",
                    help="The path to the predictions.npy")

parser.add_argument("--first-plot-name")
parser.add_argument("--second-plot-name")
parser.add_argument("--third-plot-name")
parser.add_argument("--fourth-plot-name")

# parser.add_argument("--data-dir", default="data/sliced",
#                     help="The path to the data")
parser.add_argument("--error-dir", default="all_errors", help="The directory where the computed errors should be stored")
parser.add_argument("--error-name", help="Name of the error that is being computed", required=True)
parser.add_argument("--x-axis-size", help="The size of the x axis", default=40)


def cumulative_error(error):
    num_bins = 1000
    num_vertices, x_error = np.histogram(error, bins=num_bins)
    x_error = np.convolve(x_error, 0.5 * np.ones(2), mode='valid')

    factor_error = 100 / error.shape[0]

    cumulative_num_vertices = np.cumsum(num_vertices)

    x_error_vector = np.zeros((num_bins + 1,))
    x_error_vector[1:] = x_error

    y_vec_error = np.zeros((num_bins + 1,))
    y_vec_error[1:] = factor_error * cumulative_num_vertices

    return x_error_vector, y_vec_error


def plot_error_over_vertices(first_model_error, second_model_error, third_model_error, fourth_model_error,
                             label_first_pred, label_second_pred, label_third_pred, label_fourth_pred,
                             vertices, path, x_axis_size):
    first_model_error = np.reshape(first_model_error, (-1,))
    second_model_error = np.reshape(second_model_error, (-1,))
    third_model_error = np.reshape(third_model_error, (-1,))
    fourth_model_error = np.reshape(fourth_model_error, (-1,))

    x_first, y_first = cumulative_error(first_model_error)
    x_second, y_second = cumulative_error(second_model_error)
    x_third, y_third = cumulative_error(third_model_error)
    x_fourth, y_fourth = cumulative_error(fourth_model_error)

    plt.plot(x_first, y_first, label=label_first_pred)
    plt.plot(x_second, y_second, label=label_second_pred)
    plt.plot(x_third, y_third, label=label_third_pred)
    plt.plot(x_fourth, y_fourth, label=label_fourth_pred)
    plt.ylabel('Vertices (percent)')
    plt.xlabel('Euclidean Error (mm)')

    plt.legend(loc='lower right')
    plt.xlim(0, int(x_axis_size))
    plt.grid(True)  # ,color='grey', linestyle='-', linewidth=0.5)
    plt.savefig(path)
    pass


def calculate_error(predicted_vertices, original_vertices):
    error = np.sqrt(np.sum((predicted_vertices - original_vertices) ** 2, axis=2))
    error_mean = np.mean(error)
    error_std = np.std(error)
    error_median = np.median(error)
    return error, error_mean, error_std, error_median


args = parser.parse_args()

first_prediction_name = os.path.basename(args.first_prediction).split(".")[0]
first_prediction = np.load(args.first_prediction)
date_print("Calculating errors for " + args.first_prediction)

mesh_data = data.ChokepointDataflowProvider(fit_pca=False, mesh_path=cfg.COMA.TEMPLATE_MESH_EVAL)

# First Prediction
date_print("Predicting using {}".format(first_prediction_name))
predicted_first_vertices = (first_prediction * mesh_data.std) + mesh_data.mean

# CoMA spiral
second_prediction_name = os.path.basename(args.second_prediction).split(".")[0]
second_prediction = np.load(args.second_prediction)
predicted_second_vertices = (second_prediction * mesh_data.std) + mesh_data.mean

# CoMA spiral
third_prediction_name = os.path.basename(args.third_prediction).split(".")[0]
third_prediction = np.load(args.third_prediction)
predicted_third_vertices = (third_prediction * mesh_data.std) + mesh_data.mean

# CoMA spiral
fourth_prediction_name = os.path.basename(args.fourth_prediction).split(".")[0]
fourth_prediction = np.load(args.fourth_prediction)
predicted_fourth_vertices = (fourth_prediction * mesh_data.std) + mesh_data.mean

# Plotnames
first_plot_name = args.first_plot_name
second_plot_name = args.second_plot_name
third_plot_name = args.third_plot_name
fourth_plot_name = args.fourth_plot_name

if first_plot_name is None:
    first_plot_name = first_prediction_name
if second_plot_name is None:
    second_plot_name = second_prediction_name
if third_plot_name is None:
    third_plot_name = third_prediction_name
if fourth_plot_name is None:
    fourth_plot_name = fourth_prediction_name

# Original
original_vertices = (mesh_data.val_meshes[:first_prediction.shape[0]] * mesh_data.std) + mesh_data.mean

# we want millimeters
predicted_first_vertices_mm = predicted_first_vertices * 1000
predicted_second_vertices_mm = predicted_second_vertices * 1000
predicted_third_vertices_mm = predicted_third_vertices * 1000
predicted_fourth_vertices_mm = predicted_fourth_vertices * 1000
original_vertices_mm = original_vertices * 1000

# CoMA cheb
first_model_error, first_model_error_mean, first_model_error_std, first_model_error_median = calculate_error(
    predicted_first_vertices_mm,
    original_vertices_mm)

date_print(
    "{} Error - Mean: ".format(first_prediction_name) + str(first_model_error_mean) + ", Std: " + str(
        first_model_error_std) + ", Median: " + str(
        first_model_error_median))

# CoMA spiral
second_model_error, second_model_error_mean, second_model_error_std, second_model_error_median = calculate_error(
    predicted_second_vertices_mm,
    original_vertices_mm)

date_print(
    "{} Error - Mean: ".format(second_prediction_name) + str(second_model_error_mean) + ", Std: " + str(
        second_model_error_std) + ", Median: " + str(
        second_model_error_median))

# Third Model
third_model_error, third_model_error_mean, third_model_error_std, third_model_error_median = calculate_error(
    predicted_third_vertices_mm,
    original_vertices_mm)

date_print(
    "{} Error - Mean: ".format(third_prediction_name) + str(third_model_error_mean) + ", Std: " + str(
        third_model_error_std) + ", Median: " + str(
        third_model_error_median))

# Fourth model
fourth_model_error, fourth_model_error_mean, fourth_model_error_std, fourth_model_error_median = calculate_error(
    predicted_fourth_vertices_mm,
    original_vertices_mm)

date_print(
    "{} Error - Mean: ".format(fourth_prediction_name) + str(fourth_model_error_mean) + ", Std: " + str(
        fourth_model_error_std) + ", Median: " + str(
        fourth_model_error_median))

date_print("Saving error plot")
error_plot_path = args.error_dir + "/error_plot"
if not os.path.exists(error_plot_path):
    os.makedirs(error_plot_path)

plot_error_over_vertices(first_model_error,
                         second_model_error,
                         third_model_error,
                         fourth_model_error,
                         first_plot_name,
                         second_plot_name,
                         third_plot_name,
                         fourth_plot_name,
                         original_vertices,
                         error_plot_path + "/" + args.error_name,
                         x_axis_size=args.x_axis_size)

date_print("Storing errors.")
if not os.path.exists(args.error_dir):
    os.makedirs(args.error_dir)
# Cheb CoMA
first_prediction_error_file = args.error_dir + "/" + first_prediction_name + "_" + args.error_name
with open(first_prediction_error_file + ".json", 'w') as file:
    save_params = dict()
    # save_params['coma_model_error'] = model_error
    save_params[first_prediction_name + '_model_error_mean'] = str(first_model_error_mean)
    save_params[first_prediction_name + '_model_error_std'] = str(first_model_error_std)
    save_params[first_prediction_name + '_model_error_median'] = str(first_model_error_median)
    date_print(str(save_params))
    json.dump(save_params, file)
np.save(first_prediction_error_file + "_error.npy", first_model_error)

# Spiral CoMA
second_prediction_error_file = args.error_dir + "/" + second_prediction_name + "_" + args.error_name
with open(second_prediction_error_file + ".json", 'w') as file:
    save_params = dict()
    # save_params['coma_model_error'] = model_error
    save_params[second_prediction_name + '_model_error_mean'] = str(second_model_error_mean)
    save_params[second_prediction_name + '_model_error_std'] = str(second_model_error_std)
    save_params[second_prediction_name + '_model_error_median'] = str(second_model_error_median)
    date_print(str(save_params))
    json.dump(save_params, file)
np.save(second_prediction_error_file + "_error.npy", second_model_error)

third_prediction_error_file = args.error_dir + "/" + third_prediction_name + "_" + args.error_name
with open(third_prediction_error_file + ".json", 'w') as file:
    save_params = dict()
    # save_params['coma_model_error'] = model_error
    save_params[third_prediction_name + '_model_error_mean'] = str(third_model_error_mean)
    save_params[third_prediction_name + '_model_error_std'] = str(third_model_error_std)
    save_params[third_prediction_name + '_model_error_median'] = str(third_model_error_median)
    date_print(str(save_params))
    json.dump(save_params, file)
np.save(third_prediction_error_file + "_error.npy", third_model_error)

fourth_prediction_error_file = args.error_dir + "/" + fourth_prediction_name + "_" + args.error_name
with open(fourth_prediction_error_file + ".json", 'w') as file:
    save_params = dict()
    # save_params['coma_model_error'] = model_error
    save_params[fourth_prediction_name + '_model_error_mean'] = str(fourth_model_error_mean)
    save_params[fourth_prediction_name + '_model_error_std'] = str(fourth_model_error_std)
    save_params[fourth_prediction_name + '_model_error_median'] = str(fourth_model_error_median)
    date_print(str(save_params))
    json.dump(save_params, file)
np.save(fourth_prediction_error_file + "_error.npy", fourth_model_error)
