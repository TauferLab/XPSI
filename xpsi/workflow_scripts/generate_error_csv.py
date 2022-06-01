# Script for generating a csv file containing averages and their variability
# Note that this file is intended to be modified to iterate over the correct set of tests

import csv
import numpy as np



def print_error_averages(input_dir, csvwriter):

  test_errors = np.loadtxt(input_dir+"rf_test_errors.txt")

  test_means = np.mean(test_errors, axis=1)
  test_medians = np.median(test_errors, axis=1)
  test_rmse = np.sqrt(np.mean(np.square(test_errors), axis=1))

  values = [np.min(test_means), np.median(test_means), np.max(test_means),
            np.min(test_medians), np.median(test_medians), np.max(test_medians),
            np.min(test_rmse), np.median(test_rmse), np.max(test_rmse)]

  csvwriter.writerow((round(x, 4) for x in values))



with open('averages.csv', 'w') as csvfile:
  csvwriter = csv.writer(csvfile)
#  for config in ['1n0u_39692_1e16_5epochs','1n0u_39692_1e16_10epochs','1n0u_39692_1e16_15epochs','1n0u_39692_1e16_20epochs','1n0u_39692_1e16_25epochs','1n0u_39692_1e16_30epochs','1n0u_39692_1e16_35epochs','1n0u_39692_1e16_40epochs','1n0u_39692_1e16_45epochs','1n0u_39692_1e16_50epochs','1n0u_39692_1e16_55epochs','1n0u_39692_1e16_60epochs','1n0u_39692_1e16_65epochs','1n0u_39692_1e16_70epochs','1n0u_39692_1e16_75epochs','1n0u_39692_1e16_80epochs', ]:
  for config in ['cpu_1n0u_39692_1e16_50epochs']:
    for k in []:
      print_error_averages('output/'+config+'/prediction/'+str(k)+' Nearest Neighbors/', csvwriter)
    for trees in [1, 10, 100]:
      print_error_averages('output/'+config+'/prediction/Random '+str(trees)+' Tree Forest/', csvwriter)
