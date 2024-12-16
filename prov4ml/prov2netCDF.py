import json
import netCDF4 as nc
import os, argparse
import numpy as np

if __package__ == None:
    print('This file must be run as a module using "python -m prov4ml.prov2netCDF -h" and not directly.')
    exit()

from prov4ml.utils.prov_getters import get_metric_numpy, get_metrics
from prov4ml.utils.compress_utils import compress_file, print_file_size

def json_to_netcdf(json_file, netcdf_file):
    # Load JSON data
    with open(json_file, 'r') as f:
        data = json.load(f)

    metrics = get_metrics(data, "TRAINING")
    metrics = [get_metric_numpy(data, m) for m in metrics] # [[epochs, values, time, size], [.., .., .., ..], ...]

    # Determine groups dimension
    unique_sizes = list(set([len(m[0]) for m in metrics]))

    # Stack metrics with equal sizes
    epochs = {}
    values = {}
    times = {}
    num_metrics_per_size = {}    
    for metric in metrics: 
        size = metric[3]

        # Save how many metrics we have for each size
        if size in num_metrics_per_size: 
            num_metrics_per_size[size] += 1
        else: 
            num_metrics_per_size[size] = 1
            flag = True

        # Stack
        if flag:
            epochs[size] = metric[0].copy()
            values[size] = metric[1].copy()
            times[size] = metric[2].copy()
            flag = False
        else:
            epochs[size] = np.vstack((epochs[size], metric[0]))
            values[size] = np.vstack((values[size], metric[1]))
            times[size] = np.vstack((times[size], metric[2]))

    # Resize metrics with only one row
    for size in unique_sizes:
        if epochs[size].ndim == 1:
            epochs[size].resize(1, size)
            values[size].resize(1, size)
            times[size].resize(1, size)
        
    # Create NetCDF file
    dataset = nc.Dataset(netcdf_file, 'w', format='NETCDF4')
 
    groups = {size: dataset.createGroup(f"metric_granularity_{size}") for size in unique_sizes}

    for size, group in groups.items(): 
        group.createDimension('metrics', num_metrics_per_size[size])
        group.createDimension('items', size)

        groups[size] = [
            group.createVariable('epochs', 'i4', ('metrics', 'items')),
            group.createVariable('values', 'f4', ('metrics', 'items')), 
            group.createVariable('timestamps', 'i8', ('metrics', 'items'))
        ]

        group['epochs'][:] = epochs[size]
        group['values'][:] = values[size]
        group['timestamps'][:] = times[size]

    # Add metadata
    dataset.description = 'Metrics with values, timestamps, and epochs'
    dataset.source = 'Converted from JSON'
    # dataset.processing_date = '2024-09-13'

    # To do

    # Close the dataset
    dataset.close()
    print(f'NetCDF file "{netcdf_file}" created successfully.')

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--input', help='input file path, must be .json', required=True)
    parser.add_argument('-o', '--output', help='output file path, if missing defaults to <input>.nc', required=False)

    args = parser.parse_args()
    
    input_file: str = os.path.abspath(args.input)
    output_file: str

    if not os.path.isfile(input_file):
        print("Input is not a valid file")
        exit()

    if not input_file.endswith('.json'):
        print("File type must be .json")
        exit()

    if args.output:
        output_file = os.path.abspath(args.output)

        if not output_file.endswith('.nc'):
            output_file += '.nc'
    else:
        output_file = input_file.replace('.json', '.nc')

    return input_file, output_file

if __name__ == "__main__":

    input_file, output_file = parse_args()

    json_to_netcdf(input_file, output_file)
    compress_file(output_file, output_file)
    
    print_file_size(input_file)
    print_file_size(output_file)
    print_file_size(output_file + '.gz')
