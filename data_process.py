import numpy as np
import torch

def save_aligned_statistics(pred, gt, base_filename="data_statistics"):
    """
    Calculates statistical features for each frame in the pred and gt data
    and saves the results to a separate, fixed-width formatted text file
    for each sample, ensuring columns are neatly aligned.

    Args:
        pred (np.ndarray): Predicted data, expected as a 5D array of shape
                           (num_samples, num_frames, channels, height, width).
        gt (np.ndarray): Ground truth data, expected to have the same shape
                         as pred.
        base_filename (str): The base name for the output files. Defaults to
                             "data_statistics".
    """
    assert pred.shape == gt.shape, "Shape of 'pred' and 'gt' must be identical."

    # --- Formatting Definitions ---
    # Define column widths for neat, fixed-width formatting.
    # Adjust these values if your column names or numbers require more space.
    ID_WIDTH = 10
    VALUE_WIDTH = 18

    num_samples = pred.shape[0]
    print(f"Found {num_samples} samples, starting processing...")

    for sample_idx in range(num_samples):
        output_filename = f"{base_filename}_sample_{sample_idx:03d}.txt"

        with open(output_filename, 'w') as f:
            # Create and write the formatted header string
            header = (f"{'Frame_ID':<{ID_WIDTH}}"
                      f"{'Pred_Max':<{VALUE_WIDTH}}"
                      f"{'Pred_Min':<{VALUE_WIDTH}}"
                      f"{'Pred_Mean':<{VALUE_WIDTH}}"
                      f"{'Pred_Var':<{VALUE_WIDTH}}"
                      f"{'GT_Max':<{VALUE_WIDTH}}"
                      f"{'GT_Min':<{VALUE_WIDTH}}"
                      f"{'GT_Mean':<{VALUE_WIDTH}}"
                      f"{'GT_Var':<{VALUE_WIDTH}}\n")
            f.write(header)

            num_frames = pred.shape[1]
            for frame_idx in range(num_frames):
                pred_frame = pred[sample_idx, frame_idx, :, :, :]
                gt_frame = gt[sample_idx, frame_idx, :, :, :]

                # Calculate statistics for the current frame
                stats = [
                    frame_idx,
                    np.max(pred_frame), np.min(pred_frame), np.mean(pred_frame), np.var(pred_frame),
                    np.max(gt_frame), np.min(gt_frame), np.mean(gt_frame), np.var(gt_frame)
                ]

                # Create the formatted data line string
                line = (f"{stats[0]:<{ID_WIDTH}}"
                        f"{stats[1]:<{VALUE_WIDTH}.5f}"
                        f"{stats[2]:<{VALUE_WIDTH}.5f}"
                        f"{stats[3]:<{VALUE_WIDTH}.5f}"
                        f"{stats[4]:<{VALUE_WIDTH}.5f}"
                        f"{stats[5]:<{VALUE_WIDTH}.5f}"
                        f"{stats[6]:<{VALUE_WIDTH}.5f}"
                        f"{stats[7]:<{VALUE_WIDTH}.5f}"
                        f"{stats[8]:<{VALUE_WIDTH}.5f}\n")
                
                f.write(line)

        print(f"Statistics for sample {sample_idx} saved to: {output_filename}")

    print("All samples have been processed successfully.")

def save_dataset_statistics(data, base_filename="dataset_statistics"):
    """
    Calculates statistical features for each frame in a given dataset and saves
    the results to a separate, fixed-width formatted text file for each sample.

    This function is intended for analyzing a single dataset (e.g., ground truth)
    without a corresponding prediction. The output is formatted into aligned
    columns for easy reading in any text editor.

    Args:
        data (np.ndarray): The dataset to analyze, expected as a 5D array of
                           shape (num_samples, num_frames, channels, height, width).
        base_filename (str): The base name for the output files. Defaults to
                             "dataset_statistics".
    """
    # Check if the input is a 5D array
    if data.ndim != 5:
        raise ValueError(f"Input data must be a 5D array, but got {data.ndim} dimensions.")

    # --- Formatting Definitions ---
    # Define column widths for neat, fixed-width formatting.
    # You can adjust these values if your numbers are larger or smaller.
    ID_WIDTH = 10        # Width for the Frame_ID column
    VALUE_WIDTH = 18     # Width for all statistical value columns

    num_samples = data.shape[0]
    print(f"Found {num_samples} samples, starting processing...")

    for sample_idx in range(num_samples):
        output_filename = f"{base_filename}_sample_{sample_idx:03d}.txt"

        with open(output_filename, 'w') as f:
            # Create and write the formatted header string
            header = (f"{'Frame_ID':<{ID_WIDTH}}"
                      f"{'Max':<{VALUE_WIDTH}}"
                      f"{'Min':<{VALUE_WIDTH}}"
                      f"{'Mean':<{VALUE_WIDTH}}"
                      f"{'Variance':<{VALUE_WIDTH}}\n")
            f.write(header)

            num_frames = data.shape[1]
            for frame_idx in range(num_frames):
                # Get the data for the current sample and frame
                data_frame = data[sample_idx, frame_idx, :, :, :]

                # Calculate statistics for the current frame
                stats = [
                    frame_idx,
                    np.max(data_frame),
                    np.min(data_frame),
                    np.mean(data_frame),
                    np.var(data_frame)
                ]

                # Create the formatted data line string
                # {value:<{width}} means left-align the value within the given width.
                # .5f specifies 5 decimal places for float numbers.
                line = (f"{stats[0]:<{ID_WIDTH}}"
                        f"{stats[1]:<{VALUE_WIDTH}.5f}"
                        f"{stats[2]:<{VALUE_WIDTH}.5f}"
                        f"{stats[3]:<{VALUE_WIDTH}.5f}"
                        f"{stats[4]:<{VALUE_WIDTH}.5f}\n")
                
                f.write(line)

        print(f"Statistics for sample {sample_idx} saved to: {output_filename}")

    print("All samples have been processed successfully.")

def normalize(data, method='minmax'):
    """
    Normalizes the input data using a specified method.

    Args:
        data (np.ndarray or torch.Tensor): The input data to be normalized.
        method (str): The normalization method to use. Supported methods are:
                      'minmax' for scaling to [-1, 1].
                      'zscore' for standardization (zero mean, unit variance).
                      Defaults to 'minmax'.

    Returns:
        tuple: A tuple containing:
            - normalized_data (np.ndarray or torch.Tensor): The normalized data,
              same type as input.
            - stats (dict): A dictionary containing the statistics used for
              normalization (e.g., min/max or mean/std).
    """
    # Ensure data is a torch.Tensor for computation
    is_numpy = isinstance(data, np.ndarray)
    if is_numpy:
        tensor_data = torch.from_numpy(data.copy()).float()
    else:
        tensor_data = data.clone().float()

    stats = {}
    
    if method == 'minmax':
        data_min = tensor_data.min()
        data_max = tensor_data.max()
        
        # Store stats for denormalization
        stats['min'] = data_min.item()
        stats['max'] = data_max.item()

        # Handle the edge case where all values are the same
        if data_max - data_min == 0:
            normalized_data = torch.zeros_like(tensor_data)
        else:
            normalized_data = 2 * (tensor_data - data_min) / (data_max - data_min) - 1

    elif method == 'zscore':
        mean = tensor_data.mean()
        std = tensor_data.std()
        
        # Store stats for denormalization
        stats['mean'] = mean.item()
        stats['std'] = std.item()

        # Handle the edge case where standard deviation is zero
        if std == 0:
            normalized_data = torch.zeros_like(tensor_data)
        else:
            normalized_data = (tensor_data - mean) / std
            
    else:
        raise ValueError(f"Unknown normalization method: {method}. Supported methods are 'minmax' and 'zscore'.")

    # Convert back to numpy if the original input was a numpy array
    if is_numpy:
        return normalized_data.numpy(), stats
    else:
        return normalized_data, stats

def denormalize(normalized_data, stats, method='minmax'):
    """
    Denormalizes the input data using statistics from a previous normalization.

    Args:
        normalized_data (np.ndarray or torch.Tensor): The data to be denormalized.
        stats (dict): A dictionary containing the statistics used for the original
                      normalization (e.g., {'min': 0, 'max': 1} or 
                      {'mean': 0.5, 'std': 0.1}).
        method (str): The denormalization method to use. Must match the method
                      used for normalization. Supported methods are 'minmax' and
                      'zscore'. Defaults to 'minmax'.

    Returns:
        (np.ndarray or torch.Tensor): The denormalized data, same type as input.
    """
    # Ensure data is a torch.Tensor for computation
    is_numpy = isinstance(normalized_data, np.ndarray)
    if is_numpy:
        tensor_data = torch.from_numpy(normalized_data.copy()).float()
    else:
        tensor_data = normalized_data.clone().float()

    if method == 'minmax':
        data_min = stats['min']
        data_max = stats['max']
        denormalized_data = (tensor_data + 1) * (data_max - data_min) / 2 + data_min

    elif method == 'zscore':
        mean = stats['mean']
        std = stats['std']
        denormalized_data = tensor_data * std + mean
        
    else:
        raise ValueError(f"Unknown denormalization method: {method}. Supported methods are 'minmax' and 'zscore'.")

    # Convert back to numpy if the original input was a numpy array
    if is_numpy:
        return denormalized_data.numpy()
    else:
        return denormalized_data







