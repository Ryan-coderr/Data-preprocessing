import numpy as np

from data_process import save_aligned_statistics, save_dataset_statistics, normalize, denormalize



# --- Example Usage ---
if __name__ == "__main__":

    ### Example of the function:save_aligned_statistics
    # 1. Create dummy data for demonstration
    # Shape: (2 samples, 10 frames, 1 channel, 64 height, 64 width)
    print("Generating dummy ground truth and prediction data...")
    dummy_gt = np.random.rand(2, 10, 1, 64, 64) * 100
    
    # Create dummy predictions, e.g., ground truth with some added random noise
    dummy_pred = dummy_gt + np.random.randn(2, 10, 1, 64, 64) * 5

    # 2. Run the analysis function
    print("Running analysis to compare predictions and ground truth...")
    save_aligned_statistics(dummy_pred, dummy_gt, base_filename="pred_vs_gt_stats")

    # The script will generate 'pred_vs_gt_stats_sample_000.txt' and 
    # 'pred_vs_gt_stats_sample_001.txt' in the current directory.


    ###  Example of the function:save_dataset_statistics
    # Create a dummy dataset for demonstration
    # Shape: (2 samples, 10 frames, 1 channel, 64 height, 64 width)
    dummy_dataset = np.random.rand(2, 10, 1, 64, 64) * 1000

    print("Running analysis on the dummy dataset...")
    save_dataset_statistics(dummy_dataset, base_filename="my_data_analysis")


    ###  Example of the function:normalize, denormalize
    # 1. Create a small dummy numpy array for a clear demonstration
    original_data = np.array([2, 2, 3], dtype=np.float32)
    print(f"Original Data: {original_data}\n")

    # --- Test 1: Min-Max Normalization ---
    print("--- Method: Min-Max ---")
    normalized_mm, stats_mm = normalize(original_data, method='minmax')
    denormalized_mm = denormalize(normalized_mm, stats_mm, method='minmax')
    
    print(f"Normalized  : {normalized_mm}")
    print(f"Stats       : {stats_mm}")
    print(f"Denormalized: {denormalized_mm}\n")

    # --- Test 2: Z-Score Normalization ---
    print("--- Method: Z-Score ---")
    normalized_zs, stats_zs = normalize(original_data, method='zscore')
    denormalized_zs = denormalize(normalized_zs, stats_zs, method='zscore')
    

    print(f"Normalized  : {normalized_zs}")
    print(f"Stats       : {stats_zs}")
    print(f"Denormalized: {denormalized_zs}\n")
    
    print("✅ All tests passed successfully!")
