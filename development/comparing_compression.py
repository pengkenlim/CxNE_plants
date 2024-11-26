# %%
import joblib
import os
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

def benchmark_compression(loader, loader_path, compress_levels=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), method = "lz4"):
    """
    Save the loader using different compression levels and measure loading speed.

    Args:
        loader: The object to be saved and loaded.
        loader_path: The base path to save files. The level will be appended to the name.
        compress_levels: A list or tuple of compression levels to test.

    Returns:
        A dictionary containing compression level, file size, and loading time.
    """
    results = []
    for level in compress_levels:
        # Create the file path for this compression level
        file_path = f"{loader_path}_level{level}.joblib"
        
        # Save the loader with the specified compression level
        joblib.dump(loader, file_path, compress=(method, level))
        
        # Measure file size
        file_size = os.path.getsize(file_path)
        
        # Measure loading speed
        start_time = time.time()
        _ = joblib.load(file_path)
        loading_time = time.time() - start_time

        # Append results
        results.append({
            'compression_level': level,
            'file_size': file_size,
            'loading_time': loading_time
        })
        
        # Optionally, clean up the file (uncomment to keep files)
        # os.remove(file_path)
        print(f"{level:<10}{file_size:<15}{loading_time:<15.5f}")
    # Print results
    print(f"{'Level':<10}{'Size (bytes)':<15}{'Load Time (s)':<15}{method}")
    for result in results:
        print(f"{result['compression_level']:<10}{result['file_size']:<15}{result['loading_time']:<15.5f}{method}")
    return results
# %%
coexp_adj_mat = joblib.load("/mnt/md2/ken/CxNE_plants_data/intermediate_data/taxid3702_old/coexp_adj_mat.joblib")
# %%
coexp_adj_mat_path  = "/mnt/md0/ken/test_compression_speed/coexp_adj_mat"
results = benchmark_compression(coexp_adj_mat, coexp_adj_mat_path, method = "zlib")
# %%

def load_input_data(path):
    data = joblib.load(path)
    return data

path = "/mnt/md0/ken/test_compression_speed/coexp_adj_mat_level0.joblib"

with ProcessPoolExecutor(max_workers= 2) as preprocess_executor:
    start_time = time.time()
    next_data_future_preprocess = preprocess_executor.submit(load_input_data, path)
    coexp_adj_mat = next_data_future_preprocess.result()
    loading_time = time.time() - start_time
    print(loading_time)


# %%
