import threading
import time
import queue

# Simulate Preprocess (IO-heavy task)
def Preprocess(batch):
    print(f"{time.time()-script_start_time:.2f} seconds: Preprocessing batch {batch}...")
    time.sleep(6)  # Simulating I/O delay
    output = f"output_1 for batch {batch}"
    print(f"{time.time()-script_start_time:.2f} seconds: Finished preprocessing batch {batch}")
    return output

# Simulate Calculation (CPU-heavy task)
def Calculation(output_1):
    print(f"{time.time()-script_start_time:.2f} seconds: Calculating on {output_1}...")
    time.sleep(10)  # Simulate CPU-heavy calculation
    output_2 = f"Result of {output_1}"
    print(f"{time.time()-script_start_time:.2f} seconds: Finished calculation on {output_1}")
    return output_2

# Simulate Postprocess (IO-heavy task)
def Postprocess(output_2):
    print(f"{time.time()-script_start_time:.2f} seconds: Postprocessing {output_2}...")
    time.sleep(6)  # Simulate I/O-heavy postprocessing
    output_3 = f"Final result of {output_2}"
    print(f"{time.time()-script_start_time:.2f} seconds: Finished postprocessing {output_2}")
    return output_3

def preprocessing_worker(preprocess_queue, calculation_queue):
    while True:
        batch = preprocess_queue.get()
        if batch is None:  # Sentinel to stop the worker
            break
        output_1 = Preprocess(batch)
        calculation_queue.put(output_1)
        preprocess_queue.task_done()

def calculation_worker(calculation_queue, postprocessing_queue):
    while True:
        output_1 = calculation_queue.get()
        if output_1 is None:  # Sentinel to stop the worker
            break
        output_2 = Calculation(output_1)
        postprocessing_queue.put(output_2)
        calculation_queue.task_done()

def postprocessing_worker(postprocessing_queue):
    while True:
        output_2 = postprocessing_queue.get()
        if output_2 is None:  # Sentinel to stop the worker
            break
        Postprocess(output_2)
        postprocessing_queue.task_done()

def main(batches):
    # Create queues for each stage
    preprocess_queue = queue.Queue(maxsize=1)  # Preprocessing -> Calculation
    calculation_queue = queue.Queue(maxsize=1)  # Calculation -> Postprocessing
    postprocessing_queue = queue.Queue(maxsize=1)  # Postprocessing

    # Start worker threads for each process
    preprocess_thread = threading.Thread(target=preprocessing_worker, args=(preprocess_queue, calculation_queue))
    calculation_thread = threading.Thread(target=calculation_worker, args=(calculation_queue, postprocessing_queue))
    postprocess_thread = threading.Thread(target=postprocessing_worker, args=(postprocessing_queue,))

    preprocess_thread.start()
    calculation_thread.start()
    postprocess_thread.start()

    # Add batches to preprocess queue
    for batch in batches:
        preprocess_queue.put(batch)

    # Wait for all batches to finish preprocessing, then stop the workers
    preprocess_queue.join()
    calculation_queue.join()
    postprocessing_queue.join()

    # Send sentinel to stop worker threads
    preprocess_queue.put(None)
    calculation_queue.put(None)
    postprocessing_queue.put(None)

    preprocess_thread.join()
    calculation_thread.join()
    postprocess_thread.join()

if __name__ == "__main__":
    # Example batches
    batches = ["batch1", "batch2", "batch3", "batch4", "batch5"]

    # Record the start time of the script
    script_start_time = time.time()

    # Run the main process
    main(batches)

    # Record the end time of the script
    script_end_time = time.time()

    # Calculate and print the total time elapsed
    total_time_elapsed = script_end_time - script_start_time
    print(f"\nTotal time elapsed since the script started: {total_time_elapsed:.2f} seconds")

#%%
#Code for testing
print("hello")
#%%
import wandb
import random

# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="TEST",

    # track hyperparameters and run metadata
    config={
    "learning_rate": 0.02,
    "architecture": "CNN",
    "dataset": "CIFAR-100",
    "epochs": 10,
    }
)

# simulate training
epochs = 10
offset = random.random() / 5
for epoch in range(2, epochs):
    acc = 1 - 2 ** -epoch - random.random() / epoch - offset
    loss = 2 ** -epoch + random.random() / epoch + offset

    # log metrics to wandb
    wandb.log({"acc": acc, "loss": loss})

# [optional] finish the wandb run, necessary in notebooks
wandb.finish()

# %%
from concurrent.futures import ThreadPoolExecutor
import numpy as np

def load(data_path):
    # Simulate data loading
    print(f"Loading data from {data_path}")
    return np.random.rand(1000, 1000)  # Replace with actual loading logic

def run(data):
    # Simulate multithreaded computation
    print("Running computation on data")
    return np.sum(data)  # Replace with actual computation logic

data_paths = ["path1", "path2", "path3", "path4"]

# Use ProcessPoolExecutor for background data loading
with ThreadPoolExecutor(max_workers=1) as executor:
    next_data_future = None  # Placeholder for the future
    
    for data_idx, data_path in enumerate(data_paths):
        if data_idx == 0:  # First batch
            current_data = load(data_path)
        else:
            # Wait for the next batch to finish loading
            current_data = next_data_future.result()

        # Start loading the next batch in the background
        if data_idx != len(data_paths) - 1:  # Not the last batch
            next_data_future = executor.submit(load, data_paths[data_idx + 1])
        
        # Process the current batch
        results = run(current_data)
        print(f"Results for {data_path}: {results}")
    with ThreadPoolExecutor(max_workers=1) as executor:
        next_data_future = None  # Placeholder for the future
        
        for data_idx, data_path in enumerate(data_paths):
            if data_idx == 0:  # First batch
                current_data = load(data_path)
            else:
                # Wait for the next batch to finish loading
                current_data = next_data_future.result()

            # Start loading the next batch in the background
            if data_idx != len(data_paths) - 1:  # Not the last batch
                next_data_future = executor.submit(load, data_paths[data_idx + 1])
            
            # Process the current batch
            results = run(current_data)
            print(f"Results for {data_path}: {results}")
# %%