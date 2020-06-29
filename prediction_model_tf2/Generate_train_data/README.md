## Instructions for running benchmark script for Conv in TF2

1. Create a virtual env/docker container with tensorflow2. (for anaconda prompt, use: conda create -n tf-gpu tensorflow-gpu)
Be sure to have numpy and pandas installed. If missing, just pip install.

2. Navigate to ml-performance-prediction/prediction_model_tf2/Generate_train_data

3. Run: python benchmark.py --testConv --device=GPUname --num_vals=5 --num_gpu=1
The script will run and output the .pkl file into the 'results' folder

4. Use viewPkl.ipynb to quickly view a pkl file.
