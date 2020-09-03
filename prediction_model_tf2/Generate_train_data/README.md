## Instructions for running benchmark script for Conv in TF2

1. Create a virtual env/docker container with tensorflow2. (for anaconda prompt, use: conda create -n tf-gpu tensorflow-gpu)  
Be sure to have numpy and pandas installed. If missing, just pip install.

2. Navigate to ml-performance-prediction/prediction_model_tf2/Generate_train_data

3. Run: python benchmark.py --testConv --device=GPUname --num_vals=5 --num_gpu=1  
The script will run and output the .pkl file into the 'results' folder

## Benchmark parameters

Run
```bash
python benchmark.py
```
with the following optional arguments:

#### Types of benchmarks
--testConv (Benchmark 2D convolution)<br/>
--testVGG16 (Benchmark training a VGG16 cnn on sythetic data)<br/>

#### General parameters
--backprop_ratio (Ratio of iterations with backward pass ([0..1])
--num_gpu (Number of GPUs to use, default 1)<br/>
--devlist (List of devices to use, overwrites num_gpu if set, default '')<br/>
--num_val (Number of results to compute)  
--logfile (Name of output file)  
--device (Name of device to appear on logfile)  
--iter_benchmark (Number of iterations for benchmark)  
--iter_warmup (Number of iterations for warm-up)  
--repetitions (Number of repetitions of the same experiment)  

For TF MirroredStrategy, run
```bash
python benchmark.py
```
with the same optional arguments, except for  ```--num_gpu``` which now becomes the number of GPUs to use with TF mirroredStrategy (default=2)