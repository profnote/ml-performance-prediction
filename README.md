# Practicum Project with CCCIS - Complexity estimation of Deep Learning models
Code related to the paper **Predicting the Computational Cost of Deep Learning Models**. This code allows to train a machine learning model that can predict the execution time for commonly used layers within deep neural networks - and, by combining these, for the full network.  

We forked this repo from the original researchers of the paper mentioned below, migrated the code from TF1 to TF2 and proposed a methodology for estimating training time of distributed training using TF mirroredStrategy. The concept is similar to the original method in the paper. 
We start by benchmarking training time for individual layers on a single GPU, then we also benchmark training time for the last layer using TF mirroredStrategy. 
A large proportion of the time is spent on calculating the loss over multiple devices after the last layer, so the output shape of the last layer has the most impact. 
The full model training time prediction can be computed by summing up the single-GPU predicted times of each layer except the last plus the predicted time for the last layer running on TF mirroredStrategy. 
Finally, we can adjust our predictions by the overestimation factor to get more accurate predictions.  

## Folder Breakdown

* **DataAnalysis** - Excel spreadsheets/jupyter notebooks used to analyze the data from our experiments
* **ProjectReport** - The final project report and presentation slides
* **Scripts** - Python scripts used to get the actual runtimes of various models, singleGPU and MirroredStrategy
* **prediction_model_tf2** - Code to generate training data for the model described in the above paper, a data preparation pipeline, 
and the model training procedures. This folder also contains the training data and the existing tensorflow models. Compatible with Tensorflow 2. 

#### Original files of the Researchers (Tensorflow 1)
* **prediction_model** - Code to generate training data for the model described in the above paper, a data preparation pipeline, and the model training procedures. This folder also contains the training data and the existing tensorflow models.
* **benchmark** - code for benchmarking deep neural networks as well as single layers within them
