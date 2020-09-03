# Data Analysis
Here lies the collection of Excel spreadsheets and python notebooks we used to analyze all of our experiment data.  

## Single GPU
* **replicated_results** - Comparison of Actual vs Predicted runtimes for fullmodel (using same method as researchers), model for MNIST dataset
* **benchmark_actual_ratio** - Comparison of the time produced from researcher's method against the actual run times, models for MNIST and CIFAR10 datasets
* **MS_run_Single_run_pred.xlsx** - Comparison of Mirrored Strategy runtime of full model against Single GPU runtime and Single GPU predicted time, models for MNIST and CIFAR10 datasets

## MS: Full Model Predictions using only the model from MirroredStrategy benchmark data
* **method1_fullmodel.xlsx** - Comparison of Actual vs Predicted training times for MNIST data CNN model with MirroredStrategy

## MS: Full Model Predictions using a combined model (Method 2)
* **method2_fullmodel.xlsx** - Comparison of Actual vs Predicted training times for MirroredStrategy, model for patch camyleon dataset
* **Results_Final_Prediction.xlsx** - Comparison of Actual vs Predicted training times for MirroredStrategy, with also data on how the exact same experiment can take different amount of times over different leases of the P100 GPU, model for patch camyleon dataset

## Modelling the Initialization time
* **model_init_time** - Contains a ipynb and the data used to model initialization time based on difference between first epoch runtime and the rest