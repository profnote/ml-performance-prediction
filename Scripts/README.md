# Scripts
A collection of python scripts we used to gather runtime data of various models

## Single GPU
* **mnist_single.py** - Script to get actual epoch runtimes, model for MNIST dataset
* **cifar_single.py** - Script to get actual epoch runtimes, model for CIFAR10 dataset
* **benchmark_custom.py** - Original benchmark script but with customized parameters, a copy can also be found in *prediction_model_tf2/Generate_train_data
* **patch_camelyon_custom_no_layers.py** - Script to generate model for patch camelyon data with a customized number of layers; layer 1 to layer n, or layer n to last layer

## TF MirroredStrategy
* **mnist_ms.py** - Script to get actual epoch runtimes using TF MS, model for MNIST dataset
* **patch_camelyon_ms_v2.py** - Script to get actual epoch runtimes using TF MS, model for patch camelyon dataset
