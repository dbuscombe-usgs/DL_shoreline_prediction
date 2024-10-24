# Are equilibrium shoreline models just convolutions? A proof-of-concept tinyML model

This repo contains a proof of concept univariate CNN-based model for shoreline prediction, to support the manuscript by Vitousek et al "Are equilibrium shoreline models just convolutions?"

It is based on modified codes originally from [Gomez-de la Pena, 2023](https://github.com/eduardogomezdelapena/DL_shoreline_prediction), which used a larger multivariate CNN model, as detailed in [Gomez-de la Pena et al. 2023](https://doi.org/10.5194/esurf-11-1145-2023). 

This model takes in only Hs, and uses only one convolution layer for feature extraction, with the smallest possible architecture, meaning minimum dense neurons in the regression head. This results in models only c. 50 trainable parameters (actually, between 47 and 61, depending on kernel size)


Conda recipe:

```
conda create --name cnnshorelines python=3.10 -y
conda activate cnnshorelines
python3 -m pip install nvidia-cudnn-cu11 tensorflow[and-cuda]
```

check gpus:
```
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```


```
conda install scikit-learn numpy scipy matplotlib pandas
conda install ipython
```

How to use:

cd to `1run_models` directory, and execute the script `run_cnn_models_Hsonly_OBdata.py`. It cycles through a list of hyperparameters, creating plots and metrics for each model instance. The best model is used to create Figure 6 in the Vitousek et al paper. Models are designed to be trained with a GPU for faster execution, but these codes will also run on a CPU-only version of Tensorflow/keras.