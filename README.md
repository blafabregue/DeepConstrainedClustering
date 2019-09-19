# Deep constrained clustering applied to satelliteimage time series -- Code

This is the code corresponding to the experiments conducted for the work
"Deep constrained clustering applied to satelliteimage time series" (Baptiste Lafabregue, Jonathan Weber, Pierre Gançarki & Germain Forestier)
Tis work was preseted at MACLEAN workshop at ECML/PKDD Conference 2019 (https://mdl4eo.irstea.fr/maclean-machine-learning-for-earth-observation/)

## Requirements

Experiments were done with Python 3.7 and the following packages:
 - Numpy 
 - Matplotlib 
 - Keras 
 - Pandas 
 - Scikit-learn 
 - Scipy

This code should execute correctly with last versions of these packages.

## Datasets

The dataset used for the papaer is not available but  it can be test on time-series datasets, such as as the UEA archive: http://www.timeseriesclassification.com/ univariate or multivariate. 
The script ts_to_a2cnes_format can be used to convert sk-time format files to our format.

## Usage

### Training on the UCR and UEA archives

To train a model on the Mallat dataset from the UCR archive you have to train first an autoencoder with mlp or fcnn architecture:

with fcnn: 
`python FCNN_AE.py Univariate Mallat --itr "1" --epochs=700 --batch_size=8`
with mlp:
`python MLP_SDAE.py Univariate Mallat --itr "1" --epochs=200 --epochs_final=400 --batch_size=8`

Then, you can train the constrained clustering as follow:
`python MLP_DEC.py fcnn Mallat 5 0.1 --archive_name Univariate --itr "0" --ae_weights "ae_weights/fcnn/Mallat1/Mallat-pretrain-model-700_z10.h5" --batch_size=8`
