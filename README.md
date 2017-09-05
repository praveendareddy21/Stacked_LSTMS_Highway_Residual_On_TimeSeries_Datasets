# Stacked LSTM Architecture Study on Time Series Datasets
Long short-term memory(LSTM) networks are powerful machine learning models which are cur
rently used for wide range of applications like Speech recognition, Music Composition, Human
action Recognition, Time series Prediction etc. This is made possible due to the Universal nature of
LSTM network where given enough neural units, it can properly model and compute any conven
tional computing problem, provided it has proper weight matrix.

In our study, We have stacked Layers of LSTM networks for better performance. And by taking inspirance from ConvNets for Image Classification, We mplemented Residual Connections and Highway Connections between LSTM layers. Then, We compared our models on various Datasets.


# Experiment Setup
We have used tensorflow as Deep Learning Framework and used its GPU processing facility on NVIDIA GeForce
GTX 960M Graphic card for faster parallel batch processing.

# Python Libraries Required
numpy, tensorflow, matplotlib, sklearn


# Training models and Data Handlers
main_runner.py is used for setting all hyper-parameters and running in a loop for figuring out best parameters.

classes with data_handler suffix are responsible for data input and can be configured for different datasets without altering model definition.

Also, Each model can also run by itself to test each package (BY default, HAR dataset is used)
