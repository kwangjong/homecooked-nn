# HomeCooked-NN 
[![blog](https://img.shields.io/badge/%F0%9F%93%9D%20Blog-green?&style=flat)](https://hf.space/streamlit/kwangjong/homecooked-nn/+)
[![kaggle](https://img.shields.io/badge/Kaggle-20BEFF.svg?&style=flat&logoColor=white)](https://kwangjong.github.io/2022/07/09/homecooked-nn/)
[![demo](https://img.shields.io/badge/%F0%9F%A4%97%20demo-blue?&style=flat)](https://huggingface.co/spaces/kwangjong/homecooked-nn)

My attempt on building a neural network from scratch.

## Demo
Integrated into [Huggingface Spaces](https://huggingface.co/spaces/kwangjong/homecooked-nn) 🤗 using Streamlit.

## Installing module

Download or clone neuralnet.py and import it into your python code.
```python
import neuralnet as nn
```
## API Reference

### Constructing model
```python
nn.NeuralNet(
    input_size:int, 
    hidden_size:tuple, 
    activation:tuple, 
    loss:tuple, 
    metric:list=[],
    optimizer:str='sgd', 
    random_state:int=None
)
```
Construct `NeuralNet` Class.
#### Args:
* `input_size:int`: Number of input.

* `hidden_size:tuple`: Number of units for each hidden layer including the output layer.

* `activation:tuple`: List of activation functions for each layer. 
  * `hidden_size` and `activation` should contain the same number of elements. 
  * see [here](https://github.com/Kwangjong/homecooked-nn/edit/main/README.md#activation) for a list of activation functions available.

* `loss:tuple`: Loss function. 
  * see [here] for a list of loss functions available.

* `metric:list` List of metric functions. 
  * Metrics will be measured after each epoch and results will be returned after fitting is done. 
  * see [here](https://github.com/Kwangjong/homecooked-nn/edit/main/README.md#metric) for a list of metric functions available.

* `optimizer:str`: Name of a optimizer in string. 
  * If unspecified, `optimizer` will default to `'sgd'`. 
  * see [here](https://github.com/Kwangjong/homecooked-nn/edit/main/README.md#optimizer) for a list of optimizers available.

* `random_state:int`: Integer seed used by the random number generator.
  * If unspecified, `random_state` will default to `None`.

#### Returns:
* A new instance of `NeuralNet` class

### Printing model summary
```python
NeuralNet.summary()
```
Prints model summary.
#### Args:
* No argument.
#### Returns:
* No return.

### Fitting
```python
NeuralNet.fit(
    X:numpy.ndarray, y:numpy.ndarray, 
    batch_size:int=32, 
    epochs:int=50, 
    learning_rate:float=None, 
    valid_data:tuple=None, 
    history:dict=None, 
    verbose:int=1
)
```
Trains the model for a fixed number of epochs.
#### Args:
* `X:numpy.ndarray`: Input data.

* `y:numpy.ndarray`: Target data.

* `batch_size:int`: Number of samples per batch of computation. If unspecified, `batch_size` will default to 32. 

* `epochs:int`: Number of epochs to train the model. If unspecified, `epochs` will default to 50. 

* `learning_rate:float`: Learning rate. If unspecified, `learning_rate` will default to 0.01 for `'sgd'` and 0.001 for `'adam'`. 

* `valid_data:tuple`: Validation data as a tuple `(x_val, y_val)` of numpy array.
  * Validation data is used to evaluate the loss and any model metrics at the end of each epoch. 
  * The model will not be trained on this data.
  * If unspecified, only training data will be used for evaluatation


* `history:dict`: Empty dictionary for evaluation history.
  * Loss and any model metrics are stored in `history` during fitting.
  * `history['train_loss']`, `history['valid_loss']`: Prediction losses at the end of each epoch.
  * `history['train_<metric_name>']`, `history['valid_<metric_name>']`: Metric calculated at the end of each epoch.
  * If unspecified, evaluation history will not be stored. 

* `verbose:int`: Verbosity mode.
  * If 0, do not print any progress.
  * If 1, prints and updates single line progress bar and losses.
  * If 2, prints a single line with losses and any model metrics per epoch.
  * If unspecified, `verbose` will default to 1.
  
#### Returns:
* No return.
  
### Predict
```python
NeuralNet.predict(X:numpy.ndarray):
```
Generates output predictions for the input samples.
#### Args:
* `X:numpy.ndarray`: input samples.

#### Returns:
* Numpy array of predictions.

### Miscellaneous
```python
NeuralNet.recompile(metric:list=None, optimizer:str=None):
```
Recompile model with new metrics and optimizer. Weights will be reset.
#### Args:
* `metric:list`: List of metric functions. 
  * If unspecified, `metric` will remain unchanged.
* `optimizer:str`: Name of a optimizer in string. 
  * If unspecified, `optimizer` will remain unchanged.

#### Returns:
* no return.

```python
NeuralNet.reset_weights(self, random_state:int=None)
```
Reset weights.
#### Args:
* `random_state:int`: Integer seed used by the random number generator.
  * If unspecified, `random_state` will default to `None`.

#### Returns:
  * no return.

## List of model parameters available
#### Activation
* `nn.relu`: ReLU
* `nn.linear`: Linear
* `nn.sigmoid`: Sigmoid

#### Loss
* `nn.mse`: Mean Squared Error
* `nn.bce`: Binary Cross Entropy

#### Metric
* `nn.bacc`: Binary Accuracy

#### Optimizer
* `'sgd'`: Stochastic Gradient Descent
* `'adam`: Adam Optimizer



## Example
### Import module
```python
import neuralnet as nn
```
### Construct model
```python
model = nn.NeuralNet(
    input_size=1, 
    hidden_size=(4,4,1),
    activation=[nn.relu, nn.relu, nn.linear],
    loss=nn.mse,
    optimizer='sgd',
    random_state=0
)
```
### Print model summary
```python
model.summary()
```
`Output:`
```
loss='mean_square_error', optimizer='sgd'
--------------------------------------------------
     input:     1 unit(s)
  hidden_0:     4 unit(s), activation='relu'
  hidden_1:     4 unit(s), activation='relu'
    output:     1 unit(s), activation='linear'
--------------------------------------------------
                   total_unit: 3  total_param: 33 
```
### Fitting
```
history = {}
model.fit(
    train_X, train_y, 
    batch_size=32,
    epochs=800,
    learning_rate=0.02,
    valid_data=(valid_X, valid_y),
    history=history, 
    verbose=1
)
```
### Show history
```python
import matplotlib.pyplot as plt

plt.plot(history['train_loss'][100:], label='train loss')
plt.plot(history['valid_loss'][100:], label='validation loss')
plt.show()
```
