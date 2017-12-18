
# CIFAR-10 Example

This example uses a simple [Keras](https://keras.io/) model (with Tensorflow backend, based on [this example](https://github.com/keras-team/keras/blob/master/examples/cifar10_cnn.py)) to train an image classification model on [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html).

It supports:
 - training on CPU or a *single* GPU
 - hyperparemeter search

To perform a training run, execute (in this directory):
```
riseml train
```
Note: The default command downloads the training data from inside the container on each run.

To perform hyperparameter optimization:
```
riseml train -f riseml_hyper.yml
```

Feel free to adjust the `riseml.yml` to your needs, e.g., change resource requirements.


## Download Training Data

To download training data, create a `cifar` folder on your RiseML `data` volume:

```
mkdir cifar
wget http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
tar xvfz cifar-10-python.tar.gz --strip=1
```
Adjust the run command in the `riseml.yml` and pass the directory `/data/cifar` as first argument.
For example:
```
python cifar10.py /data/cifar --epochs 2
```
