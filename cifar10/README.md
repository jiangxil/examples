
# Train a Keras model on CIFAR-10 data

Supports
 - training on CPU or a *single* GPU
 - hyperparemeter search

To perform a training run:
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

To download trainig data, create a `cifar` folder on your RiseML `data` volume:

```
mkdir cifar
wget http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
tar xvfz cifar-10-python.tar.gz --strip=1
```
Adjust the run command in the `riseml.yml` and pass the directory `/data/cifar` as first argument.
For example:
```
python cifar10.py /data/cifar --lr 0.0001 --lr-decay 0.000001 --epochs 2 --tensorboard-dir 3
```
