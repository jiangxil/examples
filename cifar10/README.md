
# Train a Keras model on CIFAR-10 data

Supports
 - training on CPU or a single GPU
 - hyperparemeter search

The default is to download the training data from inside the container.
To change this switch download_data with get_data in the code.

To download data:
```
wget http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
tar xvfz cifar-10-python.tar.gz --strip=1
```

To run:
```
python cifar10.py <DATA_DIR> --lr 0.0001 --lr-decay 0.000001 --epochs 2 --tensorboard-dir 3
```
