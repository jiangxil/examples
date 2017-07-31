
# Train a Keras model on CIFAR-10 data

Supports
 - training on CPU or a single GPU
 - hyperparemeter search

To download data:
```
wget http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
tar xvfz cifar-10-python.tar.gz --strip=1
```

To run:
```
python cifar10.py <DATA_DIR> --lr 0.0001 --lr-decay 0.000001 --epochs 2 --tensorboard-dir 3
```