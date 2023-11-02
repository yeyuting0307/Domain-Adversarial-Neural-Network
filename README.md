Domain-Adversarial-Neural-Network
===

![](https://i.imgur.com/AxpyUVB.png)


## Paper Link
- [Domain-Adversarial Training of Neural Networks](https://arxiv.org/abs/1505.07818)


---

## Dataset
- [MNIST](https://pytorch.org/vision/stable/generated/torchvision.datasets.MNIST.html)
- [MNIST-M](https://github.com/mashaan14/MNIST-M)


---

## Environment
- macOS Ventura : `13.2.1`
- GPU : `Apple M2 Pro`
- python :  `3.9.16`
- conda : `23.1.0`

---

## Train
Train the model and save the weights in checkpoints
```
python train.py
```

## Test
Evaluate the saved model in checkpoints

```
python test.py
```

```
Epoch: 100
loss: 0.695, label_loss: 0.080, domain_loss: 0.616

Test-Set Accuracy:
[MNIST Label] 9929 / 10000 = 99.29%
[MNIST Domain] 9408 / 10000 = 94.08%
[MNIST-M Label] 8191 / 9001 = 91.00%
[MNIST-M Domain] 4081 / 9001 = 45.34%
```

---

## Reference

### Model Algorithm

![](https://i.imgur.com/DbkrzAc.png)

### Model Architecture

![](https://i.imgur.com/t9ym0Uk.png)

### Backward

![](https://i.imgur.com/Qy4DMvU.png)

### Adjusted learning rate and domain adaptation parameter

![](https://i.imgur.com/Sy9TFXF.png)

---
