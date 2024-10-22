# Implementation of muP for vanilla MLP

A simple implementation of muP parametrization and adaptive learning rates based on: 
```
@article{yang2023spectral,
  title={A spectral condition for feature learning},
  author={Yang, Greg and Simon, James B and Bernstein, Jeremy},
  journal={arXiv preprint arXiv:2310.17813},
  year={2023}
}
```

Both standard training as well as mup training are implemented in `train.py`. I mostly used `wandb` for logging, but most of the logging also works for `tensorboard`. The visualization notebooks however are for my `wandb` setting. The first part for plotting the hyperparameter transfer curves only relies in the provided csv files.

If you have any questions please don't hesitate to open an issue.

## WANDB training runs

https://wandb.ai/chrisxx/cifar10-adam?nw=nwuserchrisxx
https://wandb.ai/chrisxx/cifar10-adam-just-following-the-table
https://wandb.ai/chrisxx/cifar10-sgd?nw=nwuserchrisxx

