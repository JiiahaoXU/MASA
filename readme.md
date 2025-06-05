## Identify Backdoored Model in Federated Learning via Individual Unlearning

This is the official implementation for WACV'25 paper "Identify Backdoored Model in Federated Learning via Individual Unlearning", you can find the paper [here][paper].

[paper]: https://arxiv.org/abs/2411.01040

## Usage

If you have any issues using this repo, feel free to contact Jiahao @ jiahaox@unr.edu.

The proposed MASA algorithm is placed in `src/aggregation.py`, and you can easily take it and integrate MASA with your code.

### Environment

Our code does not rely on special libraries or tools, so it can be easily integrated with most environment settings. 

If you want to use the same settings as us, we provide the conda environment we used in `env.yaml` for your convenience.

### Dataset

All tested datasets are available on `torchvision` and will be downloaded automatically.

### Example

In our experiments, unless otherwise specified, the default attack method is `Badnet` with a data poisoning ratio `r=0.5`. The FL system contains `20` clients, and `20%` of them are malicious. No `client sampling` will be employed.

Generally, to run a case with default settings, you can easily use the following command:

```
python federated.py --data cifar10 --num_agents 20 \
--attack badnet --poison_frac 0.5 --num_corrupt 4 \
--aggr masa
```

If you want to run a case with non-IID settings, you can easily use the following command:

```
python federated.py --data cifar100 --num_agents 20 \
--attack badnet --poison_frac 0.5 --num_corrupt 4 \
--aggr masa \
--non_iid --alpha 0.5
```

Here,

| Argument        | Type       | Description   | Choice | Default |
|-----------------|------------|---------------|--------|--------|
| `aggr`         | str   | Defense method applied by the server | avg, masa, rlr, mkrum, mul_metric, lockdown, fg, rfa| avg|
| `data`    |   str     | Main task data        | cifar10, cifar100 | cifar10|
| `num_agents`         | int | Number of clients in FL   | N/A | 20|
| `attack`         | str | Attack method   | badnet, DBA, neurotoxin, pgd, lie | badnet|
| `poison_frac`         | float | Data poisoning ratio   | [0.0, 1.0] | 0.5|
| `num_corrupt`         | int | Number of malicious clients in FL   | [0, num_agents//2-1] | 4|
| `non_iid`         | store_true | Enable non-IID settings or not      | N/A | False|
| `alpha`         | float | Data heterogeneous level     | [0.1, 1.0]| 0.5|

For other arguments, you can check the `federated.py` file where the detailed explanation is presented.

## Citation
If you find our repository is useful for your work, please cite our work:
```
@InProceedings{Xu_2025_WACV,
    author    = {Xu, Jiahao and Zhang, Zikai and Hu, Rui},
    title     = {Identify Backdoored Model in Federated Learning via Individual Unlearning},
    booktitle = {Proceedings of the Winter Conference on Applications of Computer Vision (WACV)},
    month     = {February},
    year      = {2025},
    pages     = {7949-7958}
}
```

## Acknowledgment
Our code is constructed on https://github.com/git-disl/Lockdown, big thanks to their contribution!