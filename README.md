# Deep Learing 2020 Course Project: Enhancing Interpretability of Self-Explaining Neural Networks

## Introduction
This repository contains the code for reproducing the results of our project in the 2020 Deep Learning class at ETH Zurich. Our project is based on the paper ["Towards Robust Interpretability with Self-Explaining Neural Networks"](https://arxiv.org/abs/1806.07538) [[1]](#1). Please find our project report [here](project_report.pdf).


## Installing dependencies
```bash
pip install -r requirements.txt
```

## Code structure
| Folder | Filename | Description |
|--------|----------|-------------|
| [SENN](SENN)   | [aggregators.py](SENN/aggregators.py)| definitions of aggregation functions|
|[SENN](SENN)| [conceptizers.py](SENN/conceptizers.py)| definitions of functions that encode inputs into concepts|
|[SENN](SENN)| [parametrizers.py](SENN/parametrizers.py)| definitions of functions that generate relevances from inputs|
|[SENN](SENN)| [encoders.py](SENN/encoders.py) | definitions of encoders for conceptizers|
|[SENN](SENN)| [decoders.py](SENN/decoders.py) | definitions of decoders for conceptizers|
|[SENN](SENN)| [losses.py](SENN/losses.py) | definitions of loss functions e.g., robustness (stability) loss|
|[SENN](SENN)| [models.py](SENN/models.py) | definition of model classes|
|[SENN](SENN)| [backbones.py](SENN/backbones.py) | definition of backbone models for parametrizers i.e., VGG net|
|[SENN](SENN)| [utils.py](SENN/utils.py) | helper functions e.g., dataset definitions, custom layers|
|[SENN](SENN)| [eval_utils.py](SENN/eval_utils.py) | helper functions for evaluation|
|[SENN](SENN)| [trainers.py](SENN/trainers.py) | model training utilities|
|[SENN](SENN)| [simsiam.py](SENN/simsiam.py) | definitions of siamese networks|
|[SENN](SENN)| [argparser.py](SENN/argparser.py) | argparser for different models|
|[SENN](SENN)| [invarsenn.py](SENN/invarsenn.py) | definition of InvarSENN modules|
|[SENN](SENN)| [disentanglers.py](SENN/disentanglers.py) | definition of disentangler for InvarSENN|
|[Scripts](Scripts)| [senn.py](Scripts/senn.py) | script for reproduction of results for SENN|
|[Scripts](Scripts)| [vaesenn.py](Scripts/vaesenn.py) | script for reproduction of results for VaeSENN |
|[Scripts](Scripts)| [vsiamsenn.py](Scripts/vsiamsenn.py) | script for reproduction of results for V-SiamSENN |
|[Scripts](Scripts)| [invarsenn.py](Scripts/invarsenn.py) | script for training InvarSENN |
## Reproduction of results
### Reproduction of results for SENN
```bash 
## MNIST dataset 
python Scripts/senn.py --dataset "MNIST"
## CIFAR10 dataset
python Scripts/senn.py --dataset "CIFAR10" --n_epochs 200
```
### Reproduction of results for VaeSENN
```bash 
## MNIST dataset
python Scripts/vaesenn.py --dataset "MNIST"
## CIFAR10 dataset
python Scripts/vaesenn.py --dataset "CIFAR10" --n_epochs 200
```
### Reproduction of results for VSiamSENN
```bash 
## MNIST dataset
python Scripts/vaesenn.py --dataset "MNIST"
## CIFAR10 dataset
python Scripts/vaesenn.py --dataset "CIFAR10" --n_epochs 200
```
In order to load a pretraianed model please specify path via `--path_pretrained [PATH]` flag.
## Additional model: InvarSENN 
```bash 
## MNIST dataset
python Scripts/invarsenn.py --dataset "MNIST"
## CIFAR10 dataset
python Scripts/invarsenn.py --dataset "CIFAR10" --n_epochs 200
```
## Authors
- Edward Guenther (gedward@student.ethz.ch)
- Carina Schnuck (cschnuck@student.ethz.ch)
- Massimo Hoehn (hoehnm@student.ethz.ch)

## References
<a id="1">[1]</a> [David Alvarez-Melis and Tommi S. Jaakkola. "Towards Robust Interpretability with Self-Explaining Neural Networks". 2018.](https://arxiv.org/abs/1806.07538)
