# Even Number Generator using GAN
An self-introducing project on Generative Adversarial Network (GAN) with simple even numbers generators

## Environments and Dependencies
Python 3.10.6

Install requirements
```bash
pip install -r requirements.txt
```

## Structure
- `helpers.py` functions to generate data 
- `dataloader.py` builds DataLoader model
- `model.py` defines architecture for GNet and DNet
- `train.py` training loop of the models
- `infer.py` infers with the trained model

## Inference
Simply inference by running
```bash
python infer.py
```
