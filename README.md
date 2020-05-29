# DlDegibbs

[Paper](https://arxiv.org/abs/1905.04176) | [GitHub](https://github.com/mmuckley/dldegibbs)

A deep learning model trained on ImageNet for removing noise and Gibbs artifacts from diffusion images.

This repository contains code for reproducing experiments from the paper, "[Training a Neural Network for Gibbs and Noise Removal in Diffusion MRI](https://arxiv.org/abs/1905.04176)" by M. Muckley et al.

This package was developed entirely for research purposes at the NYU School of Medicine and is not affiliated with any other entity. It has not been validated for clinical use.

## Usage

The workhorse script is ```degibbs_main.py```. Configuration files are in the ```.yaml``` format and are stored in
the ```configs/``` folder. Options specified at command line overwrite options
in the ```.yaml``` files, e.g.,

```bash
python dldegibbs_main.py magnitude --exp_dir='my_exp_dir/'
```

should overwrite

```yaml
exp_dir: "~/data/logs/diffusion/res_256_to_130/"
```

in the ```.yaml``` files.

The package was tested using PyTorch 1.1. PyTorch and other packages were
installed using Anaconda.

## Pretrained models

Pretrained models are available on [GLOBUS](https://app.globus.org/file-manager?origin_id=15c7de28-a76b-11e9-821c-02b7a92d8e58&origin_path=%2F).
Models from the pretrained experiment are saved based on best MSE on the ImageNet validation data set.
To use a pretrained model, download one of the model folders from GLOBUS (either `res_256_to_100` if you're working on 100 x 100 images or `res_256_to_130` if you're working on 130 x 130 images) and change the `configs/data.yaml` script `exp_dir` variable to where you save it in your local file system.
When running `degibbs_main.py`, you should get a message that the script found the file and loaded the model.
