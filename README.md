# 3d-pose-baseline-pytorch

PyTorch implementation of [*A simple yet effective baseline for 3d human pose estimation*](https://arxiv.org/abs/1705.03098).

Todo:
- [ ] Performance evaluation
- [ ] Provide trained models
- [ ] Provide tutorials to predict 3D pose from 2D pose input
- [ ] Train models on Stacked Hourglass output

## Performance

*Coming soon...*

## Preparation

### Human3.6M dataset

Get `h36m.zip` by following [author's repo](https://github.com/una-dinosauria/3d-pose-baseline), place it under `dataset`, and unzip it.

```
$ cd dataset
$ unzip h36m.zip
```

### Install dependencies

```
$ pip install -r requirements.txt
```

## Usage

### Training

```
$ ./tools/train.py OUTPUT_DIR ./output
```

You'll find trained weight and tensorboard event file under `./output` directory.

## Paper

**A simple yet effective baseline for 3d human pose estimation**

*Julieta Martinez, Rayat Hossain, Javier Romero, James J. Little*

[[Paper]](https://arxiv.org/abs/1705.03098)[[Author's implementation]](https://github.com/una-dinosauria/3d-pose-baseline)

```
@inproceedings{martinez_2017_3dbaseline,
  title={A simple yet effective baseline for 3d human pose estimation},
  author={Martinez, Julieta and Hossain, Rayat and Romero, Javier and Little, James J.},
  booktitle={ICCV},
  year={2017}
}
```