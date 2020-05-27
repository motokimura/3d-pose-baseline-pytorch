# 3d-pose-baseline-pytorch

PyTorch implementation of [*A simple yet effective baseline for 3d human pose estimation [Martinez+, ICCV'17]*](https://arxiv.org/abs/1705.03098).

Todo:
- [ ] Provide trained models
- [ ] Provide tutorials to predict 3D pose from 2D pose input
- [ ] Train models on Stacked Hourglass output

## Performance

### Protocol #1 (no rigid alignment in post-processing)

MPJPE [mm]:
|        | Avg      |  Direct |   Discuss   |  Eating   | Greet  | Phone  | Photo | Pose  | Purch  | Sitting   | SittingD   | Smoke   |  Wait   | WaitD   | Walk   | WalkT   |
| :-------- | --------: | --------:| :------: |--------: | --------:| :------: |--------: | --------:| :------: |--------: | --------:| ------: |--------: | --------:| ------: |------: |
| Paper     | 45.5  | 37.7 | 44.4 | 40.3 | 42.1 | 48.2 | 54.9 | 44.4 | 42.1 | 54.6 | 58.0 | 45.1 | 46.4 | 47.6 | 36.4 | 40.4 |
| This repo | 43.3  | 35.7 | 41.6 | 40.1 | 40.4 | 45.0 | 52.0 | 42.9 | 38.0 | 53.2 | 55.4 | 43.5 | 43.3 | 43.3 | 35.6 | 33.7 |

Both were trained on truth 2D poses input and multiple actions.

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

### Train model

```
$ ./tools/train.py OUTPUT_DIR ./output
```

You'll find trained weight and tensorboard event file under `./output` directory.

### Evaluate model

```
$ ./tools/test.py OUTPUT_DIR ./output MODEL.WEIGHT ${PATH_TO_WEIGHT}
```

You'll find evaluation results in a JSON file under `./output` directory.

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