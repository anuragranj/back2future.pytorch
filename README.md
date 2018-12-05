This is a Pytorch implementation of

Janai, J., Güney, F., Ranjan, A., Black, M. and Geiger, A., **Unsupervised Learning of Multi-Frame Optical Flow with Occlusions.** ECCV 2018.

[[Link to Paper](http://www.cvlibs.net/publications/Janai2018ECCV.pdf)] [[Project Page](https://avg.is.tuebingen.mpg.de/research_projects/back2future)] [[Original Torch Code](https://github.com/jjanai/back2future)]

## Requirements
- Runs and tested on [Pytorch 0.3.1](https://pytorch.org/get-started/previous-versions/), it should be compatible with higher versions with little/no modifications.
- Correlation package is taken from [NVIDIA/flownet2-pytorch](https://github.com/NVIDIA/flownet2-pytorch/) and it can be installed using
```bash
cd correlation_package
bash make.sh
```
If you are using Pytorch>0.3.1, you can use correlation layer from [here](https://github.com/ClementPinard/Pytorch-Correlation-extension).
## Usage
To use the model, go to your favorite python environment
```python
from back2future import Model
model = Model(pretrained='pretrained/path_to_your_favorite_model')
```
There are two pretrained models in `pretrained/`, that are fine tuned on Sintel and KITTI in an unsupervised way.

Refer to `demo.py` for more.

## Testing
To test performance on KITTI, use
```bash
python3 test_back2future.py --pretrained-flow path/to/pretrained/model --kitti-dir path/to/kitti/2015/root
```

## Training
Please use the [[original torch code](https://github.com/jjanai/back2future)] for training new models.

## License
This is a reimplementation. License for the original work can be found at [JJanai/back2future](https://github.com/JJanai/back2future/blob/master/LICENSE).

## While using this code, please cite
```
@inproceedings{Janai2018ECCV,
  title = {Unsupervised Learning of Multi-Frame Optical Flow with Occlusions },
  author = {Janai, Joel and G{"u}ney, Fatma and Ranjan, Anurag and Black, Michael J. and Geiger, Andreas},
  booktitle = {European Conference on Computer Vision (ECCV)},
  volume = {Lecture Notes in Computer Science, vol 11220},
  pages = {713--731},
  publisher = {Springer, Cham},
  month = sep,
  year = {2018},
  month_numeric = {9}
}
```
