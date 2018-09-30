# Code for the paper "Predicting Gaze in Egocentric Video by Learning Task-dependent Attention Transition" (ECCV2018)

This is the github repository containing the code for the paper ["Predicting Gaze in Egocentric Video by Learning Task-dependent Attention Transition"](https://arxiv.org/pdf/1803.09125) by Yifei Huang, Minjie Cai, Zhenqiang Li and Yoichi Sato.

## Requirements
The code is tested to work correctly with:

- GPU environment
- Anaconda Python 3.6.4
- [Pytorch](https://pytorch.org/) v0.4.0
- NumPy
- OpenCV
- [tqdm](https://github.com/tqdm/tqdm)

## Model architecture
<img src="https://hyf015.github.io/static/img/ECCV2018_architecture.jpg">

## Code usage
For simplicity of tuning, we separate the training of each module (SP, AT and LF)
### Dataset preparation
We use [GTEA Gaze+](http://ai.stanford.edu/~alireza/GTEA_Gaze_Website/GTEA_Gaze+.html) and [GTEA Gaze](http://ai.stanford.edu/~alireza/GTEA_Gaze_Website/GTEA_Gaze.html) dataset.

For the optical flow images, use [dense flow](https://github.com/yjxiong/dense_flow) to extract all optical flow images, and put them into `path/to/opticalflow/images` (e.g. gtea_imgflow/). The flow images will be in different sub-folders like:
```
    .
    +---gtea_imgflow
    |
        +---Alireza_American
        |   +---flow_x_00001.jpg
            +---flow_x_00002.jpg
            .
            .
            +---flow_y_00001.jpg
            .
            .
        +---Ahmad_Burger
        |   +---flow_x_00001.jpg
        .
        .
        .
```

All images should be put into `path/to/images` (e.g. gtea_images/).

The ground truth gaze image is generated from the gaze data by pointing a 2d Gaussian at the gaze position. We recommend ground truth images to have same name with rgb images. Put the ground truth gaze maps into `path/to/gt/images` (e.g. gtea_gts/). For 1280x720 image we use gaussian variance of 70. Processing reference can be seen in [data/dataset_preprocessing.py](data/dataset_preprocessing.py)

We also use predicted fixation/saccade in our model. Examples for GTEA Gaze+ dataset are in folder [fixsac](fixsac/). You may use any method to predict fixation.

### Running the code
To run the complete experiment, after preparing the data, run
```
python gaze_full.py --train_sp --train_lstm --train_late --extract_lstm --extract_late --flowPath path/to/opticalflow/images --imagePath path/to/images --fixsacPath path/to/fixac/folder --gtPath path/to/gt/images
```
The whole modle is not trained end to end. We extract data for each module and train them separatedly.

Details of args can be seen in [gaze_full.py](gaze_full.py) or by typing ``python gaze_full.py -h``.

## Publication:
Y. Huang, <u>M. Cai</u>, Z. Li and Y. Sato, &quot;Predicting Gaze in Egocentric Video by Learning Task-dependent Attention Transition,&quot; <i>European Conference on Computer Vision (**ECCV**)</i>, to appear, 2018. (<font color="blue">oral presentation, acceptance rate: 2%</font>)  
[[Arxiv preprint]](https://arxiv.org/pdf/1803.09125)

[[CVF Open Access]](http://openaccess.thecvf.com/content_ECCV_2018/papers/Huang_Predicting_Gaze_in_ECCV_2018_paper.pdf)

## Citation
Please cite the following paper if you feel this repository useful.
```
@inproceedings{Huang2018Predicting,
  author    = {Yifei Huang and
               Minjie Cai and
               Zhenqiang Li and
               Yoichi Sato},
  title     = {Predicting Gaze in Egocentric Video by Learning Task-dependent Attention Transition},
  booktitle   = {ECCV},
  year      = {2018},
}
```

## Contact
For any question, please contact
```
Yifei Huang: hyf@iis.u-tokyo.ac.jp
```
