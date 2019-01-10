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


## Simple test code
Output gaze prediction using one image only!

1. Download pretrained models: [spatial](https://drive.google.com/open?id=1lK8rg2987B2njtfSuuUlAlb7TqQh3Mzd), [late](https://drive.google.com/open?id=1H4TEv0Xhr3o0X0P-E4F-gY1YQaxR9bV2) and put them into ``path/to/models``.

2. Prepare some images named with ``**_img.jpg`` in ``path/to/imgs/``.

3. Run ``run_spatialstream.py --trained_model /path/to/models/spatial.pth.tar --trained_late /path/tp/models/late.pth.tar --dir /path/to/imgs/`` and see the results.

This module assumes fixation at predicted gaze position without any attention transition. Note the model is trained on [GTEA Gaze+](http://ai.stanford.edu/~alireza/GTEA_Gaze_Website/GTEA_Gaze+.html) dataset, I haven't tested images from other datasets, so images from the same dataset is recommended to use.

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
The whole modle is not trained end to end. We extract data for each module and train them separatedly. We reccomend to first train the spatial and temporal stream separatedly, and then train the full SP module using pretrained spatial and temproal models. Direct training of SP result in slightly worse final results but better SP results.

Details of args can be seen in [gaze_full.py](gaze_full.py) or by typing ``python gaze_full.py -h``.

### Pre-trained model
You can find pre-trained SP module [here](https://drive.google.com/open?id=14-HTsPIN0s7NHdypY_EnxkJszl8x823o)

The module is trained using leave-one-subject-out strategy, this model is trained with 'Alireza' left out.

## Publication:
Y. Huang, <u>M. Cai</u>, Z. Li and Y. Sato, &quot;Predicting Gaze in Egocentric Video by Learning Task-dependent Attention Transition,&quot; <i>European Conference on Computer Vision (**ECCV**)</i>, to appear, 2018. (<font color="blue">oral presentation, acceptance rate: 2%</font>)  
[[Arxiv preprint]](https://arxiv.org/pdf/1803.09125)

[[CVF Open Access]](http://openaccess.thecvf.com/content_ECCV_2018/papers/Huang_Predicting_Gaze_in_ECCV_2018_paper.pdf)

## Citation
Please cite the following paper if you feel this repository useful.
```
@article{huang2018predicting,
  title={Predicting Gaze in Egocentric Video by Learning Task-dependent Attention Transition},
  author={Huang, Yifei and Cai, Minjie and Li, Zhenqiang and Sato, Yoichi},
  journal={arXiv preprint arXiv:1803.09125},
  year={2018}
}
```

## Contact
For any question, please contact
```
Yifei Huang: hyf(.at.)iis.u-tokyo.ac.jp
```
