# SG-PGM
Yaxu Xie, Alain Pagani, Didier Stricker

[Full paper](https://arxiv.org/abs/2403.19474) | [Video](https://youtu.be/_xqNhcBC-jI?si=zPrnwZY_Q97FTfB0) | Webpage (no, I am not doing it ^_^ )


<div style="width:80%; margin: auto;">

![](assets/teaser.png)
</div>

**Abstract**: *Scene graphs have been recently introduced into 3D spatial understanding as a comprehensive representation of the scene. The alignment between 3D scene graphs is the first step of many downstream tasks such as scene graph aided point cloud registration, mosaicking, overlap checking, and robot navigation. In the paper, we treat 3D scene graph alignment as a partial graph matching problem and propose to solve it with a graph neural network. 
We reuse the geometric features learned by the point cloud registration method and associate the clustered point-level geometric features with the node-level semantic feature via our designed feature fusion module. Partial matching is enabled by using a learnable method to select the top-K similar node pairs. Subsequent downstream tasks such as point cloud registration are achieved by running a pre-trained registration network within the matched regions.We further propose a point-matching rescoring method, that uses the node-wise alignment of the 3D scene graph to reweight the matching candidates from a pre-trained point cloud registration method. It reduces the false point correspondence estimated especially in low-overlapping cases. Experiments show that our method improves the alignment accuracy by 10~20\% in low-overlap and random transformation scenarios and outperforms the existing work in multiple downstream tasks.*

<section class="section" id="BibTeX">
  <div class="container is-max-desktop content">
    <h2 class="title">BibTeX</h2>
    <pre><code>@article{xie2024sg,
  title={SG-PGM: Partial Graph Matching Network with Semantic Geometric Fusion for 3D Scene Graph Alignment and Its Downstream Tasks},
  author={Xie, Yaxu and Pagani, Alain and Stricker, Didier},
  journal={arXiv preprint arXiv:2403.19474},
  year={2024}
}</code></pre>
  </div>
</section>


## Environment setup
Build conda environment
```
conda env create -n sg-pgm -f environment.yml
```
Install the grid sampling ext for points and semantic labels
```
python setup.py build develop
```

Pretrain model download:
Click the links below to download the our trained model.

`SG-PGM Models`:  [SG-PGM](https://drive.google.com/drive/folders/1E60DFV7uTdvwgIxf6_eTIbayQH7uYZzP?usp=drive_link)

`GeoTransformer`: [GeoTr pretrained on 3DMatch](https://github.com/qinzheng93/GeoTransformer/releases)

## Data Generation
Please follow [SGAligner](https://github.com/sayands/sgaligner) for generate data samples for alignment, registration, overlap checking and alignment with semantic noises etc.. To obtain the exact same data samples, please set ``np.random.seed(42)`` in their data generation scripts.

After that, to generate data samples for evaluating on 3D scene graph alignment with dynamics \[ 'sub2rescan', 'sub2scan', 'sub2sub' \], please run:
```
python prepare_data/gen_dynamic_scenes.py --split='val' --dynamics=sub2rescan
```

## Evaluation
Inference for 3D Scene Graph Alignment and Downstream Tasks.
### 3D Scene Graph Alignment and Point Cloud Registration
```
python scripts/inference_reg.py --trained_model=./weights/the_weights.pth --rand_trans
```

### Scene Fragments Overlap Checking
```
python scripts/inference_overlap.py --trained_model=./weights/the_weights.pth --at3
```

### 3D Scene Graph Alignment on Scene with Dynamics
```
python scripts/inference_dynamics.py --trained_model=./weights/the_weights.pth
```

## Training
To train SG-PGM on 3RScan subscans, you can use:
```
python script/train.py --rand_trans
```
We provide config file in `data/` directory. Please change the parameters in the configuration files, if you want to tune the hyper-parameters.

## Acknowledgements
- [SGAligner](https://github.com/sayands/sgaligner)
- [GeoTransformer](https://github.com/qinzheng93/GeoTransformer)
- [RGM](https://github.com/fukexue/RGM)
- [AFAT](https://github.com/Thinklab-SJTU/ThinkMatch)
- [SceneGraphFusion](https://github.com/ShunChengWu/3DSSG)
- [3RScan](https://github.com/WaldJohannaU/3RScan)


