
<div align="center">

  <h1 align="center">AlignGS: Aligning Geometry and Semantics for Robust Indoor Reconstruction from Sparse Views</h1>
  

[Yijie Gao](https://gaoyijie-sjtu.github.io/), [Houqiang Zhong](https://waveviewer.github.io/), Tianchi Zhu, Zhengxue Cheng, Qiang Hu✉, Li Song✉ <br />


### [[`Project Page`](https://gaoyijie-sjtu.github.io/AlignGS/)]
</div>

<!-- <p align="center">
<img src="./static/im/teaser.png" width=100% height=100% 
class="center">
</p> -->

## Abstract

The demand for semantically rich 3D models of indoor scenes is rapidly growing, driven by 
applications in augmented reality, virtual reality, and robotics. However, creating them 
from sparse views remains a challenge due to geometric ambiguity. Existing methods often 
treat semantics as a passive feature painted on an already-formed, and potentially flawed, geometry. 
We posit that for robust sparse-view reconstruction, semantic understanding instead be an active, guiding force. 
This paper introduces AlignGS, a novel framework that actualizes this vision by pioneering a 
synergistic, end-to-end optimization of geometry and semantics. Our method distills rich priors from 
2D foundation models and uses them to directly regularize the 3D representation through 
a set of novel semantic-to-geometry guidance mechanisms, including depth consistency and 
multi-faceted normal regularization. Extensive evaluations on standard benchmarks demonstrate 
that our approach achieves state-of-the-art results in novel view synthesis and produces 
reconstructions with superior geometric accuracy. The results validate that leveraging 
semantic priors as a geometric regularizer leads to more coherent and complete 3D models from 
limited input views.

## Overview
<p align="center">
<img src="img/teaser.png" width=100% height=100% 
class="center">
</p>

We present the overview of the AlignGS pipeline, starting with initialization and 
the subsequent geometry and semantics joint optimization. This framework enables 
the end-to-end joint optimization of all geometric and semantic attributes of the 
Gaussian primitives, ensuring a synergistic refinement of both the scene's geometric 
structure and its semantic understanding.

## Citation
If you find this project useful in your research, please consider cite:

```

```
