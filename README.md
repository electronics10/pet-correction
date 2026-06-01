## Introduction

This is a project dedicated for scatter correction for preclinical positron emission tomography (PET). We wish to train a deep-learning based scatter estimation model to outperform traditional single scatter simulation (SSS). The model should predict scatter sinograms from prompted sinograms (true + scatter). To train the model, we require Monte Carlo simulation that can produce distinct true and scatter bins. [MCGPU-PET](https://github.com/DIDSR/MCGPU-PET.git) is utilized to replace [GATE](http://www.opengatecollaboration.org/) as a faster backend Monte Carlo simulator. 

To make the model generalize enough, we plan on generating dataset that varies across different phantoms and PET scanners. Also, currently, attenuation map is considered to be not used in training (since attenuation map may NOT always be acquirable in practice). 

For 3D reconstruction, [Parallelproj](https://github.com/KUL-recon-lab/parallelproj) is utilized to acquire the forward and backward operator (which could probably used in advance ML architecture design to compensate the shortage of attenuation map).

## Installation
This project runs on Ubuntu 24.04 (other Linux should work; macOS/Windows untested).

**Prerequisites**
- git
- [pixi](https://pixi.sh) (`curl -fsSL https://pixi.sh/install.sh | sh`)
- an NVIDIA GPU with a driver supporting CUDA 12

**Setup**
```bash
git clone https://github.com/electronics10/pet-correction.git
cd pet-correction
pixi install
```

## Pipeline 

TODO (under development)
1. Python interface of MCGPU-PET ([this](notes/mcgpu-pet.md))
2. Machine learning architecture (pending)
3. Reconstruction algorithm ([this](notes/parallelproj.md))

