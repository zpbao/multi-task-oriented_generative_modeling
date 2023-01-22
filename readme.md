
# Generative Modeling for Multi-task Visual Learning

This is the repository for [*Generative Modeling for Multi-task Visual Learning*](https://proceedings.mlr.press/v162/bao22c.html), published at ICML 2022.  


[[Paper](https://proceedings.mlr.press/v162/bao22c.html)]
[[Slides](https://icml.cc/media/icml-2022/Slides/17652.pdf)]
[[Poster](https://icml.cc/virtual/2022/poster/17651)]
[[Talk](https://icml.cc/virtual/2022/spotlight/17652)]

## Dataset architecture
```      
root 
	- rgb 
		- scene_class_0
			- 001.png
			- 002.png
			- ...
		- scene_class_1
	- ss
		- scene_class_0
			- 001.npy
			- ...
	- de
		- scene_class_0
			- 001.npy
			- ...
	- sn
		- scene_class_0
			- 001.png
			- ...
	- msk
		- scene_class_0
			- 001.png
			- ...

```


## Full MGM model training

To run the full model, due to the training mismatch of the GAN and discriminative model, we need to first pre-train the GAN and then train the full model. See `train.py` and `train.sh` for sampled training process.



## Light-weight MGM model training
The refinement network and the self-supervision networks are the key in the paper. These modeules can help as long as we have some weakly-labelled data. Thus, we provide a light-weight MGM model -- we did not include a generative model in the framework, but assume we have some weakly-labeled data (no matter it is real or synthesized) to train the model. 

see `MGM_light` in `mgm.py` and `train_light.py` for a sample training.


## Acknowledgement
This work is built on [self-attention GAN](https://github.com/rosinality/sagan-pytorch) and [taskgrouping](https://github.com/tstandley/taskgrouping) repo.

## Citation

```
@inproceedings{bao2022generative,
    Author = {Bao, Zhipeng and Hebert, Martial and Wang, Yu-Xiong},
    Title = {Generative Modeling for Multi-task Visual Learning},
    Booktitle = {ICML},
    Year = {2022},
}
```


