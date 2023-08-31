# This is the code of WWW-23 paper "Balancing Unobserved Confounding with a Few Unbiased Ratings in Debiased Recommendations".

## Datasets
We provide Coat dataset and Music dataset in the "datasets" folder. "random.txt" denotes unbiased data, and "user.txt" denotes biased data. 

## To run the code
Taking BAL-Autodebias as an example, the following command can be run directly: \
`python BAL-AD_coat.py`\
`python BAL-AD_music.py`\
All methods can be run in a similar way, and we also include several baseline methods in the "baselines" folder.

## Citation
If you find our code helpful, please kindly cite:
```
@inproceedings{li2023balancing,
  title={Balancing unobserved confounding with a few unbiased ratings in debiased recommendations},
  author={Li, Haoxuan and Xiao, Yanghao and Zheng, Chunyuan and Wu, Peng},
  booktitle={Proceedings of the ACM Web Conference 2023},
  pages={1305--1313},
  year={2023}
}
```

## Reference
```
@inproceedings{chen2021autodebias,
  title={AutoDebias: Learning to debias for recommendation},
  author={Chen, Jiawei and Dong, Hande and Qiu, Yang and He, Xiangnan and Xin, Xin and Chen, Liang and Lin, Guli and Yang, Keping},
  booktitle={Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval},
  pages={21--30},
  year={2021}
}
```
