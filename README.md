# This is the code of WWW-23 paper "Balancing Unobserved Confounding with a Few Unbiased Ratings in Debiased Recommendations".

## Datasets
We provide Coat dataset and Music dataset in the "datasets" folder. "random.txt" denotes unbiased data, and "user.txt" denotes biased data. 

## To run the code
Taking BAL-Autodebias as an example:


- For Coat:


Run the code:

```shell
python BAL-AD.py --dataset coat
```

- For Music:


Run the code:

```shell
python BAL-AD.py --dataset music
```


All methods can be run in a similar way, and we also include several baseline methods in the "baselines" folder.


## Environment Requirement

The code runs well at python 3.8.18. The required packages are as follows:
-   pytorch == 1.9.0
-   numpy == 1.24.4 
-   scipy == 1.10.1
-   pandas == 2.0.3
-   scikit-learn == 1.3.2


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

