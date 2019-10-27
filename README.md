

In this project, we utilized the Spark engine to build and evaluate a music recommender system. Relying on implicit feedback modeling, we conducted experiments on building the baseline model and did hyperparamter tuning on the rank of latent factors, regularization parameter and the scaling for handling implicit feedbacks. Further, we evaluated several modification strategies on implicit feedback data and the efficiency gain achieved on accelerated query search from utilizing spatial data structure by using the Annoy package.

## Evaluation

* **Precision at *k***: This is a measure of how many of the predicted items are truly relevant, i.e. appear in the true relevant items set, regardless of the ordering of the recommended items. The mathematical definition is the following

  ![](https://latex.codecogs.com/gif.latex?p%28k%29%3D%5Cfrac%7B1%7D%7BM%7D%20%5Csum_%7Bi%3D0%7D%5E%7BM-1%7D%20%5Cfrac%7B1%7D%7Bk%7D%20%5Csum_%7Bj%3D0%7D%5E%7B%5Cmin%20%28%7CD%7C%2C%20k%29-1%7D%20%5Coperatorname%7Brel%7D_%7BD_%7Bi%7D%7D%5Cleft%28R_%7Bi%7D%28j%29%5Cright%29%20%5Cquad%20%5Ctext%20%7B%20where%20%7D%20%5Coperatorname%7Brel%7D_%7BD%7D%28r%29%3D%5Cleft%5C%7B%5Cbegin%7Barray%7D%7Bll%7D%7B1%7D%20%26%20%7B%5Ctext%20%7B%20if%20%7D%20r%20%5Cin%20D%7D%20%5C%5C%20%7B0%7D%20%26%20%7B%5Ctext%20%7B%20otherwise%20%7D%7D%5Cend%7Barray%7D%5Cright.)
  
* **Mean Average Precision (MAP)**: This is also a measure of how many of the predicted items also appear in the true relevant items set. However, the MAP score accounts for the order of the recommender. It will impose higher penalty for highly relevant items not being recommended with high relevance by the model, *i.e.* the item appears near the end of recommender list or even doesnt appear in the list. The mathematical definition is the following:

  ![](https://latex.codecogs.com/png.latex?%5Cmathrm%7BMAP%7D%3D%5Cfrac%7B1%7D%7BM%7D%20%5Csum_%7Bi%3D0%7D%5E%7BM-1%7D%20%5Cfrac%7B1%7D%7B%5Cleft%7CD_%7Bi%7D%5Cright%7C%7D%20%5Csum_%7Bj%3D0%7D%5E%7BQ-1%7D%20%5Cfrac%7B%5Cmathrm%7Brel%7D_%7BD_%7Bi%7D%7D%5Cleft%28R_%7Bi%7D%28j%29%5Cright%29%7D%7Bj%2B1%7D)
