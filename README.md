# Autonomous-active-learning

Active learning comparison on MNIST data by appending labels to a continuously growing data set. Here we compare the difference of appending data points with the highest and lowest prediction certainty. For the first method we start of with a data set of unlabeled MNIST images of size $N=10000$. An initial size $n_{initial}=1000$ is labeled. With this inital data, a model is trained on the $n_{initial}$ data points. The trained model then predicts on the remaining $N-n_{initial}$ unlabeled data points and we append $n=100$ of these predictions with the highest confidence (class probability $p_i$) and use the model's prediction as a label to our growing data set. This process is repeated with but now with a data set with size $n_{initial}+n$. The main idea is that a larger data set should result in better performance, however labeling a large data set is time consuming, hence why we only label a subset of $n_{initial}$ and try to autonomously label the remaining data. This method is subject to missclassification in the growing data set. However, machine learning models are robust enough to generalize with a small % of missclassified labels in the data set. We compare this method to the classical active learning method where we manually label the $n$ data points with the lowest prediction confidence and append them to our data set. The dynamically growing data set is defined in <code>ActiveLearningDataset</code> in the <code>data.py</code> module. 


---
**Analysis**

Three separate runs with 40 iterations was simulated for both methods and the $F_1$ score on a held-out test set was used to benchmark the different methods. 

![image1](https://i.imgur.com/imquG0K.png)
