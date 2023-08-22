# Autonomous-active-learning

Active learning comparison on MNIST data by appending labels to a continuously growing data set. Here we compare the difference of appending data points with the highest versus lowest prediction confidence. For the first method we start of with a data set of unlabeled MNIST images of size $N=10000$. An initial size $n_{initial}=1000$ is labeled. With this inital data, a model is trained on the $n_{initial}$ data points. The trained model then predicts on the remaining $N-n_{initial}$ unlabeled data points and we append $n=100$ of these predictions with the highest confidence (class probability $p_i$) and use the model's prediction as a label to our growing data set. This process is repeated with but now with a data set with size $n_{initial}+n$. The main idea is that a larger data set should result in better performance, however labeling a large data set is time consuming, hence why we only label a subset of $n_{initial}$ and try to autonomously label the remaining data. This method is subject to missclassifications in the growing data set. However, machine learning models are robust enough to generalize with a small % of missclassified labels in the data set. We compare this method to the classical active learning method where we manually label the $n$ data points with the lowest prediction confidence and append them to our data set. The dynamically growing data set is defined in <code>ActiveLearningDataset</code> in the <code>data.py</code> module. 


---
**Analysis**

Five separate runs with 40 iterations was simulated for both methods and the $F_1$ score on a held-out test set was used to benchmark the different methods. As expected the classical method achieves a higher increase in performance over the run of the simulation. This is because adding labels to data points in which the model has low prediction confidence on, should yield more  information to the data set. However, we still see an increase in performance for the greedy method that is completelty autonomous in its labeling process. 

![image1](https://i.imgur.com/nI4tsKE.png)

As mentioned, the greedy method will introduce missclassified data to the data set. The error rate as a function of data size is shown in the figure below. It's worth noting that the parameters $n_initial$ and $n$ directly affect the error_rate progression. For large $n_initial$ the initial model will have a high generalization performance which lowers the probability of introducing missclassified labels. For the used parameters the error rate is growing proportional to $E \sim \sqrt{x}$ and it looks to plateau near the end of the simulation. This is most likely due to the fact that towards the end, the performance of the ML model is high enough so the rate at which new data is appended to the data set is faster than the introduction of missclassified labels. 

![image2](https://i.imgur.com/VGNOcg8.png)
