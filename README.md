# Autonomous-active-learning

Active learning comparison on MNIST data by appending labels to continuously growing data set. Comparison of active learning method. We start of with a data set of unlabeled MNIST images of size $N$. An initial size $n_{initial} << N$ is labeled. With this data a model is trained on the $n_{initial}$ data points. The trained model predicts on the remaining $N-n_{initial}$ unlabeled points and we select $n$ of these predictions and use the model's prediction as label. This process is repeated with but now with a data set with size $n_{initial}+n$. 
