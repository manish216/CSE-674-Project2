Project2:
Explainable AI
==========================
Manish Reddy Challamala,
April 04, 2019 ,manishre@buffalo.edu

For detailed explaination, please visit the below link:
[link for report pdf](https://github.com/manish216/CSE-672-Project2/blob/master/proj2.pdf)

## Abstract

The objective of this project is to develop a machine learning model which
can learn the explainable features for a task domain and learn to answer the
variety of queries in that domain.

## 1 Introduction

XAI (explainable Artificial Intelligence), is the new perspective of machine learning, to
produce more explainable models that can establish the trust by enabling the users to
understand how the model is predicting the output by characterizing the features strength and
weakness.
In this project we are going to combine both deep learning and Probabilistic graphical model,
to achieve that a model could learn and answer variety of queries.
There are three different datasets of features:

1. Human determined features.
2. Deep learning features.
3. Explainable deep learning features

## 2 Theory

**2.1. Probabilistic Graphical Models**

Probabilistic graphical model is a structured model which represents the conditional
independencies between random variables.

There are two types of network structures:

2.1.1. Bayesian network

2 .1.2. Markov network

**2.1.1 Bayesian Network**

- Bayesian network is probabilistic graphical model for representing the multivariate
    probability distributions, in which nodes represent the random variables and the edges
    represent conditional probability distributions [CPD’s] tables between random
    variables, which are used to calculate the probability dependencies between the
    variables.
- Bayesian networks are also called as belief or causal networks because they are
    directed acyclic graphs [DAG], even with a change of CPD at one node can affect the
    whole networks performance.
- The probability distribution is defined in the form of factors:

...
     ![Probability distribution](https://latex.codecogs.com/gif.latex?P%28X_%7B1%7D&plus;X_%7B2%7D&plus;X_%7B3%7D&plus;...&plus;X_%7BN%7D%29%20%3D%20%5Cprod_%7Bi%3D1%7D%5E%7BN%7D%20P%28X_%7Bi%7D%20%7C%20%5Cprod%20X_%7Bi%7D%29)

Where ![](https://latex.codecogs.com/gif.latex?%5Cprod_%7Bi%3D1%7D%5E%7BN%7D%20P%28X_i%20%7C%20%5Cprod%20X_i%29) is P (node | parent(node))
...

**2.1.2 Markov network**

- Markov networks are undirected acyclic graphs which are similar to the Bayesian
    network. As the graphs are undirected, instead of edes they have cliques which
    connect each node with their neighboring nodes.
- The probability distribution is defined in the form of factors of potential functions.
    Which can be written in the log-linear form.
- As there is no topological order for the Markov network, we use potential functions
    for each clique in the graphs.
- Joint distribution in the Markov network is proportional to the product of clique
    potentials.
- The conditional probability for Markov network is defined as

...
![](https://latex.codecogs.com/gif.latex?P%28y%7C%20%5Ctheta%29%20%3D%20%5Cfrac%7B1%7D%7BZ%28%5Ctheta%29%7D%20%5Cprod_c%20%5Cphi%28y_c%20%7C%20%5Ctheta_c%29)

Where ![](https://latex.codecogs.com/gif.latex?Z%28%5Ctheta%29) is the partition function derives as summation of products of all
potential factors.
...

**2.2 Deep Learning:**

Deep learning is the representation of multi-layer artificial neural network which can
automatically learn the feature representations from the data provided. As the algorithms are
learning the features from the data, we can apply the deep learning algorithms to supervised,
unsupervised and semi-supervised learning problems.

We can use two types of models

2.2.1 Siamese network

2.2.2 Auto-Encoders

**2.2.1 Siamese network**

The idea of Siamese is twin network, where the two networks can share the same weights
among them to learn the useful features that can help to compare between the inputs of the
respective subnets.

Mostly, Siamese networks use the binary classification has activation function at the output,
to classify whether the two inputs are the same class or not.

**2.2.2 Auto-encoder**

- An auto encoder is the unsupervised learning algorithm that applies backpropagation,
    so that setting of target variable is equal to input variable.
- The auto encoder tries to learn an approximation to a function such that output x^ is
    similar to input x.
- We can divide the autoencoder model into two structures encoder and decoder.
- The encoder model is forced to learn the compressed knowledge representations of
    the input, for example given an input image the encoder will detect the most important
    features of the input.
- The decoder on the other hand takes these features from encoder as input and try to
    reconstruct the input.
- So, from the above two statements, we can understand that the encoder and decoder
    share same structure.

## 3 Explanation:

**3.1 Task1: Data Annotation**.

In this task, we are manually going through all of the images in the dataset and assigning each
feature of the image to specific class. In total there are 15 features each consists of different
class. The 15 features and no of classes each feature contains is displayed below.


Table 1: Representation of no of classes each feature can take


| Feature           | No of classes |
| ---               | ---           |
|Pen_pressure       |2              |
|Letter_spacing     |3              |
|Size               |3              |
|Dimensions         |3              |
|Is_lowercase       |2              |
|Is_continuous      |2              |
|Slatness           |4              |
|Tilt               |2              |
|Entry stroke_’a’   |2              |
|Staff of ‘a’       |4              |
|Formation of ‘n’   |2              |
|Staff of ‘d’       |3              |
|Exit stroked d     |4              |
|Word formation     |2              |
|constancy          |2              |


**3.1 Task 2 : Sample Verification of PGM:**

In this task, we are using the Bayesian networks to create a model and
infer the value, which says weather the two images are similar or not.
Creating five Bayesian networks:

- We are creating a Bayesian network structure [f], and making a copy
    of the structure ‘f’ and naming the other structure as [g].
- The edges for a Bayesian network structure is selected by using the
    Correlation values.

[For detailed explaination of the evaluation results please vist the link](https://github.com/manish216/CSE-672-Project2/blob/master/proj2.pdf)

**3. 2 Task 3 : Deep learning Inference
3.2.1 Siamese network**

Create a Convolution 2D Siamese network:
    - A convolution 2D Siamese network is created using keras.
    - Inputs to the networks will be the pair of images. [left image
       to left Siamese network and right image to right Siamese
       network.]
    - The network will have many layers which will perform
       convolution, max pooling operations with ‘relu’ and sigmoid as
       activation function, optimizer used is adadelta and loss
       function as binary_ cross entropy.
Issue with Siamese network:
- While working with Siamese network the training and
validation accuracy is ranging from 0.5 to 0. 65
(approximately). The problem might be that the model is not
training properly.

3.2.2 Auto-encoder

Creating the Auto-encoder
- An auto-encoder is created using convolution 2d layers, where
    the model performs the convolution 2D and max pooling
    operations with relu and sigmoid as activation function.
- The encoder section of the model learns the knowledge
    representation of the input.
- The output of the encoder will be a vector of 1 x512.
- The decoder section of the model will take these 512 features
    as input and tries to reconstruct the image.
- Advantage of using the auto-encoder is that as we are
    reconstructing the images, we can keep a check whether the
    model is properly trained or not.
- The model is trained using the ‘fit_generator()’ method.


[For detailed explaination of the evaluation results please vist the link](https://github.com/manish216/CSE-672-Project2/blob/master/proj2.pdf)

