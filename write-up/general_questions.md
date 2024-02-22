# General Questions

### Q1: What is the importance of using a validation set?

A validation set could be used to identify the overfitting issue of the model since the model has never seen the validation set before. In addition to this, it could be used to estimate the prediction error, giving a sense of accuracy of the model.

### Q2: Assuming a binary classification setting, describe what false negative rate and false positive rate represents in the context of this dataset.

False negative rate represents the rate of the model falsely identifying attack instances as benign within all attack data packets. False positive rate represents the rate of falsely identifying benign data packets as attack data packets within all benign packets.

### Q3: What differences did you observe between the decision tree models and the neural networks? What are some pros and cons of using each model?

We can observe the internal structure of the decision tree models, however, we cannot do the same using neural networks.

Pros of decision tree models
- able to visualize the decision

Cons of decision tree models
- too many branches of the tree could be hard to interpret
- prone to overfitting

Pros of Neural Networks

- Much better when training on audio, image, or other unstructued data
- Able to understand complex non-linear relationships

Cons of Neural Networks

- Lack of interpretability
- Requires extensitve computation




### Q4: Between decision trees and neural networks, which model would you recommend using in security sensitive domains and why? There are no wrong answers here, but answer should be supported with sufficient justification.

I would recommend neural networks because of its ability possibly able to detect unseen attack and able to understand complex non-linear relationships. Combined with other features of the network packets such as unstructured data, which cannot be trained on a decision tree, the model could be drastically improved.

### Q5: What did you notice when switching from the binary classification to multiclass classification setting?

The accuracy of the model decreases using the same hyperparameter settings and we could not rely on ROC curve to check the model accuracy. Because of limited data of R2L and U2R, the accuracy of these 2 classification is not satisfying.
