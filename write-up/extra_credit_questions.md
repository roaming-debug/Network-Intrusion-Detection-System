# Extra Credit Questions - PyTorch model (can be answered as long as extra credit model was attempted)

Complete with command to run on test set: `./642test-$(uname -p) exp5_model.pt` 

### Q1: If you attempted the extra credit (whether you finished it or not), what difficulties did you encounter?

I have to recall what I learned from CS540 and I almost forgot everything. Relearning the concepts and implementing it took a significant amount of my time even. I also tried to adjust few hyperparameters and added new functionality to the training process, but unexpectedly, the accuracy of the model decreases. After tuning for hours, I gave up and reverted to my original rudimentary code. Also my model would not output the label 4 using either my validation data set or testing data set even after I have adjusted the weights of each label in the optimizer.

### Q1: What batch size did you select when using the DataLoader? What is the trade off between using smaller vs larger batch sizes? How does being able to control the batch size enable us to train larger models or use larger datasets?

I chose the default value - 1. Using larger batch sizes would take a large amount of memory space and requires more epochs to achive the same accuracy as using a smaller batch size. Utlizing larger batch sizes could result in faster convergence but less likely to converge on optimal solution. Being able to control batch size could help us balance between computational effeciency, the speed of convergence, finding optimal solution.