# Network Intrusion Detection System Machine Learning Project

This project aimed at creating machine learning models to predict future network traffic, detecting if it is benign or malicious. Various experiments were performed with different models and classification settings - training, adjusting, and evaluating the efficiency of each model.

This project gives an understanding of how to build and evaluate predictive machine learning models particularly in the field of Information Security. It also provides a glance into the importance of precision and recall when determining the efficiency of a model.

## Set up

i. Build the Docker image

```bash
docker build -t network-intrusion .devcontainer/
```

ii. Every time you want to test your code and if you have exited the container you had previously created, you will have to deploy a new Docker container:

```bash
docker run --rm -it -v ${PWD}:/workspace/project2 \
    -v ${HOME}/.gitconfig:/home/vscode/.gitconfig \
    -v ${HOME}/.ssh:/home/vscode/.ssh \
    -w /workspace/project2 \
    --entrypoint bash cs642-project2-docker-image:latest
```
To execute your program, run `python3 main.py --exp <EXP_NUM>`. This will run the function corresponding to that experiment number and will save the trained model to `exp<EXP_NUM>_model.pkl` (or `exp<EXP_NUM>_model.pt` for the extra credit experiment).

Once you are happy with your model, you can submit it to the test program to be tested against the hidden test set. 

To test your models on the hidden test set, run `./642test-$(uname -p)
<MODEL_FILE_NAME>`. Optionally, for binary classification settings you can also specify a threshold (default is 0.5) for classifying samples as attacks. In this case, run `./642test-$(uname -p) <MODEL_FILE_NAME> <THRESHOLD>` to include the threshold.

The test program will output a variety of statistics generated from `sklearn.metrics.classification_report`. Models will be given full points for the testing portion of the grade once they reach at least 88% accuracy on the test program.

## Conclusion

The report of the model evaluations: [evaluation](./write-up/experiment_questions.md)
