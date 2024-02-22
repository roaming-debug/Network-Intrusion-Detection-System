#   Author        : *** Ezra Ge ***
#   Last Modified : *** DATE ***

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sklearn
import torch
import argparse
import helper
import pandas as pd
from sklearn import tree
from sklearn import metrics
from sklearn import neural_network
from sklearn import model_selection
import os


def exp1(data, labels):
    """STUDENT CODE BELOW"""
    if not os.path.exists('image'):
        os.makedirs('image')
    # define model architecure
    model = tree.DecisionTreeClassifier(max_depth=7)
    X_train, X_test, y_train, y_test = model_selection.train_test_split(data, labels, test_size=0.3, random_state=10)
    model.fit(X_train, y_train)
    
    # validation
    y_pred = model.predict(X_test)
    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
    
    # draw decision tree
    plt.figure(figsize=(20,20))
    tree.plot_tree(model, fontsize=5, feature_names=model.feature_names_in_)
    plt.savefig('image/tree_1', dpi=200)
    
    # draw feature importance plot
    feature_importance = model.feature_importances_
    sorted_idx = np.argsort(feature_importance)
    plt.figure(figsize=(12, 12))
    plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx], align='center')
    plt.yticks(range(len(sorted_idx)), np.array(X_test.columns)[sorted_idx])
    plt.title('Feature Importance')
    plt.savefig('image/feature_importance_1')

    # Confusion matrix plot
    cm = metrics.confusion_matrix(y_test, y_pred)
    ax= plt.subplot()
    sns.heatmap(cm, annot=True, fmt='g', ax=ax)

    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix')
    ax.xaxis.set_ticklabels(['benign', 'malicious'])
    ax.yaxis.set_ticklabels(['benign', 'malicious'])
    plt.savefig('image/confusion_matrix_1')
    
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)
    roc_auc = metrics.auc(fpr, tpr)
    display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc)
    display.plot()
    plt.savefig('image/ROC_curve_1')
    """STUDENT CODE ABOVE"""
    return model


def exp2(data, labels):
    """STUDENT CODE BELOW"""
    if not os.path.exists('image'):
        os.makedirs('image')
    # define model architecture
    model = tree.DecisionTreeClassifier(max_depth=10)
    X_train, X_test, y_train, y_test = model_selection.train_test_split(data, labels, test_size=0.3, random_state=10)
    model.fit(X_train, y_train)
    
    # validation
    y_pred = model.predict(X_test)
    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
    
    # draw decision tree
    plt.figure(figsize=(20,20))
    tree.plot_tree(model, fontsize=5, feature_names=model.feature_names_in_)
    plt.savefig('image/tree_2', dpi=200)
    
    # draw feature importance plot
    feature_importance = model.feature_importances_
    sorted_idx = np.argsort(feature_importance)
    plt.figure(figsize=(12, 12))
    plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx], align='center')
    plt.yticks(range(len(sorted_idx)), np.array(X_test.columns)[sorted_idx])
    plt.title('Feature Importance')
    plt.savefig('image/feature_importance_2')

    # Confusion matrix plot
    cm = metrics.confusion_matrix(y_test, y_pred)
    ax= plt.subplot()
    sns.heatmap(cm, annot=True, fmt='g', ax=ax)

    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix')
    ax.xaxis.set_ticklabels(['benign', 'DoS', 'Probe', 'R2L', 'U2R'])
    ax.yaxis.set_ticklabels(['benign', 'DoS', 'Probe', 'R2L', 'U2R'])
    plt.savefig('image/confusion_matrix_2')
    """STUDENT CODE ABOVE"""
    return model


def exp3(data, labels):
    """STUDENT CODE BELOW"""
    if not os.path.exists('image'):
        os.makedirs('image')
    # define model architecture
    model = neural_network.MLPClassifier(hidden_layer_sizes=(50,20), random_state=99)
    X_train, X_test, y_train, y_test = model_selection.train_test_split(data, labels, test_size=0.3, random_state=10) 
    model.fit(X_train, y_train)

    # validation
    y_pred = model.predict(X_test)
    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

    # Loss curve
    plt.plot(model.loss_curve_)
    plt.savefig('image/loss_curve_3')
    plt.clf()
    
    # Confusion matrix plot
    cm = metrics.confusion_matrix(y_test, y_pred)
    ax= plt.subplot()
    sns.heatmap(cm, annot=True, fmt='g', ax=ax)
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix')
    ax.xaxis.set_ticklabels(['benign', 'malicious'])
    ax.yaxis.set_ticklabels(['benign', 'malicious'])
    plt.savefig('image/confusion_matrix_3')
    
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)
    roc_auc = metrics.auc(fpr, tpr)
    display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc)
    display.plot()
    plt.savefig('image/ROC_curve_3')
    """STUDENT CODE ABOVE"""
    return model


def exp4(data, labels):
    """STUDENT CODE BELOW"""
    if not os.path.exists('image'):
        os.makedirs('image')
    # define model architecture
    model = neural_network.MLPClassifier(hidden_layer_sizes=(50, 20), random_state=199)
    X_train, X_test, y_train, y_test = model_selection.train_test_split(data, labels, test_size=0.3, random_state=10) 
    model.fit(X_train, y_train)
    
    # validation
    y_pred = model.predict(X_test)
    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

    # Loss curve
    plt.plot(model.loss_curve_)
    plt.savefig('image/loss_curve_4')
    plt.clf()

    # Confusion matrix plot
    cm = metrics.confusion_matrix(y_test, y_pred)
    ax= plt.subplot()
    sns.heatmap(cm, annot=True, fmt='g', ax=ax)

    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix')
    ax.xaxis.set_ticklabels(['benign', 'DoS', 'Probe', 'R2L', 'U2R'])
    ax.yaxis.set_ticklabels(['benign', 'DoS', 'Probe', 'R2L', 'U2R'])
    plt.savefig('image/confusion_matrix_4')
    """STUDENT CODE ABOVE"""
    return model


def exp5(data, labels):
    """STUDENT CODE BELOW"""
    # convert data to pytorch dataset
    dataset = helper.convert_to_pytorch_dataset(data, labels)
    # define model architecture
    model = torch.nn.Sequential(
        torch.nn.Linear(40, 50),
        torch.nn.ReLU(),
        torch.nn.Linear(50, 10),
        torch.nn.ReLU(),
        torch.nn.Linear(10, 5),
    )
    trainloader = torch.utils.data.DataLoader(dataset)
    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    for epoch in range(2):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0

    """STUDENT CODE ABOVE"""
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=int, default=1, help="Experiment number")
    args = parser.parse_args()
    save_name = f"exp{args.exp}_model" + (".pt" if args.exp == 5 else ".pkl")
    if args.exp == 1:
        model = exp1(*helper.load_dataset(multiclass=False, normalize=False))
    elif args.exp == 2:
        model = exp2(*helper.load_dataset(multiclass=True, normalize=False))
    elif args.exp == 3:
        model = exp3(*helper.load_dataset(multiclass=False, normalize=True))
    elif args.exp == 4:
        model = exp4(*helper.load_dataset(multiclass=True, normalize=True))
    elif args.exp == 5:
        model = exp5(*helper.load_dataset(multiclass=True, normalize=True))
    else:
        print("Invalid experiment number")
    helper.save_model(model, save_name)
