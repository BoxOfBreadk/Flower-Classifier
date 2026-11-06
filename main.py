import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay

iris = load_iris(as_frame=True)
X = iris.data
y = iris.target
features = iris.data
labels = iris.target
species_names = iris.target_names

flowers = features.copy()
label_to_species = {
    0: 'setosa',
    1: 'versicolor',
    2: 'virginica'
}

