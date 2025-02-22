import os
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, recall_score
from imblearn.over_sampling import SMOTE


class NaiveBayesClassifier:
    def __init__(self):
        self.class_priors = {}
        self.feature_means = {}
        self.feature_stds = {}
        self.feature_probs = {}

    def fit(self, X, y):
        self.class_priors = {label: count / len(y) for label, count in y.value_counts().items()}

        for label in self.class_priors:
            X_label = X[y == label]
            self.feature_means[label] = X_label[continuous_features].mean().to_dict()
            self.feature_stds[label] = X_label[continuous_features].std(ddof=0).replace(0, 1e-9).to_dict()

            self.feature_probs[label] = {}
            for feature in categorical_features:
                feature_counts = X_label[feature].value_counts(normalize=True).to_dict()
                self.feature_probs[label][feature] = feature_counts

    def _gaussian_likelihood(self, x, mean, std):
        return norm.pdf(x, mean, std)
    
    def _categorical_likelihood(self, feature, value, label):
        return self.feature_probs[label][feature].get(value, 1e-9)
    
    def predict(self, x):
        epsilon = 1e-9
        predictions = []

        for _, row in x.iterrows():
            posteriors = {}
            for label in self.class_priors:
                prior_log_prob = np.log(self.class_priors[label] + epsilon)
                continuous_log_likelihood = sum(
                    np.log(self._gaussian_likelihood(row[feature], self.feature_means[label][feature], self.feature_stds[label][feature]) + epsilon)
                    for feature in continuous_features
                )
                categorical_log_likelihood = sum(
                    np.log(self._categorical_likelihood(feature, row[feature], label) + epsilon)
                    for feature in categorical_features
                )
                posteriors[label] = prior_log_prob + continuous_log_likelihood + categorical_log_likelihood

            predictions.append(max(posteriors, key=posteriors.get))
        return np.array(predictions)


current_file_path = os.path.abspath(__file__)
current_directory = os.path.dirname(current_file_path)
os.chdir(current_directory)

print("Current working directory:", os.getcwd())

print("Train file exists:", os.path.exists('adult_train.txt'))
print("Test file exists:", os.path.exists('adult_test.txt'))

train_file_path = 'adult_train.txt'
test_file_path = 'adult_test.txt'

column_names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']

train_data = pd.read_csv(train_file_path, names=column_names, na_values='?', delimiter=',')
test_data = pd.read_csv(test_file_path, names=column_names, na_values='?', delimiter=',')

train_data.dropna(inplace=True)
test_data.dropna(inplace=True)

train_data.drop(columns=['native-country', 'fnlwgt'], inplace=True)
test_data.drop(columns=['native-country', 'fnlwgt'], inplace=True)
# education-num与captain-gain以及hours-per-week相关性到达0.14左右可以选择删除
# train_data.drop(columns=['education'], inplace=True) 
# test_data.drop(columns=['education'], inplace=True)

train_data['income'] = train_data['income'].map({' <=50K': 0, ' >50K': 1})
test_data['income'] = test_data['income'].map({' <=50K.': 0, ' >50K.': 1})

combined_data = pd.concat([train_data, test_data], ignore_index=True)

y_combined = combined_data['income']
X_combined = combined_data.drop(columns=['income'])

X_combined = pd.get_dummies(X_combined)

X_train = X_combined.iloc[:len(train_data)]
X_test = X_combined.iloc[len(train_data):]
y_train = train_data['income']
y_test = test_data['income']

continuous_features = ["age", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
categorical_features = list(set(X_combined.columns) - set(continuous_features))

smote = SMOTE(random_state=123)
X_train, y_train = smote.fit_resample(X_train, y_train)

nb = NaiveBayesClassifier()
nb.fit(X_train, y_train)

y_pred = nb.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print(f"AUC: {auc:.4f}")
print(f"F1-score: {f1:.4f}")
print(f"Recall: {recall:.4f}")
