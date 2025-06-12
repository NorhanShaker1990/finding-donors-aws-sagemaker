import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import accuracy_score ,fbeta_score, mean_absolute_error, recall_score, precision_score
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder



def distribution(data, transformed=False):
    """
       Visualization code for displaying skewed distributions of features
       """

    # Create figure

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for i, feature in enumerate(['capital-gain', 'capital-loss']):
        ax[i].hist(data[feature], bins=25, color='Blue')
        ax[i].set_title("Feature Distribution " + feature, fontsize=14)
        ax[i].set_xlabel("Value")
        ax[i].set_ylabel("Number of Records")
        ax[i].set_ylim((0, 2000))
        ax[i].set_yticks([0, 500, 1000, 1500, 2000])
        ax[i].set_yticklabels([0, 500, 1000, 1500, ">2000"])

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    CensusData = pd.read_csv('census.csv')
    # D = CensusData.describe(include = 'all') ##Remove duplicates
    Features = CensusData.drop('income', axis=1)
    income = CensusData['income']
    le = LabelEncoder()
    income = le.fit_transform(income)  # Converts '>50K' and '<=50K' to 1 and 0
    extractedFeatures = ['capital-gain', 'capital-loss']
    distribution(Features)
    # skewed data
    skewed_CensusData = pd.DataFrame(Features)
    skewed_CensusData[extractedFeatures] = Features[extractedFeatures].apply(lambda x: np.log(x + 1))
    distribution(skewed_CensusData)
    # convert numerical feature to all 0-1 range
    Scaler = MinMaxScaler()
    numerical = ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
    categoricalColumns = skewed_CensusData.drop(numerical, axis=1)
    categorical = categoricalColumns.columns
    skewed_CensusData[numerical] = Scaler.fit_transform(skewed_CensusData[numerical])
    CategoricalFeaturesAfterOneHotEncoding = pd.get_dummies(skewed_CensusData[categorical])
    skewed_CensusData = pd.concat([skewed_CensusData[numerical], CategoricalFeaturesAfterOneHotEncoding], axis=1)

    xtrain, xtest, ytrain, ytest = train_test_split(skewed_CensusData,
                                                    income,
                                                    test_size=0.2,
                                                    random_state=0)
    log_model = LogisticRegression(random_state=0)
    xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    rf_model = RandomForestClassifier(random_state=0)
    xtrain = xtrain.fillna(0)
    xtest = xtest.fillna(0)

    num_folds = 5
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

    print("-------Start Logistic Regression Cross Validation---------")
    LogisticRegression_cross_val_results = cross_val_score(log_model, xtrain, ytrain, cv=kf, scoring='accuracy')
    print("-------Start XGB Cross Validation---------")
    XGB_cross_val_results = cross_val_score(xgb_model, xtrain, ytrain, cv=kf, scoring='accuracy')
    print("-------Start Random Forest Cross Validation---------")
    RandomForestClassifier_cross_val_results = cross_val_score(rf_model, xtrain, ytrain, cv=kf, scoring='accuracy')

    CrossValidationResults = {}
    CrossValidationResults['XGB'] = np.mean(XGB_cross_val_results)

    CrossValidationResults['LogisticRegression'] = np.mean(LogisticRegression_cross_val_results)

    CrossValidationResults['RandomForestClassifier'] = np.mean(RandomForestClassifier_cross_val_results)

    ###Work on Random Forest separately (GridSearch)
    parameters = {'n_estimators': [50, 150, 70], 'min_samples_split': [15, 5, 20]}
    grid_obj = GridSearchCV(rf_model, param_grid=parameters)

    grid_fit = grid_obj.fit(xtrain, ytrain)

    # Get the estimator
    best_clf = grid_fit.best_estimator_

    joblib.dump(best_clf, "findingDonors.joblib")

    # Make predictions using the unoptimized and model
    best_predictions = best_clf.predict(xtest)

    print("Final accuracy score on the testing data: {:.4f}".format(accuracy_score(ytest, best_predictions)))
    print("Final F-score on the testing data: {:.4f}".format(fbeta_score(ytest, best_predictions, beta=0.5)))
