# === Explainability ===
import keras
import numpy as np
import pandas as pd
import json
import collections
import shap
import scipy
from scipy.special import softmax
import sklearn
import torch
import tensorflow as tf


result = collections.namedtuple('result', 'score properties')
info = collections.namedtuple('info', 'description value')


def analyse(clf, train_data, test_data, config, factsheet):
    
    #function parameters
    target_column = factsheet["general"].get("target_column")
    clf_type_score = config["score_algorithm_class"]["clf_type_score"]["value"]
    ms_thresholds = config["score_model_size"]["thresholds"]["value"]
    cf_thresholds = config["score_correlated_features"]["thresholds"]["value"]
    high_cor = config["score_correlated_features"]["high_cor"]["value"]
    fr_thresholds = config["score_feature_relevance"]["thresholds"]["value"]
    threshold_outlier = config["score_feature_relevance"]["threshold_outlier"]["value"]
    penalty_outlier = config["score_feature_relevance"]["penalty_outlier"]["value"]
    
    output = dict(
        algorithm_class     = algorithm_class_score(clf, clf_type_score),
        correlated_features = correlated_features_score(train_data, test_data, thresholds=cf_thresholds, target_column=target_column, high_cor=high_cor ),
        model_size          = model_size_score(clf, test_data, ms_thresholds),
        feature_relevance   = feature_relevance_score(clf, train_data, test_data ,target_column=target_column, thresholds=fr_thresholds,
                                                     threshold_outlier =threshold_outlier,penalty_outlier=penalty_outlier )
                 )

    scores = dict((k, v.score) for k, v in output.items())
    properties = dict((k, v.properties) for k, v in output.items())

    print("***** Explainability Metrics *****")
    print(scores)
    return  result(score=scores, properties=properties)


def algorithm_class_score(clf, clf_type_score):

    try:
        clf_name = type(clf).__name__
        exp_score = clf_type_score.get(clf_name,np.nan)
        properties= {"dep" :info('Depends on','Model'),
            "clf_name": info("model type",clf_name)}

        return  result(score=exp_score, properties=properties)
    except Exception as e:
        print("ERROR in algorithm_class_score(): {}".format(e))
        return result(score=np.nan, properties={"Non computable because": str(e)})


def correlated_features_score(train_data, test_data, thresholds=[0.05, 0.16, 0.28, 0.4], target_column=None, high_cor=0.9):

    try:
        test_data = test_data.copy()
        train_data = train_data.copy()

        if target_column:
            X_test = test_data.drop(target_column, axis=1)
            X_train = train_data.drop(target_column, axis=1)
        else:
            X_test = test_data.iloc[:,:-1]
            X_train = train_data.iloc[:,:-1]


        df_comb = pd.concat([X_test, X_train])
        corr_matrix = df_comb.corr().abs()

        # Select upper triangle of correlation matrix
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

        # Find features with correlation greater than 0.95
        to_drop = [column for column in upper.columns if any(upper[column] > high_cor)]

        pct_drop = len(to_drop)/len(df_comb.columns)

        bins = thresholds
        score = 5-np.digitize(pct_drop, bins, right=True)
        properties= {"dep" :info('Depends on','Training Data'),
            "pct_drop" : info("Percentage of highly correlated features", "{:.2f}%".format(100*pct_drop))}

        return  result(score=int(score), properties=properties)

    except Exception as e:
        print("ERROR in correlated_features_score(): {}".format(e))
        return result(score=np.nan, properties={"Non computable because": str(e)})


def model_size_score(clf, test_data, thresholds = np.array([10,30,100,500])):

    try:
        # New Suggestions to improve the Model Size Metric
        if isinstance(clf, sklearn.ensemble.RandomForestClassifier):
            # Get the overall maximum depth
            overall_max_depth = max([estimator.tree_.max_depth for estimator in clf.estimators_])
            print(overall_max_depth)

            # Get the number of trees
            nr_trees = clf.n_estimators
            print(nr_trees)

            # Calculate the total number of nodes
            total_nodes = sum([dt.tree_.node_count for dt in clf.estimators_])
            print(total_nodes)
        elif isinstance(clf, sklearn.neighbors.KNeighborsClassifier):
            # Get the "k" value
            k_value = clf.n_neighbors
            print(k_value)
        elif isinstance(clf, sklearn.svm.SVC):
            # Get the number of support vectors
            total_support_vectors = len(clf.support_vectors_)
            print(total_support_vectors)
        elif isinstance(clf, torch.nn.Module):
            # Calculate the number of parameters of a PyTorch DNN
            total_params = sum(p.numel() for p in clf.parameters() if p.requires_grad)
            print(total_params)
            # Get the number of layers in the PyTorch model

            num_layers = len(list(clf.children()))
            print(num_layers)
        elif isinstance(clf, tf.keras.models.Sequential) or isinstance(clf, tf.keras.models.Model):
            # Calculate the number of parameters of a Keras/Tensorflow DNN
            total_params = clf.count_params()
            print(total_params)

            # number of layers
            num_layers = len(clf.layers)
            print(num_layers)
        else:
            print("The type of the classifier is not supported.")

        dist_score = 5- np.digitize(test_data.shape[1]-1, thresholds, right=True)

        return result(score=int(dist_score), properties={"dep" :info('Depends on','Training Data'),
            "n_features": info("number of features", test_data.shape[1]-1)})
    except Exception as e:
        print("ERROR in model_size_score(): {}".format(e))
        return result(score=np.nan, properties={"Non computable because": str(e)})

def feature_relevance_score(clf, train_data, test_data, target_column=None, threshold_outlier = 0.03, penalty_outlier = 0.5, thresholds = [0.05, 0.1, 0.2, 0.3]):

    try:
        scale_factor = 1.5
        distri_threshold = 0.6
        importance_old = None
        if (type(clf).__name__ == 'LogisticRegression') or (type(clf).__name__ == 'LinearRegression'):
            pd.options.mode.chained_assignment = None
            train_data = train_data.copy()
            if target_column:
                X_train = train_data.drop(target_column, axis=1)
                y_train = train_data[target_column]
            else:
                X_train = train_data.iloc[:, :-1]
                y_train = train_data.iloc[:, -1:]

            #normalize
            #for feature in X_train.columns:
            #    X_train.loc[feature] = X_train[feature] / X_train[feature].std()
            clf.max_iter =1000
            clf.fit(X_train, y_train.values.ravel())
            importance_old = clf.coef_[0]
            #pd.DataFrame(columns=feat_labels,data=importance_old.reshape(1,len(importance_old))).plot.bar()

        if (type(clf).__name__ == 'RandomForestClassifier') or (type(clf).__name__ == 'DecisionTreeClassifier'):
             importance_old = clf.feature_importances_

        # Removed "else" clause - Feature Relevance is now quantifiable for all ML models - Computation using SHAP follows
        # else:
        #     return result(score= np.nan, properties={"dep" :info('Depends on','Training Data and Model')})

        if test_data is None:
            importance = importance_old
            feat_labels = [""]*100
            feat_labels = pd.Series(feat_labels)
        else:
            # Compute Feature Importance using SHAP Explainer
            nr_random_samples = 20
            rand_idxs = np.random.randint(0, test_data.shape[0], nr_random_samples)
            sample_data = test_data.iloc[rand_idxs]
            bg = sample_data.drop(target_column, axis=1)

            # Fits the explainer
            explainer = shap.Explainer(clf.predict, bg)
            shap_values = explainer(bg)
            # Calculates the feature importance (mean absolute shap value) for each feature
            importance = []
            for i in range(shap_values.values.shape[1]):
                importance.append(np.mean(np.abs(shap_values.values[:, i])))
            importance = scipy.special.softmax(np.asarray(importance))

            if importance_old is not None:
                # Compare the two approaches
                importances = pd.DataFrame([importance], columns=bg.columns, index=["SHAP Prediction"])
                importances.loc["Classifier Prediction"] = importance_old

            feat_labels = test_data.drop(target_column,axis=1).columns


        # absolut values
        importance = abs(importance)

        indices = np.argsort(importance)[::-1]
        feat_labels = feat_labels[indices]

        importance = importance[indices]

        # calculate quantiles for outlier detection
        q1, q2, q3 = np.percentile(importance, [25, 50 ,75])
        lower_threshold , upper_threshold = q1 - scale_factor*(q3-q1),  q3 + scale_factor*(q3-q1)

        #get the number of outliers defined by the two thresholds
        n_outliers = sum(map(lambda x: (x < lower_threshold) or (x > upper_threshold), importance))

        # percentage of features that concentrate distri_threshold percent of all importance
        pct_dist = sum(np.cumsum(importance) < 0.6) / len(importance)

        dist_score = np.digitize(pct_dist, thresholds, right=False) + 1

        if n_outliers/len(importance) >= threshold_outlier:
            dist_score -= penalty_outlier

        score =  max(dist_score,1)
        properties = {"dep" :info('Depends on','Training Data and Model'),
            "n_outliers":  info("number of outliers in the importance distribution",int(n_outliers)),
                      "pct_dist":  info("percentage of feature that make up over 60% of all features importance", "{:.2f}%".format(100*pct_dist)),
                      "importance":  info("feature importance", {"value": list(importance), "labels": list(feat_labels)})
                      }

        return result(score=int(score), properties=properties)
        # import seaborn as sns
        # sns.boxplot(data=importance)

    except Exception as e:
        print("ERROR in feature_relevance_score(): {}".format(e))
        return result(score=np.nan, properties={"Non computable because": str(e)})
