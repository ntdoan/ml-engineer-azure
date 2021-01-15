# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Summary
This dataset was obtained from a banking campaign, consisting of 20 features, including various demographic information about the clients. The goal is to predict whether the client would subscribe a term deposit (variable y)

The best performing model was a pipeline composed of a data transformation step and an ensemble. This model was found using AutoML. 

## Scikit-learn Pipeline and hyperparameter tuning
The best performing model was a logistic regression model with `C` of 1 and `max_iter` of 300. 

Before model training, the data are cleaned using the `clean_data` function implemented in `train.py`. The pipeline is simply consisted of fitting the Logistic Regression itself. To search for the optimal parameter set (`C` and `max_iter`) for this model, we used hyperparameter tunning functionality as provided in `HyperDrive`. A grid of the hyperparameter space was created for the set of hyperparameters. 

The `RandomParameterSampling` was used as it randomly searches the parameter space and thus is less time consuming compared to e.g. `GridParameterSampling` which performs exhausive search over the entire space. RandomParameterSampling also supports both continous and discrete hyperparamters. 

The `Bandit` policy was used for early stopping policy. It terminates runs where the primary metric, `accuracy` in this case, is not within the predefined a slack amount compared to the best performing run. 

## AutoML
The model returned by AutoML was a pipeline consisting of a `datatransformer`, which performs basic transformations of date features, text features etc., and a `prefittedsoftvotingclassifier`, which is an ensemble of 9 estimators, including a `xgboostclassifier`. 


## Pipeline comparison
The model returned with hyperparamter tuning yielded 0.91 accuracy, whereas the one returned by AutoML yielded a higher accuracy of 0.919. The difference may or may not be significant depending on the business use cases. 

A variation of the above AutoML run, which used a cross validation setting of the training data, with parameter `n_cross_validations`, instead of using the validation dataset, yielded a very similar accuracy (0.917) and the best model with the same architecture. 

The best AutoML model is a soft voting ensemble. The final class label is the result of a weighted average of votings from 9 estimators. Each estimator is a `sklearn` pipeline. The estimator with the highest weight (`w=0.27`) is a pipeline of a `maxabsscaler` followed by a `xgboostclassifier`. The remaining 8 estimators have an equal weight of `0.09`.

Ensemble of estimators is known be among the best classifiers for several applications. It is worth noting that AutoML with a simpler run configuration, i.e. less number of parameters to set (no search space to specify) compared to the HyperDrive run, returned a model which is fairly easy to interpret with a higher accuracy. 

In hyperparameter tuning, the type of model experimented, as well as the hyperparameter search space, was manually selected. On the other hand, AutoML automatically searched through a collection of available models and combination of processing steps and models, which in theory should have a higher likelihood of returning a better model. The potential disadvantages of AutoML include the time it takes to run the search, which can be huge depending on the use case and the amount of data, and the possibility that the returned model may not be highly interpretable. 

## Future work
It could be possible that we did not find the optimal parameter sets. Since this is a small dataset, increasing the search space of the parameters and the `GridParameterSampling` approach may increase the likelihood that the search would converge to the optimal. 

## Proof of cluster clean up
The cluster is deleted in the notebook.


