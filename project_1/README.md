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
The model returned by AutoML was a pipeline consisting of a `datatransformer`, which performs basic transformations of date features, text features etc., and a `prefittedsoftvotingclassifier`, which is an ensemble of 20 estimators, including a `xgboostclassifer`. 

## Pipeline comparison
The model returned with hyperparamter tuning yielded 0.91 accuracy, whereas the one returned by AutoML yielded a higher accuracy of 0.919. The difference may or may not be significant depending on the business use case. 

In terms of architecture, in hyperparameter tuning, the type of model experimented, as well as the hyperparameter search space, was manually selected. On the other hand, AutoML automatically searched through a collection of available models and combination of processing steps and models, which in theory should have a higher likelihood of returning a better model. The potential disadvantages of AutoML include the time it takes to run the search, which can be huge depending on the use case and the amount of data, and the possibility that the returned model may not be highly interpretable. 

## Future work
It could be possible that we did not find the optimal parameter sets. Since this is a small dataset, increasing the search space of the parameters and the `GridParameterSampling` approach may increase the likelihood that the search would converge to the optimal. 

## Proof of cluster clean up
All experiments contained in the notebooks were run on a personal Azure account. 


