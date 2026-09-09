from src.data_handling import preprocessing
from src.recommenders import popularity
from src.recommenders import collaborative
from src.recommenders import content_based
from src.recommenders import hybrid, ranking

from src.output import evaluation
from src.data_handling import exploration, feature_engineering

from tests import sample_data

import pandas as pd

# TESTING THAT BASIC RECOMMENDERS WORK INDEPENDENTLY

#first load/merge/clean/typecast/process final data

loaded_datasets = preprocessing.load_datasets()
merged_datasets = preprocessing.merge_datasets(loaded_datasets)
cleaned_datasets = preprocessing.clean_datasets(merged_datasets)
typecast_datasets = preprocessing.typecasting_datasets(cleaned_datasets)

processed_datasets = preprocessing.processing_datasets(typecast_datasets)

evaluated_student = 11391

def check_baselines_working():
    #recommender 1: popularity
    popularity_recs = popularity.popularity_scores(evaluated_student,processed_datasets["interaction_data"])

    print("Popularity Recommender: \n")
    print(popularity_recs.head(5))

    # next is collaborative recommender, for student id 11391
    collaborative_recs = collaborative.collaborative_scores(evaluated_student,processed_datasets["interaction_data"])
    print("\nCollaborative Filtering Recommender: \n")
    print(collaborative_recs.head(5))

    #last is content based recommender, for student id 11391
    content_recs = content_based.content_scores(evaluated_student,processed_datasets["resource_data"],processed_datasets["interaction_data"])
    print("\nContent-based Filtering Recommender: \n")
    print(content_recs.head(5))

    hybrid_recs = hybrid.hybrid_scores(popularity_recs,content_recs,collaborative_recs)
    print("\nHybrid Recommender: \n")
    print(hybrid_recs.head(5))

# create function generate FINAL recommendations with all resource columns + all score columns



# check_baselines_working()

#BASELINE RECOMMENDERS EVALUATION for top 20 recommendation
# history_interactions, future_interactions = evaluation.temporal_split(processed_datasets["vle_data"],86)
# k = 20

# evaluation.evaluate_baseline_models_single(history_interactions,future_interactions,processed_datasets["resource_data"],evaluated_student,k)

# OLD, uses >86 for evaluation. will use >150 to evaluate later on, same as ML model
# results, summary = evaluation.evaluate_baseline_models_overall(history_interactions,future_interactions,processed_datasets["resource_data"],k, True)
# print(summary)

#save visualizations in images folder
# exploration.explore_student_data(processed_datasets["student_data"])
# exploration.explore_assessment_data(processed_datasets["assessment_data"])
# exploration.explore_vle_data(processed_datasets["vle_data"])
# exploration.explore_resource_data(processed_datasets["resource_data"])
# exploration.explore_interaction_data(processed_datasets["interaction_data"])

# test sample data for testing works and looks correct
# test_data = sample_data.create_datasets()
# print(test_data)


# ML RANKING MODEL

# cutoff days, median (86) and less than max (269)
# training_cutoff = 86
# test_cutoff = 150

# 1. cutoff into ML periods
# (training_history_vle,training_future_vle,test_history_vle,test_future_vle) = evaluation.ml_temporal_split(processed_datasets["vle_data"],training_cutoff,test_cutoff)

#2. aggregate vle into interaction data so that recommenders can use in producing scores
# training_history = evaluation.aggregate_interaction_data(training_history_vle)
# training_future = evaluation.aggregate_interaction_data(training_future_vle)
# test_history = evaluation.aggregate_interaction_data(test_history_vle)
# test_future = evaluation.aggregate_interaction_data(test_future_vle)

# baseline models evaluation, using same testing cutoff so all training data before it used for recs then after it for target (no need training because its not a real model)
# results, summary = evaluation.evaluate_baseline_models_overall(test_history,test_future,processed_datasets["resource_data"],k, True)
# print("Baseline models results",summary)

#3. create student features for training/test as per cutoffs
# training_student_features = feature_engineering.create_features(processed_datasets, training_cutoff)
# test_student_features = feature_engineering.create_features(processed_datasets, test_cutoff)

# check if it works
# print("training history",training_history.head())
# print("training student features",training_student_features.head())

# 4. prepare ML dataset for all student-course rows, save to csv to prevent running again and again
# training_dataset = ranking.create_ml_dataset(training_history,training_future,processed_datasets["resource_data"],training_student_features)
# training_dataset.to_csv("data/processed/ml_training.csv", index=False)

# test_dataset = ranking.create_ml_dataset(test_history,test_future,processed_datasets["resource_data"],test_student_features)
# test_dataset.to_csv("data/processed/ml_test.csv", index=False)

# after I run once, just read it from the file rather than running again, specify data type for disability bceause keep getting warning for it
# training_dataset = pd.read_csv("data/processed/ml_training.csv", dtype={'disability':'boolean'})
# test_dataset = pd.read_csv("data/processed/ml_test.csv", dtype={'disability':'boolean'})

# check if it works
# print("target",training_dataset["target"].value_counts())
# print("content score",test_dataset["content_score"].value_counts())

# 5. create X/y for training/testing from the ML dataset
# X_train, y_train, X_test, y_test = ranking.prepare_model_input(training_dataset,test_dataset)

# check it works
# print("X",X_train.head(5))
# print("y",y_test.head(5))
# print("x train shape", X_train.shape)
# print("y train shape", y_train.shape)
# print("x test shape", X_test.shape)
# print("y test shape", y_test.shape)

# 6. create model and fit with training data
# random_forest_model = ranking.create_ML_model(X_train,y_train,'random_forest')
# print('random forest model fit complete')
# ranking.save_model(random_forest_model, 'models/random_forest_model.joblib')

# 7. evaluate model with testing data and print results
# random_forest_model = ranking.load_model('models/random_forest_model.joblib')

# overall classifier results, checking recs across only generated candidates
# random_forest_results = evaluation.evaluate_ML_classifier(random_forest_model,X_test,y_test, 'random_forest','random_forest_classifier_results')
# print("Classifier results",random_forest_results)

# # @20 recommender results, must use interactions because checking recs across all interactions
# (ML_results,ML_summary) = evaluation.evaluate_ML_recommender(random_forest_model,X_test_data=X_test,test_dataset=test_dataset,history_interactions=test_history,future_interactions=test_future,model_type='random_forest',k=20)
# evaluation.save_ML_recommender_results(ML_results,ML_summary,"random_forest_recommender_results","random_forest",saveSummary=True)
# print("Recommender summary results",ML_summary)
# print('sample of ML results',ML_results.head(5))

#8. compare ML models with new sample input, 10k samples
# sample_training_dataset = ranking.sample_training_data(training_dataset,10000)

# test not touched
# X_train_sample, y_train_sample, X_test, y_test = ranking.prepare_model_input(sample_training_dataset,test_dataset)

# create models
# rf_model = ranking.create_ML_model(X_train_sample,y_train_sample,"random_forest")
# ranking.save_model(rf_model, 'models/random_forest_sample_model.joblib')
# svc_model = ranking.create_ML_model(X_train_sample,y_train_sample,"SVC")
# ranking.save_model(svc_model, 'models/SVC_sample_model.joblib')

# rf_model = ranking.load_model('models/random_forest_sample_model.joblib')
# svc_model = ranking.load_model('models/SVC_sample_model.joblib')

# evaluate and save
# overall
# (RF_results,RF_summary) = evaluation.evaluate_ML_recommender(rf_model,X_test_data=X_test,test_dataset=test_dataset,history_interactions=test_history,future_interactions=test_future,model_type='random_forest',k=20)
# evaluation.save_ML_recommender_results(RF_results,RF_summary,"random_forest_sample_results","random_forest",saveSummary=False)

# (SVC_results,SVC_summary) = evaluation.evaluate_ML_recommender(svc_model,X_test_data=X_test,test_dataset=test_dataset,history_interactions=test_history,future_interactions=test_future,model_type='SVC',k=20)
# evaluation.save_ML_recommender_results(SVC_results,SVC_summary,"SVC_sample_results","SVC",saveSummary=False)

# evaluation.save_ML_comparison_results(RF_summary,SVC_summary)

# by classifier (but sampled)
# rf_results = evaluation.evaluate_ML_classifier(rf_model,X_test,y_test, 'random_forest','random_forest_sample_classifier_results')
# svc_results = evaluation.evaluate_ML_classifier(svc_model,X_test,y_test, 'SVC','SVC_sample_classifier_results')

# CROSS VALIDATION EVALUATION
# cutoff_days = [60,120,180]
# test_size = 60

# baseline_cv_results, baseline_cv_summary = evaluation.temporal_cv_baselines(processed_datasets["vle_data"],processed_datasets["resource_data"],cutoff_days,test_size,20)
# print(baseline_cv_summary)

# ML_cv_results, ML_cv_summary = evaluation.temporal_cv_ML_model(processed_datasets,cutoff_days,test_size,"random_forest",20)
# print(ML_cv_summary)