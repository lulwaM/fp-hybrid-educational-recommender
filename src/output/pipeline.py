from src.data_handling import preprocessing
from src.recommenders import popularity
from src.recommenders import collaborative
from src.recommenders import content_based
from src.recommenders import hybrid, ranking

from src.output import evaluation, explanation
from src.data_handling import exploration, feature_engineering

from tests import sample_data

import pandas as pd

# ------------------------------- DATA HANDLING AND EXPLORATION -----------------------------------------

# 1. run preprocessing
# does the entire pipeline and returns processed result
def run_preprocessing():

    loaded_datasets = preprocessing.load_datasets()
    merged_datasets = preprocessing.merge_datasets(loaded_datasets)
    cleaned_datasets = preprocessing.clean_datasets(merged_datasets)
    typecast_datasets = preprocessing.typecasting_datasets(cleaned_datasets)

    processed_datasets = preprocessing.processing_datasets(typecast_datasets)

    print("Datasets have been preprocessed and stored.")

    return processed_datasets

# 2. run and save visualizations in images/exploration folder 
def run_visualizations(processed_datasets):

    #save visualizations in images folder
    exploration.explore_student_data(processed_datasets["student_data"])
    exploration.explore_assessment_data(processed_datasets["assessment_data"])
    exploration.explore_vle_data(processed_datasets["vle_data"])
    exploration.explore_resource_data(processed_datasets["resource_data"])
    exploration.explore_interaction_data(processed_datasets["interaction_data"])

    print("Visualizations have been generated and stored.")



# ------------------------------- BASELINE MODELS EVALUATION -----------------------------------------


# 3. run baseline evaluation for single student
# using median for cutoff defualt (same as API)
def run_baseline_evaluation_single(processed_datasets,cutoff_day=86,evaluated_student=6516, k=20):

    #BASELINE RECOMMENDERS EVALUATION for top 20 recommendation
    history_interactions, future_interactions = evaluation.temporal_split(processed_datasets["vle_data"],cutoff_day)

    evaluation.evaluate_baseline_models_single(history_interactions,future_interactions,processed_datasets["resource_data"],evaluated_student,k)

    print("Baseline models have been evaluated for single student.")


# 4. run baseline evaluation for all students (hold out validation)
# using ML testing cutoff for cutoff default (same as ML model, comparable)
def run_baseline_evaluation_holdout(processed_datasets,cutoff_day=150, k=20):

    #BASELINE RECOMMENDERS EVALUATION for top 20 recommendation
    history_interactions, future_interactions = evaluation.temporal_split(processed_datasets["vle_data"],cutoff_day)

    results, summary = evaluation.evaluate_baseline_models_overall(history_interactions,future_interactions,processed_datasets["resource_data"],k, saveData=True)

    print(summary)

    print("Baseline models have been evaluated for all student using hold out validation.")

    return (results, summary)


# 5. run baseline evaluation for all students (cross validation, 3 folds)
def run_baseline_cross_validation(processed_datasets,cutoff_days=[60,120,180], test_size=60,k=20):

    baseline_cv_results, baseline_cv_summary = evaluation.temporal_cv_baselines(processed_datasets["vle_data"],processed_datasets["resource_data"],cutoff_days,test_size,k)
    print(baseline_cv_summary)

    print("Baseline models have been evaluated for all student using 3 fold cross validation.")

    return (baseline_cv_results, baseline_cv_summary)



# ------------------------------- ML MODELS EVALUATION -----------------------------------------


# 6. prepare data for ML model evaluation
# cutoff days same as used in original evaluation
def run_prepare_ML_data(processed_datasets,training_cutoff=86,test_cutoff=150,k=20):

    # 1. cutoff into ML periods
    (training_history_vle,training_future_vle,test_history_vle,test_future_vle) = evaluation.ml_temporal_split(processed_datasets["vle_data"],training_cutoff,test_cutoff)

    # 2. aggregate vle into interaction data so that recommenders can use in producing scores
    training_history = evaluation.aggregate_interaction_data(training_history_vle)
    training_future = evaluation.aggregate_interaction_data(training_future_vle)
    test_history = evaluation.aggregate_interaction_data(test_history_vle)
    test_future = evaluation.aggregate_interaction_data(test_future_vle)

    # 3. create student features for training/test as per cutoffs
    training_student_features = feature_engineering.create_features(processed_datasets, training_cutoff)
    test_student_features = feature_engineering.create_features(processed_datasets, test_cutoff)

    # 4. prepare ML dataset for all student-course rows, save to csv to prevent running again and again
    training_dataset = ranking.create_ml_dataset(training_history,training_future,processed_datasets["resource_data"],training_student_features,k)
    training_dataset.to_csv("data/processed/ML/ml_training.csv", index=False)

    test_dataset = ranking.create_ml_dataset(test_history,test_future,processed_datasets["resource_data"],test_student_features,k)
    test_dataset.to_csv("data/processed/ML/ml_test.csv", index=False)

    print("ML data has been prepared into training and test datasets.")

    return training_dataset, test_dataset


# 7. load saved ML data helper function
def run_load_ML_data():

    # after I run once, just read it from the file rather than running again, specify data type for disability bceause keep getting warning for it
    training_dataset = pd.read_csv("data/processed/ML/ml_training.csv", dtype={'disability':'boolean'})
    test_dataset = pd.read_csv("data/processed/ML/ml_test.csv", dtype={'disability':'boolean'})

    print("ML data of training and test datasets have been loaded.")

    return training_dataset, test_dataset

# 8. train ML model function
def run_train_ML(model_type='random_forest',sample_size=None):

    training_dataset, test_dataset = run_load_ML_data()

    if sample_size is not None:
        training_dataset = ranking.sample_training_data(training_dataset=training_dataset,sample_size=sample_size)


    # create X/y for training/testing from the ML dataset
    X_train, y_train, X_test, y_test = ranking.prepare_model_input(training_dataset,test_dataset)

    # create model and fit with training data
    model = ranking.create_ML_model(X_train,y_train,model_type)

    if sample_size is None:
        ranking.save_model(model, f'models/{model_type}_model.joblib')
    else:
        ranking.save_model(model, f'models/{model_type}_sample_model.joblib')

    print("ML model has been trained and saved.")

    # return it
    return model


# 9. evalaute as recommender for ML function
def run_evaluate_ML_recommender(processed_datasets, file_name ='random_forest_results' ,model_type='random_forest',model=None,model_path='models/random_forest_model.joblib',test_cutoff=150,k=20,saveData=True, saveSummary=True):

    training_dataset, test_dataset = run_load_ML_data()

    # create X/y for training/testing from the ML dataset
    X_train, y_train, X_test, y_test = ranking.prepare_model_input(training_dataset,test_dataset)

    # split according to test cutoff for evaluation, already trained with training cutoff earlier
    # entire VLE data beacuse for testing < test cutoff is ALL history used to generate recs and > test cutoff for labels
    test_history, test_future = evaluation.temporal_split(processed_datasets["vle_data"],test_cutoff)


    if model is None:
        model =ranking.load_model(model_path)

    (ML_results,ML_summary) = evaluation.evaluate_ML_recommender(model,X_test_data=X_test,test_dataset=test_dataset,history_interactions=test_history,future_interactions=test_future,model_type=model_type,k=k,file_name=file_name,saveData=saveData,saveSummary=saveSummary)

    print("ML model has been evaluated as a recommender.")

    return (ML_results,ML_summary)

# 10. save SVC vs Random Forest sampled recommender comparison function
def run_save_ML_comparison_results(RF_summary, SVC_summary):
    evaluation.save_ML_comparison_results(RF_summary,SVC_summary)

    print("Sample ML model comparison has been saved.")

# 11. evaluate as classifier for ML function
# only model info needed, because no need to rank recs/do entire process as recommender does
def run_evaluate_ML_classifier(model=None,model_type='random_forest',model_path='models/random_forest_model.joblib',file_name='random_forest_classifier_results'):

    training_dataset, test_dataset = run_load_ML_data()

    # create X/y for training/testing from the ML dataset
    X_train, y_train, X_test, y_test = ranking.prepare_model_input(training_dataset,test_dataset)

    if model is None:
        model =ranking.load_model(model_path)
        
    ML_results = evaluation.evaluate_ML_classifier(model,X_test,y_test, model_type,file_name)

    print("ML model has been evaluated as a classifier.")

    return (ML_results)

# 12. evaluate ML via cross validation as recommender function
def run_evaluate_ML_cross_validation(processed_datasets,cutoff_days=[60,120,180], test_size=60,k=20,model_type='random_forest'):

    ML_cv_results, ML_cv_summary = evaluation.temporal_cv_ML_model(processed_datasets,cutoff_days,test_size,model_type,k)
    print(ML_cv_summary)

    print("ML model has been evaluated as a recommender with cross validation.")

    return (ML_cv_results, ML_cv_summary)


# ------------------------------- RECOMMENDATIONS GENERATION -----------------------------------------


# 13. create function generate FINAL recommendations with all resource columns + all score columns
# input is the identifier for student/course + preprocessed datasets + model +top k results returned
# NO CREATION OF FILES FOR INTERACTIONS/FEATURES, LIGHTWEIGHT just loaded from memory with no cutoff day because already created
def generate_API_recommendations(datasets, model,id_student=6516,code_module='AAA',code_presentation='2014J', k=20):

    # load from api datasetes (reduced size, not tfull preprocessed datasets)
    history_interactions = datasets["history_interactions"]
    student_features = datasets["student_features"]


    # filter for particular student-course record
    student_course_record = history_interactions[(history_interactions["id_student"]==id_student) & (history_interactions["code_module"]==code_module) & (history_interactions["code_presentation"]==code_presentation)]

    # if does not exist, cannot produce recs
    if student_course_record.empty:
        raise ValueError("Student has no historical interactions with this course")

    # generate candidate recs, no need to do entire pipeline of creating ML training/test because model is ready
    candidates = ranking.generate_candidates(id_student=id_student,interaction_data=history_interactions,resource_data=datasets["resource_data"],code_module=code_module,code_presentation=code_presentation,k=k)

    # if no recommendations, return empty dataframe (no error, just no recs)
    if candidates.empty:
        return pd.DataFrame({})

    # create student's features (for ML model input) and only extract those for student being evaluated (not all)
    student_features = student_features[(student_features["id_student"]==id_student) & (student_features["code_module"]==code_module) & (student_features["code_presentation"]==code_presentation)]

    # attach student features manually instead of calling prepare ML input / create ML dataset function (because no need to create training/test)
    ML_dataset = candidates.merge(student_features, on=["id_student", "code_module", "code_presentation"],how="left")

    # prepare the dataset input into X to feed to model using helper function
    X = ranking.prepare_prediction_input(ML_dataset)

    # predict relevance scores
    ML_dataset["ML_score"] = ranking.predict_ML_scores(model=model,X_test_data=X,model_type="random_forest")

    # rank recommendations by ML score and only get top k
    #ascending false to return highest first
    recommendations = ML_dataset.sort_values(by="ML_score",ascending=False)

    # only top k and drop index because has no meaning
    recommendations = recommendations.head(k).reset_index(drop=True).copy()

    # add interpretability via attaching explanation, note that axis=1 to apply to each recommendation row and specifying weights from function parameters to 1 (so raw baseline scores rather than fixed weight hybrid defaults)
    recommendations["explanation"] = recommendations.apply(explanation.explain_recommendation,axis=1,popularity_weight=1,content_weight=1,collaborative_weight=1)

    # return only relevant columns of identifiers for student/resource/course +scores+explanation
    recommendations = recommendations[["id_student", "id_site","code_module", "code_presentation","popularity_score","collaborative_score","content_score","ML_score","explanation"]]

    print("API recommendations have been generated.")

    return recommendations

#14. create function generate FINAL recommendations with all resource columns + all score columns
# input is the identifier for student/course + preprocessed datasets + model + cutoff day for final prediction cutoff +top k results returned
# GENERAL, re-creates data does not load based on cutoff day
def generate_recommendations(datasets,model,id_student=6516,code_module='AAA',code_presentation='2014J',cutoff_day=86,k=20):

    # using simple temporal split, will just use the median here in function call becuase it was tested already in hold out validation and provides enough data
    # NOTE: not using future interactions because no evaluation done here, just using history to produce recommendations
    history_interactions, future_interactions = evaluation.temporal_split(datasets["vle_data"],cutoff_day)

    # filter for particular student-course record
    student_course_record = history_interactions[(history_interactions["id_student"]==id_student) & (history_interactions["code_module"]==code_module) & (history_interactions["code_presentation"]==code_presentation)]

    # if does not exist, cannot produce recs
    if student_course_record.empty:
        raise ValueError("Student has no historical interactions with this course")

    # generate candidate recs, no need to do entire pipeline of creating ML training/test because model is ready
    candidates = ranking.generate_candidates(id_student=id_student,interaction_data=history_interactions,resource_data=datasets["resource_data"],code_module=code_module,code_presentation=code_presentation,k=k)

    # if no recommendations, return empty dataframe (no error, just no recs)
    if candidates.empty:
        return pd.DataFrame({})

    # create student's features (for ML model input) and only extract those for student being evaluated (not all)
    student_features = feature_engineering.create_features(datasets, cutoff_day=cutoff_day)
    student_features = student_features[(student_features["id_student"]==id_student) & (student_features["code_module"]==code_module) & (student_features["code_presentation"]==code_presentation)]

    # attach student features manually instead of calling prepare ML input / create ML dataset function (because no need to create training/test)
    ML_dataset = candidates.merge(student_features, on=["id_student", "code_module", "code_presentation"],how="left")

    # prepare the dataset input into X to feed to model using helper function
    X = ranking.prepare_prediction_input(ML_dataset)

    # predict relevance scores
    ML_dataset["ML_score"] = ranking.predict_ML_scores(model=model,X_test_data=X,model_type="random_forest")

    # rank recommendations by ML score and only get top k
    #ascending false to return highest first
    recommendations = ML_dataset.sort_values(by="ML_score",ascending=False)

    # only top k and drop index because has no meaning
    recommendations = recommendations.head(k).reset_index(drop=True).copy()

    # add interpretability via attaching explanation, note that axis=1 to apply to each recommendation row and specifying weights from function parameters to 1 (so raw baseline scores rather than fixed weight hybrid defaults)
    recommendations["explanation"] = recommendations.apply(explanation.explain_recommendation,axis=1,popularity_weight=1,content_weight=1,collaborative_weight=1)

    # return only relevant columns of identifiers for student/resource/course +scores+explanation
    recommendations = recommendations[["id_student", "id_site","code_module", "code_presentation","popularity_score","collaborative_score","content_score","ML_score","explanation"]]

    print("Recommendations have been generated.")

    return recommendations

# 15. save sample recommendations function
def save_sample_recommendations(recommendations,file_name):

    recommendations.to_csv(f"outputs/{file_name}.csv",index=False)

    print("Sample recommendations saved.")