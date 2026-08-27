# for data storage
import pandas as pd

# data to be used in creation of features without data leakage
def filter_data_by_cutoff(datasets,cutoff_day):

    # not directly editing variable, makign copy of dictionary via loop (good practice)
    cutoff_datasets = {key:dataframe.copy() for key,dataframe in datasets.items()}

    # 1. for assessment data, cutoff by date submitted
    cutoff_datasets["assessment_data"] = cutoff_datasets["assessment_data"][cutoff_datasets["assessment_data"]["date_submitted"]<=cutoff_day].copy()

    # 2. for vle data, cutoff by date
    cutoff_datasets["vle_data"] = cutoff_datasets["vle_data"][cutoff_datasets["vle_data"]["date"]<=cutoff_day].copy()

    # 3. for student data, cutoff by registration date
    cutoff_datasets["student_data"] = cutoff_datasets["student_data"][cutoff_datasets["student_data"]["date_registration"]<=cutoff_day].copy()

    #for resource data, no cutoff needed (as they simply store info no temporal info)

    # drop data leaking features, will return its features in laters functions up to the cutoff date:

    # drop interaction data as it contains data from future interactions, will derive from vle data
    cutoff_datasets.pop("interaction_data",None)

    # remove registration duration because students could still be enrolled at time of cutoff, alongside final results
    cutoff_datasets["student_data"] = cutoff_datasets["student_data"].drop(columns=["registration_duration","final_result","withdrew","date_unregistration"])

    # similarly for assessment data's average score, could be from assessments not submitted yet
    cutoff_datasets["assessment_data"] = cutoff_datasets["assessment_data"].drop(columns="average_score")

    return cutoff_datasets

def create_assessment_features(assessment_data):

    #NOTE: average scores already created by processing dataset, but will replace according to cutoff data provided in input

    #code inspired by: https://www.kaggle.com/code/veenajoe/veena-vit-project-msc-ds-june-2026#4.2-Aggregating-students-based-on-id

    # create features for EACH STUDENT assessment details
    # new columnName =  (column from assessment data used in aggregation, aggregation function)
    assessment_features = assessment_data.groupby(["id_student","code_module","code_presentation" ]).agg(
        average_score=("score","mean"),min_score=("score","min"),max_score=("score","max"),
        assessments_submitted=("id_assessment","count"), average_submission_date = ("date_submitted","mean"), average_assessment_weight=("weight","mean"), assessment_type_diversity=("assessment_type","nunique")).reset_index()

    # end inspired code

    return assessment_features

def create_vle_features(vle_data):

    #code inspired by: https://www.kaggle.com/code/veenajoe/veena-vit-project-msc-ds-june-2026#4.2-Aggregating-students-based-on-id

    # create features for EACH student's interaction details
    # new columnName =  (column from assessment data used in aggregation, aggregation function)
    vle_features = vle_data.groupby(["id_student","code_module","code_presentation"]).agg(
        total_resource_clicks=("sum_click","sum"), average_resource_clicks = ("sum_click","mean"),first_used=("date","min"),last_used=("date","max"),
        resources_used=("id_site","nunique")).reset_index()
    # end inspired code

    vle_features["interaction_duration"] = vle_features["last_used"] - vle_features["first_used"]

    return vle_features


# returns overall student features with all features combined
def create_features(datasets,cutoff_day=86):

    # historical datasets as per cutoff day
    historical_datasets = filter_data_by_cutoff(datasets,cutoff_day)

    # get features up to the cutoff (prevents data leakage)
    assessment_features = create_assessment_features(historical_datasets["assessment_data"])
    vle_features = create_vle_features(historical_datasets["vle_data"])

    # combine features to student data so each student has its own features to learn from, note that merge is left to keep all student records
    student_features = historical_datasets["student_data"].merge(assessment_features,on=["id_student","code_module","code_presentation"],how="left")
    student_features = student_features.merge(vle_features,on=["id_student","code_module","code_presentation"],how="left")

    # small additional student feature for registration durration (up to the cutoff day)
    student_features["days_registered"] = cutoff_day - student_features["date_registration"]

    # some engineered features naturally may be NaN because they might not have interactions/assessments as per cutoff date, so fill with 0:

    # for assessment data
    student_features[["average_score","min_score","max_score","assessments_submitted","average_submission_date","average_assessment_weight","assessment_type_diversity"]] = student_features[["average_score","min_score","max_score","assessments_submitted","average_submission_date","average_assessment_weight","assessment_type_diversity"]].fillna(0)
    # for vle data 
    student_features[["total_resource_clicks","average_resource_clicks","first_used","last_used","resources_used","interaction_duration"]] = student_features[["total_resource_clicks","average_resource_clicks","first_used","last_used","resources_used","interaction_duration"]].fillna(0)

    # single dataframe with all student features returned
    return student_features




