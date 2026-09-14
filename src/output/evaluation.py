import pandas as pd
import numpy as np

from src.recommenders import popularity
from src.recommenders import collaborative
from src.recommenders import content_based
from src.recommenders import hybrid
from src.recommenders import ranking

from src.output import explanation

from src.data_handling import feature_engineering

# for ML model evaluation
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

# for saving
from pathlib import Path

# global pandas settings to display dataframe in terminal without truncation
# code copied from: https://builtin.com/data-science/pandas-show-all-columns
pd.set_option("display.max_columns", None)
pd.set_option("display.max_colwidth", None)
# end copied code

def ml_temporal_split(vle_data, training_cutoff,test_cutoff):

    # split 1: for training dataset
    training_history_vle = vle_data[vle_data["date"]<=training_cutoff].copy()
    training_future_vle = vle_data[(vle_data["date"]> training_cutoff) & (vle_data["date"]<=test_cutoff)].copy()

    # split 2: for test dataset
    # NOTE: history available for full recommendations
    test_history_vle = vle_data[vle_data["date"]<=test_cutoff].copy()
    # untouched future (final target)
    test_future_vle = vle_data[vle_data["date"]>test_cutoff].copy()

    return (training_history_vle,training_future_vle,test_history_vle,test_future_vle)

def aggregate_interaction_data(vle_data):
    #SAME AS PREPROCESSING STEP for interaction data
    #code inspired by: https://www.kaggle.com/code/veenajoe/veena-vit-project-msc-ds-june-2026#4.2-Aggregating-students-based-on-id

    #grouped so each row has one student and their intercation with 1 module resource, each aggregation column specifies 
    # new columnName =  (column from VLE data used in aggregation, aggregation function)
    #main feature used for collaboartive filtering is the total resource clicks
    interaction_data = vle_data.groupby(["id_student","code_module","code_presentation","id_site","activity_type"]).agg(total_resource_clicks=("sum_click","sum"),first_used=("date","min"),last_used=("date","max")).reset_index()

    # end inspired code

    interaction_data["click_duration"] = interaction_data["last_used"] - interaction_data["first_used"]

    return interaction_data

def temporal_split(vle_data,cutoff_day):

    history_vle = vle_data[vle_data["date"]<=cutoff_day].copy()
    future_vle = vle_data[vle_data["date"]> cutoff_day].copy()

    # aggregate intereactions to be used in recommenders
    history_interactions = aggregate_interaction_data(history_vle)
    future_interactions = aggregate_interaction_data(future_vle)

    #to store in 2 separate variables during evaluaton
    return (history_interactions, future_interactions)

#KEY EVALUATION METRICS, input is pandas series of relevant/recommended id of resources
#code copied from: https://giorgi.tech/blog/offline-metrics-for-recommender-systems/
def precision_k(relevant, recommended,k):

    #protect against k error
    if(k<=0):
        raise ValueError("Value of k must be greater than zero")

    #protect against divison by zero error if no recommendations
    if(len(recommended[:k])==0):
        return 0

    # measures number of relevant items in k / total number of items in k
    return len(set(relevant).intersection(recommended[:k])) /k

def recall_k(relevant, recommended, k):

    #protect against k error
    if(k<=0):
        raise ValueError("Value of k must be greater than zero")

    #protect against divison by zero error if no relevant
    if(len(relevant)==0):
        return 0

    # measures number of relevant items in k / total number of relevant items
    return len(set(relevant).intersection(recommended[:k])) / len(relevant)

#end copied code

# derived from well known formula, must name it helper to not clash with existing sci-kitlearn f1_score function
def f1_score_helper(precision,recall):
    #protect against divison by zero error in denominator
    if((precision+recall)==0):
        return 0

    return 2 * ((precision*recall)/(precision+recall))

#function for random model to act as baseline to compare model performances
def random_scores(id_student, interaction_data):

    #only provide recommendations within the student's already taken courses, dropped duplicates as the goal is to provide a list of taken courses
    student_courses = interaction_data[interaction_data["id_student"]==id_student][["code_module","code_presentation"]].drop_duplicates()

    if(student_courses.empty):
        #cold start users, give recommendations from all modules
        candidate_interactions = interaction_data
    else:
        #otherwise only give recommendations from already taken courses
        candidate_interactions = interaction_data.merge(student_courses, on=["code_module","code_presentation"], how="inner")


    #possible recommendations for resources with required info, no duplicates
    scores = candidate_interactions[["id_site","code_module","code_presentation","activity_type"]].drop_duplicates().copy()

    #remove used resources to not recommend
    used_resources = candidate_interactions[candidate_interactions["id_student"]==id_student]["id_site"]
    scores = scores[~scores["id_site"].isin(used_resources)]

    # for reproducible random scores, create seed (based on student id so that students with identically same candidate recs not same score)
    #note that default rng is modern version of setting seed as per numpy documentation, not seed method
    random_seed = np.random.default_rng(10+int(id_student))

    #assign random score to each rec, between 0-1 like other recommenders
    scores["random_score"] = random_seed.random(len(scores))

    #display highest score first, drop old indices for cleaner display
    scores = scores.sort_values(by="random_score", ascending=False).reset_index(drop=True)

    return scores

#USED FOR SINGLE STUDENT EVALUATION
def evaluate_baseline_models_single(history_interactions,future_interactions,resource_data,evaluated_student,k):

    #items from interactions, returns series
    history_items = history_interactions[history_interactions["id_student"]==evaluated_student]["id_site"]
    future_items = future_interactions[future_interactions["id_student"]==evaluated_student]["id_site"]

    #removed used resources in history from future items, as goal is to recommend new items, filter series
    relevant_items = future_items[~future_items.isin(history_items)].unique()

    #evaluate on history data only
    eval_popularity_recs = popularity.popularity_scores(evaluated_student,history_interactions)
    eval_collaborative_recs = collaborative.collaborative_scores(evaluated_student,history_interactions)
    #note that resource data is not leaking, its simply listing the resources
    eval_content_recs = content_based.content_scores(evaluated_student,resource_data,history_interactions)
    eval_random_recs = random_scores(evaluated_student,history_interactions)

    #for fixed weights hybrid model
    eval_hybrid_recs = hybrid.hybrid_scores(eval_popularity_recs,eval_content_recs,eval_collaborative_recs)

    #first for popularity recommender
    popularity_recommended_items = eval_popularity_recs["id_site"].head(k)
    print("Popularity Model Evaluation:")

    popularity_precision = precision_k(relevant_items,popularity_recommended_items,k)
    print("Precision = ",popularity_precision)

    popularity_recall = recall_k(relevant_items,popularity_recommended_items,k)
    print("Recall = ",popularity_recall)

    popularity_f1 = f1_score_helper(precision=popularity_precision,recall=popularity_recall)
    print("F1 = ",popularity_f1)

    #second is collaborative model
    collaborative_recommended_items = eval_collaborative_recs["id_site"].head(k)
    print("\nCollaborative Model Evaluation:")

    collaborative_precision = precision_k(relevant_items,collaborative_recommended_items,k)
    print("Precision = ",collaborative_precision)

    collaborative_recall = recall_k(relevant_items,collaborative_recommended_items,k)
    print("Recall = ",collaborative_recall)

    collaborative_f1 = f1_score_helper(precision=collaborative_precision,recall=collaborative_recall)
    print("F1 = ",collaborative_f1)

    #next is content based model
    content_recommended_items = eval_content_recs["id_site"].head(k)
    print("\nContent-based Model Evaluation:")

    content_precision = precision_k(relevant_items,content_recommended_items,k)
    print("Precision = ",content_precision)

    content_recall = recall_k(relevant_items,content_recommended_items,k)
    print("Recall = ",content_recall)

    content_f1 = f1_score_helper(precision=content_precision,recall=content_recall)
    print("F1 = ",content_f1)

    #next is hybrid fixed weights model 
    hybrid_recommended_items = eval_hybrid_recs["id_site"].head(k)
    print("\nHybrid Model Evaluation:")

    hybrid_precision = precision_k(relevant_items,hybrid_recommended_items,k)
    print("Precision = ",hybrid_precision)

    hybrid_recall = recall_k(relevant_items,hybrid_recommended_items,k)
    print("Recall = ",hybrid_recall)

    hybrid_f1 = f1_score_helper(precision=hybrid_precision,recall=hybrid_recall)
    print("F1 = ",hybrid_f1)

    #final is random model (for baseline)
    random_recommended_items = eval_random_recs["id_site"].head(k)
    print("\nRandom Model Evaluation:")

    random_precision = precision_k(relevant_items,random_recommended_items,k)
    print("Precision = ",random_precision)

    random_recall = recall_k(relevant_items,random_recommended_items,k)
    print("Recall = ",random_recall)

    random_f1 = f1_score_helper(precision=random_precision,recall=random_recall)
    print("F1 = ",random_f1)

    #new: also adding hybrid model explanations, default weights, note that axis=1 to apply to each recommendation row 
    eval_hybrid_recs["hybrid_explanation"] = eval_hybrid_recs.apply(explanation.explain_recommendation,axis=1)

    #finally printing all top-K recommendations with all scores + explanations
    print(eval_hybrid_recs.head(k))

#USED FOR OVERALL EVALUATION, all eligible students
#note that resource data required for content based filtering
def evaluate_baseline_models_overall(history_interactions, future_interactions,resource_data,k, saveData=True, saveSummary=True):

    #get dataframe of non-duplicate students in historica and future interactions separately
    historical_students = history_interactions[["id_student","code_module","code_presentation"]].drop_duplicates()
    future_students = future_interactions[["id_student","code_module","code_presentation"]].drop_duplicates()

    # only evaluate students with future interactions to evaluate via inner join
    evaluated_students = historical_students.merge(future_students, on=["id_student","code_module","code_presentation"],how="inner")

    #for 20 students check
    # evaluated_students = evaluated_students.sample(100)

    popularity_results = []
    collaborative_results = []
    content_results = []
    hybrid_results = []
    random_results = []

    #iterate over each evaluated student (values are index,row where row is each evaluated student record with keys id/coures module/etc)
    #iteration code inspired by: https://stackoverflow.com/questions/16476924/how-can-i-iterate-over-rows-in-a-pandas-dataframe
    for index,evaluated_student in evaluated_students.iterrows():

        #only print after every 100 loops, prevents heavy console logging
        if(index % 100 == 0):
            print("evaluating:",index)

    #end inspired code
        #items from interactions, returns series
        history_items = history_interactions.loc[(history_interactions["id_student"]==evaluated_student["id_student"]) & 
                                                 (history_interactions["code_module"]==evaluated_student["code_module"]) & 
                                                 (history_interactions["code_presentation"]==evaluated_student["code_presentation"]),"id_site"]

        future_items = future_interactions.loc[(future_interactions["id_student"]==evaluated_student["id_student"]) & 
                                                 (future_interactions["code_module"]==evaluated_student["code_module"]) & 
                                                 (future_interactions["code_presentation"]==evaluated_student["code_presentation"]),"id_site"]

        #removed used resources in history from future items, as goal is to recommend new items, filter series
        relevant_items = future_items[~future_items.isin(history_items)].unique()

        #prevents divison by 0 error
        if(len(relevant_items) == 0):
            #student has used all future resource recommendation, no scores so skip
            continue

        #evaluate on history data only, make sure from same course being evaluated only (similar to what was done for relevant items extraction and ML model generating candidates)
        eval_popularity_recs = popularity.popularity_scores(evaluated_student["id_student"],history_interactions)
        eval_popularity_recs = eval_popularity_recs[(eval_popularity_recs["code_module"]==evaluated_student["code_module"]) & (eval_popularity_recs["code_presentation"]==evaluated_student["code_presentation"])].copy()
        eval_collaborative_recs = collaborative.collaborative_scores(evaluated_student["id_student"],history_interactions)
        eval_collaborative_recs = eval_collaborative_recs[(eval_collaborative_recs["code_module"]==evaluated_student["code_module"]) & (eval_collaborative_recs["code_presentation"]==evaluated_student["code_presentation"])].copy()
        #note that resource data is not leaking, its simply listing the resources
        eval_content_recs = content_based.content_scores(evaluated_student["id_student"],resource_data,history_interactions)
        eval_content_recs = eval_content_recs[(eval_content_recs["code_module"]==evaluated_student["code_module"]) & (eval_content_recs["code_presentation"]==evaluated_student["code_presentation"])].copy()
        eval_random_recs = random_scores(evaluated_student["id_student"],history_interactions)
        eval_random_recs = eval_random_recs[(eval_random_recs["code_module"]==evaluated_student["code_module"]) & (eval_random_recs["code_presentation"]==evaluated_student["code_presentation"])].copy()

        #for hybrid model
        eval_hybrid_recs = hybrid.hybrid_scores(eval_popularity_recs,eval_content_recs,eval_collaborative_recs)
        eval_hybrid_recs = eval_hybrid_recs[(eval_hybrid_recs["code_module"]==evaluated_student["code_module"]) & (eval_hybrid_recs["code_presentation"]==evaluated_student["code_presentation"])].copy()

        #first for popularity recommender
        popularity_recommended_items = eval_popularity_recs["id_site"].head(k)
        popularity_precision = precision_k(relevant_items,popularity_recommended_items,k)
        popularity_recall = recall_k(relevant_items,popularity_recommended_items,k)
        popularity_f1 = f1_score_helper(precision=popularity_precision,recall=popularity_recall)

        #second is collaborative model
        collaborative_recommended_items = eval_collaborative_recs["id_site"].head(k)
        collaborative_precision = precision_k(relevant_items,collaborative_recommended_items,k)
        collaborative_recall = recall_k(relevant_items,collaborative_recommended_items,k)
        collaborative_f1 = f1_score_helper(precision=collaborative_precision,recall=collaborative_recall)

        #next is content based model
        content_recommended_items = eval_content_recs["id_site"].head(k)
        content_precision = precision_k(relevant_items,content_recommended_items,k)
        content_recall = recall_k(relevant_items,content_recommended_items,k)
        content_f1 = f1_score_helper(precision=content_precision,recall=content_recall)

        #next is hybrid model
        hybrid_recommended_items = eval_hybrid_recs["id_site"].head(k)
        hybrid_precision = precision_k(relevant_items,hybrid_recommended_items,k)
        hybrid_recall = recall_k(relevant_items,hybrid_recommended_items,k)
        hybrid_f1 = f1_score_helper(precision=hybrid_precision,recall=hybrid_recall)

        #last is random for comparison
        random_recommended_items = eval_random_recs["id_site"].head(k)
        random_precision = precision_k(relevant_items,random_recommended_items,k)
        random_recall = recall_k(relevant_items,random_recommended_items,k)     
        random_f1 = f1_score_helper(precision=random_precision,recall=random_recall)

        #append to results for each baseline
        popularity_results.append({"id_student":evaluated_student["id_student"],
                                   "code_module": evaluated_student["code_module"],
                                   "code_presentation": evaluated_student["code_presentation"],
                                   "precision":popularity_precision,
                                   "recall": popularity_recall,
                                   "f1":popularity_f1})
        collaborative_results.append({"id_student":evaluated_student["id_student"],
                                   "code_module": evaluated_student["code_module"],
                                   "code_presentation": evaluated_student["code_presentation"],
                                   "precision":collaborative_precision,
                                   "recall": collaborative_recall,
                                   "f1":collaborative_f1})
        content_results.append({"id_student":evaluated_student["id_student"],
                                   "code_module": evaluated_student["code_module"],
                                   "code_presentation": evaluated_student["code_presentation"],
                                   "precision":content_precision,
                                   "recall": content_recall,
                                   "f1":content_f1})
        hybrid_results.append({"id_student":evaluated_student["id_student"],
                                   "code_module": evaluated_student["code_module"],
                                   "code_presentation": evaluated_student["code_presentation"],
                                   "precision":hybrid_precision,
                                   "recall": hybrid_recall,
                                   "f1":hybrid_f1})
        random_results.append({"id_student":evaluated_student["id_student"],
                                   "code_module": evaluated_student["code_module"],
                                   "code_presentation": evaluated_student["code_presentation"],
                                   "precision":random_precision,
                                   "recall": random_recall,
                                   "f1":random_f1})

    #convert all results to df for vectorized aggregation operation (mean)
    popularity_results_df = pd.DataFrame(popularity_results)
    collaborative_results_df = pd.DataFrame(collaborative_results)
    content_results_df = pd.DataFrame(content_results)
    hybrid_results_df = pd.DataFrame(hybrid_results)
    random_results_df = pd.DataFrame(random_results)

    #create summary of results for each baseline
    summary = {
        "popularity": {
            "precision": popularity_results_df["precision"].mean(),
            "recall": popularity_results_df["recall"].mean(),
            "f1": popularity_results_df["f1"].mean(),
        },
            "collaborative": {
            "precision": collaborative_results_df["precision"].mean(),
            "recall": collaborative_results_df["recall"].mean(),
            "f1": collaborative_results_df["f1"].mean(),
        },
            "content": {
            "precision": content_results_df["precision"].mean(),
            "recall": content_results_df["recall"].mean(),
            "f1": content_results_df["f1"].mean(),
        },
        "hybrid":{
            "precision": hybrid_results_df["precision"].mean(),
            "recall": hybrid_results_df["recall"].mean(),   
            "f1": hybrid_results_df["f1"].mean(),
        },
        "random":{
            "precision": random_results_df["precision"].mean(),
            "recall": random_results_df["recall"].mean(),   
            "f1": random_results_df["f1"].mean(),
        }
    }

    #organize individual results for easy access
    results = {"popularity":popularity_results_df,"collaborative":collaborative_results_df,
               "content":content_results_df, "hybrid":hybrid_results_df, "random": random_results_df}

    if saveData == True:
        #saving each baseline individual results in csv files for storage,removing index as they hold no meaning
        results["popularity"].to_csv("outputs/baselines/popularity_results.csv",index=False)
        results["collaborative"].to_csv("outputs/baselines/collaborative_results.csv",index=False)
        results["content"].to_csv("outputs/baselines/content_results.csv",index=False)
        results["hybrid"].to_csv("outputs/baselines/hybrid_results.csv",index=False)
        results["random"].to_csv("outputs/baselines/random_results.csv",index=False)

    if saveSummary == True:
    #also saving summary in a csv file for storage, but must convert to dataframe first to use the to_csv pandas method

        # using helper function, iterate over each recommender and add to file
        for recommender, metrics in summary.items():

            save_recommender_summary(summary=metrics,recommender=recommender)   

    return (results,summary)

# OVERALL CLASSIFIER PERFORMANCE, not top 20. it checks if task learned, if the model correctly classified used candidates (within candidates only, not all interactions)
# checking against y test candidates rather than full interactions
def evaluate_ML_classifier(model, X_test_data, y_test_data,model_type,file_name):

    if model_type == 'random_forest':
        # relevance score, get as single list for all recs
        relevance_scores = model.predict_proba(X_test_data)[:,1]
        predictions = model.predict(X_test_data)

    elif model_type == 'SVC':
        relevance_scores = model.decision_function(X_test_data)
        predictions = (relevance_scores>0).astype(int)
    else:
        raise ValueError("Model type must be random_forest or SVC")

    # all using sci kit learn's actual functions, not own helper functions
    results = {
        "model": model_type,
        'classifier_precision': precision_score(y_true=y_test_data,y_pred=predictions),
        'classifier_recall': recall_score(y_true=y_test_data,y_pred=predictions),
        'classifier_f1': f1_score(y_true=y_test_data,y_pred=predictions),
        # uses probaiblities rather than binary predictions
        'classifier_roc-auc': roc_auc_score(y_true=y_test_data,y_score=relevance_scores)
    }

    # save to csv after converting to dataframe (within list to prevent conversion to series)
    results_df = pd.DataFrame([results])
    results_df.to_csv(f"outputs/ML_classifier/{file_name}.csv",index=False)

    return results

# COMPARABLE RECOMMENDER PERFORMANCE TO BASELINES, out of the candidate records take the top 20 and check whether student used in ENTIRE future interactions
# checking again interactions rather than y test candidates
# as such, requires history/future interactions and also test_dataset for the student-course recommendation records
def evaluate_ML_recommender(model, X_test_data, test_dataset,history_interactions,future_interactions, model_type='random_forest',k=20, file_name='random_forest_recommender_results',saveData=True,saveSummary=True):

    # extract needed info for recommendations from dataset
    recommendation_data = test_dataset[["id_student","code_module","code_presentation","id_site"]].copy()

    # calculate ML scorefrom X of test dataset
    recommendation_data["ML_score"] = ranking.predict_ML_scores(model, X_test_data,model_type)

    # same process as baseline models evaluation:

    # get unique student-course records for evaluated students, reset index for correct index tracking in loop
    evaluated_students = recommendation_data[["id_student","code_module","code_presentation"]].drop_duplicates().reset_index(drop=True)

    # will store each individual score here
    ML_results = []

    #iterate over each evaluated student (values are index,row where row is each evaluated student record with keys id/coures module/etc)
    #iteration code inspired by: https://stackoverflow.com/questions/16476924/how-can-i-iterate-over-rows-in-a-pandas-dataframe
    for index,evaluated_student in evaluated_students.iterrows():

        #only print after every 100 loops, prevents heavy console logging
        if(index % 100 == 0):
            print("evaluating:",index)

    #end inspired code
        #items from interactions, returns series
        history_items = history_interactions.loc[(history_interactions["id_student"]==evaluated_student["id_student"]) & 
                                                 (history_interactions["code_module"]==evaluated_student["code_module"]) & 
                                                 (history_interactions["code_presentation"]==evaluated_student["code_presentation"]),"id_site"]

        future_items = future_interactions.loc[(future_interactions["id_student"]==evaluated_student["id_student"]) & 
                                                 (future_interactions["code_module"]==evaluated_student["code_module"]) & 
                                                 (future_interactions["code_presentation"]==evaluated_student["code_presentation"]),"id_site"]

        #removed used resources in history from future items, as goal is to recommend new items, filter series
        relevant_items = future_items[~future_items.isin(history_items)].unique()

        #prevents divison by 0 error
        if(len(relevant_items) == 0):
            #student has used all future resource recommendation, no scores so skip
            continue

        # NEW: generate recommendations by ranking by highest ML score, first extract specific student's candidates
        eval_ML_recs = recommendation_data[(recommendation_data["id_student"]==evaluated_student["id_student"]) & 
                                                 (recommendation_data["code_module"]==evaluated_student["code_module"]) & 
                                                 (recommendation_data["code_presentation"]==evaluated_student["code_presentation"])].copy()

        # do ranking, ascending false to return highest first
        eval_ML_recs = eval_ML_recs.sort_values(by="ML_score",ascending=False)

        # only top k
        ML_recommended_items = eval_ML_recs["id_site"].head(k)

        # generate metric results
        ML_precision = precision_k(relevant_items,ML_recommended_items,k)
        ML_recall = recall_k(relevant_items,ML_recommended_items,k)     
        ML_f1 = f1_score_helper(precision=ML_precision,recall=ML_recall)

        #append to results for each baseline
        ML_results.append({"id_student":evaluated_student["id_student"],
                                   "code_module": evaluated_student["code_module"],
                                   "code_presentation": evaluated_student["code_presentation"],
                                   "precision":ML_precision,
                                   "recall": ML_recall,
                                   "f1":ML_f1})

    #convert results to df for vectorized aggregation operation (mean)
    ML_results_df = pd.DataFrame(ML_results)

    #create summary of results for ML model, single object because not multiple models data stored
    ML_summary = {
            "precision": ML_results_df["precision"].mean(),
            "recall": ML_results_df["recall"].mean(),
            "f1": ML_results_df["f1"].mean(),
        }

    if saveData == True:

        ML_results_df.to_csv(f"outputs/ML_recommender/{file_name}.csv",index=False)

    if saveSummary == True:
        save_recommender_summary(ML_summary,model_type)

    return (ML_results_df,ML_summary)

# add to summary file
def save_recommender_summary(summary,recommender):

    # code to check if file exists copied from: https://mimo.org/tutorials/python/how-to-check-if-a-file-exists-in-python
    summary_path = Path("outputs/recommender_summary.csv")
    # end copied code

    if summary_path.exists():
        # read existing evaluation summary from baseline models, index is first column where model names held
        recommender_summary_df = pd.read_csv('outputs/recommender_summary.csv', index_col=0)

    else:
        # create empty with required columns
        recommender_summary_df = pd.DataFrame(columns=['precision','recall','f1'])


    # create new row with model name and column values in exact same order
    recommender_summary_df.loc[recommender] = [summary["precision"],summary["recall"],summary["f1"]]

    # overwrite old evaluation summary with updated one
    recommender_summary_df.to_csv(summary_path)

def save_ML_comparison_results(RF_summary, SVC_summary):
    comparison = {
        "random_forest": {
            "precision": RF_summary["precision"],
            "recall": RF_summary["recall"],
            "f1": RF_summary["f1"]
        },
        "SVC": {
            "precision": SVC_summary["precision"],
            "recall": SVC_summary["recall"],
            "f1": SVC_summary["f1"]
        },
    }


    #note that it is transposed so that the indices represent each model rather than the precision/recall values, easier to understand
    comparison_df = pd.DataFrame(comparison).T
    # index included because it represents ML model type
    comparison_df.to_csv("outputs/ml_recommender_sample_comparison.csv")

# for cross validation evaluation, basically follows the logic of sklearn's TimeSeriesSplit but manually (https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html)

# instead of passing history/future like in earlier evaluation of hold out, passing entire VLE to do separate split for each fold
def temporal_cv_baselines(vle_data,resource_data,cutoff_days,test_size,k=20):

    results = []

    # code structure of iteration inspired by: https://blog.devgenius.io/how-to-implement-time-series-cross-validation-in-python-1ce8aa6f9a6c
    # keeping track of index and value via enumerate, setting index to start from 1 instead of 0 because fold 1/2/3 not fold 0/1/2
    for fold, cutoff_day in enumerate(cutoff_days,start=1):

        print("fold",fold)

        # final day for current fold
        test_end = cutoff_day + test_size

        # restrict interactions to be max= history + test size
        fold_vle = vle_data[vle_data["date"]<=test_end].copy()

        # split fold data into history/future interactions for recommenders as per cutoff day
        history_interactions, future_interactions = temporal_split(fold_vle,cutoff_day)

        # run evaluation, not saving results because saving for fold later
        fold_results, fold_summary = evaluate_baseline_models_overall(history_interactions,future_interactions,resource_data,k, False, False)

        # storing results from current fold, note that summary is object of objects where outer key is recommender type and inner key evaluation metric
        for recommender, metrics in fold_summary.items():
            results.append({'fold':fold,
                            'cutoff_day': cutoff_day,
                            "test_end": test_end,
                            'recommender':recommender,
                            'precision':metrics["precision"],
                            "recall":metrics["recall"],
                            "f1":metrics["f1"]})

    # end inspired code

    # once populated, convert to dataframe
    results_df = pd.DataFrame(results)

    # save overall results for all folds, removing idnex no meaning
    results_df.to_csv("outputs/cross_validation/baselines_cv_results.csv", index=False)

    # get summary by averaging results across all folds, grouping by each recommender folds, extracting the metrics, and calling mean
    average_results = results_df.groupby("recommender")[["precision","recall","f1"]].mean()

    # get std
    std_results = results_df.groupby("recommender")[["precision","recall","f1"]].std()

    # rename columns for clarity
    average_results.columns = ["precision_mean","recall_mean","f1_mean"]
    std_results.columns = ["precision_std","recall_std","f1_std"]

    # combine horizontally via concat
    summary_results = pd.concat([average_results,std_results],axis=1)

    # save summary
    summary_results.to_csv("outputs/cross_validation/baselines_cv_summary.csv")

    # return individual and overall results
    return results_df, summary_results

# similar structure to baseline models with additional logic for training/testing cutoff
# also note that entire datasets passed as parameter because needed in creating student features for learning model
def temporal_cv_ML_model(datasets,cutoff_days,test_size,model_type,k=20):

    results = []

    # code structure of iteration inspired by: https://blog.devgenius.io/how-to-implement-time-series-cross-validation-in-python-1ce8aa6f9a6c
    # keeping track of index and value via enumerate, setting index to start from 1 instead of 0 because fold 1/2/3 not fold 0/1/2
    # DIFFERENCE FOR ML: each cutoff day refers to test cutoff, where test end > days > test cutofff are evaluated for predictions
    for fold, test_cutoff in enumerate(cutoff_days,start=1):

        print("fold",fold)

        # final day for current fold
        test_end = test_cutoff + test_size

        # must calculate training cutoff, where split history (before test cutoff) into 2 for history learning and future labels. BUT test cutoff and above until test end reserved for predictions (no leakage)
        # the future will have same size as test size with the rest starting from earliest VLE date as history learning, design choice
        # so training structure will be where train history is everything < training cutoff and train future is training cutoff < train future < test cutoff
        training_cutoff = test_cutoff - test_size

        # restrict interactions to be max= test cutoff + test size
        fold_vle = datasets["vle_data"][datasets["vle_data"]["date"]<=test_end].copy()

        # now follow exact same process of pipelien evaluation

        #1. this split for history/future already done by function call, where test history is everything < test cutoff and test future is test cutoff < test future <= test end by using fold vle
        (training_history_vle,training_future_vle,test_history_vle,test_future_vle) = ml_temporal_split(fold_vle,training_cutoff,test_cutoff)

        #2. aggregate vle into interaction data so that recommenders can use in producing scores
        training_history = aggregate_interaction_data(training_history_vle)
        training_future = aggregate_interaction_data(training_future_vle)
        test_history = aggregate_interaction_data(test_history_vle)
        test_future = aggregate_interaction_data(test_future_vle)

        #3. create student features for training/test as per cutoffs used for no data leakage
        training_student_features = feature_engineering.create_features(datasets, training_cutoff)
        test_student_features = feature_engineering.create_features(datasets, test_cutoff)

        #4. prepare ML dataset for all student-course rows, save to csv to prevent running again and again if already present
        # code to check if file exists copied from: https://mimo.org/tutorials/python/how-to-check-if-a-file-exists-in-python
        training_dataset_path = Path(f"data/processed/cross_validation/ml_training_fold_{fold}.csv")
        test_dataset_path = Path(f"data/processed/cross_validation/ml_test_fold_{fold}.csv")
        # end copied code

        if training_dataset_path.exists():
            training_dataset = pd.read_csv(training_dataset_path, dtype={'disability':'boolean'})
        else:
            training_dataset = ranking.create_ml_dataset(training_history,training_future,datasets["resource_data"],training_student_features,k)
            training_dataset.to_csv(training_dataset_path, index=False)

        if test_dataset_path.exists():
            test_dataset = pd.read_csv(test_dataset_path, dtype={'disability':'boolean'})
        else:
            test_dataset = ranking.create_ml_dataset(test_history,test_future,datasets["resource_data"],test_student_features,k)
            test_dataset.to_csv(test_dataset_path, index=False)

        # 5. create X/y for training/testing from the ML dataset
        X_train, y_train, X_test, y_test = ranking.prepare_model_input(training_dataset,test_dataset)

        # 6. create model and fit with training data. no need to save because running this is not very time consuming, only the dataset really
        ML_model = ranking.create_ML_model(X_train,y_train,model_type)

        # 7. evaluate model with testing data (as recommender, not classifier, also not saving yet will save later
        # # @20 recommender results, must use interactions because checking recs across all interactions
        (ML_results,ML_summary) = evaluate_ML_recommender(ML_model,X_test_data=X_test,test_dataset=test_dataset,history_interactions=test_history,future_interactions=test_future,model_type=model_type,k=k,saveData=False,saveSummary=False)

        results.append({'fold':fold,
                        'test_cutoff': test_cutoff,
                        'training_cutoff': training_cutoff,
                        "test_end": test_end,
                        'recommender':model_type,
                        'precision':ML_summary["precision"],
                        "recall":ML_summary["recall"],
                        "f1":ML_summary["f1"]})

    # end inspired code

    # once populated, convert to dataframe
    results_df = pd.DataFrame(results)

    # save overall results for all folds, removing idnex no meaning
    results_df.to_csv(f"outputs/cross_validation/ML_cv_{model_type}_results.csv", index=False)

    # get summary by averaging, no need to group because all folds belong to single ML model recommender
    # must transpose after converting to dataframe because it is series to display nicely
    # code copied and adapted from: https://stackoverflow.com/questions/43517338/transpose-a-pandas-series
    average_results = results_df[["precision","recall","f1"]].mean().to_frame().T


    std_results = results_df[["precision","recall","f1"]].std().to_frame().T
    # end adapted code

    # rename columns for clarity
    average_results.columns = ["precision_mean","recall_mean","f1_mean"]
    std_results.columns = ["precision_std","recall_std","f1_std"]

    # use concat to combine horizontally in one row
    summary_results = pd.concat([average_results,std_results],axis=1)


    # save summary
    summary_results.to_csv(f"outputs/cross_validation/ML_cv_{model_type}_summary.csv",index=False)

    # return individual and overall results
    return results_df, summary_results
