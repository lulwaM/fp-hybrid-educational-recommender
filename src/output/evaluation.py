import pandas as pd
import numpy as np

from src.recommenders import popularity
from src.recommenders import collaborative
from src.recommenders import content_based
from src.recommenders import hybrid

def temporal_split(vle_data,cutoff_day):

    history_vle = vle_data[vle_data["date"]<=cutoff_day].copy()
    future_vle = vle_data[vle_data["date"]> cutoff_day].copy()

    #SAME AS PREPROCESSING STEP for interaction data
    #code inspired by: https://www.kaggle.com/code/veenajoe/veena-vit-project-msc-ds-june-2026#4.2-Aggregating-students-based-on-id

    #grouped so each row has one student and their intercation with 1 module resource, each aggregation column specifies 
    # new columnName =  (column from VLE data used in aggregation, aggregation function)
    #main feature used for collaboartive filtering is the total resource clicks
    history_interactions = history_vle.groupby(["id_student","code_module","code_presentation","id_site","activity_type"]).agg(total_resource_clicks=("sum_click","sum"),first_used=("date","min"),last_used=("date","max")).reset_index()
    future_interactions = future_vle.groupby(["id_student","code_module","code_presentation","id_site","activity_type"]).agg(total_resource_clicks=("sum_click","sum"),first_used=("date","min"),last_used=("date","max")).reset_index()

   #end inspired code

    #to store in 2 separate variables during evaluaton
    return (history_interactions, future_interactions)

#KEY EVALUATION METRICS, input is pandas series of relevant/recommended id of resources
#code copied from: https://giorgi.tech/blog/offline-metrics-for-recommender-systems/
def precision_k(relevant, recommended,k):

    #protect against k error
    if(k<0):
        raise ValueError("Value of k must be greater than zero")

    #protect against divison by zero error if no recommendations
    if(len(recommended[:k])==0):
        return 0

    # measures number of relevant items in k / total number of items in k
    return len(set(relevant).intersection(recommended[:k])) /k

def recall_k(relevant, recommended, k):

    #protect against k error
    if(k<0):
        raise ValueError("Value of k must be greater than zero")

    #protect against divison by zero error if no relevant
    if(len(relevant)==0):
        return 0

    # measures number of relevant items in k / total number of relevant items
    return len(set(relevant).intersection(recommended[:k])) / len(relevant)

#end copied code

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

    #assign random score to each rec, between 0-1 like other recommenders
    scores["random_score"] = np.random.rand(len(scores))

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

    #second is collaborative model
    collaborative_recommended_items = eval_collaborative_recs["id_site"].head(k)
    print("\nCollaborative Model Evaluation:")

    collaborative_precision = precision_k(relevant_items,collaborative_recommended_items,k)
    print("Precision = ",collaborative_precision)

    collaborative_recall = recall_k(relevant_items,collaborative_recommended_items,k)
    print("Recall = ",collaborative_recall)

    #next is content based model
    content_recommended_items = eval_content_recs["id_site"].head(k)
    print("\nContent-based Model Evaluation:")

    content_precision = precision_k(relevant_items,content_recommended_items,k)
    print("Precision = ",content_precision)

    content_recall = recall_k(relevant_items,content_recommended_items,k)
    print("Recall = ",content_recall)

    #next is hybrid fixed weights model 
    hybrid_recommended_items = eval_hybrid_recs["id_site"].head(k)
    print("\nHybrid Model Evaluation:")

    hybrid_precision = precision_k(relevant_items,hybrid_recommended_items,k)
    print("Precision = ",hybrid_precision)

    hybrid_recall = recall_k(relevant_items,hybrid_recommended_items,k)
    print("Recall = ",hybrid_recall)

    #final is random model (for baseline)
    random_recommended_items = eval_random_recs["id_site"].head(k)
    print("\nRandom Model Evaluation:")

    random_precision = precision_k(relevant_items,random_recommended_items,k)
    print("Precision = ",random_precision)

    random_recall = recall_k(relevant_items,random_recommended_items,k)
    print("Recall = ",random_recall)


#USED FOR OVERALL EVALUATION, all eligible students
#note that resource data required for content based filtering
def evaluate_baseline_models_overall(history_interactions, future_interactions,resource_data,k):

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

        #evaluate on history data only
        eval_popularity_recs = popularity.popularity_scores(evaluated_student["id_student"],history_interactions)
        eval_collaborative_recs = collaborative.collaborative_scores(evaluated_student["id_student"],history_interactions)
        #note that resource data is not leaking, its simply listing the resources
        eval_content_recs = content_based.content_scores(evaluated_student["id_student"],resource_data,history_interactions)
        eval_random_recs = random_scores(evaluated_student["id_student"],history_interactions)

        #for hybrid model
        eval_hybrid_recs = hybrid.hybrid_scores(eval_popularity_recs,eval_content_recs,eval_collaborative_recs)

        #first for popularity recommender
        popularity_recommended_items = eval_popularity_recs["id_site"].head(k)
        popularity_precision = precision_k(relevant_items,popularity_recommended_items,k)
        popularity_recall = recall_k(relevant_items,popularity_recommended_items,k)

        #second is collaborative model
        collaborative_recommended_items = eval_collaborative_recs["id_site"].head(k)
        collaborative_precision = precision_k(relevant_items,collaborative_recommended_items,k)
        collaborative_recall = recall_k(relevant_items,collaborative_recommended_items,k)

        #next is content based model
        content_recommended_items = eval_content_recs["id_site"].head(k)
        content_precision = precision_k(relevant_items,content_recommended_items,k)
        content_recall = recall_k(relevant_items,content_recommended_items,k)

        #next is hybrid model
        hybrid_recommended_items = eval_hybrid_recs["id_site"].head(k)
        hybrid_precision = precision_k(relevant_items,hybrid_recommended_items,k)
        hybrid_recall = recall_k(relevant_items,hybrid_recommended_items,k)

        #last is random for comparison
        random_recommended_items = eval_random_recs["id_site"].head(k)
        random_precision = precision_k(relevant_items,random_recommended_items,k)
        random_recall = recall_k(relevant_items,random_recommended_items,k)     

        #append to results for each baseline
        popularity_results.append({"id_student":evaluated_student["id_student"],
                                   "code_module": evaluated_student["code_module"],
                                   "code_presentation": evaluated_student["code_presentation"],
                                   "precision":popularity_precision,
                                   "recall": popularity_recall})
        collaborative_results.append({"id_student":evaluated_student["id_student"],
                                   "code_module": evaluated_student["code_module"],
                                   "code_presentation": evaluated_student["code_presentation"],
                                   "precision":collaborative_precision,
                                   "recall": collaborative_recall})
        content_results.append({"id_student":evaluated_student["id_student"],
                                   "code_module": evaluated_student["code_module"],
                                   "code_presentation": evaluated_student["code_presentation"],
                                   "precision":content_precision,
                                   "recall": content_recall})
        hybrid_results.append({"id_student":evaluated_student["id_student"],
                                   "code_module": evaluated_student["code_module"],
                                   "code_presentation": evaluated_student["code_presentation"],
                                   "precision":hybrid_precision,
                                   "recall": hybrid_recall})
        random_results.append({"id_student":evaluated_student["id_student"],
                                   "code_module": evaluated_student["code_module"],
                                   "code_presentation": evaluated_student["code_presentation"],
                                   "precision":random_precision,
                                   "recall": random_recall})

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
        },
            "collaborative": {
            "precision": collaborative_results_df["precision"].mean(),
            "recall": collaborative_results_df["recall"].mean(),
        },
            "content": {
            "precision": content_results_df["precision"].mean(),
            "recall": content_results_df["recall"].mean(),
        },
        "hybrid":{
            "precision": hybrid_results_df["precision"].mean(),
            "recall": hybrid_results_df["recall"].mean(),   
        },
        "random":{
            "precision": random_results_df["precision"].mean(),
            "recall": random_results_df["recall"].mean(),   
        }
    }

    #organize individual results for easy access
    results = {"popularity":popularity_results_df,"collaborative":collaborative_results_df,
               "content":content_results_df, "hybrid":hybrid_results_df, "random": random_results_df}

    #saving each baseline individual results in csv files for storage,removing index as they hold no meaning
    results["popularity"].to_csv("outputs/popularity_results.csv",index=False)
    results["collaborative"].to_csv("outputs/collaborative_results.csv",index=False)
    results["content"].to_csv("outputs/content_results.csv",index=False)
    results["hybrid"].to_csv("outputs/hybrid_results.csv",index=False)
    results["random"].to_csv("outputs/random_results.csv",index=False)

    #also saving summary in a csv file for storage, but must convert to dataframe first to use the to_csv pandas method
    #note that it is transposed so that the indices represent each model rather than the precision/recall values, easier to understand
    summary_df = pd.DataFrame(summary).T
    summary_df.to_csv("outputs/evaluation_summary.csv")


    return (results,summary)