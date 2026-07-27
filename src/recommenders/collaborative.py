# for data storage
import pandas as pd

# for collaborative and content based filtering
from sklearn.metrics.pairwise import cosine_similarity

# recommender scores 2: collaborative filtering
# input: student id to make targetted recommendations and interactions df to see similar students, output: dataframe with columns=resource id and collaborative filtering score
def collaborative_scores(id_student, interaction_data):

    #only provide recommendations within the student's already taken courses, dropped duplicates as the goal is to provide a list of taken courses
    student_courses = interaction_data[interaction_data["id_student"]==id_student][["code_module","code_presentation"]].drop_duplicates()

    if(student_courses.empty):
        #cold start users, cannot give recommendations because has no interactions with resources
        # empty dataframe, no collabarative score
        return pd.DataFrame(columns=["id_site","code_module","code_presentation","activity_type","collaborative_score"])
    else:
        #otherwise only give recommendations from already taken courses
        candidate_interactions = interaction_data.merge(student_courses, on=["code_module","code_presentation"], how="inner")

    # workflow code inspiration from: https://www.dasca.org/world-of-data-science/article/the-ultimate-guide-to-building-recommendation-systems-in-python

    # new dataframe where columns = resource id, rows = student id, values= clicks
    user_item_matrix = candidate_interactions.pivot_table(
        values="total_resource_clicks", index="id_student", columns="id_site", fill_value=0
    )

    # matrix returned here each value indicates level of similarity to student row
    # code syntax from sci kit learn documenation
    student_similarity = cosine_similarity(user_item_matrix)

    # convert matrix to dataframe for easy access, make sure rows/columns identifiers is student id, values=student cosine similarity
    student_similarity_df = pd.DataFrame(
        student_similarity, index=user_item_matrix.index, columns=user_item_matrix.index
    )

    # obtain values of student similarity
    similar_students = student_similarity_df[id_student]

    # remove similarity to own student (as it is not included in recommendation generation) and sort so most similar students first
    similar_students = similar_students.drop(id_student).sort_values(ascending=False)

    # store resource interactions of similar students
    # : .loc used because we are selecting by row not column (column is resource, row is student)
    similar_students_interactions = user_item_matrix.loc[similar_students.index]

    # end inspired code

    # weighted sum approach for score instead of mean, where score = student similarity x resource usage
    # similar_students_interactions is resource usage, similar students is student similarity
    scores = similar_students_interactions.T.dot(similar_students.values)

    # locate already used resources (student has interactions with) and remove from list of scores (to not recommend)
    used_resources = user_item_matrix.loc[id_student]
    # will be 0 because fill value = 0 when empty interaction
    scores = scores[used_resources == 0]

    # organize scores dataframe in correct format so that it has resource id and corresponding collaborative score
    # note: reset index moves the id_site to normal column instead of index
    scores = scores.reset_index()
    scores.columns = ["id_site", "collaborative_score"]

    #only keep resources that have scores
    scores = scores[scores["collaborative_score"]>0]

    # normalize scores between 0 and 1 to be understandable, if condition prevents errors with divisons by 0 or empty
    if len(scores) > 0 and scores["collaborative_score"].max() > 0:
        scores["collaborative_score"] = (
            scores["collaborative_score"] / scores["collaborative_score"].max()
        )

    # include resource details like code module/presentation in recommender function
    #first get list of resource details with no duplicates
    resource_details = candidate_interactions[["id_site","code_module","code_presentation","activity_type"]].drop_duplicates()

    #merge these details with the final resource scores
    scores = scores.merge(resource_details, on="id_site",how="left")

    #display highest score first, drop old indices for cleaner display
    scores = scores.sort_values(by="collaborative_score", ascending=False).reset_index(drop=True)

    #re-organize columns to be consistent with other recommender outputs
    scores = scores[["id_site","code_module","code_presentation","activity_type","collaborative_score"]]

    return scores