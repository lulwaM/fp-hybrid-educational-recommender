# standard import for data storage
import pandas as pd


# recommender scores function for popularity-based model
# input: id_student to remove used resources and interaction data to analyze popularity / output: dataframe with columns=id_site and popularity_scores
def popularity_scores(id_student, interaction_data):

    # only provide recommendations within the student's already taken courses, dropped duplicates as the goal is to provide a list of taken courses
    student_courses = interaction_data[interaction_data["id_student"] == id_student][
        ["code_module", "code_presentation"]
    ].drop_duplicates()

    # cold start users, give recommendations from all modules
    if student_courses.empty:
        candidate_interactions = interaction_data
    else:
        # otherwise only give recommendations from already taken courses
        candidate_interactions = interaction_data.merge(
            student_courses, on=["code_module", "code_presentation"], how="inner"
        )

    # creates new series where index=id site and value=sum clicks total (add up clicks for each resource)
    # note: mean is not used like in other projects with ratings because feature is interactions not range 1-5
    scores = candidate_interactions.groupby(
        ["id_site", "code_module", "code_presentation", "activity_type"]
    )["total_resource_clicks"].sum()

    # organize series and rename column to match score being calculated
    # note: reset index moves the id_site to normal column instead of index
    scores = scores.reset_index().rename(
        columns={"total_resource_clicks": "popularity_score"}
    )

    # remove used resources to not recommend
    used_resources = candidate_interactions[
        candidate_interactions["id_student"] == id_student
    ]["id_site"]
    scores = scores[~scores["id_site"].isin(used_resources)]

    # normalize scores between 0 and 1 to be understandable, if condition prevents errors with divisons by 0 or empty
    if len(scores) > 0 and scores["popularity_score"].max() > 0:
        scores["popularity_score"] = (
            scores["popularity_score"] / scores["popularity_score"].max()
        )

    # display highest score first, drop old indices for cleaner display
    scores = scores.sort_values(by="popularity_score", ascending=False).reset_index(
        drop=True
    )

    # return new dataframe of popularity based model recommendations
    return scores
