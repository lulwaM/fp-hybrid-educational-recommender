# for data storage
import pandas as pd

# recommender scores 1: popularity based
# input: none (becuase it is general for all students, not specific to student), output: dataframe with columns=id_site and popularity_scores
def popularity_scores(interaction_data):

    # creates new series where index=id site and value=sum clicks total (add up clicks for each resource)
    # note: mean is not used like in other projects with ratings because feature is interactions not range 1-5
    scores = interaction_data.groupby(["id_site","code_module","code_presentation","activity_type"])["total_resource_clicks"].sum()

    # organize series and rename column to match score being calculated
    # note: reset index moves the id_site to normal column instead of index
    scores = scores.reset_index().rename(columns={"total_resource_clicks": "popularity_score"})

    # normalize score between 0-1 by dividing by max (makes score more understandable and comparable on same scale)
    scores["popularity_score"] = (
        scores["popularity_score"] / scores["popularity_score"].max()
    )

    #display highest score first, drop old indices for cleaner display
    scores = scores.sort_values(by="popularity_score", ascending=False).reset_index(drop=True)

    return scores