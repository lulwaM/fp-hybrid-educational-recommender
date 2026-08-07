# for data storage
import pandas as pd

#takes as input the recommendations produced by each score + fixed weights (automatically less for content due to poor metadata)
def hybrid_scores(popularity_scores,content_scores,collaborative_scores,popularity_weight=0.4,content_weight=0.2,collaborative_weight=0.4):

    if(round(popularity_weight+content_weight+collaborative_weight) != 1):
        raise ValueError("Weights must add up to 1")

    if(popularity_weight < 0 or collaborative_weight < 0 or content_weight <0):
        raise ValueError("Weights cannot be negative")

    # not directly editing variable, makign copy of dataframes (good practice)
    popularity_data = popularity_scores.copy()
    content_data = content_scores.copy()
    collaborative_data = collaborative_scores.copy()

    #create hybrid recommendations by merging with all produced recommendations (outer to prevent loss of any resources)
    hybrid_data = popularity_data.merge(content_data,on=["id_site","code_module","code_presentation","activity_type"],how="outer")
    hybrid_data = hybrid_data.merge(collaborative_data,on=["id_site","code_module","code_presentation","activity_type"],how="outer")

    #since merged on outer, it is possible some resources that were not recommended by other models will be NaN (so fill with 0)
    hybrid_data[["popularity_score","content_score","collaborative_score"]] = hybrid_data[["popularity_score","content_score","collaborative_score"]].fillna(0)

    #calculate according to fixed weights formula
    hybrid_data["hybrid_score"] = (popularity_weight*hybrid_data["popularity_score"]) + (content_weight*hybrid_data["content_score"]) + (collaborative_weight*hybrid_data["collaborative_score"])

    #display highest score first, drop old indices for cleaner display
    hybrid_data = hybrid_data.sort_values(by="hybrid_score", ascending=False).reset_index(drop=True)

    #re-organize columns to be consistent with other recommender outputs
    hybrid_data = hybrid_data[["id_site","code_module","code_presentation","activity_type","popularity_score","content_score","collaborative_score","hybrid_score"]]

    return hybrid_data