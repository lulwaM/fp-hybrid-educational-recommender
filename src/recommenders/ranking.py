#imports

# ML ranking model will use ALL recommenders
from src.recommenders import popularity, collaborative, content_based

# generate possible recommendations for a student-course record, k = number of candidate recommendations to generate
def generate_candidates(id_student,interaction_data,resource_data, k=20, popularity_recs=None, collaborative_recs=None, content_recs=None):

    if(popularity_recs == None or collaborative_recs == None or content_recs ==None ):
        # generate scores since not generated

        popularity_recs = popularity.popularity_scores(id_student,interaction_data)
        collaborative_recs = collaborative.collaborative_scores(id_student,interaction_data)
        content_recs = content_based.content_scores(id_student,resource_data,interaction_data)


    # limit recommendations to top k 
    popularity_recs = popularity_recs.head(k)
    collaborative_recs = collaborative_recs.head(k)
    content_recs = content_recs.head(k)


    # create candidates with resource details + each score (using details from populariyt scores then extracting scores from each)
    candidates = popularity_recs[["id_site","code_module","code_presentation","activity_type","popularity_score"]].copy()
    candidates = candidates.merge(collaborative_recs[["id_site","collaborative_score"]], on="id_site",how="outer")
    candidates = candidates.merge(content_recs[["id_site","content_score"]], on="id_site",how="outer")

    # since resource details comes from popularity only, may be missing from recommendations produced only by other scores (fix: use resource data)
    candidates = candidates.drop(columns=["code_module","code_presentation","activity_type"])
    candidates = candidates.merge(resource_data[["code_module","code_presentation","activity_type"]],on="id_site",how="left")

    # outer joins may lead to NaN for some scores since not all recommenders recommend same resources, so fill with 0
    candidates[["popularity_score","collaborative_score","content_score"]] = candidates[["popularity_score","collaborative_score","content_score"]].fillna(0)


    return candidates





    

