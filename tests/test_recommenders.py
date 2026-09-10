#python testing framework
import unittest

#for data storage
import pandas as pd

#to create sample test data and preprocessed
from tests.sample_data import create_datasets
from src.data_handling import preprocessing, feature_engineering
from src.output import evaluation

#functions to be tested
from src.recommenders import popularity, collaborative, content_based, hybrid, ranking

class TestRecommenders(unittest.TestCase):

    #runs once before tests, used to store datasets to be evaluated
    def setUp(self):
        self.datasets = create_datasets()
        merged_result = preprocessing.merge_datasets(self.datasets)
        cleaned_result = preprocessing.clean_datasets(merged_result)
        typecast_result = preprocessing.typecasting_datasets(cleaned_result)

        self.processed_datasets = preprocessing.processing_datasets(typecast_result)

    #1. popularity recommender scores function

    def test_popularity_value(self):
        #since 101 id site material is most clicked on, it should be first recommendation with no records
        popularity_scores = popularity.popularity_scores(4,self.processed_datasets["interaction_data"])

        self.assertEqual(101,popularity_scores.iloc[0]["id_site"])

    #check that a dataframe is returned
    def test_popularity_dataframe(self):
        #on id student 1
        popularity_scores = popularity.popularity_scores(1,self.processed_datasets["interaction_data"])

        self.assertIsInstance(popularity_scores,pd.DataFrame)

    def test_popularity_cold_start(self):
        #on id student 4 which does not exist in sample data
        popularity_scores = popularity.popularity_scores(4,self.processed_datasets["interaction_data"])

        #returns recommendations from ALL courses
        self.assertIn("AAA",popularity_scores["code_module"].values)
        self.assertIn("BBB",popularity_scores["code_module"].values)

    def test_popularity_taken_course(self):
        #student id 1 only takes course with code module AAA, not BBB, so recommendations only from there
        popularity_scores = popularity.popularity_scores(1,self.processed_datasets["interaction_data"])

        self.assertIn("AAA",popularity_scores["code_module"].values)
        self.assertNotIn("BBB",popularity_scores["code_module"].values)

    #2. collaborative recommender scores function

    def test_collaborative_value(self):
        #student id 3 already used resource id 101 and 103, most similar to student id 2 which used resource id 102 (so should be top recommendation)
        collaborative_scores = collaborative.collaborative_scores(3,self.processed_datasets["interaction_data"])

        self.assertEqual(102,collaborative_scores.iloc[0]["id_site"])

    #check that used resources are excluded from recommendations
    def test_collaborative_used_resources(self):
        #student id 1 used resources with id 101 and 102, so should not exist
        collaborative_scores = collaborative.collaborative_scores(1,self.processed_datasets["interaction_data"])

        self.assertNotIn(101, collaborative_scores["id_site"].values)
        self.assertNotIn(102, collaborative_scores["id_site"].values)

    def test_collaborative_columns(self):
        collaborative_scores = collaborative.collaborative_scores(1,self.processed_datasets["interaction_data"])

        #should contain resource details + score
        self.assertEqual(collaborative_scores.columns.to_list(), ["id_site","code_module","code_presentation","activity_type","collaborative_score"])

    def test_collaborative_resource_data(self):
        collaborative_scores = collaborative.collaborative_scores(1,self.processed_datasets["interaction_data"])

        #no missing values for resource info
        self.assertFalse(collaborative_scores["code_module"].isna().any())
        self.assertFalse(collaborative_scores["code_presentation"].isna().any())
        self.assertFalse(collaborative_scores["activity_type"].isna().any())

    #3. content-based recommender scores function

    def test_content_value(self):
        #student id 1 already used resource 101 and 102, should recommender resoruce 103 because it is closest to used resource 103 (both url activity type)
        content_scores = content_based.content_scores(1,self.processed_datasets["resource_data"],self.processed_datasets["interaction_data"])

        self.assertEqual(103,content_scores.iloc[0]["id_site"])
    
    def test_content_normalized(self):
        content_scores = content_based.content_scores(1,self.processed_datasets["resource_data"],self.processed_datasets["interaction_data"])

        #scores between 0 and 1
        self.assertGreaterEqual(1,content_scores["content_score"].values)
        self.assertLessEqual(0,content_scores["content_score"].values)

    def test_content_sorted(self):
        content_scores = content_based.content_scores(1,self.processed_datasets["resource_data"],self.processed_datasets["interaction_data"])

        #checks if sorted, code copied from: https://stackoverflow.com/questions/17315881/how-can-i-check-if-a-pandas-dataframes-index-is-sorted
        self.assertTrue(content_scores["content_score"].is_monotonic_decreasing)
        #end copied code

    def test_content_empty(self):
        #student id 2 only used resource ID 201 from course BBB which has only this single resource, so no recommendations
        content_scores = content_based.content_scores(2,self.processed_datasets["resource_data"],self.processed_datasets["interaction_data"])

        self.assertTrue(content_scores.empty)

    #4. hybrid recommender function (fixed weight)

    def test_hybrid_value(self):
        #creating new scores to calculate according to hybrid fixed weightage formula, using single recommendation for sample only
        popularity_score = pd.DataFrame({"id_site": [101],
            "code_module": ["AAA"],
            "code_presentation": ["2013J"],
            "activity_type": ["url"],
               "popularity_score":[0.4] })

        content_score = pd.DataFrame({"id_site": [101],
            "code_module": ["AAA"],
            "code_presentation": ["2013J"],
            "activity_type": ["url"],
             "content_score":[0.5]})

        collaborative_score = pd.DataFrame({"id_site": [101],
            "code_module": ["AAA"],
            "code_presentation": ["2013J"],
            "activity_type": ["url"],
              "collaborative_score":[0.3]})

        hybrid_score = hybrid.hybrid_scores(popularity_score,content_score,collaborative_score)
        expected_score = (popularity_score.loc[0,"popularity_score"]*0.4) + (content_score.loc[0,"content_score"]*0.2) + (collaborative_score.loc[0,"collaborative_score"]*0.4)

        #calculated according to default weights
        self.assertEqual(hybrid_score.loc[0,"hybrid_score"],expected_score)

    def test_hybrid_missing(self):

        #creating new scores to calculate according to hybrid fixed weightage formula, using single recommendation for sample only
        popularity_score = pd.DataFrame({"id_site": [101],
            "code_module": ["AAA"],
            "code_presentation": ["2013J"],
            "activity_type": ["url"],
               "popularity_score":[0.4] })

        #content only recommends 102, the rest recommend 101 (will be missing scores)
        content_score = pd.DataFrame({"id_site": [102],
            "code_module": ["AAA"],
            "code_presentation": ["2013J"],
            "activity_type": ["url"],
             "content_score":[0.5]})

        collaborative_score = pd.DataFrame({"id_site": [101],
            "code_module": ["AAA"],
            "code_presentation": ["2013J"],
            "activity_type": ["url"],
              "collaborative_score":[0.3]})

        hybrid_score = hybrid.hybrid_scores(popularity_score,content_score,collaborative_score)

        #works by extracting hybrid record for a specific resource id, then locating the first entry for each score via iloc

        #for 101, will be 0 score for content
        self.assertEqual(hybrid_score[hybrid_score["id_site"]==101].iloc[0]["content_score"],0)

        #for 102, will be 0 score for both populatity and collaborative
        self.assertEqual(hybrid_score[hybrid_score["id_site"]==102].iloc[0]["collaborative_score"],0)
        self.assertEqual(hybrid_score[hybrid_score["id_site"]==102].iloc[0]["popularity_score"],0)

    def test_hybrid_error_greater(self):
        #creating new scores to calculate according to hybrid fixed weightage formula, using single recommendation for sample only
        popularity_score = pd.DataFrame({"id_site": [101],
            "code_module": ["AAA"],
            "code_presentation": ["2013J"],
            "activity_type": ["url"],
               "popularity_score":[0.4] })

        content_score = pd.DataFrame({"id_site": [101],
            "code_module": ["AAA"],
            "code_presentation": ["2013J"],
            "activity_type": ["url"],
             "content_score":[0.5]})

        collaborative_score = pd.DataFrame({"id_site": [101],
            "code_module": ["AAA"],
            "code_presentation": ["2013J"],
            "activity_type": ["url"],
              "collaborative_score":[0.3]})

        #if weights sum > 1, error returned
        #code copied from: https://stackoverflow.com/questions/129507/how-do-you-test-that-a-python-function-throws-an-exception
        with self.assertRaises(ValueError) as context:
            hybrid.hybrid_scores(popularity_score,content_score,collaborative_score,1,2,3)
        #end copied code

    def test_hybrid_error_negative(self):
        #creating new scores to calculate according to hybrid fixed weightage formula, using single recommendation for sample only
        popularity_score = pd.DataFrame({"id_site": [101],
            "code_module": ["AAA"],
            "code_presentation": ["2013J"],
            "activity_type": ["url"],
               "popularity_score":[0.4] })

        content_score = pd.DataFrame({"id_site": [101],
            "code_module": ["AAA"],
            "code_presentation": ["2013J"],
            "activity_type": ["url"],
             "content_score":[0.5]})

        collaborative_score = pd.DataFrame({"id_site": [101],
            "code_module": ["AAA"],
            "code_presentation": ["2013J"],
            "activity_type": ["url"],
              "collaborative_score":[0.3]})

        #if any weight <0, error returned
        #code copied from: https://stackoverflow.com/questions/129507/how-do-you-test-that-a-python-function-throws-an-exception
        with self.assertRaises(ValueError) as context:
            hybrid.hybrid_scores(popularity_score,content_score,collaborative_score,-0.3,0.5,0.8)
        #end copied code

    # now for ranking file functions

    # 5. generate candidates function

    def test_generate_candidates_columns(self):

        # call candidate recommendation on student id 1, who has interactions with course AAA presentation 2013J
        candidates_result = ranking.generate_candidates(id_student=1,interaction_data=self.processed_datasets["interaction_data"],resource_data=self.processed_datasets["resource_data"],code_module="AAA",code_presentation="2013J",k=20)

        # check scores generated
        self.assertIn("popularity_score",candidates_result.columns)
        self.assertIn("collaborative_score",candidates_result.columns)
        self.assertIn("content_score",candidates_result.columns)

    def test_generate_candidates_missing(self):

        # call candidate recommendation on student id 1, who has interactions with course AAA presentation 2013J
        candidates_result = ranking.generate_candidates(id_student=1,interaction_data=self.processed_datasets["interaction_data"],resource_data=self.processed_datasets["resource_data"],code_module="AAA",code_presentation="2013J",k=20)

        # any missing scores must be filled with 0, so no NaN values
        self.assertFalse(candidates_result["popularity_score"].isna().any())
        self.assertFalse(candidates_result["collaborative_score"].isna().any())
        self.assertFalse(candidates_result["content_score"].isna().any())

    # 6. create ML dataset function

    def test_ml_dataset_columns(self):

        # creating simple temporal split to produce history/future interactions for create ml dataset function, choosing cutoff = 1 because it exists for sample data
        history_interactions, future_interactions = evaluation.temporal_split(vle_data=self.processed_datasets["vle_data"],cutoff_day=1)

        # also creating features for function, same cutoff day of 1
        student_features = feature_engineering.create_features(datasets=self.processed_datasets,cutoff_day=1)

        # create ml dataset
        ml_dataset_result = ranking.create_ml_dataset(history_interactions=history_interactions,future_interactions=future_interactions,resource_data=self.processed_datasets["resource_data"],student_features=student_features,k=20)

        # ML target created (most important)
        self.assertIn("target",ml_dataset_result.columns)

        # also student features
        self.assertIn("days_registered",ml_dataset_result.columns)
        self.assertIn("average_score",ml_dataset_result.columns)
        self.assertIn("total_resource_clicks",ml_dataset_result.columns)

    def test_ml_dataset_target(self):

        # creating simple temporal split to produce history/future interactions for create ml dataset function, choosing cutoff = 1 because it exists for sample data
        history_interactions, future_interactions = evaluation.temporal_split(vle_data=self.processed_datasets["vle_data"],cutoff_day=1)

        # also creating features for function, same cutoff day of 1
        student_features = feature_engineering.create_features(datasets=self.processed_datasets,cutoff_day=1)

        # create ml dataset
        ml_dataset_result = ranking.create_ml_dataset(history_interactions=history_interactions,future_interactions=future_interactions,resource_data=self.processed_datasets["resource_data"],student_features=student_features,k=20)

        # check target either 0 or 1 (binary)
        self.assertTrue(ml_dataset_result["target"].isin([0,1]).all())


    # 7. sample training data function

    # very simple test because it simply uses sklearn well-established function

    def test_sample_training_data_size(self):

        # creating simple temporal split to produce history/future interactions for create ml dataset function, choosing cutoff = 1 because it exists for sample data
        history_interactions, future_interactions = evaluation.temporal_split(vle_data=self.processed_datasets["vle_data"],cutoff_day=1)

        # also creating features for function, same cutoff day of 1
        student_features = feature_engineering.create_features(datasets=self.processed_datasets,cutoff_day=1)

        # create ml dataset to SAMPLE
        ml_dataset = ranking.create_ml_dataset(history_interactions=history_interactions,future_interactions=future_interactions,resource_data=self.processed_datasets["resource_data"],student_features=student_features,k=20)

        # take 2 sample rows for training data (note that 2 chosen because there are low number of interactions in sample data)
        sample_result = ranking.sample_training_data(training_dataset=ml_dataset,sample_size=2)

        # check if actually 2 rows taken
        self.assertEqual(len(sample_result),2)


    # 8. prepare model input function

    def test_model_input_columns(self):

        # creating simple temporal split to produce history/future interactions for create ml dataset function, choosing cutoff = 1 because it exists for sample data
        history_interactions, future_interactions = evaluation.temporal_split(vle_data=self.processed_datasets["vle_data"],cutoff_day=1)

        # also creating features for function, same cutoff day of 1
        student_features = feature_engineering.create_features(datasets=self.processed_datasets,cutoff_day=1)

        # create ml dataset to PREPARE MODEL INPUT
        ml_dataset = ranking.create_ml_dataset(history_interactions=history_interactions,future_interactions=future_interactions,resource_data=self.processed_datasets["resource_data"],student_features=student_features,k=20)

        # NOTE: supplying same dataset for training and test because this test does not evaluate uniqueness, simply evaluates structure of data returned
        (X_train, y_train, X_test, y_test) = ranking.prepare_model_input(training_dataset=ml_dataset,test_dataset=ml_dataset)

        # identifiers removed from model input (X for features, both train/test)
        self.assertNotIn("id_student",X_train.columns)
        self.assertNotIn("id_site",X_train.columns)
        self.assertNotIn("id_student",X_test.columns)
        self.assertNotIn("id_site",X_test.columns)

        # useful features should exist
        self.assertIn("gender",X_train.columns)
        self.assertIn("region",X_train.columns)
        self.assertIn("average_score",X_test.columns)
        self.assertIn("popularity_score",X_test.columns)


    def test_model_input_type(self):

        # creating simple temporal split to produce history/future interactions for create ml dataset function, choosing cutoff = 1 because it exists for sample data
        history_interactions, future_interactions = evaluation.temporal_split(vle_data=self.processed_datasets["vle_data"],cutoff_day=1)

        # also creating features for function, same cutoff day of 1
        student_features = feature_engineering.create_features(datasets=self.processed_datasets,cutoff_day=1)

        # create ml dataset to PREPARE MODEL INPUT
        ml_dataset = ranking.create_ml_dataset(history_interactions=history_interactions,future_interactions=future_interactions,resource_data=self.processed_datasets["resource_data"],student_features=student_features,k=20)

        # NOTE: supplying same dataset for training and test because this test does not evaluate uniqueness, simply evaluates structure of data returned
        (X_train, y_train, X_test, y_test) = ranking.prepare_model_input(training_dataset=ml_dataset,test_dataset=ml_dataset)

        # check categorical columns are objects
        self.assertEqual(X_train["gender"].dtype, object) 
        self.assertEqual(X_train["region"].dtype, object) 
        self.assertEqual(X_test["highest_education"].dtype, object) 
        self.assertEqual(X_test["disability"].dtype, object) 

    # 9. create ml model function

    # very simple test because it simply uses sklearn well-established function

    def test_create_model_error(self):

        #if random_forest or SVC not model_type, error returned
        #code copied from: https://stackoverflow.com/questions/129507/how-do-you-test-that-a-python-function-throws-an-exception
        with self.assertRaises(ValueError) as context:
            ranking.create_ML_model(None,None,model_type="error")
        #end copied code

    # 10. predict ml scores function

    # very simple test because it simply uses sklearn well-established function

    def test_predict_scores_error(self):

        #if random_forest or SVC not model_type, error returned
        #code copied from: https://stackoverflow.com/questions/129507/how-do-you-test-that-a-python-function-throws-an-exception
        with self.assertRaises(ValueError) as context:
            ranking.predict_ML_scores(None,None,model_type="error")
        #end copied code

    # will not test ranking saving/loading model because they are from well-established joblib


