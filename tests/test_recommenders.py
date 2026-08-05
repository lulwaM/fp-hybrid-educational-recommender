#python testing framework
import unittest

#for data storage
import pandas as pd

#to create sample test data and preprocessed
from tests.sample_data import create_datasets
from src.data_handling import preprocessing

#functions to be tested
from src.recommenders import popularity, collaborative, content_based

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





