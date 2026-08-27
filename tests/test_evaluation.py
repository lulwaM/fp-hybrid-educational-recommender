#python testing framework
import unittest

#for data storage
import pandas as pd

#to create sample test data and preprocessed
from tests.sample_data import create_datasets
from src.data_handling import preprocessing

#functions to be tested
from src.output import evaluation

class TestEvaluation(unittest.TestCase):

    #runs once before tests, used to store datasets to be evaluated
    def setUp(self):
        self.datasets = create_datasets()
        merged_result = preprocessing.merge_datasets(self.datasets)
        cleaned_result = preprocessing.clean_datasets(merged_result)
        typecast_result = preprocessing.typecasting_datasets(cleaned_result)

        self.processed_datasets = preprocessing.processing_datasets(typecast_result)

    #NOTE: not testing evaluation of baseline models functions as they have already been tested in test_recommenders.py file

    #1. temporal split function

    def test_temporal_split_cutoff(self):
        #setting cutoff day at 22 (as per sample data) so that history interactions less than that and future interactions more than that + both exist
        history_interactions, future_interactions = evaluation.temporal_split(self.processed_datasets["vle_data"],22)

        #note that date split into 2 new variables (last/first used), each correspond to cutoff date
        self.assertTrue((history_interactions["last_used"]<=22).all())
        self.assertTrue((future_interactions["first_used"]>22).all())


    def test_temporal_split_columns(self):
        #setting cutoff day at 22 (as per sample data) so that history interactions less than that and future interactions more than that + both exist
        history_interactions, future_interactions = evaluation.temporal_split(self.processed_datasets["vle_data"],22)

        #all new variables created via aggregation + existing student/resource info (note that the order remains same as creation to prevent test fail)
        self.assertEqual(["id_student","code_module","code_presentation","id_site","activity_type","total_resource_clicks","first_used","last_used","click_duration"], history_interactions.columns.to_list())
        self.assertEqual(["id_student","code_module","code_presentation","id_site","activity_type","total_resource_clicks","first_used","last_used","click_duration"], future_interactions.columns.to_list())


    #2. precision@k function

    def test_precision_value(self):
        #not using sample data, creating own data for clarity on precision formula
        relevant = [1,5,2]
        recommended = [1,4,2,3]

        #precision = number of relevant items / total number of items = 2 / 4
        result = evaluation.precision_k(relevant,recommended,4)

        self.assertEqual(result,2/4)

    def test_precision_error_relevant(self):
        #not using sample data, creating own data for clarity on precision formula
        relevant = []
        recommended = [1,4,2,3]

        #if relevant is empty, then 0 returned according to formula
        result = evaluation.precision_k(relevant,recommended,4)

        self.assertEqual(0,result)

    #3. recall@k function

    def test_recall_value(self):
        #not using sample data, creating own data for clarity on recall formula
        relevant = [1,5,2]
        recommended = [1,4,2,3]

        #recall = number of relevant items / total number of relevant items = 2 / 3
        result = evaluation.recall_k(relevant,recommended,4)

        self.assertEqual(result,2/3)

    def test_recall_error_k(self):
        #not using sample data, creating own data for clarity on recall formula
        relevant = [1,5,2]
        recommended = [1,4,2,3]

        #if k < 0, then error returned
        #code copied from: https://stackoverflow.com/questions/129507/how-do-you-test-that-a-python-function-throws-an-exception
        with self.assertRaises(ValueError) as context:
            evaluation.recall_k(relevant,recommended,-1)
        #end copied code

    #3. random scores function
    
    def test_random_value(self):
        random_scores_1 = evaluation.random_scores(1,self.processed_datasets["interaction_data"])
        random_scores_2 = evaluation.random_scores(1,self.processed_datasets["interaction_data"])

        #verify output is truly random
        self.assertNotEqual(random_scores_1["random_score"].values,random_scores_2["random_score"].values)

    def test_random_range(self):
        random_scores = evaluation.random_scores(1,self.processed_datasets["interaction_data"])

        #scores between 0 and 1
        self.assertGreaterEqual(1,random_scores["random_score"].values)
        self.assertLessEqual(0,random_scores["random_score"].values)

