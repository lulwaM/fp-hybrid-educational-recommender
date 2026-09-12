#python testing framework
import unittest

# for mocking
from unittest.mock import patch

# for testing API
from fastapi import FastAPI
from fastapi.testclient import TestClient

#for data storage
import pandas as pd
import numpy as np

#to create sample test data and preprocessed
from tests.sample_data import create_datasets
from src.data_handling import preprocessing

#function to be tested
from src.output import pipeline

# prevents github action error by giving empty values for automatically executed function, code copied and adapted from: https://stackoverflow.com/questions/16134281/python-mocking-a-function-from-an-imported-module
with patch('src.data_handling.preprocessing.preprocess_datasets', return_value = {}), \
    patch('src.recommenders.ranking.load_model', return_value = None):
    # must use same app
    from src.output.api import app
# end copied and adapted code

class TestAPIPipeline(unittest.TestCase):

    #runs once before tests, used to store datasets to be evaluated
    def setUp(self):

        self.datasets = create_datasets()
        merged_result = preprocessing.merge_datasets(self.datasets)
        cleaned_result = preprocessing.clean_datasets(merged_result)
        typecast_result = preprocessing.typecasting_datasets(cleaned_result)

        self.processed_datasets = preprocessing.processing_datasets(typecast_result)


        # code copied and adapted to work with unittest instead of pytest: https://fastapi.tiangolo.com/tutorial/testing/#using-testclient
        # setting up client for API
        self.client = TestClient(app)

    # 1. api routes functions

    def test_api_root(self):
        response = self.client.get("/")

        # check if no error returned, success
        self.assertEqual(response.status_code, 200)
        # check if main instruction present
        self.assertIn("Please visit the /docs page to view routes and make a request.",response.json()["message"])
        # end copied and adapted code

    def test_api_invalid(self):
        # should be invalid because student id is not integer and not supplied additional path parameters
        response = self.client.get("/recommendations/invalid_student_id")

        self.assertNotEqual(response.status_code, 200)

    # 2. pipeline generate recommendations function

    def test_pipeline_invalid_student(self):

        #code copied from: https://stackoverflow.com/questions/129507/how-do-you-test-that-a-python-function-throws-an-exception
        with self.assertRaises(ValueError) as context:
            # id student 10 does not exist in sample data
            pipeline.generate_recommendations(id_student=10,code_module='AAA',code_presentation='2013J',datasets=self.processed_datasets,model=None,cutoff_day=86,k=20)
        #end copied code

    def test_pipeline_invalid_student(self):

        #code copied from: https://stackoverflow.com/questions/129507/how-do-you-test-that-a-python-function-throws-an-exception
        with self.assertRaises(ValueError) as context:
            # id student 1 exists but has no interactions with course BBB / 2013B
            pipeline.generate_recommendations(id_student=1,code_module='BBB',code_presentation='2013B',datasets=self.processed_datasets,model=None,cutoff_day=86,k=20)
        #end copied code