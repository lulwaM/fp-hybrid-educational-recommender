#python testing framework
import unittest

#for data storage
import pandas as pd

#to create sample test data
from tests.sample_data import create_datasets

#functions to be tested
from src.data_handling import preprocessing

class TestPreprocessing(unittest.TestCase):

    #runs once before tests, used to store datasets to be evaluated
    def setUp(self):
        self.datasets = create_datasets()

    #1. merge datasets function

    #test expected keys
    def test_merge_keys(self):
        merged_result = preprocessing.merge_datasets(self.datasets)

        self.assertEqual(merged_result.keys(),
                         {"student_data","assessment_data","vle_data","resource_data"})

    #test dataframe has expected columns/data
    def test_merge_student_dataframe_details(self):
        merged_result = preprocessing.merge_datasets(self.datasets)

        self.assertIn("id_student",merged_result["student_data"].columns)
        self.assertIn("code_module",merged_result["student_data"].columns)
        self.assertIn("gender",merged_result["student_data"].columns)
        self.assertIn("disability",merged_result["student_data"].columns)

    #2. clean datasets function

    #handling missing values in student data for registration data by dropping records
    def test_clean_drop_missing_student_registration(self):
        merged_result = preprocessing.merge_datasets(self.datasets)
        cleaned_result = preprocessing.clean_datasets(merged_result)

        #any method returns true if any of the registration dates are NaN, vectorized operation
        self.assertFalse(cleaned_result["student_data"]["date_registration"].isna().any())

    #handling missing values in student data for registration/unregistration date and imd band by dropping/filling records
    def test_clean_missing_student(self):
        merged_result = preprocessing.merge_datasets(self.datasets)
        cleaned_result = preprocessing.clean_datasets(merged_result)

        #any method returns true if any of the registration dates are NaN, vectorized operation
        self.assertFalse(cleaned_result["student_data"]["date_registration"].isna().any())
        self.assertFalse(cleaned_result["student_data"]["imd_band"].isna().any())
        self.assertFalse(cleaned_result["student_data"]["date_unregistration"].isna().any())

    #handling missing values in assessment data for scores/date by dropping/filling records
    def test_clean_missing_student(self):
        merged_result = preprocessing.merge_datasets(self.datasets)
        cleaned_result = preprocessing.clean_datasets(merged_result)

        #any method returns true if any of the registration dates are NaN, vectorized operation
        self.assertFalse(cleaned_result["assessment_data"]["score"].isna().any())
        self.assertFalse(cleaned_result["assessment_data"]["date"].isna().any())

    #handling missing values in vle data for dropping entire week from/to columns (does not exist)
    def test_clean_missing_vle(self):
        merged_result = preprocessing.merge_datasets(self.datasets)
        cleaned_result = preprocessing.clean_datasets(merged_result)

        self.assertNotIn("week_from",cleaned_result["vle_data"])
        self.assertNotIn("week_to",cleaned_result["vle_data"])

    #3. typecasting datasets function

    #handling categorical columns data type conversion
    def test_typecast_categorical(self):
        merged_result = preprocessing.merge_datasets(self.datasets)
        cleaned_result = preprocessing.clean_datasets(merged_result)
        typecast_result = preprocessing.typecasting_datasets(cleaned_result)

        #only selected some key categorical columns to test, note that assert instance used here because pandas variable is a class

        #for student data
        self.assertIsInstance(typecast_result["student_data"]["code_module"].dtype, pd.CategoricalDtype)
        self.assertIsInstance(typecast_result["student_data"]["gender"].dtype, pd.CategoricalDtype)
        self.assertIsInstance(typecast_result["student_data"]["final_result"].dtype, pd.CategoricalDtype)
        #for assessment data
        self.assertIsInstance(typecast_result["assessment_data"]["assessment_type"].dtype, pd.CategoricalDtype)
        #for vle data
        self.assertIsInstance(typecast_result["vle_data"]["activity_type"].dtype, pd.CategoricalDtype)
        #for resource data
        self.assertIsInstance(typecast_result["resource_data"]["code_presentation"].dtype, pd.CategoricalDtype)

    #handling boolean columns data type conversion
    def test_typecast_boolean(self):
        merged_result = preprocessing.merge_datasets(self.datasets)
        cleaned_result = preprocessing.clean_datasets(merged_result)
        typecast_result = preprocessing.typecasting_datasets(cleaned_result)

        #only selected some key boolean columns to test, note that assert equal used here because python bool is object

        #for student data
        self.assertEqual(typecast_result["student_data"]["withdrew"].dtype, bool) 
        self.assertEqual(typecast_result["student_data"]["disability"].dtype, bool) 
        #for assessment data
        self.assertEqual(typecast_result["assessment_data"]["is_banked"].dtype, bool)

    #4. processing datasets function

    def test_processing_student(self):
        merged_result = preprocessing.merge_datasets(self.datasets)
        cleaned_result = preprocessing.clean_datasets(merged_result)
        typecast_result = preprocessing.typecasting_datasets(cleaned_result)
        processed_result = preprocessing.processing_datasets(typecast_result)

        #code inspired by: https://stackoverflow.com/questions/40913113/compare-two-pandas-series-dataframes-that-are-virtually-equal
        #verify feature created is correct according to formula
        pd.testing.assert_series_equal(processed_result["student_data"]["registration_duration"],processed_result["student_data"]["date_unregistration"]-processed_result["student_data"]["date_registration"],check_names=False )
        #end inspired code

    def test_processing_assessment(self):
        merged_result = preprocessing.merge_datasets(self.datasets)
        cleaned_result = preprocessing.clean_datasets(merged_result)
        typecast_result = preprocessing.typecasting_datasets(cleaned_result)
        processed_result = preprocessing.processing_datasets(typecast_result)

        self.assertIn("average_score",processed_result["assessment_data"].columns)

    def test_processing_resource(self):
        merged_result = preprocessing.merge_datasets(self.datasets)
        cleaned_result = preprocessing.clean_datasets(merged_result)
        typecast_result = preprocessing.typecasting_datasets(cleaned_result)
        processed_result = preprocessing.processing_datasets(typecast_result)

        #code inspired by: https://stackoverflow.com/questions/40913113/compare-two-pandas-series-dataframes-that-are-virtually-equal
        #verify feature created is correct according to formula
        pd.testing.assert_series_equal(processed_result["resource_data"]["resource_text"],
                                       processed_result["resource_data"]["code_module"].astype(str) + " " + processed_result["resource_data"]["code_presentation"].astype(str) + " " + processed_result["resource_data"]["activity_type"].astype(str), check_names=False )
        #end inspired code

    def test_processing_interaction(self):
        merged_result = preprocessing.merge_datasets(self.datasets)
        cleaned_result = preprocessing.clean_datasets(merged_result)
        typecast_result = preprocessing.typecasting_datasets(cleaned_result)
        processed_result = preprocessing.processing_datasets(typecast_result)

        self.assertIn("total_resource_clicks",processed_result["interaction_data"])

        #code inspired by: https://stackoverflow.com/questions/40913113/compare-two-pandas-series-dataframes-that-are-virtually-equal
        #verify feature created is correct according to formula
        pd.testing.assert_series_equal(processed_result["interaction_data"]["click_duration"],
                                       processed_result["interaction_data"]["last_used"]-processed_result["interaction_data"]["first_used"], check_names=False )
        #end inspired code

#code copied from: https://docs.python.org/3/library/unittest.html
#used to run tests
if __name__ == '__main__':
    unittest.main()
#end copied code
    

