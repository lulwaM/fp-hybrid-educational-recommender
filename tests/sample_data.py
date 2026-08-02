# for data storage
import pandas as pd

# for np.nan simulation
import numpy as np


# creating dataframe for each csv file in dataset, copied from files
def create_datasets():

    #1. 3 assessment records belonging to the 2 courses, has missing values to test preprocessing funjctions
    assessments = pd.DataFrame(
        {
            "code_module": ["AAA", "BBB", "AAA"],
            "code_presentation": ["2013J", "2013B", "2013J"],
            "id_assessment": ["1001", "2001", "1002"],
            "assessment_type": ["CMA", "TMA", "Exam"],
            "date": [30, np.nan, 54],
            "weight": [10, 20, 100],
        }
    )

    #2. 2 courses
    courses = pd.DataFrame(
        {
            "code_module": ["AAA", "BBB"],
            "code_presentation": ["2013J", "2013B"],
            "module_presentation_length": [241, 268],
        }
    )

    #3. 4 student assessment records belonging to 3 students and 2 possible assessments, has missing data to evaluate preprocessing functions
    student_assessment = pd.DataFrame(
        {
            "id_assessment": ["1001", "2001", "1002", "1001"],
            "id_student": [1, 2, 1, 3],
            "date_submitted": [18, 64, 53, 16],
            "is_banked": [0, 1, 0, 0],
            "score": [np.nan, 60, 70, 80],
        }
    )

    #4. 3 student records, must match the codem odule with the taken assessments
    student_info = pd.DataFrame(
        {
            "code_module": ["AAA", "BBB", "AAA"],
            "code_presentation": ["2013J", "2013B", "2013J"],
            "id_student": [1, 2, 3],
            "gender": ["M", "F", "M"],
            "region": ["East Anglian Region", "Wales", "Scotland"],
            "highest_education": ["HE Qualification", "Lower Than A Level", "A Level or Equivalent"],
            "imd_band": ["90-100%", "30-40%", np.nan],
            "age_band": ["55<=", "35-55", "0-35"],
            "num_of_prev_attempts": [1,0,0],
            "studied_credits": [60, 240, 60],
            "disability": ["Y", "N", "N"],
            "final_result": ["Pass", "Withdrawn", "Fail"],
        }
    
    )

    #5. 3 student registration belonging to the 3 students, missing values to test preprocessing
    student_registration = pd.DataFrame(
        {
            "code_module": ["AAA", "BBB", "AAA"],
            "code_presentation": ["2013J", "2013B", "2013J"],
            "id_student": [1, 2, 3],
            "date_registration": [-92,np.nan,20],
            "date_unregistration": [3,-10,np.nan],   
        }
    )

    #6. 5 student VLe interactions belonging to the 3 students
    student_vle = pd.DataFrame(
        {
            "code_module": ["AAA", "BBB", "AAA","AAA","AAA"],
            "code_presentation": ["2013J", "2013B", "2013J","2013J","2013J"],
            "id_student": [1, 2, 3,1,3], 
            "id_site": [101, 201, 101,102,103],
            "date": [-10,5,3,2,1],
            "sum_click": [5,14,12,4,3],   
        }
    )

    #7. 4 VLE resources linked to the student VLE interactions
    vle = pd.DataFrame(
        {
            "id_site": [101,102,103,201],
            "code_module": ["AAA", "AAA","AAA","BBB"],
            "code_presentation": ["2013J", "2013J","2013J","2013B"],
            "activity_type": ["url", "oucontent", "resource","subpage"], 
            "week_from": [np.nan,np.nan,2,np.nan],
            "week_to": [5,np.nan,26,23],   
        }
    )

    # returned is a dictionary/object where key=dataset name value=pandas dataframe for it, exactly same as load dataset function
    return {
        "assessments":assessments,
        "courses":courses,
        "student_assessment":student_assessment,
        "student_info":student_info,
        "student_registration":student_registration,
        "student_vle":student_vle,
        "vle":vle
    }

