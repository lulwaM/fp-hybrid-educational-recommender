# standard imports for data storage and data visualizations
import pandas as pd
import matplotlib.pyplot as plt


# creates data visualizations for student data exploration
# input: dataframe of preprocessed student data / no output (simply saves image to folder)
def explore_student_data(student_data):

    # code inspiration from: https://www.kaggle.com/code/ayoubaziba/studentperformancenalysis
    # 1. gender distribution pie chart, seems to be more males than females
    student_data["gender"].value_counts().plot(kind="pie", autopct="%1.1f%%")

    # formatting for clarity + save
    plt.title("Gender Distribution for Students")
    plt.tight_layout()
    plt.savefig("images/student_data/student_gender.png")
    # end inspired code

    # clear figure
    plt.clf()

    # 2. code module distribution, seems to be 6 modules with BBB being most popular
    student_data["code_module"].value_counts().plot(kind="bar", color="orange")

    # formatting for clarity + save
    plt.title("Code Module Distribution for Students")
    plt.xlabel("code_module")
    plt.ylabel("count")
    # needed to stop cutting off the xlabel, code inspiration from: https://stackoverflow.com/questions/13073045/matplotlib-savefig-size-control
    plt.tight_layout()
    # end inspired code
    plt.savefig("images/student_data/student_modules.png")

    # clear figure
    plt.clf()

    # 3. region distribution, seems focused on Europe/UK
    student_data["region"].value_counts().plot(kind="bar", color="green")

    # formatting for clarity + save
    plt.title("Region Distribution for Students")
    plt.xlabel("region")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig("images/student_data/student_regions.png")

    # clear figure
    plt.clf()

    # 4. final results distribution, seems to be mostly pass or withdrawn

    # formatting for clarity + save
    student_data["final_result"].value_counts().plot(kind="pie", autopct="%1.1f%%")
    plt.title("Final Result Distribution for Students")
    plt.tight_layout()
    plt.savefig("images/student_data/student_final_results.png")

    # clear figure
    plt.clf()

    # 5. registration duration, seems to be a lot of outliers but concentrated around 200-300 days
    student_data.boxplot(column="registration_duration", color="blue")

    # formatting for clarity + save
    plt.title("Registration Duration Box Plot for Students")
    plt.ylabel("days")
    plt.tight_layout()
    plt.savefig("images/student_data/student_registration_duration.png")

    # clear figure
    plt.clf()

    # 6. disability distribution, seems to be mostly no disabled students
    student_data["disability"].value_counts().plot(kind="pie", autopct="%1.1f%%")

    # formatting for clarity + save
    plt.title("Disability Distribution for Students")
    plt.tight_layout()
    plt.savefig("images/student_data/student_disability.png")

    # clear figure
    plt.clf()


# creates data visualizations for assessment data exploration
# input: dataframe of preprocessed assessment data, no output (simply saves image to folder)
def explore_assessment_data(assessment_data):

    # 1. assessment type distribution, seems to be mostly tutor marked assessments

    # formatting for clarity + save
    assessment_data["assessment_type"].value_counts().plot(
        kind="pie", autopct="%1.1f%%"
    )
    plt.title("Type Distribution for Asssessments")
    plt.tight_layout()
    plt.savefig("images/assessment_data/assessment_type.png")

    # clear figure
    plt.clf()

    # 2. histogram of average assessment score, seems to be mostly 70%+ (good marks)
    # code inspiration from: https://www.kaggle.com/code/ayoubaziba/studentperformancenalysis
    assessment_data["average_score"].hist(color="purple")

    # formatting for clarity + save
    plt.title("Histogram of Assessment Average Scores")
    plt.xlabel("average_score")
    plt.ylabel("frequency")
    plt.tight_layout()
    plt.savefig("images/assessment_data/assessment_average_score.png")

    # clear figure
    plt.clf()

    # multiple assessments per student, so must group and count number of assessment records per student
    assessment_summary = assessment_data.groupby("id_student")["id_assessment"].count()

    # 3. histogram of number of assessments per student, seems to be mostly between 5-10
    assessment_summary.hist(color="green")

    # formatting for clarity + save
    plt.title("Histogram of Number of Assessments Taken")
    plt.xlabel("number_of_assessments")
    plt.ylabel("frequency")
    plt.tight_layout()
    plt.savefig("images/assessment_data/assessment_taken.png")
    # end inspired code

    # clear figure
    plt.clf()

    # 4. is banked assessment distribution, seems to be almost all no transfer (organized university result release process)
    assessment_data["is_banked"].value_counts().plot(kind="pie", autopct="%1.1f%%")

    # formatting for clarity + save
    plt.title("Distribution of Transferred Assessment Results (banked)")
    plt.tight_layout()
    plt.savefig("images/assessment_data/assessment_banked.png")

    # clear figure
    plt.clf()


# creates data visualizations for vle data exploration
# input: dataframe of preprocessed vle data, no output (simply saves image to folder)
def explore_vle_data(vle_data):

    # 1. activity type dsitribution, seems to be mostly forumng / oucontent
    vle_data["activity_type"].value_counts().plot(kind="barh", color="pink")

    # formatting for clarity + save
    plt.title("Activity Type Distribution for VLE")
    plt.tight_layout()
    plt.savefig("images/vle_data/vle_activity_type.png")

    # clear figure
    plt.clf()

    # 2. module presentation length histogram, seems to be most modules around 265-270 days
    vle_data["module_presentation_length"].hist(color="lightblue")

    # formatting for clarity + save
    plt.title("Histogram of VLE Module Presentation Length")
    plt.xlabel("module_presentation_length")
    plt.ylabel("frequency")
    plt.tight_layout()
    plt.savefig("images/vle_data/vle_presentation_length.png")

    # clear figure
    plt.clf()


# creates data visualizations for resource data exploration
# input: dataframe of preprocessed resource data, no output (simply saves image to folder)
def explore_resource_data(resource_data):

    # 1. code modules, seems to be 6 modules with FFF having the most resources
    resource_data["code_module"].value_counts().plot(kind="bar", color="orange")

    # formatting for clarity + save
    plt.title("Code Module Distribution for Resources")
    plt.xlabel("code_module")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig("images/resource_data/resource_code_module.png")

    # clear figure
    plt.clf()

    # code inspiration from: https://www.geeksforgeeks.org/python/using-pandas-crosstab-to-create-a-bar-plot/
    # 2. cross tabulation of code module against activity type to create stacked bar chart
    vle_crosstab = pd.crosstab(
        resource_data["code_module"], resource_data["activity_type"], normalize="index"
    )

    # seems to be mostly all have high proportion of hompepage and oucontent activities
    vle_crosstab.plot(kind="bar", stacked=True)

    plt.title("Distribution of Activity Types by Code Module")

    # formatting for clarity + save
    plt.xlabel("code_module")
    plt.ylabel("normalized_proportion")
    plt.legend(title="activity_type", bbox_to_anchor=(1, 1.02), loc="upper left")
    # end inspired code
    plt.tight_layout()
    plt.savefig("images/resource_data/vle_module_activity.png")

    # clear figure
    plt.clf()


# creates data visualizations for interaction data exploration
# input: dataframe of preprocessed interaction data, no output (simply saves image to folder)
def explore_interaction_data(interaction_data):

    # 1. plots total clicks per student, seems to be most students have low clicks 0-5000
    # code inspiration from:https://www.kaggle.com/code/ayoubaziba/studentperformancenalysis
    interaction_data.groupby("id_student")["total_resource_clicks"].sum().hist(
        color="lightgreen"
    )
    # end inspired code

    # formatting for clarity + save
    plt.title("Histogram of Total Resource Clicks by Student")
    plt.xlabel("total_resource_clicks")
    plt.ylabel("frequency")
    plt.tight_layout()
    plt.savefig("images/interaction_data/interaction_clicks_student.png")

    # clear figure
    plt.clf()

    # 2. plot horizontal bar graph for average click duration per activity type, seems to be significantly higher engagement for homepage
    interaction_data.groupby("activity_type")[
        "click_duration"
    ].mean().sort_values().plot(kind="barh", color="violet")

    # formatting for clarity + save
    plt.title("Average Click Duration by Activity Type")
    plt.xlabel("click_duration")
    plt.ylabel("activity_type")
    plt.tight_layout()
    plt.savefig("images/interaction_data/interaction_average_clicks_activity.png")

    # clear figure
    plt.clf()
