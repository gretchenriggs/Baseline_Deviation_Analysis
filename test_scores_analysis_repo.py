import pandas as pd
import numpy as np
import warnings
from math import sqrt
from scipy.stats import norm
import datetime
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')


def plot_all_emp_scores(df):
    ''' Plotting all employees Scores & Baselines (if exist)
        Input: df     DataFrame containing date/employee name/score/baseline
        Output: None  Function creates plots that are saved to disk as .png
    '''
    employee_arr = df['Employee'].unique()

    for employee in employee_arr:
        dates = df[df['Employee'] == employee]['Date'].values
        scores = df[df['Employee'] == employee]['Score'].values
        baselines = df[df['Employee'] == employee]['Baseline'].values
        plt.plot(dates, scores)
        plt.plot(dates, baselines)
        plt.title("Test Scores - " + employee)
        plt.xticks(rotation=25)
        plt.ylabel("Scores")
        plt.ylim(0,0.8)
        plt.savefig("plots/plot_" + employee + ".png")
        plt.clf()

def filter_df(df):
    ''' Filter incomplete or inactive data from DataFrame
        Input:  df   DataFrame containing date/employee name/score/baseline
        Output: df   DataFrame with incomplete/inactive rows filtered out
    '''
    # Removing tests times where the Score was NaN
    df = df[df['Score'].notnull()]

    # Removing tests times where the Baseline was NaN
    df = df[df['Baseline'].notnull()]

    # Removing tests of Inactive Employees
    df  = df[df['Job Role'] != "Inactive"]

    return df

def get_baseline(scores_list):
    ''' Find the baseline value using the existing algorithm for baseline
           calculation
        Input:  scores_list    array of floats, test scores
        Output: baseline_list  list of floats, baseline values
    '''

    # Create empty baseline_list to store baseline values
    baseline_list = []
    std_dev_list = []
    upd_baseline_input = False

    # Loop through test scores and compute baseline for each value
    #   Baseline just average of scores until the 36th test, then it becomes
    #   the average of the last 35 test scores.
    for i, score in enumerate(list(scores_list)):
        # When 1st & 2nd test score, baseline = test score
        if i in xrange(0,2):
            if i == 0:
                baseline = scores_list[i]
            else:
                baseline = scores_list[i-1]
            n = 1
            # Computing population standard deviation
            std_dev = np.std(scores_list[i], ddof=0)

        # For 3rd through 35th test, using the average of all tests to date for
        #    Baseline.
        elif i in xrange(2,35):
            baseline = np.mean(scores_list[0:i])
            n = len(scores_list[0:i+1])
            std_dev = np.std(scores_list[0:i], ddof=0)

        # For 36th test and going forward, using the average of the previous 35
        #    tests as the baseline
        else:
            if upd_baseline_input == False:
                baseline = np.mean(scores_list[i-35:i])
                n = len(scores_list[i-35:i])
                std_dev = np.std(scores_list[i-35:i], ddof=0)
            else:
                baseline = baseline
                std_dev = std_dev
            # Checking for outliers.  If score is an outlier, set
            #   upd_baseline_input to Boolean "True", so is flagged to use the
            #   previous baseline for the next test score, not including the
            #   outlier in the baseline computation
            if scores_list[i] < (baseline - (3.0 * std_dev)):
                upd_baseline_input = True
            elif scores_list[i] > (baseline + (3.0 * std_dev)):
                upd_baseline_input = True
            else:
                upd_baseline_input = False

        baseline_list.append(baseline)
        std_dev_list.append(std_dev)

    return baseline_list, std_dev_list

def tolerance_check(emp_df, baseline_list, std_dev_list):
    ''' Checking if test scores within tolerance range specified for job risk
           level.
        Input: emp_df           DataFrame for 1 employee
               baseline_list    List of floats, computed baseline values
               std_dev_list     List of floats, computed standard deviation
                                  values
        Output: score_in_range  List of Booleans
                second_test_chk List of Booleans

        We're performing a hypothesis test using a the normal distribution, as
            the test scores are approximately normally distributed.
        H0 = Score is the same as the baseline -> baseline = score ;
                       i.e. the score is not far enough away from the baseline
                            to be considered significantly different
        HA = Score is not the same as the baseline -> baseline != score ;
                       i.e. the score IS far enough away from the baseline to
                            be considered significantly different
        This is a 2-tailed test.
    '''
    # Creating date_upd column in DataFrame, not containing only the date, not
    #   the date & the time
    emp_df['date_upd'] = emp_df['Date'].dt.date

    # Running average baseline is computed over 35 points
    running_avg_num_pts = 35

    # The tolerance percentage, that is used to compute the acceptable range of
    #    test values.
    tolerance_pct = 0.05

    # Computing the lower & upper tolerance values for the acceptable range of
    #    test scores for this employee
    low_tolerance = np.array(baseline_list) * (1 - tolerance_pct)
    high_tolerance = np.array(baseline_list) * (1 + tolerance_pct)

    # Computing z, the number of standard deviations test score is away from
    #   baseline
    z = (np.array(baseline_list) - \
       (np.array(baseline_list) * (1 - tolerance_pct))) / np.array(std_dev_list)

    # Computing the significance of the scores outside the tolerance range
    # Can just multiply one of the computations by 2, since we're getting
    #    the high & low side tolerance significance (2-tailed test)
    significance_of_diff = (1 - norm.cdf(z)) * 2
    confidence_of_diff = 1 - significance_of_diff
    plot_sig_conf(emp_df, significance_of_diff, confidence_of_diff)
    # else:

def plot_sig_conf(emp_df, significance_of_diff, confidence_of_diff):
    ''' Plotting statistical significance and confidence for all test scores
        Input: emp_df                DataFrame of employee information
               significance_of_diff  array of statistical significance of each
                                        test score
               confidence_of_diff    array of statistical confidence level of
                                         each test score
        Output: None, plotting to .png files
    '''
    plt.clf()
    dates = emp_df['Date'].values
    plt.plot(dates, significance_of_diff)
    plt.title("Test Scores Statistical Significance Level - " + emp_df['Employee'].values[0])
    plt.xticks(rotation=25)
    plt.xlabel("Date")
    plt.ylabel("Statistical Significance (alpha) of Test Score Difference")
    plt.ylim(0, 0.5)
    plt.legend(loc="best")
    plt.savefig("plots/siglevel_with_tolerance-" + emp_df['Employee'].values[0] + ".png")
    plt.clf()

    plt.plot(dates, confidence_of_diff)
    plt.title("Test Scores Statistical Confidence Level - " + emp_df['Employee'].values[0])
    plt.xticks(rotation=25)
    plt.xlabel("Date")
    plt.ylabel("Statistical Confidence of Test Score Difference")
    plt.ylim(0.5, 1.0)
    plt.legend(loc="best")
    plt.savefig("plots/conflevel_with_tolerance-" + emp_df['Employee'].values[0] + ".png")
    plt.clf()


def plot_scores_with_baseline(emp_df, scores_list, baseline_list, std_dev_list):
    ''' Plotting 1 employee's test scores with computed baseline
        (mine vs. company's) and outlier thresholds
        Input: df          DataFrame with all employee's test information
               employee    string, employee name
        Output: None, plots employee's test scores to screen
    '''
    # Creating list of dates & company's baseline values
    dates = emp_df['Date'].values
    baselines_comp = emp_df['Baseline'].values

    # Computing thresholds for outlier detection
    low_threshold = np.array(baseline_list) - 3.0 * np.array(std_dev_list)
    high_threshold = np.array(baseline_list) + 3.0 * np.array(std_dev_list)

    # The tolerance percentage, that is used to compute the acceptable range of
    #    test values.
    tolerance_pct = 0.05

    # Computing the lower & upper tolerance values for the acceptable range of
    #    test scores for this employee
    low_tolerance = np.array(baseline_list) * (1 - tolerance_pct)
    high_tolerance = np.array(baseline_list) * (1 + tolerance_pct)

    # PLotting test scores, with baselines and outlier thresholds to .png file
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(dates, low_threshold, color="red", label="Low Outlier Threshold")
    ax.plot(dates, high_threshold, color="red", label="High Outlier Theshold")
    ax.fill_between(dates, low_threshold, high_threshold, facecolor="magenta",\
                    alpha=0.25)
    ax.plot(dates, scores_list, color="black", marker="o", label="Score")
    ax.plot(dates, baseline_list, color="blue", label="GMR Baseline",\
            linewidth=2)
    ax.plot(dates, baselines_comp, color="green", label="PS Baseline",\
            linewidth=2)
    ax.plot(dates, low_tolerance, color='yellow', label="Low Tolerance",\
            linewidth=2)
    ax.plot(dates, high_tolerance, color='yellow', label="High Tolerance",\
            linewidth=2)
    plt.title("Test Scores - " + emp_df['Employee'].values[0])
    plt.xticks(rotation=25)
    plt.xlabel("Date")
    plt.ylabel("Score")
    plt.ylim(0,0.8)
    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    plt.savefig("plots/plot_with_tolerance-" + emp_df['Employee'].values[0] + ".png")
    plt.clf()

if __name__ == '__main__':
    # Reading in Test Score Data from CSV file
    df = pd.read_csv("test_scores.csv")

    # Reformatting date from String to Date type
    df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y %H:%M:%S')

    # Plot all the Raw employee scores by date with the baseline
    plot_all_emp_scores(df)

    # Filter incomplete data from DataFrame
    df = filter_df(df)

    # Sort DataFrame by Employee:Date and reset index values
    df = df.sort(["Employee", "Date"]).reset_index()

    # Saving array of unique employee names
    employee_arr = df['Employee'].unique()

    # Iterating through employees to compute baselines, thresholds, and generate
    #    plots
    for name in employee_arr:
        # Create DataFrame with only 1 employee's information
        emp_df = df[df['Employee'] == name]

        # Creating list of employee's test scores
        scores_list = emp_df['Score'].values

        # Calling get_baseline function to get my computed baseline and standard
        #    deviations.
        baseline_list, std_dev_list = get_baseline(scores_list)

        # Computing the tolerance range and comparing test score to the
        #    tolerance.  Creating plots of significance and confidence levels
        #    for each test score.
        tolerance_check(emp_df, baseline_list, std_dev_list)

        # Plot 1 employee's scores with baselines and outlier thresholds graphed
        plot_scores_with_baseline(emp_df, scores_list, baseline_list, \
         std_dev_list)
