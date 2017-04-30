''' The test scores are approximately normally (Gaussian) distributed.

    The tolerance levels of outliers are as follows:
        Low tolerance:  baseline (average) - 3.0 * sigma (standard dev)
        High tolerance: baseline (average) + 3.0 * sigma (standard dev)
            where baseline is the running average over 35 days and
            sigma is equal to the standard deviation over the previous
            35 days.
'''

from scipy.stats import norm
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.mlab as mlab
import math


def get_confidence(low_threshold, high_threshold):
    ''' Computing the significance level for the current threshold tolerance
          levels.
        Input: Low Threshold, High Threshold - floats
        Output: Confidence Level, Significance Level - floats
    '''
    # Computing the Cumulative Density Function for each threshold level.
    low_thres_cdf = norm.cdf(low_threshold)
    high_thresh_cdf = norm.cdf(high_threshold)

    confidence_level = high_thresh_cdf - low_thres_cdf
    significance_level = 1 - confidence_level

    print("Signficance level = {}".format(significance_level))
    print("Confidence level = {}".format(confidence_level))
    print("We can say that if we took a random sample of the employee's test\n scores, {0}% of the time the mean of those random samples is going to be\n within our threshold interval with a significance level of {1}%.".format(round(confidence_level * 100, 2), round((significance_level) * 100, 2)))

    return confidence_level, significance_level

def plot_gaussian(low_threshold, high_threshold, confidence_level):
    ''' Plotting Normal (Gaussian) distribution to display region within
           distribution that will not be flagged as outliers.
        Input: Low Threshold, High Threshold - floats
        Output: None, but plots Gaussian distribution with acceptance area
                      shaded
    '''
    # Setting up mean (mu), variance, & sigma (std dev) for plotting the
    #    normalized Gaussian
    mu = 0
    variance = 1
    sigma = math.sqrt(variance)


    x = np.linspace(-high_threshold, high_threshold, 100)
    y = mlab.normpdf(x, mu, sigma)

    plt.clf()
    fig, ax = plt.subplots()
    ax.plot(x, y, color='black')
    ax.fill_between(x, 0, y, where=x > low_threshold, facecolor="yellow",\
                    alpha=0.5)
    ax.fill_between(x, 0, y, where=y < high_threshold, facecolor="blue",\
                    alpha=0.5)
    plt.xlabel("Purple area = acceptance range,\nConfidence Level = {}".format(confidence_level))
    plt.show()


if __name__ == '__main__':
    # Getting User input for threshold tolerance values
    print("\nTest Score Threshold Evaluator")
    print("------------------------------")
    print("The threshold values are the number of standard deviations from the baseline (average) to be included in the \nacceptance range of test scores.\n")

    low_threshold = float(raw_input("Please enter the low threshold to use for the test scores: "))
    high_threshold = float(raw_input("Please enter the high threshold to use for the test scores: "))
    print("The low threshold is baseline - {} x standard deviation").format(low_threshold)
    print("The high threshold is baseline + {} x standard deviation").format(high_threshold)

    # Compute Confidence and Signficance Level of values in the acceptance range
    confidence_level, significance_level = get_confidence(low_threshold,\
                                                          high_threshold)

    # Plot Normal distribution with the acceptance range of the shaded
    plot_gaussian(low_threshold, high_threshold, confidence_level)
