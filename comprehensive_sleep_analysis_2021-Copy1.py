#!/usr/bin/env python
# coding: utf-8

# # Analysis sleep pattern for one person before and after lockdown

# Sleep is an essential aspect of our lives. It plays a vital role in maintaining physical health, mental clarity, a positive mindset, and energy levels throughout the day. Nobody can resist how important sleep is, but many people rely on their personal feelings to measure their sleep quality. 
# 
# My husband used a sleep tracking ring to seek objective, unbiased insights to understand and improve his sleep. This innovative device monitors various sleep parameters, including sleep stages, heart rate, body temperature, and activity levels. The Oura Ring application offers valuable information on sleep quality, duration, and sleep stages and insights on readiness, activity levels, body temperature, heart rate, recovery, and personalized trends.
# 
# Amidst the global COVID-19 pandemic, the world underwent significant changes, including lockdown measures that confined people to their homes. During this period, my husband experienced a shift in his daily routine as he transitioned to remote work. This presented a unique opportunity to analyze and compare his sleep patterns before and after the lockdown.
# 
# In this project, I analyze collected data from the sleep tracker and use a simple method and clear visualization to tell a  sleep story.
# 
# The project's primary goal is to analyze the sleep pattern, reveal the impact of the lockdown, and provide recommendations for improving sleep. 
# 
# To accomplish these aims, let's focus on the following questions:
# 1. __Does he have enough sleep?__ We will explore the sleep duration and compare it to recommended sleep guidelines to determine if they meet it.
# 2. __How the lockdown influenced his sleep and activity?__ We will assess any differences by comparing sleep and activity metrics before and during the lockdown.
# 3. __Are there any correlations between metrics?__ We will explore the relationships between variables such as sleep duration, readiness scores, sleep scores, and activity scores to identify potential relations or dependencies.
# 

# To analyze the sleep patterns effectively, we will take a hierarchical approach. Starting with a general overview, we will examine the yearly trends, followed by monthly variations, and finally, zoom in on weekly patterns. This approach allows us to capture broad trends and specific insights, comprehensively understanding sleep patterns before and after the lockdown period.
# 
# - __Exploratory Data Analysis (EDA):__
# 
#   - Descriptive Statistics: calculate basic statistical measures.
#   - Correlation Analysis: visualize the correlation matrix using a heatmap, where colors indicate the strength of the correlation; specify features with strong positive or negative correlations, indicating potential relationships or dependencies.
#   - Finding Outliers: plot box plots to visualize the distribution of features and detect outliers; use statistical methods like the Z-score or interquartile range (IQR) to identify and flag potential outliers; explore outliers', confirming they are not errors or anomalies in the data.
#   
#   
# - __Statistical Analysis:__
#   - Perform statistical tests to understand if the difference in sleep patterns Before and After Lockdown is statistically significant
#   
# 
# - __General Overview:__
#   - Calculate basic statistical measures such as each feature's mean and median.
#   - To compare sleep patterns, calculate statistical measures for the "Before Lockdown" and "After Lockdown" periods.
#   - Plot a histogram to visualize the distribution of sleep values.
#   - Visualize the BedTime Routine, showing the starts and ends of sleep. 
#   
# 
# 
# - __Yearly Analysis:__
#   - Calculate the average sleep duration for each year.
#   - Plot the average sleep scores over the years.
#   
#   
# - __Monthly Analysis:__
#   - Plotting the average sleep duration of different sleep types over months. 
#   - Line plots of the average sleep values.
#   
#  
# - __Weekly Analysis:__
#   - Calculate the average sleep duration for each week across all years.
#   - Plot the average sleep duration over the weeks to observe any variations or trends weekly.
#   
#   
# - __Summarize__
# 
# 
# - __Reccomendations__
# 
# By analyzing the sleep data at these levels, we will progressively uncover insights and patterns that may exist. 
# 

# ## Exploratory Data Analysis
# An essential step to get acquainted with the data set is conducting __exploratory data analysis (EDA)__. 
# EDA is a first look at the dataset, identifying patterns, detecting outliers, and exploring relationships between features. Also, the EDA process intends to find missing values and other mistakes that require correction during _data cleaning_.  

# In[1]:


# Import libraries
import numpy as np
import pandas as pd

import statsmodels.api as sm
import matplotlib.dates as mdates
import scipy.stats as stats
from scipy.stats import wilcoxon


import matplotlib.pyplot as plt
import seaborn as sns

from datetime import date
from datetime import datetime
from pylab import rcParams
from matplotlib import dates

from matplotlib.dates import AutoDateLocator, AutoDateFormatter, date2num
from matplotlib.dates import DayLocator, HourLocator


# In[2]:


# Setting the plotting theme
sns.set()

# Setting the size of all plots
plt.rcParams['figure.figsize'] = 20, 8

# Setting the colors
color = ["#4b85a8", "#d98100", "#495464"]

# Setting the background color to solid white
plt.rcParams['axes.facecolor'] = '#F0F0F0'


# In[3]:


# Load dataset. Set index to date column
df = pd.read_csv("sleeping_tracking_.csv",
                 index_col=["date"],
                 parse_dates=["date"])


# In[4]:


# Inspect data
df.head(2)


# In[5]:


# Examine the index
df.index


# In[6]:


# Print data information
df.info()


# `df.info()` gives as high-level information on the data set. The data set contains __11 columns and 447 rows__. Also, it shows the data type for each feature. All numeric values are set up for `int64` except for two features, `sleep_starts` and `sleep_ends`

# In[7]:


# Inspect the missing values
df.isna().sum()


# #### Here is no missing value. But if it would be a few, I could use fillna() or dropna() methods.

# In[8]:


# Print summary statistics
df.describe()


# The summary statistics provide insights into various variables in the dataset. On average, the `sleep_duration_min` is around 440 min (7 hours 20 minutes), with a standard deviation of approximately 61 minutes (1 hour). The dataset shows variability in `sleep_duration`, ranging from 290 minutes (4 hours 50 minutes) to 673 minutes (11 hours 13 minutes.)
# 
# `light_sleep_min` duration averages around 262 minutes (4 hours and 23 minutes), with a standard deviation of about 48 minutes. `rem_sleep_min` lasts approximately 92 minutes (1 hour and 32 minutes), while `deep_sleep_min` lasts around 85 minutes (1 hour and 25 minutes). Both `rem_sleep_min` and `deep_sleep_min` have some variability.
# 
# `readiness`, `sleep_score`, and `activity_score` have average values around 80, with slight variations. The number of `steps` taken per day averages around 9,464, with a standard deviation of approximately 3,246.
# 
# 

# ___________

# ### Creating a heatmap
# Heatmap quickly displays patterns and relationships between numeric variables.

# In[9]:


# Get correlation matrix
df_corr = df.corr()

sns.heatmap(df_corr, annot=True, annot_kws={"size": 15})

# Fix ticklabel directions and size
plt.xticks(size = 14, rotation=0)
plt.yticks(size = 14,rotation=0)

# Fits plot area to the plot, "tightly"
plt.tight_layout()


# The heatmap visually displays the correlation between the numeric values of the dataset. The heatmap utilizes color coding, where lighter shades indicate a strong correlation, black represents no correlation or negative correlation, and a correlation of 1 indicates a strong positive correlation. Upon a quick examination, it is apparent that `sleep_duration_min` strongly correlates with `light_sleep_min`, `rem_sleep_min`, and `sleep_score` (0.81). Additionally, a __strong correlation__ is observed between `rem_sleep_min` and `sleep_score` (0.71). Interestingly, the correlation between `activity_score` and `steps` is surprisingly __weak__ (0.33).
# 
# A correlation coefficient of -0.39 between `light_sleep_min` and `deep_sleep_min` suggests a __moderate negative correlation.__
# That means when the duration of light sleep increases, the duration of deep sleep tends to decrease, and vice versa. However, the strength of the correlation is moderate, indicating that the relationship is not extremely strong.

# _______________

# ### Finding outliers
# Detecting outliers is essential as they can reveal data errors, provide insights, impact statistical analysis, affect decision-making, and help see anomalies.
# 
# __Boxplots__ are effective for detecting outliers and clear visual presentation, robustness to skewness, explicit outlier indicators, and quantitative thresholds based on the interquartile range (IQR). 

# In[10]:


columns = ["sleep_duration_min", "light_sleep_min", "rem_sleep_min", "deep_sleep_min"]
sns.boxplot(data=df[columns], palette = color)

# Set labels
plt.xlabel("Sleep Categories", fontsize=16)
plt.ylabel("Duration (minutes)", fontsize=16)
plt.title("Analyzing Sleep Categories for Outliers", fontsize=20, fontweight='bold')

# Fix ticklabel directions and size
plt.xticks(size = 14, rotation=0)
plt.yticks(size = 14,rotation=0)

# Calculate and display the mean, maximum, and minimum values
means = df[columns].mean()
max_vals = df[columns].max()
min_vals = df[columns].min()

for i in range(len(columns)):
    plt.text(i, max_vals[i], f"max: {max_vals[i]}", ha='center', va='bottom',fontsize = 14)
    plt.text(i, min_vals[i], f"min: {min_vals[i]}", ha='center', va='top',fontsize = 14)
    plt.text(i, means[i], f"mean: {means[i]:.1f}", ha='center', va='center', fontsize = 14, weight= "bold")

# Display the plot
plt.show()


# The __boxplot__ displays four columns `sleep_duration_min`, `light_sleep_min`, `rem_sleep_min`, and `deep_sleep_min`. It allows us to observe summary statistics such as the _mean, max, and min_ values for each. 
# 
# Individual data points outside the whiskers are indicated as __outliers.__ These outliers may mean that sleep duration varies broadly and differs from the typical range. Identifying these outliers can help understand where sleep duration is significantly long or short and disturbances in sleep patterns.

# - The dataset contains __447 rows and 11 columns,__ showing detailed sleep and activity patterns information.
# 
# - The average Sleep duration is approximately __7 hours and 20 minutes__ but with significant variables.
# 
# - Sleep contains different stages - __Light Sleep, REM Sleep, and Deep Sleep__ - with Light Sleep being the most extended, and REM and Deep Sleep are similar but shorter.
# 
# - A moderate __negative correlation__ between Light and Deep Sleep suggests a possible trade-off between these sleep stages.
# 
# - Despite good Readiness, Sleep, and Activity scores, a __weak correlation between Activity Score and Steps__ taken indicates other factors influencing activity level.
# 
# - __Outliers in Sleep Duration__ may suggest special events or conditions.

# ______
# 
# 
# _______________

# ### Feature Engineering
# A typical sleep cycle consists of three primary stages: `light_sleep_min, rem_sleep_min, and deep_sleep_min`. __Deep Sleep takes 13-23%__ of total sleep, __REM Sleep takes 20-25%,__ and __Light Sleep accounts for >50%__ of a typical night of sleep.
# _New features_ are created to understand how the Lockdown has affected each stage. 

# In[74]:


# Add a new column for the percentage of deep sleep, rem sleep
df["deep_sleep_pct"] = ((df["deep_sleep_min"] / df["sleep_duration_min"]) * 100).round(2)
df["rem_sleep_pct"] = ((df["rem_sleep_min"] / df["sleep_duration_min"]) * 100).round(2)
df["light_sleep_pct"] = ((df["light_sleep_min"] / df["sleep_duration_min"]) * 100).round(2)


# ### Statistical Analysis

# Given that the main goal of this project is to compare sleep patterns __Before and After Lockdown__ (or during the Lockdown, more accurately), it is crucial __to split the dataset into two data frames.__

# In[75]:


# Filter the DataFrame for data points before and after the lockdown date
df_before_lockdown = df[df.index < "2020-03-15"]
df_after_lockdown = df[df.index >= "2020-03-15"]


# Determining the number of data points available in each dataset is important. This provides an initial understanding of the breadth of data being worked with.

# In[76]:


# Count the number of data points Before and After Lockdown
num_before_lockdown = len(df_before_lockdown)
num_after_lockdown = len(df_after_lockdown)

# Print the results
print(f"Number of data points before the lockdown:{num_before_lockdown}\n"
      f"Number of data points after the lockdown: {num_after_lockdown}")


# The dataset's split shows an __imbalance, with 138 data points recorded Before Lockdown and 309 After Lockdown.__ This inequality could potentially affect the analysis outcomes. 
# Given this situation, it becomes critical to conduct __a statistical test.__ This process will show if the two data sets (Before and After Lockdown) are genuinely different, making a comparative analysis meaningful and correct.
# 
# __The hypotheses for this analysis are described as follows:__
#  - __Null Hypothesis (H0):__ No significant difference exists in sleep patterns Before and After Lockdown.
# 
#  - __Alternative Hypothesis (HA):__ There is a significant difference in sleep patterns Before and After Lockdown.
# 
# A significant result from the statistical test _(with a p-value less than 0.05)_ would suggest rejecting the Null Hypothesis (H0), meaning there is __a significant difference__ in sleep patterns Before and After Lockdown.
#  
# However, before running a statistical test should verify whether the datasets follow a __normal distribution.__ For that purpose, the __Shapiro-Wilk test__ is utilized. This test is preferred due to its robustness and accuracy, even when handling smaller sample sizes.
# 
# 
# 

# In[106]:


# Shapiro-Wilk test for normality
_, p_value_before = stats.shapiro(df_before_lockdown["sleep_duration_min"])
_, p_value_after = stats.shapiro(df_after_lockdown["sleep_duration_min"])

# Histogram
plt.hist(df_before_lockdown["sleep_duration_min"], bins="auto", color=color[1], alpha=0.7, label="Before Lockdown")
plt.hist(df_after_lockdown["sleep_duration_min"], bins="auto", color=color[2], alpha=0.7, label="After Lockdown")

# Set labels
plt.ylabel("Frequency", fontsize = 16)
plt.title("Shapiro-Wilk Test", fontsize=20, fontweight='bold')


# Fix ticklabel directions and size
plt.xticks(size = 14, rotation=0)
plt.yticks(size = 14,rotation=0)
plt.legend(fontsize = 16)

plt.show()

# Print Shapiro-Wilk test p-values
print(f" Shapiro-Wilk Test p-values:\n"
      f" Before Lockdown: {p_value_before}\n"
      f" After Lockdown: {p_value_after}")


# The Shapiro-Wilk test p-values __Before Lockdown (6.10e-05) and After Lockdown (1.55e-07)__ are significantly less than the established 0.05 threshold. This leads us __to reject the null hypothesis__ of the Shapiro-Wilk test that assumes a normal distribution, suggesting that the sleep patterns during these periods __do not follow a normal distribution.__

# As a result of this finding, parametric tests like a paired t-test, which assume a normal distribution, would not be suitable. Instead, we should apply a non-parametric equivalent for analyzing related samples.
# 
# An option for this would be the __Wilcoxon Signed-Rank Test__. This test, often used as a non-parametric alternative to the paired t-test, does not require the assumption of normal distribution and is, therefore, a fitting choice for the sleep pattern data analysis.

# In[79]:


# Select the corresponding data points
before_lockdown = df_before_lockdown["sleep_duration_min"][:num_before_lockdown]
after_lockdown = df_after_lockdown["sleep_duration_min"][:num_before_lockdown]

# Conduct the Wilcoxon Signed-Rank Test
w, p = wilcoxon(before_lockdown, after_lockdown)

print(f"The Wilcoxon Signed-Rank Test returns a p-value of {p}")


# In[80]:


# List of columns to compare
columns = ["sleep_duration_min", "light_sleep_min", "rem_sleep_min", "deep_sleep_min", 
           "readiness", "sleep_score", "activity_score", "steps"]

# Create an empty DataFrame to store the results
wilcoxon_test = pd.DataFrame(columns=["column", "w-statistic", "p-value"])

# Perform the Wilcoxon Signed-Rank Test for each column
for column in columns:
    before_lockdown = df_before_lockdown[column][:num_before_lockdown]
    after_lockdown = df_after_lockdown[column][:num_before_lockdown]

    # Conduct the Wilcoxon Signed-Rank Test
    w, p = wilcoxon(before_lockdown, after_lockdown)

    # Determine if the difference is statistically significant
    significant = "Yes" if p < 0.05 else "No"

    # Append the results to the DataFrame
    wilcoxon_test = wilcoxon_test.append({"column": column, "w-statistic": w, "p-value": p, 
                                    "significant": significant}, ignore_index=True)

# Print the results
wilcoxon_test


# Based on wilcoxon_test table results, the lockdown period has significantly impacted various sleep-related variables, including light sleep duration, REM sleep duration, deep sleep duration, readiness score, sleep score, activity score, and steps count.

# In[77]:


# Calculate the number of rows needed for subplots
num_rows = int(np.ceil(len(columns)/2))

# Create subplots in a grid of 2 columns
fig, axs = plt.subplots(num_rows, 2, figsize=(15, 30))

# Flatten the axes array to iterate over it
axs = axs.flatten()

# For each category
for i, column in enumerate(columns):
    # Prepare data
    data = [df_before_lockdown[column], df_after_lockdown[column]]

    # Generate box plot on specific subplot
    box_plot = sns.boxplot(data=data, palette=color, ax=axs[i])

    # Set title and labels
    box_plot.set_title(column, fontsize=20)
    box_plot.set_xticklabels(['Before Lockdown', 'After Lockdown'], size = 14)

    # Display the mean, maximum, and minimum values
    means = [df_before_lockdown[column].mean(), df_after_lockdown[column].mean()]
    max_vals = [df_before_lockdown[column].max(), df_after_lockdown[column].max()]
    min_vals = [df_before_lockdown[column].min(), df_after_lockdown[column].min()]

    box_plot.text(0, max_vals[0], f"max: {max_vals[0]}", ha='center', va='bottom',fontsize = 14)
    box_plot.text(0, min_vals[0], f"min: {min_vals[0]}", ha='center', va='top',fontsize = 14)
    box_plot.text(0, means[0], f"mean: {means[0]:.1f}", ha='center', va='center', fontsize = 14, weight= "bold")

    box_plot.text(1, max_vals[1], f"max: {max_vals[1]}", ha='center', va='bottom',fontsize = 14)
    box_plot.text(1, min_vals[1], f"min: {min_vals[1]}", ha='center', va='top',fontsize = 14)
    box_plot.text(1, means[1], f"mean: {means[1]:.1f}", ha='center', va='center', fontsize = 14, weight= "bold")



# Tight layout for better spacing
plt.tight_layout()
plt.show()


# In[ ]:





# In[ ]:





# _______

# To simplify understanding of data, let's create a function for converting minutes into minutes and hours.

# In[81]:


def converting_minutes_to_hours(minutes):
    """Get the minutes and convert to hours and minutes

    Args:
      minutes (int): The integer amount of minutes

    Returns:
      str: The formatted string representing hours and minutes
    """
    h = minutes // 60
    m = minutes % 60
    w = f"{int(h)}h:{int(m)}m"
    return (w)


# To easily calculate and print _the average sleep duration_ for different sleep types `average_sleep_duration` function was created. The function takes a specific sleep column's mean value and presents it in a more readable format. The `average_sleep_duration` function internally calls the `converting_minutes_to_hours function`, which converts the average sleep duration into hours and minutes format. This conversion helps to present the results more precisely.

# In[82]:


def plot_bar(data, xlabel, ylabel, title, period):
    """Get the data and plot vertical bars

    Args:
      data (pandas DataFrame): The DataFrame with specific columns to plot.
      xlabel (str): The label for x axis.
      ylabel (str): The label for y axis.
      title (str): The title for the plot.
      period (strftime) : The period of time


    Returns:
      plot bars
    """
    # Plotting data
    ax = data.plot.bar(color=[color[0], color[1], color[2]])
    

    # Set labels and legend
    ax.set_xticklabels(data.index.strftime(period), rotation=0, fontsize=14)
    ax.set_xlabel(xlabel, fontsize=16)
    ax.set_ylabel(ylabel, fontsize=16)
    ax.set_title(title, fontsize=20)

    ax.legend(loc='best')
    plt.rc('legend', fontsize=13)

    # Patches is everything inside of the chart
    for rect in ax.patches:
        # Find where everything is located
        height = rect.get_height()
        width = rect.get_width()
        x = rect.get_x()
        y = rect.get_y()

        # The height of the bar is the data value and can be used as the label
        label_text = f'{height:.0f}'

        # ax.text(x, y, text)
        label_x = x + width/2
        label_y = y + height/2

        # plot only when height is greater than specified value
        if height > 0:
            ax.text(
                label_x,
                label_y,
                label_text,
                ha='center',
                va='center',
                fontsize=13
                )


# In[117]:


def plot_line(data, xlabel, ylabel, title,convert_to_hours=False):
    """Get the data and plot it in lines

    Args:
      data (pandas DataFrame): The data to plot.
      xlabel (str): The label for x axis.
      ylabel (str): The label for y axis.
      title (str): The title for the plot.
      labels (dict, optional): Custom labels for the line plots; defaults to None. 
      convert_to_hours (bool, optional): Whether to convert values into hours and minutes; defaults to False.

    Returns:
      plot
    """

    # Plotting data
    
    ax = data.plot.line(grid="white",marker='o', markersize=7, linewidth=2, color=color)

    # Set axis labels and legend
    ax.set_xlabel(xlabel, fontsize=16)
    ax.set_ylabel(ylabel, fontsize=16)
    ax.set_title(title, fontsize=20)
    
    # Set tick label
    plt.xticks(size = 14, rotation=0)
    plt.yticks(size = 14,rotation=0)

    # Add a red vertical line for pointing when lockdown started
    x = "2020-03-15"
    ax.axvline(x, color="red")
    trans = ax.get_xaxis_transform()
    plt.text(x, .05, 'Lockdown Started', transform=trans)
    ax.axvline(x="2020-03-15", color="red")

    ax.legend(loc='best')
    plt.rc('legend', fontsize=13)

   # Add labels to line plots
    for col in data.columns:
        for x, y in zip(data.index, data[col]):
            if convert_to_hours:
                value = converting_minutes_to_hours(y)
            else:
                value = y
            if pd.notnull(value):
                value_label = f"{value}" if convert_to_hours else str(value)
                ax.annotate(value_label, (x, y),
                             textcoords="offset points",
                             xytext=(1, 10),
                             ha='center',fontsize=12)

    plt.show()


# In[84]:


def plotting_distplots(data,title,ncols):
    """Get the data and plot distplot with median and mean

    Args:
      data (pandas DataFrame): The DataFrame with specific columns to plot.


    Returns:
      distplot
    """
    # Initalize a figure and axes
    fig, axes = plt.subplots(ncols=ncols, nrows=1)
    fig.suptitle(title, fontsize=20)

    # Plot the data
    for i, col in enumerate(data.columns):
        sns.histplot(data[col], ax=axes[i], color=color[2], alpha=0.5)
        
        axes[i].axvline(
              x=data[col].median(),
              color="#d98100",
              label="Median: " + str(int(data[col].median())),
              linestyle='--', linewidth=2
              )

        axes[i].axvline(
              x=data[col].mean(),
              color="#d98100",
              label="Mean: " + str(int(data[col].mean())),
              linestyle='-',
              linewidth=2
              )

        axes[i].legend()

    plt.show()


# # GENERAL OVERVIEW

# ##  Does the person have enough sleep?

# The sleep range for adults (26-64 years old)  remains __7-9 hours or 420 - 540 minutes__. [How Much Sleep Do We Really Need?](https://www.sleepfoundation.org/press-release/national-sleep-foundation-recommends-new-sleep-times)
# \
# Also, deep and REM sleep are crucial stages. __REM__ sleep constitutes of __20-25%__ of total sleep. While __deep__ sleep takes __13-23%__ of total sleep

# ### Analyzing Sleep Duration: examining REM, deep, light, and total sleep

# In[85]:


def average_sleep_duration(data, column, kind):
    """Get the mean of a column
       and convert the result from minutes to hours and minutes

    Args:
      column (pandas Series): The data to calculate.
      kind (str): The kind of sleep

    Returns:
      str
    """
    # Calculating average sleep duration
    average_sleep_duration_column = data[column].mean()

    # Converting minutes to hours
    a = converting_minutes_to_hours(average_sleep_duration_column)
    # Print average sleep duration

    print (f"Average {kind} sleep is {a}")


# In[86]:


# Calculate average duration of sleep for all periods
average_sleep_duration(df, "sleep_duration_min", kind="duration")
average_sleep_duration(df, "deep_sleep_min", kind="deep")
average_sleep_duration(df, "rem_sleep_min", kind="REM")
average_sleep_duration(df, "light_sleep_min", kind="light")


# From the summary statistics above `df.describe()`, we have already glanced at the average sleep duration. Now, let's delve deeper into this part. 
# - The __average sleep duration is 7 hours 21 minutes__, which falls within the recommended range of 7-9 hours for adults aged 18-64, as the National Sleep Foundation suggested.
# - __Average deep sleep is 1 hour 25 min__, within a typical range of 1 to 1.5 hours. Deep sleep is essential for physical restoration, memory consolidation, and sleep quality. 
# - __An average REM sleep duration of 1 hour and 32 minutes__ is generally considered within a healthy range. REM (Rapid Eye Movement) sleep is a critical stage of the sleep cycle associated with dreaming, memory consolidation, and emotional processing. Adults spend approximately 20-25% of their total sleep time in REM sleep. While no universally specified "good" or "bad" duration exists for REM sleep, a range of 1 to 2 hours is commonly observed for healthy adults. 
# - The average duration of light sleep varies among individuals, but it commonly falls within the range of 4 to 6 hours for adults. Light sleep is an important stage of the sleep cycle that facilitates transitions between wakefulness and deeper sleep stages.
# An average __light sleep duration of 4 hours and 23 minutes__ is typical for healthy adults. 

# In[87]:


# Average sleep duration for the entire period
average_sleep = df["sleep_duration_min"].mean()

# Average sleep duration before lockdown
average_sleep_before = df_before_lockdown["sleep_duration_min"].mean()

# Average sleep duration after lockdown
average_sleep_after = df_after_lockdown["sleep_duration_min"].mean()

print(f"The average sleep duration for the entire time period is {int(average_sleep)} min"
      f"or {converting_minutes_to_hours(average_sleep)}. \n"
      f"Before the lockdown, the average time was {int(average_sleep_before)} "
      f"or {converting_minutes_to_hours(average_sleep_before)}. \n"
      f"However, after the lockdown, the time increased to {int(average_sleep_after)} min"
      f" or {converting_minutes_to_hours(average_sleep_after)}"
      f" which is longer by {round(average_sleep_after - average_sleep_before)} minutes.")


# __Reminder:__  Deep sleep takes __13-23%__ of total sleep.

# In[88]:


# The average percentage of deep sleep
avg_deep_pct = df['deep_sleep_pct'].mean()

# The average percentage of deep sleep before lockdown
avg_deep_before_pct = df_before_lockdown['deep_sleep_pct'].mean()

# The average percentage of deep sleep after lockdown
avg_deep_after_pct = df_after_lockdown['deep_sleep_pct'].mean()

print(
    f"The average percentage of deep sleep is {int(avg_deep_pct)}%.\n"
    f"Before lockdown, percentage was {int(avg_deep_before_pct)}%,\n"
    f"but after lockdown average percent of deep sleep decreased "
    f"and equaled {int(avg_deep_after_pct)}% of total sleep duration."
      )


# __Reminder:__ REM sleep takes  __20-25%__ of total sleep.

# In[89]:


# The average percentage of REM sleep
avg_rem_pct = df['rem_sleep_pct'].mean()

# The average percentage of REM sleep before lockdown
avg_rem_before_pct = df_before_lockdown['rem_sleep_pct'].mean()

# The average percentage of REM sleep after lockdown
avg_rem_after_pct = df_after_lockdown['rem_sleep_pct'].mean()

print(
    f"The average percentage of REM sleep is {int(avg_rem_pct)}%. "
    f"Before lockdown, the average percentage was "
    f"{int(avg_rem_before_pct)}%,\n"
    f"but after lockdown average percent of REM sleep increased to "
    f"{int(avg_rem_after_pct)}%"
    f" of total sleep.")


# Based on the intermediate analysis of sleep patterns, the __average percentage of Deep Sleep is 19%,__ which is in the necessary _range of 13-23 % of total sleep_. Deep sleep is associated with physical restoration and rejuvenation.
# __Before Lockdown__, the average percentage of __Deep Sleep was at 22%.__ However, __After Lockdown,__ there was a slight decrease in the average percentage of Deep Sleep, which __equaled 18% of the total Sleep duration__, but still within the acceptable range. This decrease may indicate changes in sleep patterns or factors that influenced the quality of Deep Sleep After Lockdown. 
# 
# The __REM Sleep average percentage is 20%__, in the proper _range of 20-25% of total sleep._ Before the Lockdown, the average percentage of REM sleep was 19%, indicating a relatively balanced distribution of REM sleep during that time. Interestingly, After the Lockdown, there was an increase in the average percentage of REM sleep, reaching 21% of the total sleep duration, which can have positive implications for cognitive processes and overall sleep quality.

# ### Analysis of sleep score, activity score, and readiness

# Readiness Score ranges from 0 to 100. It provides a quick assessment of a person's current state and the need for recovery and rest:
# - 85 or higher: Optimal
# - 70-84: Good, the person has recovered well enough
# - Under 70: Pay attention; the person is not fully recovered

# In[90]:


# Calculate average scores for sleep, activity, and readiness
avg_sleep_score = df['sleep_score'].mean()
avg_sleep_score_before = df_before_lockdown['sleep_score'].mean()
avg_sleep_score_after = df_after_lockdown['sleep_score'].mean()

avg_activity_score = df['activity_score'].mean()
avg_activity_score_before = df_before_lockdown['activity_score'].mean()
avg_activity_score_after = df_after_lockdown['activity_score'].mean()

avg_readiness_score = df['readiness'].mean()
avg_readiness_score_before = df_before_lockdown['readiness'].mean()
avg_readiness_score_after = df_after_lockdown['readiness'].mean()

avg_steps = df['steps'].mean()
avg_steps_before = df_before_lockdown['steps'].mean()
avg_steps_after = df_after_lockdown['steps'].mean()

# Create a DataFrame for the average scores
data = {
    'avg_sleep_score': [avg_sleep_score, avg_sleep_score_before, avg_sleep_score_after],
    'avg_activity_score': [avg_activity_score, avg_activity_score_before, avg_activity_score_after],
    'avg_readiness_score': [avg_readiness_score, avg_readiness_score_before, avg_readiness_score_after],
    'avg_steps': [avg_steps, avg_steps_before, avg_steps_after]
}

index = ['Overall', 'Before Lockdown', 'After Lockdown']

average_scores_df = pd.DataFrame(data, index=index).round(1)


# In[91]:


# Display the table
average_scores_df


# In[92]:


# Plot displots for sleep, activity and readiness scores
plotting_distplots(df[["sleep_score", "activity_score", "readiness","steps"]], "Displots",4)


# In[93]:


# Plot displots for sleep, activity and readiness scores
plotting_distplots(df_before_lockdown[["sleep_score", "activity_score", "readiness","steps"]],"Scores Before ",4)


# In[94]:


# Plot displots for sleep, activity and readiness scores
plotting_distplots(df_after_lockdown[["sleep_score", "activity_score", "readiness","steps"]], "Scores After",4)


# __Based on the provided data, the average Sleep Score is 78%, the average Activity Score is 80.8%, and the average Readiness Score is 81.1%.__
# 
# Before the lockdown, the average Sleep Score was 75%, the average Activity Score was 78.8%, and the average Readiness Score was 77.8%.
# 
# However, After the Lockdown, there was an improvement in all scores. The average Sleep Score increased to 79.7%, suggesting better sleep quality. The average Activity Score increased to 81.6%, indicating higher physical activity levels. The average Readiness score increased to 82.5%, suggesting a higher state of alertness and preparedness.
# These improvements in scores After the Lockdown indicate positive changes in sleep quality, activity levels, and overall readiness.
# 
# _But it's important to note that all scores Before and After Lockdown lie between 75 and 84, which is considered Good and means that person rests well enough. But it is still room for improvement to transfer into a higher range of 85 and higher which is Optimal._
# 

# ### Bed time routine

# In[123]:


# Convert start and end times to numerical values
start_of_sleep = mdates.datestr2num(df["sleep_starts"])
end_of_sleep = mdates.datestr2num(df["sleep_ends"])

# Create the plot
fig, ax = plt.subplots(figsize=(15, 8))

# Plot the start and end times
plt.plot_date(start_of_sleep, df.index, color=color[2], alpha=0.5, marker='o', markersize=8)
plt.plot_date(end_of_sleep, df.index,color=color[1],alpha=0.5, marker='o', markersize=8)

# Set the format of the major x-ticks
majorFmt = mdates.DateFormatter('%H:%M')
ax.xaxis.set_major_locator(mdates.HourLocator())
ax.xaxis.set_major_formatter(majorFmt)

# Set labels and legend
months = pd.date_range(start=df.index[0], end=df.index[-1], freq='M').strftime('%Y-%b')
ytick_positions = pd.date_range(start=df.index[0], end=df.index[-1], freq='M')
ax.set_yticks(ytick_positions)
ax.set_yticklabels(months, rotation=0, fontsize=12)

ax.set_xlabel("Time of Day", fontsize=15)
ax.set_ylabel("Months", fontsize=15)
ax.set_title("Bed Time", fontsize=18)

plt.show()


# In[ ]:





# In[96]:


# Convert start and end times to numerical values
start_of_sleep_before = mdates.datestr2num(df_before_lockdown["sleep_starts"])
end_of_sleep_before = mdates.datestr2num(df_before_lockdown["sleep_ends"])
start_of_sleep_after = mdates.datestr2num(df_after_lockdown["sleep_starts"])
end_of_sleep_after = mdates.datestr2num(df_after_lockdown["sleep_ends"])

# Create two plots: Before Lockdown and After Lockdown
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8), sharex=True)

# Plot the start and end times before lockdown
ax1.plot_date(start_of_sleep_before, df_before_lockdown.index,color=color[2], alpha=0.5, marker='o', markersize=8)
ax1.plot_date(end_of_sleep_before, df_before_lockdown.index,color=color[1],alpha=0.5, marker='o', markersize=8)
ax1.set_title("Bed Time - Before Lockdown", fontsize=18)

# Plot the start and end times after lockdown
ax2.plot_date(start_of_sleep_after, df_after_lockdown.index,color=color[2], alpha=0.5, marker='o', markersize=8)
ax2.plot_date(end_of_sleep_after, df_after_lockdown.index,color=color[1],alpha=0.5, marker='o', markersize=8)
ax2.set_title("Bed Time - After Lockdown", fontsize=18)

# Set the format of the major x-ticks
majorFmt = mdates.DateFormatter('%H:%M')
ax1.xaxis.set_major_locator(mdates.HourLocator())
ax1.xaxis.set_major_formatter(majorFmt)
ax2.xaxis.set_major_locator(mdates.HourLocator())
ax2.xaxis.set_major_formatter(majorFmt)

# Set x-axis limits from 00:00 to 23:00
ax1.set_xlim(mdates.datestr2num("00:00"), mdates.datestr2num("23:00"))
ax2.set_xlim(mdates.datestr2num("00:00"), mdates.datestr2num("23:00"))

# Set y-axis format for both plots
ax1.yaxis.set_major_locator(plt.MaxNLocator(6))
ax2.yaxis.set_major_locator(plt.MaxNLocator(6))

# Set labels and legend for the last plot
months = pd.date_range(start=df_after_lockdown.index[0], end=df_after_lockdown.index[-1], freq='M').strftime('%Y-%b')
ytick_positions = pd.date_range(start=df_after_lockdown.index[0], end=df_after_lockdown.index[-1], freq='M')
ax2.set_yticks(ytick_positions)
ax2.set_yticklabels(months, rotation=0, fontsize=12)
ax2.set_xlabel("Time of Day", fontsize=15)
ax2.set_ylabel("Months", fontsize=15)

plt.tight_layout()
plt.show()


# In[97]:


# Convert "sleep_starts" and "sleep_ends" columns to datetime format
df_before_lockdown.loc[:, "sleep_starts"] = pd.to_datetime(df_before_lockdown["sleep_starts"])
df_before_lockdown.loc[:, "sleep_ends"] = pd.to_datetime(df_before_lockdown["sleep_ends"])
df_after_lockdown.loc[:, "sleep_starts"] = pd.to_datetime(df_after_lockdown["sleep_starts"])
df_after_lockdown.loc[:, "sleep_ends"] = pd.to_datetime(df_after_lockdown["sleep_ends"])

# Calculate the most common start and end sleep times for Before Lockdown
most_common_start_time_before = df_before_lockdown["sleep_starts"].dt.strftime("%H:%M").mode()[0]
most_common_end_time_before = df_before_lockdown["sleep_ends"].dt.strftime("%H:%M").mode()[0]

# Calculate the most common start and end sleep times for After Lockdown
most_common_start_time_after = df_after_lockdown["sleep_starts"].dt.strftime("%H:%M").mode()[0]
most_common_end_time_after = df_after_lockdown["sleep_ends"].dt.strftime("%H:%M").mode()[0]

print("Before Lockdown:")
print(f"Most common start sleep time: {most_common_start_time_before}")
print(f"Most common end sleep time: {most_common_end_time_before}")

print("\nAfter Lockdown:")
print(f"Most common start sleep time: {most_common_start_time_after}")
print(f"Most common end sleep time: {most_common_end_time_after}")


# Before the Lockdown, the most common time to __fall asleep was 22:35__, and the usual waking up time was 06:00. This shows a relatively consistent sleep routine. 
# 
# However, After Lockdown changed these patterns slightly. After the Lockdown, the most common time to go to sleep __shifted earlier to 22:08__, while the most common time to wake up remained almost unchanged at 06:01.

# # Yearly Analysis

# In[37]:


# Convert the index to datetime format if it's not already
df.index = pd.to_datetime(df.index)

# Extract the year from the index
df['year'] = df.index.year

# Count the number of unique days with data for each year
days_with_data = df.groupby('year').size()

# Print the results
for year, count in days_with_data.items():
    print(f"Year {year}: {count} days with data")


# In the sleep data analysis, we have data for three consecutive years: 2019, 2020, and 2021. It is important to note that the number of days with available data varies across these years, which can impact the overall analysis and interpretation.
# For 2019, we have data for 70 days, providing insights into sleep patterns during that period. This limited data coverage should be considered when drawing conclusions or comparing to other years.
# In 2020, we had a more extensive dataset with data available for 327 days. This allows for a more comprehensive analysis of sleep patterns and provides a better understanding of the sleep trends during that year.
# However, for 2021, the dataset only includes data for 50 days. This limited data coverage may restrict the scope of analysis and the ability to capture the complete picture of sleep patterns for that particular year.
# Given the variation in data availability across the years, it is crucial to interpret the findings cautiously. While we can still gain valuable insights from the available data, it is critical to acknowledge the limitations imposed by varying days with data for each year.
# In conclusion, the sleep analysis covers a range of years but with different data coverage for each year. This variability in data availability should be considered when interpreting the results and drawing conclusions regarding sleep patterns throughout the years.

# In[121]:


# Resample the average scores to be yearly
sleep_duration_y = df.resample('Y').mean()[["sleep_duration_min"]].round(2)

# Plot average sleep score
plot_line(
    sleep_duration_y,
    xlabel="Year",
    ylabel="Minutes",
    title="Average Sleep Duration, HH:MM", convert_to_hours=True
    )


# In[39]:


# Resample the average score to be yearly
scores_by_years = df.resample('Y').mean()[[
                                  'sleep_score',
                                  'activity_score',
                                  'readiness'
                                  ]]

# Plot scores by years
plot_bar(
    scores_by_years,
    xlabel='Years', ylabel="Scores",
    title="Scores by Years", period="%Y"
    )


# In[ ]:





# # Monthly 

# In[40]:


# Resample the data to monthly frequency and calculate the mean for each month
df_m = df.resample('M').mean().round(0)


# In[120]:


# Plot the average percentage of deep sleep monthly
plot_line(
    df_m[["sleep_duration_min"]],
    xlabel="Month",
    ylabel="Minutes",
    title="The Average Sleep Duration by Month",
    convert_to_hours=True
    )


# On the line graph, we can observe the average sleep duration by month. At a glance, we can see that the average sleep duration increased after the Lockdown. Let's calculate the average sleep duration before and after the Lockdown.

# In[112]:


# Plotting the average sleep duration of different sleep types over months
sleep_dur_m = df_m[["light_sleep_pct","deep_sleep_pct","rem_sleep_pct"]]

ax = sleep_dur_m.plot.bar(stacked=True, color=color)

# Set x-axis and y-axis tick labels
ax.set_xticklabels(sleep_dur_m.index.strftime("%b"),
                   rotation=0,
                   fontsize=14)
ax.set_yticklabels(ax.get_yticks().astype(int), fontsize=14)

# Set labels and legend
ax.set_xlabel("Month", fontsize=16)
ax.set_ylabel("Percent,%", fontsize=16)
ax.set_title("Average Percent of Sleep Type by Month", fontsize=20)

ax.legend(loc="lower right")
plt.rc("legend", fontsize=13)

# Patches is everything inside of the chart
for rect in ax.patches:
    
    # Find where everything is located
    height = rect.get_height()
    width = rect.get_width()
    x = rect.get_x()
    y = rect.get_y()

    # The height of the bar is the data value and can be used as the label
    label_text = int(height)
    

    # ax.text(x, y, text)
    label_x = x + width / 2
    label_y = y + height / 2

    # Plotting only when height is greater than the specified value
    if height > 0:
        ax.text(label_x, label_y,
                label_text, ha='center',
                va='center', fontsize=14)


# In[119]:


# Plot the average percentage of deep sleep monthly
plot_line(
    df_m[['light_sleep_min', 'rem_sleep_min',
       'deep_sleep_min']],
    xlabel="Month",
    ylabel="Minutes",
    title="The Average Sleep Duration by Month",
    convert_to_hours=True
    )


# In[118]:


# Plot the average scores lines plot
plot_line(
    (df_m[["sleep_score", "readiness", "activity_score"]]).round(1),
    xlabel="Months",
    ylabel="Score",
    title="Average Sleep Scores"
    )


# _Analysis amount of steps_

# In[46]:


# Plot average steps in months
average_steps_by_months = df.resample('M').mean()[['steps']].round(2)

# Use plot_line function for plotting 
plot_line(average_steps_by_months, xlabel = "Months", ylabel = "Steps", title = "Average monthly steps")


# _________

# __________

# # WEEKDAYS

# In[47]:


# Define the desired order of weekdays
weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

# Convert weekdays column to Categorical data type with the specified order
df['weekdays'] = pd.Categorical(df['weekdays'], categories=weekday_order, ordered=True)


# In[48]:


# Group the average sleep duration by weekdays
sleep_weekdays = df.groupby('weekdays')[
                 'sleep_duration_min'].mean().sort_values(ascending=True)

# Group the average sleep duration by weekdays before lockdown
sleep_before = df_before_lockdown.groupby('weekdays')['sleep_duration_min'].mean().sort_values(
                ascending=False)

# Group the average sleep duration by weekdays after lockdown
sleep_after = df_after_lockdown.groupby(
              'weekdays')['sleep_duration_min'].mean().sort_values(
               ascending=False)


# In[49]:


# Concatenate all dataframes in one
sleep_weekdays_merge = pd.concat([
                         sleep_weekdays,
                         sleep_before,
                         sleep_after], axis=1)

sleep_weekdays_merge = sleep_weekdays_merge.set_axis([
                         'sleep_duration_all_period',
                         'sleep_duration_before_lockdown',
                         'sleep_duration_after_lockdown'],
                          axis='columns')
# Sort the DataFrame by weekday_order
sleep_weekdays_merge = sleep_weekdays_merge.reindex(weekday_order)


# In[50]:


# Check the result of merge
sleep_weekdays_merge


# In[100]:


# Plot the average steps duration by weekdays before and after lockdown
ax = sleep_weekdays_merge.plot.line(marker='o', markersize=7, linewidth=2)

# Set labels and legend
ax.set_xticks(range(len(sleep_weekdays_merge)))
ax.set_xticklabels(sleep_weekdays_merge.index, rotation=0, fontsize=12)
ax.set_xlabel("Weekdays", fontsize=15)
ax.set_ylabel("Avg Sleep", fontsize=15)
ax.set_title("Average Sleep By Weekdays", fontsize=18)

ax.legend(loc='best')
plt.rc('legend', fontsize=13)

# Add labels to the data points
for i, col in enumerate(sleep_weekdays_merge.columns):
    for x, y in enumerate(sleep_weekdays_merge[col]):
        label_text = f"{converting_minutes_to_hours(y)}"

        ax.annotate(label_text, (x, y), textcoords='offset points', xytext=(1,12), ha='center', fontsize=12)

plt.show()


# In[101]:


# Calculate average sleep duration for Weekday Before Lockdown
weekday_before_avg_sleep = df_before_lockdown.loc[df_before_lockdown.index.weekday < 5, 'sleep_duration_min'].mean()

# Calculate average sleep duration for Weekdays After Lockdown
weekday_after_avg_sleep = df_after_lockdown.loc[df_after_lockdown.index.weekday < 5, 'sleep_duration_min'].mean()

# Calculate average sleep duration for Weekend Before Lockdown
weekend_before_avg_sleep = df_before_lockdown.loc[df_before_lockdown.index.weekday >= 5, 'sleep_duration_min'].mean()

# Calculate average sleep duration for Weekend After Lockdown
weekend_after_avg_sleep = df_after_lockdown.loc[df_after_lockdown.index.weekday >= 5, 'sleep_duration_min'].mean()

print("Average Sleep Duration:")
print("Weekday Before Lockdown:", converting_minutes_to_hours(weekday_before_avg_sleep))
print("Weekday After Lockdown:", converting_minutes_to_hours(weekday_after_avg_sleep))
print("Weekend Before Lockdown:", converting_minutes_to_hours(weekend_before_avg_sleep))
print("Weekend After Lockdown:", converting_minutes_to_hours(weekend_after_avg_sleep))


# Weekday Before Lockdown: The average sleep duration on weekdays before the lockdown was 6 hours and 30 minutes. This indicates a moderate amount of sleep during the weekdays.
# Weekday After Lockdown: The average sleep duration on weekdays after lockdown increased to 7 hours and 21 minutes. This suggests slightly longer sleep durations on weekdays after the lockdown period.
# Weekend Before Lockdown: On weekends before lockdown, the average sleep duration was 8 hours and 16 minutes. This indicates a relatively longer sleep duration during the weekends.
# Weekend After Lockdown: The average sleep duration on weekends decreased to 7 hours and 55 minutes after the lockdown. Although there was a slight decrease, the average sleep duration remained healthy during the weekends.
# Overall, it can be observed that the average sleep duration increased on weekdays after the lockdown, while there was a slight decrease in sleep duration on weekends after the lockdown. 

# In[ ]:





# In[ ]:





# In[102]:


# Group the average steps amount by weekdays
steps_by_weekdays = df.groupby('weekdays')['steps'].mean().sort_values(ascending=False)

# Group the average steps amount by weekdays before lockdown
steps_weekdays_before = df_before_lockdown.groupby('weekdays')['steps'].mean().sort_values(ascending=False)

# Group the average steps amount by weekdays after lockdown
steps_weekdays_after = df_after_lockdown.groupby('weekdays')['steps'].mean().sort_values(ascending=False)

# Concatenate the three datasets into one
avg_steps_weekdays = pd.concat([steps_by_weekdays, steps_weekdays_before, steps_weekdays_after], axis=1).round(1)
avg_steps_weekdays.columns = ['whole_time', 'before_lockdown', 'after_lockdown']

# Reorder the rows based on the specified weekday order
avg_steps_weekdays = avg_steps_weekdays.reindex(weekday_order)

print(avg_steps_weekdays)


# In[103]:


# Plot the average steps duration by weekdays before and after lockdown
ax = avg_steps_weekdays.plot.line(marker='o', markersize=7, linewidth=2)

# Set labels and legend
ax.set_xticks(range(len(avg_steps_weekdays)))
ax.set_xticklabels(avg_steps_weekdays.index, rotation=0, fontsize=12)
ax.set_xlabel("Weekdays", fontsize=15)
ax.set_ylabel("Steps", fontsize=15)
ax.set_title("Average Steps By Weekdays", fontsize=18)

ax.legend(loc='best')
plt.rc('legend', fontsize=13)

# Add labels to the data points
for i, col in enumerate(avg_steps_weekdays.columns):
    for x, y in enumerate(avg_steps_weekdays[col]):
        label_text = f"{int(y)}"

        ax.annotate(label_text, (x, y), textcoords='offset points', xytext=(1,12), ha='center', fontsize=12)

plt.show()


# In[ ]:





# ------------

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ## Summary
# 

# The numbers show that he had enough sleep duration, which equals __7 hours 21 minutes__. But after lockdown (25th March 2020), his sleep increased up to __7 hours 31 minutes.__
# 
# But if we would compare average sleep duration before and after lockdown, then average sleep duration increases from 6 hours 57 minutes up to 7 hours 31 minutes (8.15%).
# 
# The interesting thing is that REM sleep increased with the sleep duration after lockdown. However, deep sleep decreased after lockdown.
# 
# Here we judge the activity of a person by the number of steps per day. The average daily number of steps is 9463. Before lockdown, the average number was __8151 steps.__ But after the lockdown, the average number of steps increased up to __10029 steps.__
# 
# The most active day is Sunday. That might be connected to the person's extended sleep on Sunday. But the correlation between sleep duration and steps is __0.18__, which is not significant.
# 

# 
# ## Recommendations: 
# - Establishing a consistent sleep schedule __by going to sleep and waking up at the same time every day__, regardless of the day of the week, offers several benefits. It helps regulate your internal clock, promotes better sleep quality, and improves overall health.
# 
# 
# - Aim __sleep duration of 7.5 to 8 hours each night__. This range allows for sufficient rest and prevents the accumulation of fatigue, reducing the temptation to oversleep on weekends. 
# 
# Remember, finding the proper bedtime and wake-up time may require trial and error to determine what works best for your needs. Pay attention to how you feel in the morning and throughout the day to gauge whether you are getting adequate rest. 

# ______

# ---------
