#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 15 22:13:40 2025

@author: tomasgutierrez
"""

"""
Format: The project consist of your answers to 10 (equally-weighed, grade-wise) questions. Each answer
*must* include some text (describing both what you did and what you found, i.e. the answer to the
question), a figure that illustrates the findings and some numbers (e.g. test statistics, confidence
intervals, p-values or the like). Please save it as a pdf document. This document should be 4-6 pages long
(arbitrary font size and margins). About ½ a page/question is reasonable. In addition, open your
document with a brief statement as to how you handled preprocessing (e.g. data cleaning), as this will
apply to all answers. Make sure to include your name.
Deliverables: Upload two files to the Brightspace portal by the due date in the sittyba:
*A pdf (the “project report”) that contains your answers to the questions, as well as an introductory
paragraph about preprocessing, how you seeded the RNG, etc.
*A .py file with the code that performed the data analysis and created the figure

[...]

Description of dataset: The datafile rmpCapstoneNum.csv contains 89893 records. Each of these
records (rows) corresponds to information about one professor.
The columns represent the following information, in order:
1: Average Rating (the arithmetic mean of all individual quality ratings of this professor)
2: Average Difficulty (the arithmetic mean of all individual difficulty ratings of this professor)
3: Number of ratings (simply the total number of ratings these averages are based on)
4: Received a “pepper”? (Boolean - was this professor judged as “hot” by the students?)
5: The proportion of students that said they would take the class again
6: The number of ratings coming from online classes
7: Male gender (Boolean – 1: determined with high confidence that professor is male)
8: Female (Boolean – 1: determined with high confidence that professor is female)

There is a second datafile rmpCapstoneQual.csv that has the same number of 89893 records in the
same order, but only 3 columns containing qualitative information:
1: Major/Field
2: University
3: US State (2 letter abbreviation)

With this dataset in hand, we would like you to answer the following questions:
1. Activists have asserted that there is a strong gender bias in student evaluations of professors, with
male professors enjoying a boost in rating from this bias. While this has been celebrated by ideologues,
skeptics have pointed out that this research is of technically poor quality, either due to a low sample
size – as small as n = 1 (Mitchell & Martin, 2018), failure to control for confounders such as teaching
experience (Centra & Gaubatz, 2000) or obvious p-hacking (MacNell et al., 2015). We would like you to
answer the question whether there is evidence of a pro-male gender bias in this dataset.
Hint: A significance test is probably required.
2. Is there an effect of experience on the quality of teaching? You can operationalize quality with
average rating and use the number of ratings as an imperfect – but available – proxy for experience.
Again, a significance test is probably a good idea.
3. What is the relationship between average rating and average difficulty?
4. Do professors who teach a lot of classes in the online modality receive higher or lower ratings than
those who don’t? Hint: A significance test might be a good idea, but you need to think of a creative but
suitable way to split the data.
5. What is the relationship between the average rating and the proportion of people who would take
the class the professor teaches again?
6. Do professors who are “hot” receive higher ratings than those who are not? Again, a significance
test is indicated.
7. Build a regression model predicting average rating from difficulty (only). Make sure to include the R2
and RMSE of this model.
8. Build a regression model predicting average rating from all available factors. Make sure to include
the R2 and RMSE of this model. Comment on how this model compares to the “difficulty only” model
and on individual betas. Hint: Make sure to address collinearity concerns.
9. Build a classification model that predicts whether a professor receives a “pepper” from average
rating only. Make sure to include quality metrics such as AU(RO)C and also address class imbalances.
10. Build a classification model that predicts whether a professor receives a “pepper” from all available
factors. Comment on how this model compares to the “average rating only” model. Make sure to
include quality metrics such as AU(RO)C and also address class imbalances.
Extra credit: Tell us something interesting about this dataset that is not trivial and not already part of
an answer (implied or explicitly) to these enumerated questions [Suggestion: Do something with the
qualitative data, e.g. major, university or state by linking the two data file
"""

#%%
# Data Preprocessing

import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns

seed = 11233917 # my N-number
random.seed(seed)

# both csvs have no column names/a header which is frustrating, I will add them myself
data_num = pd.read_csv("rmpCapstoneNum.csv", header=None)
data_num.columns = ['avg_rating', 
                    'avg_difficulty', 
                    'num_ratings', 
                    'received_pepper',
                    'would_take_again_pct', 
                    'num_online_ratings', 
                    'is_male', 
                    'is_female']


data_qual = pd.read_csv("rmpCapstoneQual.csv", header=None)
data_qual.columns = ['major',
                     'university',
                     'state'
                    ]

print("Numeric shape:", data_num.shape)
print("Qualitative shape:", data_qual.shape)
# Numeric shape: (89893, 8)
# Qualitative shape: (89893, 3)

# Time to merge/concatenate the data the data

merged_data = pd.concat([data_num, data_qual], axis=1)

# Now, we need to think about how to remove NaNs.
# Firstly, let's remove rows from merged_data that are completely NaN for non-boolean columns. 
# This will allow us to keep the records in order for the master dataframe
# If we want to work with larger sample size of the boolean columns after
# We can take the data from the original CSV file and clean it seperately

non_bool_cols = [
    'avg_rating', 'avg_difficulty', 'num_ratings',
    'received_pepper', 'would_take_again_pct', 
    'num_online_ratings', 'major', 'university', 'state'
]

merged_data = merged_data.dropna(how='all', subset=non_bool_cols).reset_index(drop=True)
print(merged_data.shape)
# (70004, 11)

# For significance tests, lets use alpha level = 0.005
alpha = 0.005

#%%
"""
1. Activists have asserted that there is a strong gender bias in student evaluations of professors, with
male professors enjoying a boost in rating from this bias. While this has been celebrated by ideologues,
skeptics have pointed out that this research is of technically poor quality, either due to a low sample
size – as small as n = 1 (Mitchell & Martin, 2018), failure to control for confounders such as teaching
experience (Centra & Gaubatz, 2000) or obvious p-hacking (MacNell et al., 2015). We would like you to
answer the question whether there is evidence of a pro-male gender bias in this dataset.
Hint: A significance test is probably required.
"""

# H0 - There isnt a difference between male professor and female professor ratings
# H1 - Male professors receive higher ratings

# First thing we need to do is remove rows where is_male and is_female are both 0
# And where avg_rating is NaN
gendered_data = merged_data[(merged_data['is_male'] + merged_data['is_female']) == 1].reset_index(drop=True)
gendered_data = gendered_data.dropna(subset=['avg_rating']).reset_index(drop=True)

print(gendered_data.shape)
# (52089, 11)

num_males = gendered_data['is_male'].sum()
num_females = gendered_data['is_female'].sum()

print(f"Number of Male Professors: {num_males}")
print(f"Number of Female Professors: {num_females}")
# Number of Male Professors: 27163
# Number of Female Professors: 24926

# Add 'gender' column
gendered_data['gender'] = gendered_data['is_male'].map({1: 'Male'})
gendered_data.loc[gendered_data['is_female'] == 1, 'gender'] = 'Female'

# Now, let's take the means
male_mean = gendered_data[gendered_data['gender'] == 'Male']['avg_rating'].mean()
female_mean = gendered_data[gendered_data['gender'] == 'Female']['avg_rating'].mean()

print(f"Male average rating: {male_mean:.3f}")
print(f"Female average rating: {female_mean:.3f}")
# Male average rating: 3.878
# Female average rating: 3.811

# Ok, now we need to think of a significance test that checks if the underlying distributions
# Are different

# What are the assumptions of the significance tests, and which ones do our data break?

# Our sample size is great enough to not worry about CLT not kicking in, but
# we obviously cannot use z-test here, since we do not have access to population parameters

# Let's visualize the male and female ratings distributions 
sns.histplot(gendered_data[gendered_data['is_male'] == 1]['avg_rating'], kde=True, color='blue', label='Male')
sns.histplot(gendered_data[gendered_data['is_female'] == 1]['avg_rating'], kde=True, color='red', label='Female')
plt.legend()
plt.title('Distribution of Average Ratings by Gender')
plt.show()

# there is a disproportionate amount of high ratings, let's keep them to avoid selection bias,
# and keep the data natural

# Check skewness
male_skew = gendered_data[gendered_data['is_male'] == 1]['avg_rating'].skew()
female_skew = gendered_data[gendered_data['is_female'] == 1]['avg_rating'].skew()
# Male skew: -0.99, Female skew: -0.90

# Not ideal skew but also our sample size is large enough where it could still be acceptable to use t-test
# So, should we use student or Welch's?

# Let's check for unequal variance 

from scipy.stats import levene

male_ratings = gendered_data[gendered_data['is_male'] == 1]['avg_rating']
female_ratings = gendered_data[gendered_data['is_female'] == 1]['avg_rating']

stat, p = levene(male_ratings, female_ratings)
print(f"Levene’s test p-value: {p:.15f}")
# Levene’s test p-value: 0.0000
# This means we should definitely use Welch's - unequal variances

from scipy.stats import ttest_ind

t_stat, p_val_t = ttest_ind(male_ratings, female_ratings, equal_var=False)
print(f"t-statistic: {t_stat:.4f}")
print(f"p-value: {p_val_t:.6f}")

# Even though sample is large and Welch's t-test justified well
# let's try Mann-Whitney U

from scipy.stats import mannwhitneyu

u_stat, p_u = mannwhitneyu(male_ratings, female_ratings, alternative='greater')
print(f"Mann-Whitney U: {u_stat:.2f}, p = {p_u:.6f}")
# Mann-Whitney U: 346129258.50, p = 0.000004

# Let's get the effect size:
mean_diff = np.mean(male_ratings) - np.mean(female_ratings)
# Pooled variance (unbiased) is not used here; we use average of group variances Since we use Welch's
avg_var = (np.var(male_ratings, ddof=1) + np.var(female_ratings, ddof=1)) / 2
cohens_d = mean_diff / np.sqrt(avg_var)
print(cohens_d)
# [Cohen's d: 0.059]

# There is a small but significance difference between male and female avg_ratings
#Let's quickly get the confidence interval for the difference between male and female ratings

# Means and standard deviations
male_mean = male_ratings.mean()
female_mean = female_ratings.mean()
mean_diff = male_mean - female_mean

male_std = male_ratings.std(ddof=1)
female_std = female_ratings.std(ddof=1)

n_male = len(male_ratings)
n_female = len(female_ratings)

# Standard error for Welch's t-test
se_diff = np.sqrt((male_std**2 / n_male) + (female_std**2 / n_female))

# Degrees of freedom for Welch's t-test
df = ((male_std**2/n_male + female_std**2/n_female)**2) / (((male_std**2/n_male)**2 / (n_male-1))  + ((female_std**2/n_female)**2 / (n_female-1)))

# 95% CI
confidence_level = 0.95
t_crit = stats.t.ppf((1 + confidence_level) / 2, df)
ci_lower = mean_diff - t_crit * se_diff
ci_upper = mean_diff + t_crit * se_diff

print(f"Mean Difference 95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
# Mean Difference 95% CI: [0.0476, 0.0860]

# Let's plot the data now
sns.set_style('whitegrid')
plt.figure(figsize=(8, 6))
sns.boxplot(x='gender', y='avg_rating', data=gendered_data, palette='pastel', width=0.5)
plt.title('Average Professor Ratings by Gender', fontsize=14)
plt.xlabel('Gender', fontsize=12)
plt.ylabel('Average Rating', fontsize=12)
plt.ylim(0, 5)
plt.show()

#%%
"""
2. Is there an effect of experience on the quality of teaching? You can operationalize quality with
average rating and use the number of ratings as an imperfect – but available – proxy for experience.
Again, a significance test is probably a good idea.
"""

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import root_mean_squared_error
import statsmodels.api as sm
from scipy.stats import pearsonr, spearmanr, kruskal

# Lets first clean data appropriately for this problem
experience_data = merged_data.dropna(subset=['num_ratings', 'avg_rating']).reset_index(drop=True)
print(experience_data.shape)

# Let's visualize this
plt.figure(figsize=(8,6))
sns.scatterplot(x='num_ratings', y='avg_rating', data=experience_data, alpha=0.3)
plt.title('Average Rating vs. Number of Ratings')
plt.xlabel('Number of Ratings')
plt.ylabel('Average Rating')
plt.show()

# Based on the graph, there seems to be a very subtle sort of linear trend showing that as num_rating increase,
# avg_rating increase 

# Let's first check this with correlation (Obviously doesn't establish causality)
# Linear relationship: Pearson’s r
# Monotonic relationship: Spearman’s ⍴
# Later we can do an actual regression that will provide is the statistical significance of a given predictor

pearson_corr, pearson_p = pearsonr(experience_data['num_ratings'], experience_data['avg_rating'])
print(f"Pearson correlation: {pearson_corr:.4f}, p-value: {pearson_p:.6f}")
# 0.0374, p-value: 0.000000

spearman_corr, spearman_p = spearmanr(experience_data['num_ratings'], experience_data['avg_rating'])
print(f"Spearman correlation: {spearman_corr:.4f}, p-value: {spearman_p:.6f}")
# Spearman correlation: -0.0632, p-value: 0.000000

# Interesting that both pearson and spearman imply different relationships between the two variables
# This could be a result of the data being nonlinear and/or outliers
# Also notice, correlation can only give us association between variables, it cannot mean causality

# We are probably getting conflicting values here because the relationship appears positively linear until
# roughly 4.7, then it decreases. Just for sanity, let's run a spearman without values above 4.5

# Filter data to exclude avg_rating > 4.5
filtered_data = experience_data[experience_data['avg_rating'] <= 4.5].reset_index(drop=True)

# Spearman correlation
spear_corr_filtered, p_value_spear = spearmanr(filtered_data['num_ratings'], filtered_data['avg_rating'])
print(f"Spearman correlation (filtered): {spear_corr_filtered:.4f}, p-value: {p_value_spear:.6f}")
# Spearman correlation (filtered): 0.1120, p-value: 0.000000

# Pearson correlation
pearson_corr_filtered, p_value_pearson = pearsonr(filtered_data['num_ratings'], filtered_data['avg_rating'])
print(f"Pearson correlation  (filtered): {pearson_corr_filtered:.4f}, p-value: {p_value_pearson:.6f}")
# Pearson correlation  (filtered): 0.1154, p-value: 0.000000

# This is starting to make more sense, again, a small but statistically significant result
# positively correlated from ratings 4.5 

# Let's do linear regression with statsmodels as it allows us to see confidence interval
# and p-value

X = sm.add_constant(experience_data['num_ratings'])
y = experience_data['avg_rating']
model_ols = sm.OLS(y, X).fit()

print(model_ols.summary())  # Shows coef, p-value, CI, etc.

# Predict and evaluate
y_pred_ols = model_ols.predict(X)
rmse_ols = root_mean_squared_error(y, y_pred_ols)
print(f"OLS RMSE: {rmse_ols:.4f}")
# OLS RMSE: 1.1261

# Plot linear regression fit
plt.figure(figsize=(10, 6))
sns.scatterplot(x='num_ratings', y='avg_rating', data=experience_data, alpha=0.3, label='Data')
plt.plot(experience_data['num_ratings'], y_pred_ols, color='red', label='Linear Regression Fit')
plt.title('Linear Regression: Experience vs. Teaching Quality')
plt.xlabel('Number of Ratings')
plt.ylabel('Average Rating')
plt.ylim(1, 5)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Log transform

experience_data['log_num_ratings'] = np.log1p(experience_data['num_ratings'])

# Prepare data for statsmodels
X_log_sm = sm.add_constant(experience_data['log_num_ratings'])  # Add intercept
y_log = experience_data['avg_rating']

# Fit OLS model
model_log_sm = sm.OLS(y_log, X_log_sm).fit()
print(model_log_sm.summary())

# Predict and calculate RMSE
y_pred_log = model_log_sm.predict(X_log_sm)
rmse_log = root_mean_squared_error(y_log, y_pred_log)
print(f"Log-Transformed OLS RMSE: {rmse_log:.4f}")

# Plot the fit
plt.figure(figsize=(10, 6))
sns.scatterplot(x='log_num_ratings', y='avg_rating', data=experience_data, alpha=0.05, label='Data')
plt.plot(experience_data['log_num_ratings'], y_pred_log, color='red', linewidth=2, label='Log-Linear Fit (statsmodels)')
plt.title(f'Log-Transformed Linear Regression\nRMSE: {rmse_log:.4f}')
plt.xlabel('Log(Number of Ratings + 1)')
plt.ylabel('Average Rating')
plt.legend()
plt.ylim(1, 5)
plt.grid(True)
plt.tight_layout()
plt.show()

#%%
# Let's try something different for #2
# Break up professors into 3 parts based on num ratings, and runn a KS test

# STEP 1: Create 3 equal-sized experience tiers
experience_data['experience_tier'] = pd.qcut(
    experience_data['num_ratings'], q=3, labels=['low', 'medium', 'high']
)

# STEP 2: Kruskal-Wallis test across the 3 groups
low_ratings = experience_data[experience_data['experience_tier'] == 'low']['avg_rating']
mid_ratings = experience_data[experience_data['experience_tier'] == 'medium']['avg_rating']
high_ratings = experience_data[experience_data['experience_tier'] == 'high']['avg_rating']

h_stat, p_value = kruskal(low_ratings, mid_ratings, high_ratings)
print(f"Kruskal–Wallis H-statistic: {h_stat:.4f}, p-value: {p_value:.6f}")
# Kruskal–Wallis H-statistic: 298.0125, p-value: 0.000000
# This means at least one experience tier differs from the rest

# Let's just try each group against eachother with a mann whitney u test

comparisons = [
    ('low', 'medium'),
    ('low', 'high'),
    ('medium', 'high')
]

print("\nPairwise Mann–Whitney U tests with Bonferroni correction:")

for group1, group2 in comparisons:
    ratings1 = experience_data[experience_data['experience_tier'] == group1]['avg_rating']
    ratings2 = experience_data[experience_data['experience_tier'] == group2]['avg_rating']
    
    u_stat, p = mannwhitneyu(ratings1, ratings2, alternative='two-sided')
    p_bonf = min(p * len(comparisons), 1.0)  # Bonferroni correction
    
    print(f"{group1} vs {group2} → U = {u_stat:.2f}, raw p = {p:.6f}, corrected p = {p_bonf:.6f}")

# low vs medium → U = 314468428.00, raw p = 0.000000, corrected p = 0.000000
# low vs high → U = 327994134.50, raw p = 0.000000, corrected p = 0.000000
# medium vs high → U = 206977374.00, raw p = 0.145614, corrected p = 0.436841


#%%
"""
3. What is the relationship between average rating and average difficulty?
"""

# First, drop rows that have NaNs in either column

difficulty_data = merged_data.dropna(subset=['avg_rating', 'avg_difficulty']).reset_index(drop=True)
print(difficulty_data.shape)
# (70004, 11)

# Let's visualize with a scatter plot
plt.figure(figsize=(8,6))
sns.scatterplot(x='avg_difficulty', y='avg_rating', data=difficulty_data, alpha=0.3)
plt.xlabel('Average Difficulty')
plt.ylabel('Average Rating')
plt.title('Average Rating vs. Average Difficulty')
plt.legend()
plt.grid(True)
plt.show()

# Definitely seems to be a negative linear relationship, let's get the correlation coefs

pearson_corr, pearson_p = pearsonr(difficulty_data['avg_difficulty'], difficulty_data['avg_rating'])
spearman_corr, spearman_p = spearmanr(difficulty_data['avg_difficulty'], difficulty_data['avg_rating'])

print(f"Pearson correlation: {pearson_corr:.4f}, p-value: {pearson_p:.6f}")
# Pearson correlation: -0.5368, p-value: 0.000000

print(f"Spearman correlation: {spearman_corr:.4f}, p-value: {spearman_p:.6f}")
# Spearman correlation: -0.5114, p-value: 0.000000

# These results definitely align with what I expected, let's just run an OLS reg for more info
X = sm.add_constant(difficulty_data['avg_difficulty'])
y = difficulty_data['avg_rating']
model = sm.OLS(y, X).fit()
print(model.summary())

# RMSE
y_pred = model.predict(X)
rmse = root_mean_squared_error(y, y_pred)
print(f"RMSE: {rmse:.4f}")
# avg_difficulty    -0.6103      0.004   -168.325      0.000      -0.617      -0.603
# R-squared:                       0.288
# RMSE: 0.9508

# Let's plot the regression line over the scatterplot
plt.figure(figsize=(8, 6))
sns.scatterplot(x='avg_difficulty', y='avg_rating', data=difficulty_data, alpha=0.3, label='Data')
x_vals = np.linspace(difficulty_data['avg_difficulty'].min(), difficulty_data['avg_difficulty'].max(), 100)
X_line = sm.add_constant(x_vals) 
y_line = model.predict(X_line)
plt.plot(x_vals, y_line, color='red', linewidth=2, label='OLS Regression Line')
plt.xlabel('Average Difficulty')
plt.ylabel('Average Rating')
plt.title('Average Rating vs. Average Difficulty')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

#%%
"""
4. Do professors who teach a lot of classes in the online modality receive higher or lower ratings than
those who don’t? Hint: A significance test might be a good idea, but you need to think of a creative but
suitable way to split the data.
"""

# First, let's drop rows where where avg_rating,num_online_ratings, num_ratings are NaN

online_data = merged_data.dropna(subset=['avg_rating', 'num_online_ratings', 'num_ratings']).reset_index(drop=True)

# To split up the data, let's make a new column that categorizes a professor as
# High Online, or low Online based on if their num_online_ratings is greater than or less than
# The median of num_online_ratings
threshold = online_data['num_online_ratings'].median()
online_data['teaches_online'] = np.where(online_data['num_online_ratings'] > threshold, 'High Online', 'Low Online')

# Now let's visualize the data to compare the avg_rating between the two groups
sns.boxplot(x='teaches_online', y='avg_rating', data=online_data, palette='pastel')
plt.title('Average Ratings by Online Teaching Group')
plt.ylabel('Average Rating')
plt.xlabel('Online Teaching Volume')
plt.grid(True)
plt.tight_layout()
plt.show()

# Now, let's check for equal variance in the groups
group_high = online_data[online_data['teaches_online'] == 'High Online']['avg_rating']
group_low = online_data[online_data['teaches_online'] == 'Low Online']['avg_rating']

stat, p_levene = levene(group_high, group_low)
print(f"Levene’s test p-value: {p_levene:.6f}")
# Levene’s test p-value: 0.000000

# This means we should assum unequal variances, and use Welch's t-test

t_stat, p_val_t = ttest_ind(group_high, group_low, equal_var=False)
print(f"Welch’s t-test: t = {t_stat:.4f}, p = {p_val_t:.6f}")
# Welch’s t-test: t = -13.3398, p = 0.000000

# negative t-score means that the mean of online teachers ratings is less than not online

u_stat, p_val_u = mannwhitneyu(group_high, group_low)
print(f"Mann-Whitney U test: U = {u_stat:.2f}, p = {p_val_u:.6f}")
# Mann-Whitney U test: U = 295344689.00, p = 0.000000

# Let's do Cohen's D and confidence interval now

mean_diff = group_high.mean() - group_low.mean()
avg_var = (np.var(group_high, ddof=1) + np.var(group_low, ddof=1)) / 2
cohens_d = mean_diff / np.sqrt(avg_var)
print(f"Cohen's d: {cohens_d:.3f}")
# Cohen's d: -0.141

n_high = len(group_high)
n_low = len(group_low)
std_high = group_high.std(ddof=1)
std_low = group_low.std(ddof=1)
se_diff = np.sqrt((std_high**2 / n_high) + (std_low**2 / n_low))
df = ((std_high**2/n_high + std_low**2/n_low)**2) / (
    ((std_high**2/n_high)**2 / (n_high - 1)) +
    ((std_low**2/n_low)**2 / (n_low - 1))
)
t_crit = stats.t.ppf(0.975, df)
ci_lower = mean_diff - t_crit * se_diff
ci_upper = mean_diff + t_crit * se_diff
print(f"Mean Difference 95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
# Mean Difference 95% CI: [-0.1848, -0.1374]

#%%
"""
5. What is the relationship between the average rating and the proportion of people who would take
the class the professor teaches again?
"""

# Essentially same process as question 3
again_data = merged_data.dropna(subset=['avg_rating', 'would_take_again_pct']).reset_index(drop=True)
print(again_data.shape)
# (12160, 11)

plt.figure(figsize=(8,6))
sns.scatterplot(x='would_take_again_pct', y='avg_rating', data=again_data, alpha=0.3)
plt.xlabel('Proportion of Students Who Would Take Again')
plt.ylabel('Average Rating')
plt.title('Average Rating vs. Would Take Again %')
plt.grid(True)
plt.tight_layout()
plt.show()

# Appears to be a positive linear association other than professors with 100 students who
# Would take again. This seems like ceiling effect. It would be sma

again_data = again_data[again_data['would_take_again_pct'] < 100.0].reset_index(drop=True)
print(again_data.shape)
# (8111, 11)

plt.figure(figsize=(8,6))
sns.scatterplot(x='would_take_again_pct', y='avg_rating', data=again_data, alpha=0.3)
plt.xlabel('Proportion of Students Who Would Take Again')
plt.ylabel('Average Rating')
plt.title('Average Rating vs. Would Take Again %')
plt.grid(True)
plt.tight_layout()
plt.show()

pearson_corr, pearson_p = pearsonr(again_data['would_take_again_pct'], again_data['avg_rating'])
spearman_corr, spearman_p = spearmanr(again_data['would_take_again_pct'], again_data['avg_rating'])

print(f"Pearson correlation: {pearson_corr:.4f}, p-value: {pearson_p:.6f}")
# Pearson correlation: 0.8452, p-value: 0.000000
print(f"Spearman correlation: {spearman_corr:.4f}, p-value: {spearman_p:.6f}")
# Spearman correlation: 0.8261, p-value: 0.000000

# Very strong positive correlation confirmed by the corr coefs!

# Let's do OLS regression
X = sm.add_constant(again_data['would_take_again_pct'])
y = again_data['avg_rating']
model = sm.OLS(y, X).fit()
print(model.summary())

# Predict and calculate RMSE
y_pred = model.predict(X)
rmse = root_mean_squared_error(y, y_pred)
print(f"RMSE: {rmse:.4f}")

# would_take_again_pct     0.0300      0.000    142.395      0.000       0.030       0.030
# R-squared:                       0.714
# RMSE: 0.4339

# Now let's plot the OLS reg over the scatterplot
plt.figure(figsize=(8, 6))
sns.scatterplot(x='would_take_again_pct', y='avg_rating', data=again_data, alpha=0.3, label='Data')
x_vals = np.linspace(again_data['would_take_again_pct'].min(), again_data['would_take_again_pct'].max(), 100)
X_line = sm.add_constant(x_vals)
y_line = model.predict(X_line)
plt.plot(x_vals, y_line, color='red', linewidth=2, label='OLS Regression Line')
plt.xlabel('Proportion of Students Who Would Take Again (%)')
plt.ylabel('Average Rating')
plt.title('Average Rating vs. Would Take Again %')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

#%%
"""
6. Do professors who are “hot” receive higher ratings than those who are not? Again, a significance
test is indicated.
"""
# First, let's clean the data
pepper_data = merged_data.dropna(subset=['avg_rating', 'received_pepper']).reset_index(drop=True)
print(pepper_data.shape)
# (70004, 11)

pepper_counts = pepper_data['received_pepper'].value_counts()
print(pepper_counts)
# received_pepper
# 0.0    50408
# 1.0    19596

# Clearly some class imbalance here. Shouldn't matter for something like Welch's t-test
# But will definitely matter in Q9,Q10.
# Welch's t-test if fine if sample sizes are unequal and variance is unequal
# Let's just make groups for hot and not hot, and check for unequal variance anyways

hot = pepper_data[pepper_data['received_pepper'] == 1.0]['avg_rating']
not_hot = pepper_data[pepper_data['received_pepper'] == 0.0]['avg_rating']

stat, p = levene(hot, not_hot)
print(f"Levene’s test p-value: {p:.6f}")
# Levene’s test p-value: 0.000000

# Unequal variances! So let's now run Welch's t-test, and then Mann Whitney U

t_stat, p_val = ttest_ind(hot, not_hot, equal_var=False)
print(f"Welch’s t-test: t = {t_stat:.4f}, p = {p_val:.6f}")
# Welch’s t-test: t = 113.1073, p = 0.000000

u_stat, p_u = mannwhitneyu(hot, not_hot, alternative='two-sided')
print(f"Mann-Whitney U test: U = {u_stat:.2f}, p = {p_u:.6f}")
# Mann-Whitney U test: U = 684243055.50, p = 0.000000

# Let's get mean difference, cohen's D, and confidence interval

mean_hot = np.mean(hot)
mean_not_hot = np.mean(not_hot)
std_hot = np.std(hot, ddof=1)
std_not_hot = np.std(not_hot, ddof=1)

# Sample sizes
n_hot = len(hot)
n_not_hot = len(not_hot)

# Mean difference
mean_diff = mean_hot - mean_not_hot
print(f"Mean difference (Hot - Not Hot): {mean_diff:.4f}")
# Mean difference (Hot - Not Hot): 0.7978

# Cohen's d (pooled variance)
pooled_var = (std_hot**2 + std_not_hot**2) / 2
cohens_d = mean_diff / np.sqrt(pooled_var)
print(f"Cohen's d: {cohens_d:.3f}")
# Cohen's d: 0.831

# Standard error of difference
se_diff = np.sqrt((std_hot**2 / n_hot) + (std_not_hot**2 / n_not_hot))

# Degrees of freedom for Welch's t-test
df = ((std_hot**2 / n_hot + std_not_hot**2 / n_not_hot)**2) / (
    ((std_hot**2 / n_hot)**2 / (n_hot - 1)) + ((std_not_hot**2 / n_not_hot)**2 / (n_not_hot - 1))
)

# 95% confidence interval
confidence_level = 0.95
t_crit = stats.t.ppf((1 + confidence_level) / 2, df)
ci_lower = mean_diff - t_crit * se_diff
ci_upper = mean_diff + t_crit * se_diff
print(f"Mean Difference 95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
# Mean Difference 95% CI: [0.7839, 0.8116]

# Let's visualize
plt.figure(figsize=(8, 6))
sns.boxplot(x='received_pepper', y='avg_rating', data=pepper_data, palette='pastel', width=0.5)
plt.title('Average Ratings by Hotness Status')
plt.xlabel('Received Pepper (Hotness)')
plt.ylabel('Average Rating')
plt.xticks([0, 1], ['Not Hot', 'Hot'])
plt.ylim(0, 5)
plt.grid(True)
plt.tight_layout()
plt.show()

#%%
"""
7. Build a regression model predicting average rating from difficulty (only). Make sure to include the R2
and RMSE of this model
"""

# Ok, for this one, let's use ScikitLearn's linear regression for the sake of the coding 
# sessions. Also, we won't necessarily need confidence intervals for betas. 

from sklearn.metrics import r2_score

# Clean data
q7_data = merged_data.dropna(subset=['avg_rating', 'avg_difficulty']).reset_index(drop=True)

# define vars
x = q7_data[['avg_difficulty']]
y = q7_data['avg_rating']

# Fit linear regression model
model_sk = LinearRegression()
model_sk.fit(x, y)

# Make predictions
y_pred = model_sk.predict(x)

# get RMSE and r-squared
r2 = r2_score(y, y_pred)
rmse = root_mean_squared_error(y, y_pred)

# Print results
print(f"Intercept: {model_sk.intercept_:.4f}")
print(f"Coefficient for avg_difficulty: {model_sk.coef_[0]:.4f}")
print(f"R-squared: {r2:.4f}")
print(f"RMSE: {rmse:.4f}")
# Intercept: 5.5564
# Coefficient for avg_difficulty: -0.6103
# R-squared: 0.2881
# RMSE: 0.9508

# Let's visualize it now
plt.figure(figsize=(8, 6))
sns.scatterplot(x='avg_difficulty', y='avg_rating', data=q7_data, alpha=0.3, label='Data')
plt.plot(q7_data['avg_difficulty'], y_pred, color='red', linewidth=2, label='Regression Line')
plt.xlabel('Average Difficulty')
plt.ylabel('Average Rating')
plt.title('Predicting Rating from Difficulty')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

#%%
"""
8. Build a regression model predicting average rating from all available factors. Make sure to include
the R2 and RMSE of this model. Comment on how this model compares to the “difficulty only” model
and on individual betas. Hint: Make sure to address collinearity concern
"""

from sklearn.linear_model import RidgeCV, LassoCV
from sklearn.preprocessing import StandardScaler

# Let's not use the qualitative data, only numerical data for the regression
# Also, let's use ridge regression as it is designed to handle multicollinearity

features = ['avg_difficulty', 'num_ratings', 'received_pepper',
            'would_take_again_pct', 'num_online_ratings', 'is_male', 'is_female']
target = 'avg_rating'

regression_data = merged_data[[target] + features].dropna().reset_index(drop=True)

# Separate predictors and target
X = regression_data[features]
y = regression_data[target]

# Standardize just incase
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Fit Ridge
alphas = [0.01, 0.1, 1.0, 10.0, 100.0]
ridge_cv = RidgeCV(alphas=alphas, store_cv_values=True)
ridge_cv.fit(X_scaled, y)

# Predict
y_pred = ridge_cv.predict(X_scaled)
rmse = root_mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)

# Output
print(f"Best alpha: {ridge_cv.alpha_}")
print(f"RMSE: {rmse:.4f}")
print(f"R²: {r2:.4f}")
print("\nCoefficients:")
for feat, coef in zip(features, ridge_cv.coef_):
    print(f"{feat}: {coef:.4f}")
    
# Best alpha: 1.0
# RMSE: 0.3687
# R²: 0.8101

# Coefficients:
# avg_difficulty: -0.1473
# num_ratings: -0.0031
# received_pepper: 0.1022
# would_take_again_pct: 0.6220
# num_online_ratings: -0.0010
# is_male: 0.0251
# is_female: 0.0129

# Let's plot the betas
plt.figure(figsize=(8,6))
coef_series = pd.Series(ridge_cv.coef_, index=features)
coef_series.sort_values().plot(kind='barh', color='skyblue')
plt.title('Ridge Regression Coefficients')
plt.xlabel('Standardized Coefficient')
plt.grid(True)
plt.tight_layout()
plt.show()

#%%
"""
9. Build a classification model that predicts whether a professor receives a “pepper” from average
rating only. Make sure to include quality metrics such as AU(RO)C and also address class imbalances
"""

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, classification_report, confusion_matrix, precision_recall_curve, f1_score
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.utils import resample

# Prepare data
classification_data = merged_data[['avg_rating', 'received_pepper']].dropna().reset_index(drop=True)
X = classification_data[['avg_rating']]
y = classification_data['received_pepper']
print(pepper_data['received_pepper'].value_counts())
print(X.shape, y.shape)

# Train-test split with n-number
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed)
print(X_train.shape, y_train.shape)
# (49002, 1) (49002,)

# Let's undersample the majority class to handle class imbalance
data = pd.concat([X_train, y_train], axis=1)
majority = data[data['received_pepper'] == 0]
minority = data[data['received_pepper'] == 1]

majority_downsampled = resample(majority, 
                                replace=False, 
                                n_samples=len(minority), 
                                random_state=seed)
balanced_train_data = pd.concat([majority_downsampled, minority])
print(majority_downsampled.shape, minority.shape)
# (13846, 2) (13846, 2)

# Let's do the same for the test set
# Downsample majority class in test set
test_data = pd.concat([X_test, y_test], axis=1)
majority_test = test_data[test_data['received_pepper'] == 0]
minority_test = test_data[test_data['received_pepper'] == 1]

majority_test_downsampled = resample(majority_test,
                                     replace=False,
                                     n_samples=len(minority_test),
                                     random_state=seed)
balanced_test_data = pd.concat([majority_test_downsampled, minority_test])
X_test_balanced = balanced_test_data[['avg_rating']]
y_test_balanced = balanced_test_data['received_pepper']

# Separate features and labels after resampling
X_train_balanced = balanced_train_data[['avg_rating']]
y_train_balanced = balanced_train_data['received_pepper']
print(X_train_balanced.shape, y_train_balanced.shape)

# Fit logistic regression model on balanced data
model = LogisticRegression()
model.fit(X_train_balanced, y_train_balanced)

# Predict on test data
y_pred = model.predict(X_test_balanced)
y_pred_prob = model.predict_proba(X_test_balanced)[:, 1]
auc_score = roc_auc_score(y_test_balanced, y_pred_prob)

# Print evaluation metrics
print("Balanced Model Evaluation at 0.5 Threshold")
print(f"AUC Score: {auc_score:.4f}")
print("Confusion Matrix:")
print(confusion_matrix(y_test_balanced, y_pred))
print("\nClassification Report:")
print(classification_report(y_test_balanced, y_pred))

"""
Balanced Model Evaluation at 0.5 Threshold
AUC Score: 0.6910
Confusion Matrix:
[[3425 2325]
 [1468 4282]]

Classification Report:
              precision    recall  f1-score   support

         0.0       0.70      0.60      0.64      5750
         1.0       0.65      0.74      0.69      5750

    accuracy                           0.67     11500
   macro avg       0.67      0.67      0.67     11500
weighted avg       0.67      0.67      0.67     11500
"""

# Plot ROC curve
fpr, tpr, thresholds = roc_curve(y_test_balanced, y_pred_prob)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc_score:.4f})', color='darkorange')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Predicting “Hotness” from Average Rating')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Let's try to optimize for precision

# Calculate precision, recall, thresholds
precision, recall, thresholds = precision_recall_curve(y_test_balanced, y_pred_prob)

precision_cut = precision[:-1]
# Find the threshold that maximizes precision
optimal_idx = precision_cut.argmax()
optimal_threshold = thresholds[optimal_idx]

# Predict using the optimal threshold
y_pred_optimal = (y_pred_prob >= optimal_threshold).astype(int)

# Print the optimal threshold and precision
print(f"Optimal Threshold for Precision: {optimal_threshold:.4f}")
print(f"Precision at Optimal Threshold: {precision[optimal_idx]:.4f}")

# Confusion Matrix at optimal threshold
cm_optimal = confusion_matrix(y_test_balanced, y_pred_optimal)
print("\nConfusion Matrix at Optimal Threshold:")
print(cm_optimal)

# Detailed classification report
print("\nClassification Report at Optimal Threshold:")
print(classification_report(y_test_balanced, y_pred_optimal))



#%%
"""
10. Build a classification model that predicts whether a professor receives a “pepper” from all available
factors. Comment on how this model compares to the “average rating only” model. Make sure to
include quality metrics such as AU(RO)C and also address class imbalances
"""

features = ['avg_rating', 'avg_difficulty', 'num_ratings', 
            'would_take_again_pct', 'num_online_ratings', 
            'is_male', 'is_female']
target = 'received_pepper'

# Drop NaNs
classification_data = merged_data[features + [target]].dropna().reset_index(drop=True)

X = classification_data[features]
y = classification_data[target]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed)

# Downsample the majority class in training set
train_data = pd.concat([X_train, y_train], axis=1)
majority = train_data[train_data[target] == 0]
minority = train_data[train_data[target] == 1]

majority_downsampled = resample(majority,
                                replace=False,
                                n_samples=len(minority),
                                random_state=seed)

balanced_train_data = pd.concat([majority_downsampled, minority])

X_train_balanced = balanced_train_data[features]
y_train_balanced = balanced_train_data[target]

# Fit logistic regression
model = LogisticRegression(max_iter=1000)
model.fit(X_train_balanced, y_train_balanced)

# Evaluate on test set
y_pred = model.predict(X_test)
y_pred_prob = model.predict_proba(X_test)[:, 1]
auc_score = roc_auc_score(y_test, y_pred_prob)

print("Logistic Regression with All Features")
print(f"AUC Score: {auc_score:.4f}")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
"""
Logistic Regression with All Features
AUC Score: 0.7959
Confusion Matrix:
[[1251  734]
 [ 322 1341]]

Classification Report:
              precision    recall  f1-score   support

         0.0       0.80      0.63      0.70      1985
         1.0       0.65      0.81      0.72      1663

    accuracy                           0.71      3648
   macro avg       0.72      0.72      0.71      3648
weighted avg       0.73      0.71      0.71      3648
"""

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'AUC = {auc_score:.4f}', color='darkorange')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve (All Features Model)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

#%%
"""
Extra credit: Tell us something interesting about this dataset that is not trivial and not already part of
an answer (implied or explicitly) to these enumerated questions [Suggestion: Do something with the
qualitative data, e.g. major, university or state by linking the two data files
"""

# Some stem and humanities fields
stem_fields = ['Biology', 'Computer Science', 'Mathematics', 'Engineering', 'Physics', 'Information Technology']
humanities_fields = ['English', 'History', 'Fine Arts', 'Spanish', 'Humanities', 'Linguistics']

# Create new boolean columns
merged_data['is_stem'] = merged_data['major'].isin(stem_fields).astype(int)
merged_data['is_humanities'] = merged_data['major'].isin(humanities_fields).astype(int)

# Count how many fall into each group
stem_count = merged_data['is_stem'].sum()
humanities_count = merged_data['is_humanities'].sum()
print(stem_count, humanities_count)
# 13368, 9804

# Let's filter the data to not include rows where neither is_stem or is_humanities is true
discipline_data = merged_data[(merged_data['is_stem'] + merged_data['is_humanities']) == 1]
# Also filter if there is no corresponding avg_rating in row
discipline_data = discipline_data.dropna(subset=['avg_rating']).reset_index(drop=True)

# Split ratings into two groups
stem_ratings = discipline_data[discipline_data['is_stem'] == 1]['avg_rating']
humanities_ratings = discipline_data[discipline_data['is_humanities'] == 1]['avg_rating']

# Let's run the Levene tests to check for unequal variances
stat_levene, p_levene = levene(stem_ratings, humanities_ratings)
print(f"Levene’s Test p-value: {p_levene:.6f}")
# Levene’s Test p-value: 0.000000

# So we should use Welch's t-test
t_stat, p_ttest = ttest_ind(stem_ratings, humanities_ratings, equal_var=False)
print(f"Welch’s t-statistic: {t_stat:.4f}, p-value: {p_ttest:.6e}")
# Welch’s t-statistic: -21.3465, p-value: 4.277729e-100

# Mann-Whitney U test
u_stat, p_mwu = mannwhitneyu(stem_ratings, humanities_ratings, alternative='two-sided')
print(f"Mann-Whitney U: {u_stat:.2f}, p = {p_mwu:.6e}")
# Mann-Whitney U: 55110424.50, p = 8.312029e-96

# Effect size and confidence interval
mean_diff = stem_ratings.mean() - humanities_ratings.mean()
print(f"Mean Difference (STEM - Humanities): {mean_diff:.4f}")

pooled_var = (np.var(stem_ratings, ddof=1) + np.var(humanities_ratings, ddof=1)) / 2
cohens_d = mean_diff / np.sqrt(pooled_var)
print(f"Cohen's d: {cohens_d:.3f}")
# Cohen's d: -0.282

std_stem = stem_ratings.std(ddof=1)
std_humanities = humanities_ratings.std(ddof=1)
n_stem = len(stem_ratings)
n_humanities = len(humanities_ratings)

se_diff = np.sqrt((std_stem**2 / n_stem) + (std_humanities**2 / n_humanities))
df = ((std_stem**2 / n_stem + std_humanities**2 / n_humanities) ** 2) / (
    ((std_stem**2 / n_stem) ** 2) / (n_stem - 1) + ((std_humanities**2 / n_humanities) ** 2) / (n_humanities - 1)
)
t_crit = stats.t.ppf(0.975, df)
ci_lower = mean_diff - t_crit * se_diff
ci_upper = mean_diff + t_crit * se_diff
print(f"95% CI for Mean Difference: [{ci_lower:.4f}, {ci_upper:.4f}]")
# 95% CI for Mean Difference: [-0.3413, -0.2839]

# Seems as though STEM professors have statistically significant lower ratings than
# Humanities ones

# Let's get a boxplot

discipline_data['discipline'] = np.where(discipline_data['is_stem'] == 1, 'STEM', 'Humanities')

plt.figure(figsize=(8, 6))
sns.boxplot(x='discipline', y='avg_rating', data=discipline_data, palette='pastel', width=0.5)
plt.title('Average Ratings by Discipline')
plt.xlabel('Discipline')
plt.ylabel('Average Rating')
plt.ylim(0, 5)
plt.grid(True)
plt.tight_layout()
plt.show()



