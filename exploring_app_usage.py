#!/usr/bin/env python
# coding: utf-8

# Eugen Klein, August 2019


# load all libraries neccessary for the analysis
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
get_ipython().run_line_magic('matplotlib', 'inline')


# The dataset user_item_data describes events that are recorded when a user interacts
# with the app. Each event corresponds with a user watching a single video, and
# answering a corresponding question. During each session, a user will
# watch several videos.

# load and sort user data
user_data = pd.read_csv("Desktop/data/user_item_data.csv", parse_dates=['created_at'])
user_data.sort_values(by=['uuid','created_at'], inplace=True)
user_data.head()


# User interactions have to be grouped into sessions. A session is defined as a sequence of
# user interactions with the same user ID, ordered by timestamp,
# such that the time difference between any consecutive pair of interactions is at most one
# hour. 

# label sessions based on two conditions: 1 hour interval between two timestamps and different user id
cond_1 = user_data.created_at.diff() > pd.Timedelta(1, 'h')
cond_2 = user_data.uuid != user_data.uuid.shift(1)
user_data['session_id'] = (cond_1|cond_2).cumsum()

# label days based on two conditions: session id and date / user id change
cond_1 = user_data.session_id != user_data.session_id.shift(1)
# convert time stamps series to data frame to be able to extract the date (day)
created_at = pd.to_datetime(user_data['created_at'])
cond_2 = (created_at.dt.day.astype(float) != created_at.dt.day.shift(1))|(user_data.uuid != user_data.uuid.shift(1))
user_data['day_id'] = (cond_1 & cond_2).cumsum()

user_data.head()


# Havind defined the sessions, we can answer the following questions:
# - What is the distribution of session lengths?
# - How many users do several sessions a day?

# get durations between each activity within each session
user_data['activity_duration'] = user_data.groupby('session_id')['created_at'].diff()
user_data.head(10)

# aggregate session data with user ids, session ids, day ids, and session durations
session_data = user_data.groupby(['uuid','session_id', 'day_id'])['activity_duration'].sum().reset_index()
session_data.head()

# define a threshold for duration outliers based on the standard deviation
print('The mean duration for one session is: ' + str(session_data['activity_duration'].mean()) + '.')
threshold_outlier = 5*session_data['activity_duration'].std()
print('The threshold for outliers is: ' + str(session_data['activity_duration'].mean() + threshold_outlier) + '.')


# A threshold of about 2 hours for a session duration seems like a reasonable choice.


# check for outliers (very long session durations) in session durations: Z-score higher than 3 times the
# standard deviation (SD)
session_duration_outliers = ((session_data['activity_duration']-session_data['activity_duration'].mean()) >
                             threshold_outlier)

print('There were ' + str(session_duration_outliers.describe()[0]-session_duration_outliers.describe()[3]) +
      ' outliers at ' + str(session_duration_outliers.describe()[0]) + ' sessions overall.')


# exclude outliers from the set of session durations 
session_durations_clean = session_data['activity_duration'][~session_duration_outliers]


# plot session durations as a histogramm (approx. 3.5 minute intervals per bin)
session_durations_clean.astype('timedelta64[m]').hist(bins=40)
plt.ylabel('Frequency of observations')
plt.xlabel('Session duration (min)')
plt.title('Distribution of session lengths')


session_durations_median = round(session_durations_clean.astype('timedelta64[s]').median()/60)
session_durations_max = int(round(session_durations_clean.astype('timedelta64[s]').max()/60))

print('The median value for the session duration is about ' + str(session_durations_median)
                                                          + ' minutes.')
print('The max value for the session duration is about ' + str(session_durations_max)
                                                          + ' minutes.')


# The distribution of session durations can be intuitive visualized by means of a empirical
# cumulative distribution function. Looking at the graph below, we can conclude that about 80 percent
# of completed sessions had a length of 22 minutes or less.


x = np.sort(session_durations_clean.astype('timedelta64[m]').values)
y = np.arange(1, len(x) +1) / len(x)

plt.plot(x, y, marker = '.', linestyle='none')
plt.axhline(y=0.8, color = 'red')
plt.axvline(x=22, color = 'red')
plt.ylabel('Percentage of sessions')
plt.xlabel('Session duration (min)')
plt.title('Empirical cumulative distribution')


# Now, we can turn to the question of how many people do several learning sessions per day.

# aggregate session numbers split by user id and by day id
sessions_per_day = session_data.groupby(['uuid', 'day_id'])['session_id'].count().reset_index()
sessions_per_day.head()

# sanity check comparing the initial data frame with aggregated data frame
print ('Overall number of unique users (unique uuids) in the original data set: '
       + str(user_data['uuid'].nunique()))
print ('Overall number of unique users (unique uuids) in the aggregated data set: ' 
       + str(sessions_per_day['uuid'].nunique()))


# select rows with multiple sessions per day
multiple_sessions_per_day = sessions_per_day.loc[sessions_per_day['session_id'] > 1]
print ("Number of all users (unique uuids) who did 2 sessions per day at least once: "
       + str(multiple_sessions_per_day['uuid'].nunique()))
print(str(round(multiple_sessions_per_day['uuid'].nunique()/sessions_per_day['uuid'].nunique()*100))
      + "% of users did 2 sessions per day at least once.")


# We can also look closer at the question how many of these users completed at least once
# exactly 2, 3, 4, or more sessions per day.

uuids_with_multiple_sessions = multiple_sessions_per_day.groupby('session_id')['uuid'].nunique().reset_index()
uuids_with_multiple_sessions.columns = ['sessions_per_day', 'number of users']
print("Number of unique user ids for different number of sessions per day:")
uuids_with_multiple_sessions


# To test a new app feature, it is released as an A/B test with 40% of the users seeing the
# new feature (test group) and 60% not seeing it (control group).
# We can answer two questions with the A/B test:
# - What is the proportion of users per group who complete more than one session on at
# least one day?
# - What is the median time per session for each group?

#load experimental groups assignment and merge with session data
experiment_group = pd.read_csv("Desktop/data/test_groups.csv")
experiment_group.sort_values(by=['uuid'], inplace=True)
experiment_session_data = session_data.merge(experiment_group, how='left')
experiment_session_data.head()


# check for the number of users assigned to the control and the test group (58.6 vs 41.4 %)
experiment_session_data.groupby('test_group')['uuid'].nunique().reset_index()

# check for outliers in session durations: Z-score higher than 3 times the standard deviation (SD)
experiment_session_duration_outliers = ((experiment_session_data['activity_duration']-experiment_session_data['activity_duration'].mean()).abs() >
                             5*experiment_session_data['activity_duration'].std())

# exclude outliers from the set of session durations 
experiment_session_duration_clean = experiment_session_data['activity_duration'][~experiment_session_duration_outliers]

#define the group index excluding rows with outliers
group_index = experiment_session_data['test_group'][~experiment_session_duration_outliers]

group_medians = experiment_session_duration_clean.astype('timedelta64[m]').groupby(group_index).median()
group_medians = group_medians.to_frame().reset_index()
group_medians.columns = ['test_group', 'session_median']
print('The session medians for the two tested groups are:')
group_medians


# Based on the median values of session durations (in minutes), we are not able to observe a difference
# between the control and the test group. To assure ourselves, let us compare the cumulative
# distribution function for both groups below.

control_indx = np.where(group_index == 'control')[0]
control_x = experiment_session_duration_clean.astype('timedelta64[m]')[control_indx]

x_c = np.sort(control_x.values)
y_c = np.arange(1, len(x_c) +1) / len(x_c)

test_indx = np.where(group_index == 'test')[0]
test_x = experiment_session_duration_clean.astype('timedelta64[m]')[test_indx]

x_t = np.sort(test_x.values)
y_t = np.arange(1, len(x_t) +1) / len(x_t)

plt.plot(x_c, y_c, marker = '.', linestyle='none', color = 'blue')
plt.plot(x_t, y_t, marker = '+', linestyle='none', color = 'red', alpha = 0.5)

blue_patch = mpatches.Patch(color='blue', label='Control group')
red_patch = mpatches.Patch(color='red', label='Test group')
plt.legend(handles=[blue_patch, red_patch], loc='lower right')

plt.ylabel('Percentage of sessions')
plt.xlabel('Session duration (min)')
plt.title('Empirical cumulative distribution')


# merge the experimental assignment to the frame containing session numbers higher than 1
experiment_multiple_sessions_per_day = multiple_sessions_per_day.merge(experiment_group, how='left')
experiment_multiple_sessions_per_day.head()


experiment_uuids_with_multiple_sessions = experiment_multiple_sessions_per_day.groupby(['session_id', 'test_group'])['uuid'].nunique().reset_index()
experiment_uuids_with_multiple_sessions.columns = ['sessions_per_day', 'test_group', 'number of users']
print("Number of unique user ids for different number of sessions per day split by the test group:")
experiment_uuids_with_multiple_sessions


# Eyeballing the data set, the number of users assigned to each experimental group appears to be
#balanced with respect to the observed number of sessions per day. This is also true when we look at
# absolute frequency numbers for sessions rather than just unique user IDs: 

# frequency table for the number of session
freq_sessions = experiment_multiple_sessions_per_day.groupby(['session_id', 'test_group'])['uuid'].count().reset_index()
freq_sessions.columns = ['sessions_per_day', 'test_group', 'number of sessions']
freq_sessions

experiment_multiple_sessions_per_day['session_id'].hist(bins= 4, by = group_index)
plt.ylabel('Frequency of observations')
plt.xlabel('Number of sessions')


# Let us now take a look at the percentage of control and test users completing multiple sessions per
# day at least once.

# aggregate unique user IDs with all / multiple sessions per day split by the experimental group
uuids_per_group = experiment_session_data.groupby('test_group')['uuid'].nunique()
uuids_multiple_per_group = experiment_multiple_sessions_per_day.groupby('test_group')['uuid'].nunique()

print ("Number of users in the control group who did 2 sessions per day at least once: "
       + str(round(uuids_multiple_per_group[0]/uuids_per_group[0]*100)) + '%.')
print ("Number of users in the test group who did 2 sessions per day at least once: "
       + str(round(uuids_multiple_per_group[1]/uuids_per_group[1]*100)) + '%.')


# Based on the explorative analysis of the control and test group distributions as well as the number
# of users who completed multiple sessions per day, we are not able to observe a difference between the
# both investigated groups. I would recommend to look into alternative features to increase the desired
# performance metrics. For instance, investigating the distribution of session durations in more detail,
# it appears than most users interrupt their sessions at designated break points. Consider the graph
# below: 

# plot session durations as a histogramm with approx. 2 minute intervals per bin
session_durations_clean.astype('timedelta64[m]').hist(bins=70)
plt.xlim(0,40) # look closer at sessions with durations under 30 minutes
plt.ylabel('Frequency of observations')
plt.xlabel('Session duration (min)')
plt.title('Distribution of session lengths')


# Note here that the session durations can be fit into intervals dividable roughly by 5 minutes, e.g.
# sessions below 5 minutes, sessions below 10 minutes, sessions below 15 minutes and so on. The
# frequency of sessions leading up to the interval break points are higher compared to the moments
# right after these break points. For instance, the number of sessions with duration of 7, 8, 9, or 10
# minutes is higher than the number of sessions with duration of 11, 12, or 13 minutes. Perhaps, it
# would make sense to display short messages of encouragement to users before an upcoming break point
# to keep them engaging with the app below this break point.
# This would be especially important for intervals below 5 minutes as most users' sessions last less
# than 5 minutes. Going further, it should be possible to calculate individual break points for each
# user based on their average usage behavior.   
