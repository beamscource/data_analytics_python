# check for outliers in session numbers: Z-score higher than 3 times the standard deviation (SD)
#session_number_outliers = ((multiple_sessions_per_day['session_id']-multiple_sessions_per_day['session_id'].mean()).abs()
#                           > 3*multiple_sessions_per_day['session_id'].std())

#print('There were ' + str(session_number_outliers.describe()[0]-session_number_outliers.describe()[3]) +
#      ' outliers at ' + str(session_number_outliers.describe()[0]) + ' sessions overall.')

#min_outlier= multiple_sessions_per_day['session_id'][session_number_outliers].min()
#max_outlier = multiple_sessions_per_day['session_id'][session_number_outliers].max()

#print('The min and max numbers of sessions in the outlier list was ' + str(min_outlier) + ' and '
#      + str(max_outlier) + '.')

# exclude outliers from the set of session durations
#multiple_sessions_per_day_clean = multiple_sessions_per_day['session_id'][~session_number_outliers]

min_duration_outlier= round(session_data['session_id'][session_duration_outliers].min()/60)
max_duration_outlier = round(session_data['session_id'][session_duration_outliers].max()/60)

print('The min and max numbers of sessions in the outlier list was ' + str(min_duration_outlier) + ' and '
      + str(max_duration_outlier) + '.')

two_sessions_per_day = sessions_per_day.loc[sessions_per_day['session_id'] == 2]
two_sessions_per_day['uuid'].nunique()

# plot session durations as a histogramm split by the test group assignment
experiment_session_duration_clean.astype('timedelta64[m]').hist(bins=80, by=group_index)
plt.ylabel('Frequency of observations')
plt.xlabel('Session duration (min)')

uuids_with_multiple_sessions.columns = ['sessions_per_day', 'number of users']
print("Number of unique user ids for different number of sessions per day:")
uuids_with_multiple_sessions

# number of days with multiple sessions in the whole data set
experiment_multiple_sessions_per_day.groupby('uuid')['session_id'].value_counts().hist()
plt.ylabel('Frequency of observations')
plt.xlabel('Number of days with multiple sessions')
plt.title('Number of days with multiple sessions')