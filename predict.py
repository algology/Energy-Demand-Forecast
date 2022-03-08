print('Starting')

# Pandas and numpy for data manipulation
import pandas as pd
import numpy as np

# Imputing missing values in temp and value
from sklearn.impute import SimpleImputer

# Best practice to scale features
from sklearn.preprocessing import MinMaxScaler

# Models used for prediction
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor

from itertools import chain
import matplotlib.dates as dates

# Turn off setting with copy warning
pd.options.mode.chained_assignment = None

print('Loading Data')

# Read in the dataframes for training and testing
train = pd.read_csv("..\\train_corrected.csv")
test = pd.read_csv("..\\test_corrected.csv")
predictions = pd.read_csv("..\\submission_filename.csv")

predictions = predictions.sort_values(["ForecastId", "Timestamp"])
test = test.sort_values(["ForecastId", "Timestamp"])


from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import matplotlib.dates as dates
from sklearn.metrics import matthews_corrcoef
import numpy as np


#Site 120#
Site120_pred = predictions.loc[predictions["SiteId"]==120]
Site120_true = test.loc[test["SiteId"]==120]

Site120_pred = Site120_pred.set_index(["Timestamp"])
Site120_true = Site120_true.set_index(["Timestamp"])

Site120_pred_values = Site120_pred["Value"].to_numpy()
Site120_true_values = Site120_true["Value"].to_numpy()

Site120_R2 = r2_score(Site120_true_values, Site120_pred_values)

print ("Site 120 R2 performance= ", round(Site120_R2,2))

#Site 302#
Site302_pred = predictions.loc[predictions["SiteId"]==302]
Site302_true = test.loc[test["SiteId"]==302]

Site302_pred = Site302_pred.set_index(["Timestamp"])
Site302_true = Site302_true.set_index(["Timestamp"])

Site302_pred_values = Site302_pred["Value"].to_numpy()
Site302_true_values = Site302_true["Value"].to_numpy()

Site302_R2 = r2_score(Site302_true_values, Site302_pred_values)

print ("Site 302 R2 performance= ", round(Site302_R2,2))


fill_train_values = SimpleImputer(missing_values= np.nan, strategy='median')
fill_train_values.fit(test[['Value']])
test['Value'] = fill_train_values.transform(test[['Value']])


fig, ax1=plt.subplots(figsize=(15,5))

ax1.set_xlabel("Timestamp", fontsize=15)
ax1.grid()
ax1.set_facecolor("#DCDCDC")
ax1.set_ylabel("Energy Usage", color="blue", fontsize=15)
ax1.tick_params(axis="y", labelcolor="blue")
ax1.tick_params(axis="x", rotation=45)
ax1.scatter( Site120_pred.index, Site120_pred["Value"], label="Predicted Data")
ax1.scatter( Site120_true.index, Site120_true["Value"], label="Test Data")
ax1.legend()

date_fmt="%H:%M:%S"
formatter=dates.DateFormatter(date_fmt)
ax1.xaxis.set_major_formatter(formatter)
ax1.xaxis.set_major_locator(dates.HourLocator())


Site_120pred=no120.loc["2015-08-26":"2015-08-27"].sort_index()

no120hourlyweather=no120weather.loc["2015-08-26":"2015-08-27"].sort_index()


fig, ax6=plt.subplots(figsize=(15,5))

ax6.set_xlabel("Timestamp", fontsize=15)
ax6.grid()
ax6.set_facecolor("#DCDCDC")
ax6.set_ylabel("Energy Usage", color="blue", fontsize=15)
ax6.tick_params(axis="y", labelcolor="blue")
ax6.tick_params(axis="x", rotation=45)
ax6.scatter( Site120_pred["Timestamp"], Site120_pred["Value"])
ax6.scatter( Site120_true["Timestamp"], Site120_true["Value"])

ax6.set_title("Site 120 Daily Energy Usage Relation with Temperature", fontsize=20)

ax6t.set_xlabel("Timestamp")
ax6t.set_ylabel("Temperature(°C)", color="red", fontsize=15)
ax6t.tick_params(axis="y", labelcolor="red")
ax6t.ticklabel_format(axis="y", style="sci")
ax6t.plot(Site120_true.index, Site120_true[["Temperature"]], color="red")

date_fmt="%H:%M:%S"
formatter=dates.DateFormatter(date_fmt)
ax1.xaxis.set_major_formatter(formatter)
ax1.xaxis.set_major_locator(dates.HourLocator())

test = test.sort_values(["ForecastId", "Timestamp"])
test = test.reset_index()


pred_values = predictions["Value"].to_numpy()
true_values = test["Value"].to_numpy()
R2_value = r2_score(true_values, pred_values)
MAE = mean_absolute_error(true_values, pred_values)
MAPE = (np.mean(np.abs((true_values - pred_values) / true_values)) * 100)

#Test all errors#
for siteIterator in allforecasts:
    
    allsites = 1
    
    site_error_check_df = predictions.loc[predictions["ForecastId"]==5 & predictions.loc[predictions["SiteId"]==1]]
    site_true_df = test.loc[test["ForecastId"]==siteIterator]

    site_error_check_df = site_error_check_df.reset_index()
    site_true_df = site_true_df.reset_index()
        
    site_error_check_df_values = site_error_check_df["Value"].to_numpy()
    site_true_df_values = site_true_df_values["Value"].to_numpy()
    
    R2_value_120 = r2_score(site_true_df_values, site_error_check_df_values)
    
    error_df = error_df.append(site_error_check_df)
    
    allsites += 1
	    
    R2_value_120 = r2_score(site_true_values, site_error_check_df_values)
  
    error_df.append(site_error_check_df)


# Convert to datetimes
train['Timestamp'] = pd.to_datetime(train['Timestamp'])
test['Timestamp'] = pd.to_datetime(test['Timestamp'])

print('Data Loaded')

# Takes in a site id and returns a formatted training and testing set


def process_site(site_id, data='train'):
    
    if data == 'train':
        df = train_weather[train_weather['SiteId'] == site_id].sort_values(['Timestamp', 'Distance'])
        
    else:
        df = test_weather[test_weather['SiteId'] == site_id].sort_values(['Timestamp', 'Distance'])
    
    min_date = min(train_weather[train_weather['SiteId'] == site_id]['Timestamp'])
    
    df['Temperature'] = df['Temperature'].fillna(-99)
    
    df = df.drop_duplicates('Timestamp', keep = 'first')
    
    labels = df['Value'].fillna(method='ffill')
    labels = labels.fillna(0)
    
    df = df.drop(columns = ['Unnamed: 0', 'Distance', 'SiteId', 'ForecastId', 'Value'])
    
    df['Timestamp'] = [(time.days*3600*24) for time in (df['Timestamp'] - min_date)] 
    df = df.fillna(-99)
    
    return df, labels

rf_predictions = []

from sklearn.ensemble import RandomForestRegressor

for i, id in enumerate(set(test['SiteId'])):

    print('Percentage Complete: {:.2f}'.format(100 * i / len(set(test['SiteId']))))
    tree_reg = RandomForestRegressor(n_estimators=500)
    train_x, train_y = process_site(id, data = 'train')
    test_x, test_y = process_site(id, data = 'test')
    
    tree_reg.fit(train_x, train_y)
    predicted = lin_model.predict(test_x)
    clear_output()
    rf_predictions.append(predicted)



ForecastId_List = list(set(train['ForecastId']))

site =305

def process(site):

	# Testing data
    
    for ForecastId_Counter in ForecastId_List:
        test_df = test[test['ForecastId'] == ForecastId_Counter].sort_values(['Timestamp', 'Distance'])
        train_df = train[train['ForecastId'] == ForecastId_Counter].sort_values(['Timestamp', 'Distance'])
        
	test_df = test_df.drop_duplicates(['Timestamp'], keep='first')

	# Training data
	train_df = train[train['ForecastId'] == site].sort_values(['Timestamp', 'Distance'])
	train_df = train_df.drop_duplicates(['Timestamp'], keep='first')

	# Only use past training data
	train_df = train_df[train_df['Timestamp'] < test_df['Timestamp'].min()]

	# If all training temperatures are missing, drop temperatures from both training and testing
	if (np.all(np.isnan(train_df['Temperature']))) or (np.all(np.isnan(test_df['Temperature']))):
		train_df = train_df.drop(labels = 'Temperature', axis=1)
		test_df = test_df.drop(labels= 'Temperature', axis=1)

	# Otherwise impute the missing temperatures
    
    from sklearn.impute import SimpleImputer
    
    
	else:
        
		temp_median_imputer = SimpleImputer(missing_values=np.nan, strategy='median')
		temp_median_imputer.fit(train_df[['Temperature']])
		train_df['Temperature'] = temp_median_imputer.transform(train_df[['Temperature']])
		test_df['Temperature'] = temp_median_imputer.transform(test_df[['Temperature']])

	fill_train_values = SimpleImputer(missing_values= np.nan, strategy='median')
	fill_train_values.fit(train_df[['Value']])
    
    fill_train_values = SimpleImputer(missing_values= np.nan, strategy='median')
	fill_train_values.fit(test_df[['Value']])


	if pd.isnull(train_df['Value']).all():
		train_df['Value'] = 0
	else:
		train_df['Value'] = fill_train_values.transform(train_df[['Value']])

	# Find the minimum date for converting timestamp to numeric
	train_df["Timestamp"]=train_df["Timestamp"].dt.tz_localize(None)
    min_date = min(train_df['Timestamp'])
   

	# Convert timestamp to numeric
    
	train_df['Timestamp'] = (train_df['Timestamp'] - min_date).dt.total_seconds()
    test_df["Timestamp"]=test_df["Timestamp"].dt.tz_localize(None)
	test_df['Timestamp']  = (test_df['Timestamp'] - min_date).dt.total_seconds()

	# Interval between measurements
	train_df['time_diff'] = train_df['Timestamp'].diff().fillna(0)
	test_df['time_diff'] = test_df['Timestamp'].diff().fillna(0)
 
	# Extract labels
	train_var = train_df['Value']

	# Drop columns
	train_df = train_df.drop(columns = ['Distance', 'SiteId', 'ForecastId', 'Value'])
	test_df =   test_df.drop(columns = ['Distance', 'SiteId', 'ForecastId', 'Value'])

  
  
	# Scale the features between 0 and 1 (best practice for ML)
    
	scaler = MinMaxScaler()

	train_df.loc[:, :] = scaler.fit_transform(train_df.loc[:, :])
	test_df.loc[:, :] = scaler.transform(test_df.loc[:, :])

	return train_df, train_labels, test_df


    
    
# Trains and predicts for all datasets, makes predictions one site at a time
def predict():

	# List of trees to use in the random forest and extra trees model
	trees_list = [300, 350, 400, 450, 500, 550]

	# List of site ids
	site_list = list(set(train['ForecastId']))

	predictions = []

	# Keep track of the sites run so far
	number = len(site_list)
	count = 0
    
       	# Iterate through every site
	for site in site_list:

		# Features and labels
		train_x, train_y, test_x = process(site)





		# Make sure only training on past data
		assert train_x['Timestamp'].max() < test_x['Timestamp'].min(), 'Training Data Must Come Before Testing Data'

		# Initialize list of predictions for site
		_predictions = np.array([0. for _ in range(len(test_x))])

		# Iterate through the number of trees
		for tree in trees_list:

			# Create a random forest and extra trees model with the number of trees
			model1 = RandomForestRegressor(n_estimators=tree, n_jobs=-1)
			model2 = ExtraTreesRegressor(n_estimators=tree, n_jobs=-1)

			# Fitting the model
			model1.fit(train_x, train_y)
			model2.fit(train_x, train_y)

			# Make predictions with each model
			_predictions += np.array(model1.predict(test_x))
			_predictions += np.array(model2.predict(test_x))

		# Average the predictions
		_predictions = _predictions / (len(trees_list) * 2)

		# Add the predictions to the list of all predictions
		predictions.append(list(_predictions))

		# Iterate the count
		count = count + 1

		# Keep track of number of buildings process so far
		if count % 100 == 0:
			print('Percentage Complete: {:.1f}%.'.format(100 * count / number))

	# Flatten the list
	predictions = list(chain(*predictions))

	return predictions

# Make a submission file given the list of predictions and name for the submission
def make_submission_file(predictions, name):

	# Read in the submission dataframe
	submit_df = pd.read_csv('../data/submission_format.csv')

	# Assign the predictions as the value
	submit_df['Value'] = predictions

	# Save the submissions to the folder of final submissions
	submit_df.to_csv('../submissions/%s.csv' % name, index = False)
	print('Predictions saved to ../submissions/%s.csv' % name)

print('Starting Predictions')

# Make predictions
predictions = predict()

# Save predictions with a sensible name
make_submission_file(predictions, 'submission_filename')

