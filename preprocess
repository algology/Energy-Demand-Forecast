import pandas as pd
import numpy as np
from datetime import datetime
from dateutil.parser import parse
import pytz
import matplotlib.pyplot as plt


#Reading Data

train = pd.read_csv("..\\train.csv", sep=";", index_col=0)
test = pd.read_csv("..\\test.csv", index_col=0)
weather = pd.read_csv("..\\weather.csv", index_col=0)
holidays = pd.read_csv ("..\\holidays.csv", sep=";", index_col=0)
meta = pd.read_csv ("..\\metadata.csv", sep=",", index_col=0)

weather.isnull().sum()
meta=meta.drop(columns="Sampling")

#Standardizing Timestamp Index For All Datasets

datetime.strptime(train["Timestamp"], "%Y-%m-%d")
train["Timestamp"]=pd.to_datetime(train["Timestamp"], format="%Y-%m-%d")
no42.index=no42.index.tz_localize(None)

### END OF SECTION ###


train["Timestamp"]=pd.to_datetime(train["Timestamp"])
train=train.set_index("Timestamp")


test["Timestamp"]=pd.to_datetime(test["Timestamp"])
test=test.set_index("Timestamp")

weather["Timestamp"]=pd.to_datetime(weather["Timestamp"])
weather=weather.set_index("Timestamp")

holidays.index=pd.to_datetime(holidays.index)


no42=train.loc[train["SiteId"]==42].sort_values(by=no42.index)
no42weather=weather.loc[weather["SiteId"]==42].sort_values(by=no42weather.index)

no42.dtypes
no42weather.dtypes

no42raw=no42.join(no42.set_index(no42.index.year), left_index=True, right_index=True)

raw=pd.concat([no42,no42weather])



no150=train.loc[train["SiteId"]==150]

noyear150=no150.loc["2015-01-01":"2016-01-01"]

no150weather=weather.loc[weather["SiteId"]==150]
noyearweather150=no150weather.loc["2015-01-01":"2016-01-01"]

fig, ax1=plt.subplots(figsize=(15,5))
ax1t=ax1.twinx()

#ax1 plot notation#
ax1.set_xlabel("Timestamp")
ax1.grid()
ax1.set_facecolor("#DCDCDC")
ax1.set_ylabel("Energy Usage", color="blue")
ax1.tick_params(axis="y", labelcolor="blue")
ax1.scatter( noyear150.index, noyear150[["Value"]])
ax1.set_title("Transportation Infrastructure Cost vs Country Population", fontsize=15)
#ax2 plot notation#
ax1t.set_xlabel("Timestamp")
ax1t.set_ylabel("Temperature", color="red")
ax1t.tick_params(axis="y", labelcolor="red")
ax1t.ticklabel_format(axis="y", style="sci")
ax1t.scatter(no150weather.index, no150weather[["Temperature"]], marker="o", color="red", alpha=0.2, s=1)

#Feature Engineering for Timestamp

#Decomposing Timestamp

def timestamp_decompose (data):
    
    data["Timestamp"]=pd.to_datetime(data["Timestamp"])
    data = data.set_index("Timestamp")
    
    data["year"]=data.index.year
    data["month"]=data.index.month
    data["day_of_year"]=data.index.dayofyear
    data["hour"]=data.index.hour
    data["min"]=data.index.minute
    data["day_of_month"]=data.index.day
    data["day_of_week"]=data.index.dayofweek
    
    data["time_num"] = data["hour"] + (data["min"]/60)
    data = data.drop(columns=["hour", "min"])

#Timestamp Cyclical Features
    
#Cycle of 12 Months#
    data["month_sin"] = np.sin(2*np.pi*data["month"]/11)
    data["month_cos"] = np.cos(2*np.pi*data["month"]/11)
    
#Cycle of 365 Days#
    data["day_of_year_sin"] = np.sin(2*np.pi*data["day_of_year"]/364)
    data["day_of_year_cos"] = np.cos(2*np.pi*data["day_of_year"]/364)
    
#Cycle of days of weeks#    
    data["day_of_week_sin"] = np.sin(2*np.pi*data["day_of_week"]/6)
    data["day_of_week_cos"] = np.cos(2*np.pi*data["day_of_week"]/6)
    
#Cycle of Clock Daily#    
    data["time_num_sin"] = np.sin(2*np.pi*data["time_num"]/23)
    data["time_num_cos"] = np.sin(2*np.pi*data["time_num"]/23)
    

    data=data.reset_index(level=0)
    return data



train_ts=timestamp_decompose(train)
test_ts=timestamp_decompose(test)

#Weather

#test_case#
weather_302 = weather.loc[weather["SiteId"]==302]
weather_302_avg = weather_302.set_index('Timestamp').resample('H').mean()
#end_test_case#


weather["Timestamp"]=pd.to_datetime(weather["Timestamp"])
weather = weather.set_index("Timestamp")
    
#Round the  weather data to the nearest hours 
weather.index = weather.index.round(freq='H')
rounded_weather = weather.reset_index(level=0)
rounded_weather[rounded_weather.isna().any(axis=1)].sum()

train_ts_weth = pd.merge(train_ts, rounded_weather, how="left", on=["Timestamp", "SiteId"])

train_ts["Timestamp"]=pd.to_datetime(train_ts["Timestamp"]).dt.tz_localize(None)

###train_ts' ten daha yüksek değeri vardı duplike olma durumu için çıkarıldı ve sonuç olarak aynı sayıya ulaşıldı
train_ts_weth = train_ts_weth.drop_duplicates(['Timestamp', 'SiteId'], keep='first')

df = df.sort_values(['Timestamp', 'SiteId', 'Distance'])



##Working Days Per SiteId###
    # Create a new dataframe for the site
    site_meta['wday'] = [0, 1, 2, 3, 4, 5, 6]
    site_meta["SiteId"] = site
    site_meta.loc[5, "off"] 
    float(meta_slice['SaturdayIsDayOff'])
    
    # Record the days off
    if meta_slice["MondayIsDayOff"] is True:
        site_meta.loc[0, 'off'] = 1
    else:
        site_meta.loc[0, "off"] = 0
        
    if meta_slice["TuesdayIsDayOff"]==True:
        site_meta.loc[0, 'off'] = 1
    else:
        site_meta.loc[0, "off"] = 0
    
    if meta_slice["WednesdayIsDayOff"]==True:
        site_meta.loc[0, 'off'] = 1
    else:
        site_meta.loc[0, "off"] = 0
        
    if meta_slice["ThursdayIsDayOff"]==True:
        site_meta.loc[0, 'off'] = 1
    else:
        site_meta.loc[0, "off"] = 0
        
    if meta_slice["FridayIsDayOff"]==True:
        site_meta.loc[0, 'off'] = 1
    else:
        site_meta.loc[0, "off"] = 0    
    
     if meta_slice["SaturdayIsDayOff"]==True:
        site_meta.loc[0, 'off'] = 1
    else:
        site_meta.loc[0, "off"] = 0  
        
    if meta_slice["SundayIsDayOff"]==True:
        site_meta.loc[0, 'off'] = 1
    else:
        site_meta.loc[0, "off"] = 0  
    
    site_meta.loc[1, 'off'] = float(meta_slice['TuesdayIsDayOff'])
    site_meta.loc[2, 'off'] = float(meta_slice['WednesdayIsDayOff'])
    site_meta.loc[3, 'off'] = float(meta_slice['ThursdayIsDayOff'])
    site_meta.loc[4, 'off'] = float(meta_slice['FridayIsDayOff'])
    site_meta.loc[5, 'off'] = float(meta_slice['SaturdayIsDayOff'])
    site_meta.loc[6, 'off'] = float(meta_slice['SundayIsDayOff'])
    
    # Append the resulting dataframe to all site dataframe
    all_meta = all_meta.append(site_meta) 
    
train_ts_weth["non_working"] = train_ts_wet.apply(lambda)
