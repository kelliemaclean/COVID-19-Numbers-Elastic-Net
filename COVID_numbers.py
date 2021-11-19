#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import RobustScaler
from matplotlib import rc
import matplotlib.pyplot as plt
from pylab import rcParams
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")
sns.set(style="whitegrid", palette='muted', font_scale=1.5)
rcParams['figure.figsize']= 16, 8
from tabulate import tabulate
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV



# In[2]:


# File Path from GitHub - John Hopkins Covid Data
file_path_cases= 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv'
file_path_deaths = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv'
#read csv file
covid_raw_cases = pd.read_csv(file_path_cases, error_bad_lines=False)
covid_raw_deaths = pd.read_csv(file_path_deaths, error_bad_lines=False)


# In[4]:


def clean_transpose_data(dataframe_name, raw_data):
    dataframe = raw_data.loc[covid_raw_cases['Country/Region']== "Canada"]
    dataframe = dataframe.drop(['Country/Region', 'Lat', 'Long'], axis=1)
    dataframe.head()
    #Transpose the Data for Cases
    dataframe = dataframe.T
    #Make first row index the column headers
    new_header = dataframe.iloc[0] 
    dataframe = dataframe[1:]
    dataframe.columns = new_header 
    dataframe.index = pd.to_datetime(dataframe.index)#change index to datetime formate
    dataframe = dataframe.drop(['Diamond Princess','Grand Princess', 
                                          'Nunavut', 'Repatriated Travellers', 'Yukon', 'Northwest Territories'], axis= 1)
    dataframe = dataframe.reset_index()
    return dataframe


# In[5]:


cases = clean_transpose_data("cases",covid_raw_cases)


# In[6]:


cases1 = cases.drop("index", axis =1)


# ### Adding Predictor Variable Column for Yesterday's Cases
# 
# To have yesterdays case count for each province as predicting variable, the orginal dataset had to be copied and shifted by one row. The column names were changed to indicate it is the yesterday count and then merged on the index with orginal dataset. 

# In[7]:


def resetindex_orginal_df(df_name, dataframe1):
    
    #drop first row & reset index
    df_name = dataframe1
    df_name = df_name[1:].reset_index()
    df_name = df_name.drop("index", axis = 1)
    return df_name

def add_yesterday(new_df_name, dataframe):

    #create new data for yesterdays numbers
    new_df_name = dataframe
    cols = {"Alberta":"AB_yest",'British Columbia':"BC_yest","Monitoba":'Man_yest',"New Brunswick":'NB_yest',
        "Manitoba":'MB_yest', 'Nova Scotia': "NS_yest", 'Ontario':'ON_yest',
        "Newfoundland and Labrador":'NFL_yest', "Prince Edward Island":'PEI_yest',"Quebec":'QB_yest', 
        "Saskatchewan":'Sask_yest'
              }
    new_df_name = new_df_name.rename(columns=cols)
    return new_df_name


# In[8]:


#call functions
cases_df = resetindex_orginal_df("cases_df", cases1)
yesterdays_numbers = add_yesterday("yesterdays_numbers", cases)


# In[9]:


#join orginal data with yesterdays number variable
cases_w_yesterday = pd.merge(cases_df, yesterdays_numbers, left_index=True, right_index=True)
#reset timeseries index
cases_w_yesterday = cases_w_yesterday.set_index("index")
cases_w_yesterday.index = pd.to_datetime(cases_w_yesterday.index)
cases_w_yesterday = cases_w_yesterday.astype("float")


# As we would expect and can see below, the day prior's cummulative case count has a direct linear relationship with  that day's case count.

# In[10]:


# Snap Shot of Covid Case Trends in Alberta based on yesterdays case counts
sns.lineplot(x="AB_yest", y = 'Alberta', data=cases_w_yesterday)
plt.show()


# ### Adding Additional Predictor Variables
# 
# Three added predictor vairbale include the vaccine phases for the roll out to the general population, the current season, and the day of the week (Mon, Tues, Wed, etc). The vaccine phases in theory should have a effect on the number of occuring cases due to the number susceptiable individuals deacreasing. For the season effect on the model, it is commonly knowledge for most infectious disease cycles to be dependent on weather patterns. Warmer tempature can have several effects on the spread of the Sar-CoV-2 due to viral particals becoming unstable in warmer tempatures/ UV exposure and individuals spend less time in doors in poorly circulating air. Lastly, the forth variable potentially has a impact on the case numbers due to when people are more commonly contracting the virus(eg. over the weekends), getting tested, and when the data is getting updated across institutions and provinces. 
# 
# ----------------------------------
# 
# Vaccine Phases:
# 
# Phase 0 - No/little vaccine available to public (2020)
# 
# Phase 1 - ICU, Seniors, LTC (Jan 2021 - April 2021)
# 
# Phase 2 - Chronic illness, health care, above 65+ (April 2021 - Present)
# 
# -----------------------------------
# Seasons:
# 
# Winter(Nov 2 - March 1) = 0
# 
# Spring(Feb 2 - June 1) = 1
# 
# Summmer(June 2 - Sept 1) = 2
# 
# Fall (Sept 2 - Nov 1) = 3
# 
# -----------------------------------
# 
# Day of the week: 
# 
# Monday = 0 - Sunday = 6
# 
# -----------------------------------
# 
# The function below creates the above variable in the data based on the timestamp index, where pandas datatime function was utilized for simple variable creatiion. 

# In[11]:


def predict_variables(dataframe):
    
    #Vaccine Phases
    dataframe.loc[dataframe.index[0:345], 'Vaccine Phase'] = "0"
    dataframe.loc[dataframe.index[345: 436], 'Vaccine Phase'] = "1"
    dataframe.loc[dataframe.index[436::], 'Vaccine Phase'] = "2"
    
    #Month Seasons
    dataframe["Season"] = dataframe.index.month
    winter = [11,12,1,2,3] 
    spring = [4,5]
    summer = [6,7,8]
    fall = [9,10]
    dataframe["Season"].replace(winter, 0 , inplace = True)
    dataframe["Season"].replace(spring, 1 , inplace = True)
    dataframe["Season"].replace(summer, 2 , inplace = True)
    dataframe["Season"].replace(fall, 3, inplace = True)
    pd.set_option('display.max_rows', None)
    
    #day of the week
    dataframe["Day of Week"] = dataframe.index.dayofweek
    pd.set_option('display.max_rows', None)
    
    return dataframe


# In[12]:


# Call fucntion and change datatypes
cases_w_yesterday = predict_variables(cases_w_yesterday)
cases_w_yesterday = cases_w_yesterday.astype("float")


# ## Visual Exploration
# 
# ### Case count trends:
# 
# From the plot below, we can see that there is clear exponential relationship with the cummlative case count trends over time. The slope of the line of best fit becomes more extreme overtime. At the time stamp of November of 2020 (2020-11) we see the tren to begin more of a linear progression. From this, to create a better predictive linear model that has a more consistant slope, the data from November 2020 to present was used in the final model. 

# In[13]:


# Snap Shot of Covid Case Trends in Alberta
sns.lineplot(x=cases_w_yesterday.index, y = 'Alberta', data=cases_w_yesterday).set_title('Cummulative Case Trends in Alberta')
plt.show()


# In[14]:


# Snap Shot of Covid Case Trends in Alberta November 2020 on
sns.lineplot(x=cases_w_yesterday.index[284::], y = 'Alberta', data=cases_w_yesterday[284::]).set_title('Cummulative Case Trends in Alberta After November 2020')
plt.show()


# ### Impact of Vaccine Phases
# 
# From the plot below, we can see that there is a potential that the vaccine phase variable with have an impact on the case count prediction of our model. The transition from the first phase (0) to the second phase (1), we see the rate in which cases increased over time to slow after the beggining of the vaccine roll out into the high supceptable popualtions (eg. seniors, ICU & LTC health care workers)

# In[15]:


#How vaccine phases have affected case numbers over time
sns.lineplot(x="Vaccine Phase", y = 'Alberta', data = cases_w_yesterday).set_title('Cummulative Case Trends in Alberta Based on Vaccine Phases')
plt.show()


# ### Impact of Seasons
# 
# From the plot below, we can see that season may have a small impact on the cummulative case count especially into the fall and winter seaon.

# In[154]:


## look at seasonality of 2020 and covid cases
sns.lineplot(x="Season", y = 'Alberta', data = cases_w_yesterday[0:280]).set_title('Cummulative Case Trends in Alberta Based on Season (Year 2020)')
plt.show()


# ### Correlation of Predictive variables
# 
# From the Cooorelation Matrix we can see that the vaccine phase and the yesterday's case count variable are highly correlated. This correlation will casue an issue for simple linear regession models due to assumption of multicollinarity not being present. 

# In[16]:


df = cases_w_yesterday[["Vaccine Phase", "AB_yest", "Day of Week", "Season"]]

corr = df.corr(method='spearman')

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
fig, ax = plt.subplots(figsize=(6, 5))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True, sep=100)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmin=-1, vmax=1, center=0, linewidths=.5)

fig.suptitle('Correlation matrix of features', fontsize=15)
ax.text(0.77, 0.2, 'aegis4048.github.io', fontsize=13, ha='center', va='center',
         transform=ax.transAxes, color='grey', alpha=0.5)

fig.tight_layout()


# ## Building Model
# 
# During the model development process, the first model that was attempted was a simple linear regression using the sklearn package. The train data for this model was first scaled to see if the scaling added predictive value, however, with the further investigation it was found that scaled training data verse non scaled train data did not provide any benefit. From this, it was decided to leave the training data unscaled for the model training. 
# 
# To futher improve the model, the simple linear regression model was upgraded to an ElasticNet. This model type has several benifits. One being that the ElasticNet will eliminate unused features while including the benefit of feature coefficient reduction. The ElasticNet model, due to the combination of the benefits from both the Lasso and ridge model is well suited for model that have collineality present, such that was shown above in the collinearity marix. 
# 
# To predict future case counts, an additional line was created to input feature variable characteristics for that day. 

# In[17]:


# assigning train and test data
train = predict_variables(cases_w_yesterday)
# set the train data to after Nov 2020
train = train[284::] 

# Creating a extra column for Monday, April 19
test = pd.DataFrame({'date': "2021/04/19",'Alberta': [0], 'British Columbia': [0], 'Manitoba': [0], 'New Brunswick': [0], 
                    'Newfoundland and Labrador': [0], 'Nova Scotia': [0],'Ontario': [0],
                     'Prince Edward Island': [0],'Quebec': [0], 'Saskatchewan': [0], 
                     'AB_yest':[170795], 'BC_yest':[117080],'MB_yest':[36159],'NB_yest':[1788], 'NFL_yest':[1043], 'NS_yest':[1807],
                     'ON_yest':[42854],'PEI_yest':[165], 'QB_yest':[336952], 'Sask_yest':[38160],
                     'Vaccine Phase': [2], 'Season': [1],'Day of Week': [5]})
test['date'] = pd.to_datetime(test.date)    
test = test.set_index('date')
test


# In[18]:


# Looking at the end of the training data
train.tail()


# ### Hypertuning Paramters
# 
# Hypertuning the aplha and lamda paramters in the model are import for ElasticNet regression to find the best ratio between L1 and L2 penalties. Below Sklearn's Gridsearch was used to hypertune these parameters. 

# In[19]:


X_train = train[["Season","Vaccine Phase","Day of Week"]].to_numpy()
y_train = train["Alberta"].to_numpy()
X_test = test[["Season","Vaccine Phase", "Day of Week"]].to_numpy()
y_test = test["Alberta"].to_numpy()


# In[20]:


param_grid = [{'alpha': [float(x) for x in np.arange(0.1,1,0.1)], 'l1_ratio': [float(x) for x in np.arange(0.1,1,0.1)],
               'random_state':[int(x) for x in np.arange(0, 1, 0.01)]}]
gs = GridSearchCV(ElasticNet(), param_grid)
gs.fit(X_train, y_train)
print("best parama:",gs.best_params_)


# ### Model
# 
# For the model, a function was created for the ability to train and predict all selected provinces simultaneously. A yesterday count variable with a counter was created to iterate over the list of the yesterday count variables to create the X_train data and the a province variable was created to iterate over the list of province names to create the target (y) data. The y_test were used for model evaluation in early stages the model development.

# In[21]:


# Defing ElasticNet Regressiion Function
def model_pred(province, yesterday):
    
    # set counter for yesterday case province list
    i = 0
    
    #train-test split
    X_train = train[["Season","Vaccine Phase","Day of Week", yesterday[i]]].to_numpy() 
    y_train = train[province].to_numpy()
    X_test = test[["Season","Vaccine Phase", "Day of Week", yesterday[i]]].to_numpy()
    y_test = test[province].to_numpy()#not used in future prediction due to actual being unknown
    
    i = i + 1
    
    #model
    model = ElasticNet(**gs.best_params_, positive=True)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return y_pred


# In[22]:


# Province list for creatiion of y vairbale
province_list = ['Alberta', 'British Columbia', 'Manitoba','New Brunswick','Newfoundland and Labrador','Nova Scotia',
                 'Ontario','Prince Edward Island','Quebec', 'Saskatchewan']

# Province yester day case list for creation of X_train
yesterday_cases_list = ['AB_yest', 'BC_yest','NB_yest', 'NFL_yest', 'PEI_yest', 'QB_yest', 'Sask_yest']

# calling function to predict cases
prediction_19 = model_pred(province_list, yesterday_cases_list).flatten()
print(prediction_19)


# ### Multi-day prediction
# 
# To be able to have a multi-day prediction, the predicted number from the intial prediction set for April 19 needed to be added to the training set for the prediction of April 20th cases. To do this, a row was added to the bottom of the train set indicating the date, the prediction numbers from the prediction_19 list, the previous day's case count, and the appropriate numbers for the additional predictive variables. 
# 
# For the new creation of the test data, the prediction_19 list of values was added to the yesterday's case variable in the data appropriate data columns. 

# In[23]:


#Adding the new preddictions to the training data
pred = pd.DataFrame({'date': "2021/04/19",'Alberta': prediction_19[0], 'British Columbia': prediction_19[1], 
                     'Manitoba': prediction_19[2], 'New Brunswick': prediction_19[3], 
                    'Newfoundland and Labrador': prediction_19[4], 'Nova Scotia': prediction_19[5],
                     'Ontario': prediction_19[6],'Prince Edward Island': prediction_19[7],'Quebec': prediction_19[8], 
                     'Saskatchewan': prediction_19[9],
                     'AB_yest':[170795], 'BC_yest':[117080],'MB_yest':[36159],'NB_yest':[1788], 'NFL_yest':[1043], 'NS_yest':[1807],
                     'ON_yest':[42854],'PEI_yest':[165], 'QB_yest':[336952], 'Sask_yest':[38160],
                     'Vaccine Phase': [2], 'Season': [1],'Day of Week': [0]})
pred['date'] = pd.to_datetime(pred.date)    
pred = pred.set_index('date')
train = train.append(pred)

#creating a new test set for April 20th with Predctions for April 19 added to the yesterday's cases
test = pd.DataFrame({'date': "2021/04/20",'Alberta': [0], 'British Columbia': [0], 'Manitoba': [0], 'New Brunswick': [0], 
                    'Newfoundland and Labrador': [0], 'Nova Scotia': [0],'Ontario': [0],
                     'Prince Edward Island': [0],'Quebec': [0], 'Saskatchewan': [0], 
                     'AB_yest':prediction_19[0], 'BC_yest':prediction_19[1],'MB_yest':prediction_19[2],
                     'NB_yest':prediction_19[3], 'NFL_yest':prediction_19[4], 'NS_yest':prediction_19[5],
                     'ON_yest':prediction_19[6],'PEI_yest':prediction_19[7], 'QB_yest':prediction_19[8],
                     'Sask_yest':prediction_19[9],
                     'Vaccine Phase': [2], 'Season': [1],'Day of Week': [1]})
test['date'] = pd.to_datetime(test.date)    
test = test.set_index('date')
test


# In[24]:


# Train data with the added row for April 19ths Predictions
train.tail()


# In[25]:


# call function to predict April 20th cases
prediction_20 = model_pred(province_list,yesterday_cases_list).flatten()


# ### Final Output of Predicted Cases

# In[26]:


# Create Dateframe to hold case predctions 
province = pd.DataFrame({"Province":['Alberta', 'British Columbia', 'Manitoba','New Brunswick','Newfoundland and Labrador','Nova Scotia',
                 'Ontario','Prince Edward Island','Quebec', 'Saskatchewan']})
cases_predicted =  pd.DataFrame({'April 19':prediction_19,'April 20':prediction_20})
cases_predicted = pd.merge(province, cases_predicted, left_index=True, right_index=True)
cases_predicted


# In[27]:


print(tabulate(cases_predicted, headers='keys', tablefmt='fancy_grid'))


# In[28]:


train.tail()


# ### Path Input Required to Download CSV. 

# In[30]:


cases_predicted.to_csv(r'/Users/kelliemaclean/Desktop/confirmed_predicted.csv', index = None, header=True)


# ## Death Predictions
# 
# To predict the cummulative death counts for the selected provinces the same methodology was used. The death prediction dataset was clean and transformed using the same functions as previously used for the case data. 

# In[167]:


# Call functions to clean data
deaths = clean_transpose_data("deaths",covid_raw_deaths)

deaths1 = deaths.drop("index", axis =1)
deaths1.tail()
                
# call functions to create yesterday's deaths    
death_df = resetindex_orginal_df("death_df", deaths1)
yesterdays_numbers_deaths = add_yesterday("yesterdays_numbers_death", deaths)


# In[168]:


death_df.tail()


# In[169]:


yesterdays_numbers_deaths.tail()


# In[170]:


#join orginal data with yesterdays number variable
deaths_w_yesterday = pd.merge(death_df, yesterdays_numbers_deaths, left_index=True, right_index=True)
#reset timeseries index
deaths_w_yesterday = deaths_w_yesterday.set_index("index")
deaths_w_yesterday.index = pd.to_datetime(cases_w_yesterday.index)
deaths_w_yesterday.tail()


# In[171]:


# add additional predicting variables (vaccine phase, day of week, season) and change datatypes to float
deaths_w_yesterday = predict_variables(deaths_w_yesterday)
deaths_w_yesterday= deaths_w_yesterday.astype("float")
deaths_w_yesterday.tail()


# ### Visual Exploration
# 
# With the plot below we can see that death counts follow similar patterns to the cummulative number of case counts over time. From this, there is clear change in the rate of change after November 2020. Due to this trend, again, the data to train the death model will only include after the November 2020. 

# In[172]:


# Snapshot look at death cases in Alberta
sns.lineplot(x=deaths_w_yesterday.index, y = 'Alberta', data=deaths_w_yesterday).set_title('Cummulative Death Trends in Alberta')
plt.show()


# In[173]:


# Snap Shot of Covid Case Trends in Alberta November 2020 on
sns.lineplot(x=cases_w_yesterday.index[284::], y = 'Alberta', data=cases_w_yesterday[284::]).set_title('Cummulative Death Trends in Alberta After November 2020')
plt.show()


# Again we see that the vaccine phase varibable likely has an impact on the rate in which the cummulative death counts increase. However, it seems that the season may play an even lesser role in the prediction of the cummulative deaths in the data.

# In[174]:


#How vaccine phases have affect numbers over time
sns.lineplot(x="Vaccine Phase", y = 'Alberta', data = deaths_w_yesterday).set_title('Cummulative Death Trends in Alberta Based on Vaccine Phases')
plt.show()


# In[175]:


## look at seasonality of 2020 and covid deaths
sns.lineplot(x="Season", y = 'Alberta', data = deaths_w_yesterday[0:280]).set_title('Cummulative Case Trends in Alberta Based on Season')
plt.show()


# ### Building the Model

# In[176]:


deaths_w_yesterday.tail()


# In[177]:


# assign train and test data
train = predict_variables(deaths_w_yesterday)
# Use only data from after Nov 2020
train = train[284::]

# create new row in df for prediction
test = pd.DataFrame({'date': "2021/04/19",'Alberta': [0], 'British Columbia': [0], 'Manitoba': [0], 'New Brunswick': [0], 
                    'Newfoundland and Labrador': [0], 'Nova Scotia': [0],'Ontario': [0],
                     'Prince Edward Island': [0],'Quebec': [0], 'Saskatchewan': [0], 
                     'AB_yest':[2040], 'BC_yest':[1530.0],'MB_yest':[959],'NB_yest':[33.0], 'NFL_yest':[6.0], 'NS_yest':[67.0],
                     'ON_yest':[7703],'PEI_yest':[0.0], 'QB_yest':[10802], 'Sask_yest':[465.0],
                     'Vaccine Phase': [2], 'Season': [1],'Day of Week': [0]})
test['date'] = pd.to_datetime(test.date)    
test = test.set_index('date')
test


# ### Hypertune Model Based on Death Data
# 
# The model parameters and the L1/L2 ratio was retuned based on the cummulative death data. 

# In[178]:


X_train = train[["Season","Vaccine Phase","Day of Week"]].to_numpy()
y_train = train["Alberta"].to_numpy()
X_test = test[["Season","Vaccine Phase", "Day of Week"]].to_numpy()
y_test = test["Alberta"].to_numpy()


# In[179]:


param_grid = [{'alpha': [float(x) for x in np.arange(0.1,1,0.1)], 'l1_ratio': [float(x) for x in np.arange(0.1,1,0.1)],
               'random_state':[int(x) for x in np.arange(1, 20,1)]}]
gs = GridSearchCV(ElasticNet(), param_grid)
gs.fit(X_train, y_train)
print("best parama:",gs.best_params_)


# In[180]:


# Defing ElasticNet Regressiion Function
def model_pred(province, yesterday):
    
    # set counter for yesterday case province list
    i = 0
    
    #train-test split
    X_train = train[["Season","Vaccine Phase","Day of Week", yesterday[i]]].to_numpy() 
    y_train = train[province].to_numpy()
    X_test = test[["Season","Vaccine Phase", "Day of Week", yesterday[i]]].to_numpy()
    y_test = test[province].to_numpy()#not used in future prediction due to actual being unknown
    
    i = i + 1
    
    #model
    model = ElasticNet(**gs.best_params_, positive=True)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return y_pred


# ### Model Predictions
# 
# The same function was used to for the model as the case data where new variables were added for the training sets and new the hypertuned parameters are applied.

# In[181]:


prediction_19_death = model_pred(province_list,yesterday_cases_list).flatten()
prediction_19_death


# ### Multi-day Prediction

# In[182]:


#Adding the new preddictions to the training data
pred = pd.DataFrame({'date': "2021/04/19",'Alberta': prediction_19_death[0], 'British Columbia': prediction_19_death[1], 
                     'Manitoba': prediction_19_death[2], 'New Brunswick': prediction_19_death[3], 
                    'Newfoundland and Labrador': prediction_19_death[4], 'Nova Scotia': prediction_19_death[5],
                     'Ontario': prediction_19_death[6],'Prince Edward Island': prediction_19_death[7],'Quebec': prediction_19_death[8], 
                     'Saskatchewan': prediction_19_death[9],
                     'AB_yest':[2040], 'BC_yest':[1530.0],'MB_yest':[959],'NB_yest':[33.0], 'NFL_yest':[6.0], 'NS_yest':[67.0],
                     'ON_yest':[7703],'PEI_yest':[0.0], 'QB_yest':[10802], 'Sask_yest':[465.0],
                     'Vaccine Phase': [2], 'Season': [1],'Day of Week': [0]})
pred['date'] = pd.to_datetime(pred.date)    
pred = pred.set_index('date')
train = train.append(pred)

#creating a new test set for April 20th
test = pd.DataFrame({'date': "2021/04/20",'Alberta': [0], 'British Columbia': [0], 'Manitoba': [0], 'New Brunswick': [0], 
                    'Newfoundland and Labrador': [0], 'Nova Scotia': [0],'Ontario': [0],
                     'Prince Edward Island': [0],'Quebec': [0], 'Saskatchewan': [0], 
                     'AB_yest':prediction_19_death[0], 'BC_yest':prediction_19_death[1],'MB_yest':prediction_19_death[2],
                     'NB_yest':prediction_19_death[3], 'NFL_yest':prediction_19_death[4], 'NS_yest':prediction_19_death[5],
                     'ON_yest':prediction_19_death[6],'PEI_yest':prediction_19_death[7], 'QB_yest':prediction_19_death[8],
                     'Sask_yest':prediction_19_death[9],
                     'Vaccine Phase': [2], 'Season': [1],'Day of Week': [1]})
test['date'] = pd.to_datetime(test.date)    
test = test.set_index('date')
test


# In[183]:


prediction_20_death = model_pred(province_list, yesterday_cases_list).flatten()


# ### Output of Cummulative Case Data

# In[184]:


# create dataframe for predicted cases
province = pd.DataFrame({"Province":['Alberta', 'British Columbia', 'Manitoba','New Brunswick','Newfoundland and Labrador','Nova Scotia',
                 'Ontario','Prince Edward Island','Quebec', 'Saskatchewan']})
deaths_predicted =  pd.DataFrame({'April 19':prediction_19_death,'April 20':prediction_20_death})
deaths_predicted = pd.merge(province, deaths_predicted, left_index=True, right_index=True)
deaths_predicted


# In[185]:


print(tabulate(deaths_predicted, headers='keys', tablefmt='fancy_grid'))


# ### Path Input Required to Download CSV

# In[65]:


deaths_predicted.to_csv(r'/Users/kelliemaclean/Desktop/deaths_predicted.csv', index = None, header=True)


# ## Is this even a feasible prediction problem?

# Predicting the next two days of Covid 19 cases is a feasible prediction problem. We see with a simple regression model we are able to provide numbers that are relatively accurate. However, in the real world, if a client posed this data science question, I likely would not use machine learning for this problem. With this type of time series cummlative data that seems to follow a linear trend, the closest prediction for April 19th and April 20 would just be based on the current numbers on April 18 and the daily count that occured for that day. That is, I would just add the same number of the daily increase of cases that occured on the 18th to the current count to estimate a prediction for April 19 and do repeat for April 20. Not every problem needs complex machine learing models, in this case, even a simple linear regression is over complicated for a simple pattern question such as the next day prediction of covid numbers. If we are looking for cummulative case number for next month and beyond, I would consider using more complex machine learning such as LSTM.   

# ## Thank you!
