# COVID-19-Numbers-Elastic-Net
Final ML project for COVID-19 Numbers in Canada
 
 # Prediction Problem

## The objective is to predict the cumulative numbers of confirmed cases and deaths for April 19-20, 2020 (i.e., the two days immediately following the project deadline) for each of the 10 Canadian provinces

## Methodology

- Total number of cases (cummulative) are first cleaned to include only selcted Canadian Provinces and engineered for data exploration and model predictions. 

- Training data is created by creating the following features for each data point (day) of the dataset: 

        1)The vaccine roll out Phases, best dates for periods were generalized and estimated based on publicly available information.

        2)Seasons were based on the majority of the provinces seaonal weather changes.

        3)Day of the Week: Monday - Friday, may have an impact on numbers due to weekend events,testing delays, and data updates.

        4)Yesterday's case numbers, used a predicting variable for next day's cases. 

- An ElasticNet linear model is trained using the data of the previous step.

        While the data is time-series and likely well suited for an LSTM model, the problem, in general, is not overly complicated. Instead, to keep the model simple and explainable, I chose to use a regression model. An Elastic Net was used to increase the model performance by potentially implimenting both L1 and L2 penialites. Additionally, I expect to have some collinearity in my predictor vairables and the Elastic Net will be able handle this feature issue. 
        
- The linear model is used to predict new cases for tomorrow and the next day. 

- Predictions are shown in the Table for April 19 & 20

## Step 1: Cleaning and Engineering:

- The data was read from github URL for daily cummlative cases and deaths

- A function was created to clean and transposed the data to only include selected Canadian provinces

- A function was created for to create yesterdays cases numbers

- A function was created for adding additional predictor variables 

### Adding Predictor Variable Column for Yesterday's Cases

To have yesterdays case count for each province as predicting variable, the orginal dataset had to be copied and shifted by one row. The column names were changed to indicate it is the yesterday count and then merged on the index with orginal dataset. 

### Adding Additional Predictor Variables

Three added predictor vairbale include the vaccine phases for the roll out to the general population, the current season, and the day of the week (Mon, Tues, Wed, etc). The vaccine phases in theory should have a effect on the number of occuring cases due to the number susceptiable individuals deacreasing. For the season effect on the model, it is commonly knowledge for most infectious disease cycles to be dependent on weather patterns. Warmer tempature can have several effects on the spread of the Sar-CoV-2 due to viral particals becoming unstable in warmer tempatures/ UV exposure and individuals spend less time in doors in poorly circulating air. Lastly, the forth variable potentially has a impact on the case numbers due to when people are more commonly contracting the virus(eg. over the weekends), getting tested, and when the data is getting updated across institutions and provinces. 

----------------------------------

Vaccine Phases:

Phase 0 - No/little vaccine available to public (2020)

Phase 1 - ICU, Seniors, LTC (Jan 2021 - April 2021)

Phase 2 - Chronic illness, health care, above 65+ (April 2021 - Present)

-----------------------------------
Seasons:

Winter(Nov 2 - March 1) = 0

Spring(Feb 2 - June 1) = 1

Summmer(June 2 - Sept 1) = 2

Fall (Sept 2 - Nov 1) = 3

-----------------------------------

Day of the week: 

Monday = 0 - Sunday = 6

-----------------------------------

The function below creates the above variable in the data based on the timestamp index, where pandas datatime function was utilized for simple variable creatiion. 

During the model development process, the first model that was attempted was a simple linear regression using the sklearn package. The train data for this model was first scaled to see if the scaling added predictive value, however, with the further investigation it was found that scaled training data verse non scaled train data did not provide any benefit. From this, it was decided to leave the training data unscaled for the model training. 

To futher improve the model, the simple linear regression model was upgraded to an ElasticNet. This model type has several benifits. One being that the ElasticNet will eliminate unused features while including the benefit of feature coefficient reduction. The ElasticNet model, due to the combination of the benefits from both the Lasso and ridge model is well suited for model that have collineality present, such that was shown above in the collinearity marix. 

To predict future case counts, an additional line was created to input feature variable characteristics for that day. 

### Model

For the model, a function was created for the ability to train and predict all selected provinces simultaneously. A yesterday count variable with a counter was created to iterate over the list of the yesterday count variables to create the X_train data and the a province variable was created to iterate over the list of province names to create the target (y) data. The y_test were used for model evaluation in early stages the model development.

### Multi-day prediction

To be able to have a multi-day prediction, the predicted number from the intial prediction set for April 19 needed to be added to the training set for the prediction of April 20th cases. To do this, a row was added to the bottom of the train set indicating the date, the prediction numbers from the prediction_19 list, the previous day's case count, and the appropriate numbers for the additional predictive variables. 

For the new creation of the test data, the prediction_19 list of values was added to the yesterday's case variable in the data appropriate data columns.

 Province                  │   April 19 │   April 20 │
╞════╪═══════════════════════════╪════════════╪════════════╡
│  0 │ Alberta                   │ 172030     │ 173201     │
├────┼───────────────────────────┼────────────┼────────────┤
│  1 │ British Columbia          │ 116958     │ 117734     │
├────┼───────────────────────────┼────────────┼────────────┤
│  2 │ Manitoba                  │  38875.9   │  39050.4   │
├────┼───────────────────────────┼────────────┼────────────┤
│  3 │ New Brunswick             │   1828.26  │   1839.7   │
├────┼───────────────────────────┼────────────┼────────────┤
│  4 │ Newfoundland and Labrador │   1076.26  │   1083.29  │
├────┼───────────────────────────┼────────────┼────────────┤
│  5 │ Nova Scotia               │   1831.26  │   1837.3   │
├────┼───────────────────────────┼────────────┼────────────┤
│  6 │ Ontario                   │ 419265     │ 421730     │
├────┼───────────────────────────┼────────────┼────────────┤
│  7 │ Prince Edward Island      │    168.723 │    169.578 │
├────┼───────────────────────────┼────────────┼────────────┤
│  8 │ Quebec                    │ 349489     │ 351445     │
├────┼───────────────────────────┼────────────┼────────────┤
│  9 │ Saskatchewan              │  38764.2   │  39056     │
╘════╧═══════════════════════════╧════════════╧═════════
