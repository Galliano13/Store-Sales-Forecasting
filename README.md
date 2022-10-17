# Store-Sales-Forecasting
Utilized facebook prophet to perform forecasting on datasets that consist sales data from 1115 stores. Our predictive model attempts at forecasting future sales based on historical data while taking into account seasonality effects, demand, holidays, promotions, and competition.

For the dataset that i used on this project, i put it on google drive and you can see it using this link : https://drive.google.com/drive/u/0/folders/1yWxgxkqNPTcVkJBbHNefgyAjzYAv3sTP

# 1. Understand the Problem Statement and Business Case

For companies to become competitive and skyrocket their growth, they need to leverage AI/ML to develop predictive models to forecast sales in the future. Predictive models attempt at forecasting future sales based on historical data while taking into account seasonality effects, demand, holidays, promotions, and competition.

In this project, we tried to predict future daily sales based on the features of 1115 stores. We used facebook prophet for our predictive model. Facebook prophet is open source software released by Facebook's Core Data Science Team. Prophet is a procedure for forecasting time series data based on additive model where non-linear trends are fit with yearly, weekly, and daily seasonality, plus holiday effects. Facebook Prophet works best with time series that have strong seasonal effects and several seasons of historical data.

# 2. Import Libraries and Datasets

We used two csv files for our dataset, the first ones is csv contains the information of sales from 1115 stores and the second ones is csv contains the information of 1115 stores. 

## Sales Datasets

The following is display of first two rows of the datasets :

| Store  | DayofWeek | Date | Sales | Customers | Open | Promo | StateHoliday | SchoolHolidays |
| ------------- | ------------- | ------------ | ------------- |------------- |-------------  |------------- | ------------- |------------- |
| 1  | 5  | 2015-07-31 | 5263 |555  |1   | 1  | 0  |1  |
| 2  | 5  | 2015-07-31 | 6064 |625  |1   | 1  | 0  |1  |

- Id: transaction ID (combination of Store and date) 
- Store: unique store Id
- Sales: sales/day, this is the target variable 
- Customers: number of customers on a given day
- Open: Boolean to say whether a store is open or closed (0 = closed, 1 = open)
- Promo: describes if store is running a promo on that day or not
- StateHoliday: indicate which state holiday (a = public holiday, b = Easter holiday, c = Christmas, 0 = None)
- SchoolHoliday: indicates if the (Store, Date) was affected by the closure of public schools

## Stores Information Datasets

The following is display of first two rows of the datasets :

| Store  | StoreType | Assortment | CompetitionDistance | CompetitionOpenSinceMonth | Promo2 | Promo2SinceWeek | Promo2SinceYear | PromoInterval| 
| ------------- | ------------- | ------------ | ------------- |------------- |-------------  |------------- | ------------- |------------- |
| 1114  | a | c | 870.0 | NaN | 0 | NaN | NaN | NaN | NaN |
| 1112  | d | c | 5350.0 | NaN | 1 | 22.0 | 2012.0 | Mar,Jun,Sept,Dec  | NaN |

- StoreType: categorical variable to indicate type of store (a, b, c, d)
- Assortment: describes an assortment level: a = basic, b = extra, c = extended
- CompetitionDistance (meters): distance to closest competitor store
- CompetitionOpenSince [Month/Year]: provides an estimate of the date when competition was open
- Promo2: Promo2 is a continuing and consecutive promotion for some stores (0 = store is not participating, 1 = store is participating)
- Promo2Since [Year/Week]: date when the store started participating in Promo2
- PromoInterval: describes the consecutive intervals Promo2 is started, naming the months the promotion is started anew. E.g. "Feb,May,Aug,Nov" means each round starts in February, May, August, November of any given year for that store

# 3. Explore Dataset

## Explore Sales Training Data

### Checking Missing Values

Fortunately we don't have any missing values, lets proceed with data visualization

### Data Visualization

![Data Vis 1](https://user-images.githubusercontent.com/107464383/196036290-9b7c0002-ff78-410c-9809-6b11935d9213.PNG)
![Data Vis 2](https://user-images.githubusercontent.com/107464383/196036300-1b7df4c0-54f9-48ae-b288-86da6c999b94.PNG)

- Average 600 customers per day, maximum is 4500 (note that we can't see the outlier at 7388!)
- Data is equally distibuted across various Days of the week (~150000 observations x 7 day = ~1.1 million observation) 
- Stores are open ~80% of the time
- Data is equally distributed among all stores (no bias)
- Promo #1 was running ~40% of the time 
- Average sales around 5000-6000 Euros
- School holidays are around ~18% of the time

Now lets see how many stores that are open and closed

![Stores Open and Closed](https://user-images.githubusercontent.com/107464383/196036458-5a3c805c-9b74-4069-9adc-d838972e9d06.PNG)

Lets keep open stores and remove closed stores. Open column has no meaning now, lets drop the column

## Explore Stores Information Datasets

### Checking Missing Values

The following is columns with missing values and how we handle it :
- CompetitionDistance with 3 missing values (we fill them up with average values of the 'CompetitionDistance' columns)
- CompetitionOpenSinceMonth with 354 missing values (We fill them up with zero)
- CompetitionOpenSinceYear with 354 missing values (we fill them up with zero)
- Promo2SinceWeek with 544 missing values (we fill them up with zero)
- Promo2SinceYear with 544 missing values (we fill them up with zero)
- PromoInterval with 544 missing values (we fill them up with zero)

The reason we fill them up with zero because the value of promo2 column. It seems like if 'promo2' is zero, 'promo2SinceWeek', 'Promo2SinceYear', and 'PromoInterval' information is set to zero. If there are no promo, naturally there are no competition as well.

### Data Visualization

![Data Vis 3](https://user-images.githubusercontent.com/107464383/196037255-629ed61c-a7af-4ba5-9441-60441b3cdf36.PNG)

![Data Vis 4](https://user-images.githubusercontent.com/107464383/196037260-9bd1ea6f-6ce4-468b-a9ba-b9451371de16.PNG)

- half of stores are involved in promo 2
- half of the stores have their competition at a distance of 0-3000m (3 kms away)

## Explore Merged Dataset

### Merged The Dataset

We succesfully cleaned the dataset, lets merge them into one dataset. The following is first two row of merged dataset :

| Store  | DayofWeek | Date | Sales | Customers | Promo | StateHoliday | SchoolHolidays | StoreType | Assortment | CompetitionDistance | CompetitionOpenSinceMonth | Promo2 | Promo2SinceWeek | Promo2SinceYear | PromoInterval|
| ------------- | ------------- | ------------ | ------------- |-------------  |------------- | ------------- |------------- | ------------- | ------------ | ------------- |------------- |-------------  |------------- | ------------- |------------- |
| 1  | 5  | 2015-07-31 | 5263 | 555 | 1  | 0  |1  | c | a | 1270.0 | 9.0 | 2008.0 | 0 | 0.0 | 0.0 | 0 |
| 1  | 4  | 2015-07-30 | 5020 | 546 | 1  | 0  |1  | c | a | 1270.0 | 9.0 | 2008.0 | 0 | 0.0 | 0.0 | 0 |

### Data Visualization

![Corr Plot](https://user-images.githubusercontent.com/107464383/196038086-4a31e95f-2693-4b30-a8c9-aaffc0318f1a.png)

- Customers/Prmo2 and sales are strongly correlated 

Before we do another visualization, we separate month,day,and year into separate columns

![Data Vis 5](https://user-images.githubusercontent.com/107464383/196038341-d6a12ccd-aaab-454a-9e69-e2d4b6eb8ee9.PNG)

It looks like sales and number of customers peak around christmas timeframe

![Data Vis 6](https://user-images.githubusercontent.com/107464383/196038369-4629c6ad-d294-49d8-94ec-769252ace6b5.PNG)

- Minimum number of customers are generally around the 24th of the month 
- Most customers and sales are around 30th and 1st of the month

![Data Vis 7](https://user-images.githubusercontent.com/107464383/196038435-1d3714f2-a4b2-4c9d-a0cd-0e82081779b0.PNG)

It looks like sales and number of customers peak around Saturday and Sunday

![Data Vis 8](https://user-images.githubusercontent.com/107464383/196038528-78e352a2-d74b-4146-b1aa-b39443f46f85.PNG)

- Store type b is stores with highest numbers of average sales
- Store type a is stores with lowerst numbers of average sales

![Data Vis 9](https://user-images.githubusercontent.com/107464383/196038622-22041b0f-ea2b-4300-9a68-6899d68c980f.PNG)

Promo can increased the number of sales and customers

# Train the Model Part A

We utilized facebook prophet for our predictive model. We trained the model with historical data of sales from each stores. 
The following is the result of our forecasting of store number 10 sales for 60 days :

![Fb Prophet Model A-1](https://user-images.githubusercontent.com/107464383/196038967-7bba6e19-9d13-4f12-97e9-b786743799c7.PNG)

![Fb Prophet Model A-2](https://user-images.githubusercontent.com/107464383/196038976-85c826c3-1332-47f3-8755-be26b3ae80ac.PNG)

# Train the Model Part B

In this part, we incorporated the holidays information into our model.
The following is the result of our forecasing of store number 6 sales for 90 days :

![Fb Prophet Model B-1](https://user-images.githubusercontent.com/107464383/196039148-6224fa25-516c-4b87-add1-6ba3ed15d262.PNG)

![Fb Prophet Model B-2](https://user-images.githubusercontent.com/107464383/196039160-b3dd6811-8f58-413c-812b-d0ccf680c6b7.PNG)

![Fb Prophet Model B-3](https://user-images.githubusercontent.com/107464383/196039172-60cab896-c813-4434-9ae7-2665083c77f3.PNG)








