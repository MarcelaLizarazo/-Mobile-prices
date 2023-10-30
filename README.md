# Mobile-prices Project

### Table of Contents:
1. Introduction
2. Data
3. Exploratory Data Analysis (EDA)
4. Feature Engineering
5. Training and Evaluating Models
    1. Dimensional Reduction (PCA)
6. Conclusion

## Introduction

This project is based on the Mobile prices dataset taken from Kaggle,this dataset contains several factors among which we find the brand, size, weight, image quality, RAM, battery, etc. that interfere in the sales price of a Mobile.  With this data set, I want to estimate a price range that indicates how high the price is in relation to the mentioned features. For these we will apply the logistic regression and KNeighbors Classifier models.

## Data 

This dataset contains all the information related to different characteristics that interfere with the prices of a Mobile, such as:

- **battery_power:** Total energy a battery can store in one time measured in mAh
- **blue:** Has bluetooth or not
- **clock_speed:** speed at which microprocessor executes instructions
- **dual_sim:** Has dual sim support or not
- **fc:** Front Camera mega pixels
- **four_g:** Has 4G or not
- **int_memory:** Internal Memory in Gigabytes
- **m_dep:** Mobile Depth in cm
- **mobile_wt:** Weight of mobile phone
- **n_cores:** Number of cores of processor
- **pc:** Primary Camera mega pixels
- **px_height:** Pixel Resolution Height
- **px_width:** Pixel Resolution Width
- **ram:** Random Access Memory in Mega Bytes
- **sc_h:** Screen Height of mobile in cm
- **sc_w:** Screen Width of mobile in cm
- **talk_time:** longest time that a single battery charge will last when you are
- **three_g:** Has 3G or not
- **touch_screen:** Has touch screen or not
- **wifi:** Has wifi or not
- **price_range:** This is the target variable with value of 0(low cost), 1(medium cost), 2(high cost) and 3(very high cost).
