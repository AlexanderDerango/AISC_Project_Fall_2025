# Table of Contents
 - About
 - Layout
 - Features
 - Takeaways From the Project


## About
Our goal for this mini project for the fall quarter of 2025 was to make a model that can be able to predict the stock price based on the news. All of our group 
members have a large background in economics (most of us are econ/managerial economics), and we wanted to explore more of that area in the machine learning realm. Although this is more of a beginner 
project, I wanted to upload this as a way of documenting my progress in machine learning and data science!

Our Project Structure
- Find Datasets
- Cleaning (fix missing values, combine the datasets together)
- Exploratory Data Analysis
- Inputing data into a model
- Debugging, fine tuning
- Analyzing results


## Layout
- INTC - Folder containing all original datasets
- aggregated_data - Folder combining news headlines with each stock price
- exploratory_analysis - Folder contatining graphs for each company
- news_headlines - Folder containing news headlines for each company
- stock_price_history - Folder containing stock price for each company we were researching
- AAPL_headlines_price_t_plus_2.csv - Apple csv file containing combined datasets
- AISC_Google_Colab.ipynb - Main Google Colab 
- AppleTrainingREG.ipynb - Training colab
- Apple_analysis.ipynb
- INTC_prepared.csv
- INTC_test.csv
- INTC_train.csv
- INTC_training.ipynb
- McDonalds_AISC (1).ipynb - Colab for McDonalds
- McDonalds_AISC_Training.ipynb
- NVDA.ipynb
- NVDA_headlines_price_t_plus_2.csv - NVDA csv file containing combined datasets
- NVDA_headlines_price_t_plus_2.xls - Different formatting
- NVDA_training.ipynb
- aggregate_stock_data.py
- appleData.ipynb
- appleTraining.ipynb
- test.csv - Test csv for McDonalds
- theChart.ipynb - EDA for all companies
- train.csv - Train csv for McDonalds


## Features
These are a list of features that we used in our model:
- Stock News Dataset: Date, Title of the Article, Stock Symbol
- Stock Price Dataset: Date, Volume, Open Price, High Price, Low Price, Close Price
  
Although there are more features in the dataset (mostly for the stock news dataset), we decided to drop them due to a substantial amount of missing data.

## Takeaways From the Project
From this project, all of us were able to learn a lot more about what the machine learning process actually looks like from start to finish. Working with real news and stock data showed us pretty quickly that data is messy and not always easy to work with, and that preprocessing and feature selection can matter just as much as the model itself.
We also learned that predicting stock prices is much harder than it initially seems. Even when the model appeared to pick up on patterns in the data, the results were often inconsistent and not obvious, but it helped us understand how unpredictable financial markets can be and why model performance in this area should be taken with a large grain of salt.
On the technical side, this project gave us hands-on experience building and training machine learning models, debugging issues, and evaluating results. Overall, this was a good learning project for us and helped build a foundation for future work in machine learning and data science.
