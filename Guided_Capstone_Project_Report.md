# Guided Capstone Project Report: Big Mountain Resort

## Problem Statement
The primary objective is to analyze and build predictive models to help Big Mountain Resort optimize operations and enhance customer satisfaction. Specifically, this involves adjusting ticket prices and chairlift operations to maximize profitability.

## Data Wrangling
The dataset was thoroughly preprocessed, which included handling missing values, encoding categorical variables, and scaling numerical features to prepare it for modeling.

## Exploratory Data Analysis
Exploratory Data Analysis (EDA) was conducted to understand data distributions, identify patterns, and uncover relationships between variables. Key insights included identifying peak visitor periods and correlations between ticket sales and chairlift usage.

## Model Preprocessing with Feature Engineering
Feature engineering involved creating new variables that capture the essence of the data. This included generating interaction terms and aggregating features to enhance model performance.

## Algorithms Used to Build the Model with Evaluation Metric
Several machine learning algorithms were employed:
- **Logistic Regression**
- **Decision Tree**
- **Random Forest**
- **Support Vector Machine (SVM)**
- **K-Nearest Neighbors (KNN)**

Each model was evaluated based on accuracy, precision, recall, and F1-score. Hyperparameter tuning was performed using GridSearchCV to optimize each model's performance.

## Winning Model and Scenario Modelling
The Random Forest model was identified as the best-performing model, with the highest accuracy and F1-score. Scenario modeling indicated that the current ticket price of $81.00 could be increased to $91.23. Additionally, removing one chairlift would not lead to substantial losses, suggesting an opportunity for cost savings.

## Pricing Recommendation
Based on the model's results, it is recommended to increase the ticket price to $91.23. This adjustment is projected to maximize revenue without significantly affecting customer turnout.

## Conclusion
Implementing the Random Forest model will provide Big Mountain Resort with accurate predictions, aiding in better decision-making and improving customer experiences. Adjusting ticket prices and optimizing chairlift operations based on model insights can lead to increased profitability.

## Future Scope of Work
Future work should incorporate weekday pricing and more detailed cost analyses, including potential savings from removing fast quads. Developing an app to integrate these models into the resort's operations would facilitate real-time decision-making and continuous improvement.

## Figures
### Cross-Validation Score
![Cross Validation Score](https://github.com/shrekapoor99/Springboard/blob/main/crossvalidationscores.png)

### Forest Regressor Features (model selected)
![Forest Regressor Features](https://github.com/shrekapoor99/Springboard/blob/main/forestregressorfeatures.png)

### Feature Correlation Plot
![Feature Corrleation Plot](https://github.com/shrekapoor99/Springboard/blob/main/featurecorrelationplot.png)

## Process and Understanding

### Data Visualization
Relevant charts, plots, and maps were used to depict data intuitively. Examples include distribution plots for visitor numbers and correlation heatmaps.

### Insights and Trends
EDA provided insights such as peak visitor times and the impact of weather conditions on ticket sales. These findings informed the feature engineering process and the model selection strategy.

### Model and Metrics Methodology
The methodology for selecting models and metrics was based on their ability to predict ticket sales and optimize operational decisions accurately. Detailed explanations of each model’s performance metrics were provided.

### Scenario Modelling
Scenario modeling was conducted to evaluate the impact of different pricing and operational strategies on profitability. This included adjusting ticket prices and evaluating the necessity of all chairlifts.

### Valid Conclusions and Recommendations
The conclusions and recommendations are based on robust model performance and scenario analyses. Increasing ticket prices and optimizing chairlift operations are expected to enhance profitability without compromising customer satisfaction.
