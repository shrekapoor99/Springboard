 Guided Capstone Project Report: Big Mountain Resort

## Introduction
This report summarizes the findings from the modeling phase of the Guided Capstone Project, aiming to analyze and build predictive models to help Big Mountain Resort optimize their operations and improve customer satisfaction.

## Data Preparation
The dataset underwent extensive preprocessing, including handling missing values, encoding categorical variables, and scaling numerical features.

## Model Training and Evaluation
Several machine learning models were trained, including Logistic Regression, Decision Tree, Random Forest, Support Vector Machine (SVM), and K-Nearest Neighbors (KNN). Each model was evaluated based on accuracy, precision, recall, and F1-score, with hyperparameter tuning performed using GridSearchCV to optimize performance.

## Results
The Random Forest model emerged as the best-performing model with the highest accuracy and F1-score. The confusion matrix indicated balanced performance across classes, and feature importance analysis highlighted key factors influencing predictions.

## Recommendations
Based on the model's results, it is recommended that Big Mountain Resort adopt the Random Forest model for predictive tasks and focus on the most important features identified. The current ticket price of $81.00 could be increased to $91.23, and one chairlift could be removed without substantial losses to cut costs. Investing in a new chairlift could lead to greater profits overall. Continuous monitoring and updating of the model with new data is essential to maintain accuracy and relevance.

## Deficiencies and Future Work
The model's deficiency includes the exclusion of weekday prices, which could provide additional context. Another deficiency is the lack of detailed cost information on other potential cuts, such as fast quads, limiting the ability to create additional savings. The local market in Montana likely influenced conservative pricing strategies. In the future, the business could use the model to determine ticket costs and facilities to support additional customers. Deploying an app with HTML fill-in boxes for desired labels could help integrate this model into overall business implementation.

## Figures
### Model Performance Comparison
![Model Performance Comparison](URL_TO_YOUR_IMAGE_1)

### Confusion Matrix
![Confusion Matrix](URL_TO_YOUR_IMAGE_2)

### Feature Importance Plot
![Feature Importance Plot](URL_TO_YOUR_IMAGE_3)

## Conclusion
Implementing the Random Forest model will provide Big Mountain Resort with accurate predictions, aiding in better decision-making and improved customer experiences.
