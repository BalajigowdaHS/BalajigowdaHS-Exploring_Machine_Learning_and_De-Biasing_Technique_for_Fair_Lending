# Machine Learning for Fair Lending 

## Overview 

In this project, we investigate how machine learning models can be leveraged to understand and promote fairness in mortgage lending practices. The primary focus is on the Home Mortgage Disclosure Act (HMDA) 2022 Dataset, which contains mortgage application records across the U.S. The goal is to analyze these records for systemic biases and implement machine learning models that can predict loan approval/denial outcomes while ensuring fairness.

This project also explores various machine learning explainability techniques to shed light on how lending decisions are made. We specifically look at racial disparities, gender imbalances, and the influence of income in lending decisions. The project also applies de-biasing techniques to improve fairness in the models, helping ensure that lending decisions are based on relevant factors rather than biased ones.

## Dataset

The dataset utilized in this project is the HMDA 2022 dataset, obtained from the official Consumer Financial Protection Bureau (CFPB) website. This dataset includes a variety of features, such as:

**Loan Information:** Details about the loan including amount, term, property value, income, debt-to-income ratio, and more.

**Demographics:** Borrower information, including race, ethnicity, gender, and age.

**Outcome Variables:** Whether the loan was approved or denied, as well as interest rates and reasons for loan denial.

## Data Preprocessing

Before diving into analysis, the dataset required extensive preprocessing to ensure that the data was clean, consistent, and ready for analysis:

**Sampling:** Given the size of the original dataset (about 16 million records), we randomly sampled 10% of the data to make it manageable for processing and analysis.

**Data Cleaning:** We addressed missing values by imputing them with the mean (for numerical values) or mode (for categorical values). Outliers were identified and handled, and irrelevant columns were removed to keep the dataset focused.

**Feature Engineering:** We converted categorical variables into binary indicators using one-hot encoding, and standardized numerical variables to make sure each feature contributed equally to the analysis.



## Key Steps and Methodologies

### Exploratory Data Analysis:

**Exploratory Data Analysis (EDA):** Comprehensive EDA was performed on both the HMDA dataset and high-priced loans dataset. This included analyzing applicant demographics, loan characteristics, and outcomes to identify patterns and disparities.

The interactive dashboard provides an in-depth visualization of the insights derived from the HMDA 2022 dataset, enabling stakeholders to explore trends in mortgage lending and identify disparities across various demographic groups, such as race, gender, income, and age.

## Key Features and Insights:

**Loan Application Overview:** The dashboard reveals that 77.92% of the loan applications were approved, while 22.08% were denied. It also breaks down the data by gender and race, showing that male applicants made up 33.04% of the applications, while female applicants accounted for 22.25%. The majority of applicants were White (64.53%), followed by Black (8.22%) and Asian (5.85%).

![image](https://github.com/user-attachments/assets/ee701b40-8adc-4f3c-93b6-194acd45c560)


**Approval and Denial Rates by Race and Income:** The analysis shows that Black and Hispanic applicants had lower approval rates (62.79% and 72.07%, respectively), compared to White and Asian applicants, who had higher approval rates of 80.26% and 79.45%. Furthermore, approval rates increased with income, with those earning over $250,000 seeing approval rates above 85% across racial groups. However, racial disparities persisted even within similar income brackets.

![image](https://github.com/user-attachments/assets/4f78ab52-b6a8-4fd1-8c4f-90a2c1581446)

**Interest Rates and Denial Rates:** The dashboard also reveals interest rate disparities, where Hispanic applicants faced an average rate of 5.16%, higher than the 4.59% for Asian and 4.77% for White applicants. Additionally, denial rates were higher for applicants from lower income brackets, especially for Black or African American applicants, regardless of their income level.

![image](https://github.com/user-attachments/assets/dbe1ddfa-7ab0-4fb9-b1ff-acbebb480f87)


**Age and Approval Rates:** Applicants aged 25-54 had the highest approval rates, while younger (under 25) and older applicants (over 74) faced higher denial rates, often due to factors like shorter credit histories or perceived financial instability.



## Machine Learning Models and Bias Mitigation

In this project, we developed two machine learning models: one for Loan Approval/Denial Prediction and another for predicting High-Priced Loans. Both models were built using the HMDA 2022 dataset, which contains a wealth of features such as income, loan amount, debt-to-income ratio, race, gender, and others. The aim was not only to predict loan outcomes accurately but also to ensure fairness in the decision-making process by addressing any potential biases that might exist in the data.

**Loan Approval/Denial Model**

The goal of this model was to predict whether a loan application would be approved or denied. The target variable, action_taken, was trained on the five machine learning models: Logistic Regression, Decision Tree, Random Forest, XGBoost, and mXGB. After evaluating these models, XGBoost was selected as the final model, as it consistently delivered the highest accuracy and F1-score, with 89.06% accuracy and 0.83 F1-score.

![image](https://github.com/user-attachments/assets/c6350fb3-b3bb-44b7-abf0-c0d5dc7716e4)


**Bias in the Dataset**
During the evaluation of the model, we identified significant racial disparities in loan approval rates:

White applicants: 80.26% approval rate
Asian applicants: 79.45% approval rate
Black applicants: 62.79% approval rate
Hispanic or Latino applicants: 72.07% approval rate

These results indicated that Black and Hispanic applicants were less likely to be approved for loans compared to their White and Asian counterparts, suggesting potential bias in the lending process.

### Bias Mitigation Techniques
To mitigate these disparities, several debiasing techniques were applied:

**Feature Selection:**

We identified and removed features that were contributing to biased predictions. For example, property value was found to disproportionately affect racial groups, particularly Black applicants.
After removing property value, the Adverse Impact Ratio (AIR), which measures fairness between groups, improved for Black and Hispanic applicants. This led to a small decrease in F1-score by 0.02, but with an improvement in fairness.

**Hyperparameter Tuning:**

We optimized the model's hyperparameters, particularly learning rate and tree depth, to reduce bias and enhance fairness. After fine-tuning, recall for Black applicants improved by 0.04, indicating better model sensitivity for underrepresented groups.

![image](https://github.com/user-attachments/assets/bdf0a354-c814-420d-84c0-86f4870e8d25)

**Post-Processing Adjustments:**

![image](https://github.com/user-attachments/assets/94b6bb60-b2c8-47d1-8819-6c3f7689f46c)

After model training, decision thresholds were adjusted to equalize the representation of racial groups. This adjustment increased the approval rate for Black applicants from 62.79% to 65.23%.
These adjustments led to a more equitable distribution of loan approvals across racial groups, improving fairness without significant performance loss.

![image](https://github.com/user-attachments/assets/a5eb28ad-e04a-40ae-b5a5-23bac0d3adcb)



## High-Priced Loan Model

The second model aimed to predict whether a loan would be classified as high-priced based on interest rates that exceed a threshold (typically 1.5% above the average). The target variable, high_price, was predicted using XGBoost, Random Forest, and Logistic Regression, with XGBoost again emerging as the top performer, providing 87.63% accuracy and 0.82 precision.

![image](https://github.com/user-attachments/assets/e9521323-401a-4d92-9ebb-1f3754ef9670)

**Bias in the Dataset**
The analysis revealed significant disparities in the interest rates for different racial groups:

Asian applicants: Average interest rate of 4.59%
White applicants: Average interest rate of 4.77%
Black applicants: Average interest rate of 4.99%
Hispanic applicants: Average interest rate of 5.16%
Minority groups, particularly Black and Hispanic applicants, were charged higher interest rates on average compared to White and Asian applicants, even after accounting for other loan characteristics.

Bias Mitigation Techniques
To address these disparities, we applied the following debiasing techniques:

**Feature Selection:**

We identified and removed features that were contributing to unfair pricing predictions, such as debt-to-income ratio, which had a disproportionate influence on interest rates.
After feature selection, the interest rate disparity between Black and White applicants reduced from 0.22% to 0.15%. Hispanic applicants' interest rates were reduced from 5.16% to 5.05%.

**Hyperparameter Tuning:**

We fine-tuned model parameters to ensure that the model was not biased towards assigning higher rates to minority groups. This adjustment helped reduce the gap in interest rates, making the model more equitable in predicting high-priced loans.

![image](https://github.com/user-attachments/assets/f1a74f31-3b51-47e2-83ca-606d594aa976)


**Post-Processing Adjustments:**

![image](https://github.com/user-attachments/assets/303c936c-1951-4cf4-9258-dc871a2fbdb7)


Post-processing adjustments were made to interest rate thresholds to improve fairness across racial groups. After these adjustments, the interest rate for Black applicants decreased by 0.1%, helping bring their rates closer to White applicants.

![image](https://github.com/user-attachments/assets/eb580dec-c3fd-4e29-9819-fd69949db898)

## Results of Bias Mitigation

After applying the bias mitigation techniques, the interest rate gap between Hispanic and White applicants was reduced from 0.39% to 0.28%.
The gap between Black and Asian applicants narrowed from 0.40% to 0.30%, indicating a more fair distribution of interest rates.


## Conclusion

Through the application of machine learning models and bias mitigation techniques, we were able to develop models that not only predicted loan outcomes with high accuracy but also addressed potential biases in the decision-making process. The key findings include:

In the Loan Approval/Denial Model, applying feature selection, hyperparameter tuning, and post-processing adjustments helped improve fairness, with the approval rate for Black applicants increasing from 62.79% to 65.23% and reducing racial disparities in loan approvals.

In the High-Priced Loan Model, feature selection and post-processing adjustments reduced interest rate disparities, with Black applicants' rates decreasing by 0.1% and Hispanic applicants' rates dropping from 5.16% to 5.05%.

The XGBoost model was chosen as the final model for both tasks due to its superior performance and ability to handle imbalanced datasets, while still allowing for fairer predictions. These results demonstrate the importance of integrating fairness and transparency into machine learning models, ensuring that lending decisions are not only accurate but also equitable across different demographic groups.
