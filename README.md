# Marketing-MultiMix-Modeling-For-VMWare
Improving Customer Engagement at VMWare through Machine Learning on Marketing Data:

[Technologies leverage: R, Google Analytics]

•	A highly imbalanced customer dataset of more than 600 predictors and 50,000 records was explored using R Programming.

•	Predictive models (Logistic Regression, Random Forest & XG-Boost) over a 5-class predictor were developed to predict the user’s digital actions and thereby developing an efficient Marketing MultiMix Model (focused on Funnel Analysis). 	    

## Case-Specific Points:

### 1. How should such a model be internally validated so that there could be some estimates of performance that could be shared with business:
We can use historical data to internally validate the model and check for different model parameters such as accuracy, sensitivity, misclassifications etc. Looking at these measures, we can share the model performance with the business. Also, we can check if the response prediction made by the model is actually having an impact on the customer behavior or not. In case the model predictions are having an impact on the customer behavior, we can say that the model has a good performance and same details can be shared with the business.

### 2. How should the model be proven in the real world so that the business could be convinced of the benefits:
We can test the model on historical data and check for cases wherein customer behavior changed post a marketing event (email/digital etc.) We can look for the prediction made by our model in such scenario and share it with the business. Looking at the output, business can be convinced that this classification might be useful on real time data is it would look into all the customer touchpoints and predict the response which will increase the overall customer interaction across different products.

### 3. What could be future extensions to this model so that the data sciences innovations team could continue to power the performance:
For this model, we are predicting the customer response across different touchpoints and post that we can choose the desired marketing option to increase interaction.This model can be further extended to predict if the customer is likely to make a purchase along with the different business specific scenarios (price, terms etc.) based on the historical data that has been acquired. This will provide a more business centric edge to our predictive model.

### 4. VMW has identified more than 600 predictor variables. The data is also highly imbalanced. Applying logistic regression and variable reduction techniques:
From the data, we can see that VMW has combined online data (detailed product and event wise data, search engine, marketing channels etc.) along with the rich corpus of offline data (booking history, features of client firm etc.). As VMW is considering both the data channels, the number of predictor variables has increased to more than 600.
The target class 0, signifies that the website has been visited but no action was chosen. Being a B2B scenario, VWM had a single product for different customers with different needs and requirements. As the products were less personalized, the target class 0 is the majority class wherein the customers just visit the website without performing any action. Using analytics, VMW can develop a model to track the customers digital footprints/touchpoints to determine the further actions to be taken. This model can be used to increase the occurrence of other classes in future.
Further, Logistic Regression cannot be applied when the number of variables if large. For the VMW case, logistic will not be preferred because:
  1.This is a multi-class classification with highly imbalanced target classes. Hence, running a regression problem will be more complicated with a higher number of variables.
  2.As a general rule of thumb, we choose 10 or 20 predictors, basically which means having ~10 events belonging to variable in a bag.
After looking at the above two reasons, we can say that it will be better to select the variables as per their importance. For our case, we have used Burato and Ranger algorithms to choose important predictor variables. Other techniques that can be used for subset selection are – AIC, BIC, Stepwise AIC, Ridge, Lasso etc.

### 5. Efficient meta-algorithms that you can use to aggregate the model:
In a brief understanding, meta-algorithms are those algorithms that use the combination of multiple basic classification model algorithms. There are multiple meta-algorithms that could be used for multi-class classification problems similar to this case study. Some of the meta-algorithms which we used for this case study are: Random Forest (Bagging) and XGBoost. Both of these meta-algorithms combine weak individual models and refine their efficiency of prediction. Random Forest (Bagging) will form multiple decision trees with various combination of predictors in order to decorrelate the influence of each predictors in the prediction. Further, in another method ie XGBoosting, we apply an iterative process of forming decision tree aiming a stepwise reduction in residual errors. Which will result into an efficient aggregate model.

### 6. Model development and accuracy of the model using the sample data provided to develop a Random Forest model:
The very first step to produce a RF was the data cleaning. Following were the predictors which were removed while data cleaning:•Predictors with 70% or greater garbage value (e.g. values such as 9999-{70%}, "unknown" value- {50%})•Predictors having N/As greater than 70%•Predictors with zero variance•Imputing the Missing values in the remaining predictors. (only numeric predictors were imputed with the median, while in categorical predictors the missing values were made a separate class)
Further there were some specific problems with the RF model viz.:
  1. There were few predictors with more than maximum factor limit that RF could handle (thatis equal to 53); such predictors were eliminated. Also, for the remaining predictors     One-Hotencoding was implemented by creating dummy numerical predictors.
  2. Further, the imbalance of the data was adjusted using SMOTE and Up-sampling were usedfor creating separate training datasets
  3. Also, the predictor values were further normalized
  4. Post that, Ranger method was used to select top 100 important variables (predictors)
Finally, RF model was designed using nTree=20 and mTry=sqrt(ncol(data)-1).
Further, the model was tunes using tuneRF function to get best mTry and nTree which were equal to 20 & 40 respectively. The model resulted in a test Accuracy of ~97%.

### 7. Insights obtain from L1, L2 regularization on regularized logistic regression model:
In case of regression and classification problems, logistic regression can be used. Generally, there are different loss functions to evaluate the models. A generalized logistic model uses log loss of errors.A general logistic regression runs on independent variables to predict the target classes, on the other hand, the regularized logistic regression penalizes the coefficients based on the misclassifications. Lambda is the regularization term used.When the number of significant features is small and other features are nearly 0, Lasso (L1 regularization) is preferred. On the other hand, Ridge (L2 regularized) works better when number of features are more and have nearly similar values. We can also say that the model interpretability reduces for Lasso more when compared to Ridge regression. In terms of the generalized error, we can say that if we want to reduce the variance by introducing some bias, we can use L1 and L2 regularizations.From the VMW case study, we have developed different regularized models (L1 and L2) and compared their performance, we have comparatively similar accuracies for the L1, L2 case (L1 -~97%, L2- ~95%). We also cross validated the model to find the best type function (loss function for model design). After running cross validation we see that the best type value is 4 - Support vector classification by Crammer and Singer, which is giving a higher accuracy (~98%) and a higher sensitivity value.

### 8. Difference between couple of extreme gradient boosting models with different values for parameters (depth, eta, etc.):
We have created a couple of gradient boosting models with different values of parameters (as listed below) and compared their outputs:
  1. eta: It is the step size shrinkage used in update to prevent overfitting. After each boosting step, we can directly get the weight of new features, and eta shrinks the feature weights to make the boosting process more conservative.
  2. gamma: It is the minimum loss reduction required to make further partition on a leaf node.
  3. max_depth: It signifies the length of the decision stump to be created. Increasing this value will tend to fit the data in a better way.
 ##### Model 1: eta=0.001, max_depth=5, gamma=3 We ran the XGBoost model 1 using the above parameters and saw that the model fits the training data pretty well, however, we can see that there is slight overfitting as the test accuracy is comparatively less.
 ##### Model 2: eta=0.1, max_depth=2, gamma=6 We ran another XGBoost model 2 using above parameters and saw that the model underfits the data as we have chosen an eta=0.1 and max_depth =2. In this case, we have limited the tree size which is leading to underfitting.
From both the models, we can see that Model 1 performs better than Model 2.

### 9. Based on the different models results, final recommendation to create the propensity to respond model:
Based on the different models created, XGBoost will our final recommendation to create the propensity model. As the XGBoost model rebalances the classes (majority and minority) and combines multiple models by penalizing previous misclassifications, we would choose XGBoosting to predict the customer behavior for the VMW case.

### 10. Possible deployment strategies for the model results so obtained:
In the initial stage of deployment, the effectiveness of the model should be tested during a pilot phase. A small quantum of new customer should be used to form a test dataset for the prediction purpose using the developed model. Further, the predicted classes of customers should be given to the Sales & Marketing or Business Development team so that they could use their effective Digital Actions (Marketing Activities) in order to improve or ensure the future customer engagement. The real assessment of the model could be done by checking the actual conversion of the customer in those predefined classes after some-time of Intended Digital Action.

### 11. Business implications of the models developed here. Value addition to the marketing department and the sales department of VMW because of this exercise:
Such data analytics could add an edge of Digital Analytics to the Sales & Marketing of VMW. Using the Digital Analytics, the Sales & Marketing team could get a competitive head start in targeting an individual prospect via planned digital actions over email-Id or cookie level. This action could analytically help Kiran and his team meet their desired intention of improving customer engagement and maintaining over-all customer-centricity. Further, this will overall reflect in to improvised revenue generation by ensuring the optimization in advertising and marketing budget via. precise and expectedly positive marketing activities.
