# Marketing-MultiMix-Modeling-For-VMWare
Improving Customer Engagement at VMWare through Machine Learning on Marketing Data:

[Technologies leverage: R, Google Analytics]

•	A highly imbalanced customer dataset of more than 600 predictors and 50,000 records was explored using R Programming.

•	Predictive models (Logistic Regression, Random Forest & XG-Boost) over a 5-class predictor were developed to predict the user’s digital actions and thereby developing an efficient Marketing MultiMix Model (focused on Funnel Analysis). 	    

## Case-Specific Points:

### a.How should such a model be internally validated so that there could be some estimates of performance that could be shared with business:
We can use historical data to internally validate the model and check for different model parameters such as accuracy, sensitivity, misclassifications etc. Looking at these measures, we can share the model performance with the business. Also, we can check if the response prediction made by the model is actually having an impact on the customer behavior or not. In case the model predictions are having an impact on the customer behavior, we can say that the model has a good performance and same details can be shared with the business.

### b.How should the model be proven in the real world so that the business could be convinced of the benefits:
We can test the model on historical data and check for cases wherein customer behavior changed post a marketing event (email/digital etc.) We can look for the prediction made by our model in such scenario and share it with the business. Looking at the output, business can be convinced that this classification might be useful on real time data is it would look into all the customer touchpoints and predict the response which will increase the overall customer interaction across different products.

### c.What could be future extensions to this model so that the data sciences innovations team could continue to power the performance:
For this model, we are predicting the customer response across different touchpoints and post that we can choose the desired marketing option to increase interaction.This model can be further extended to predict if the customer is likely to make a purchase along with the different business specific scenarios (price, terms etc.) based on the historical data that has been acquired. This will provide a more business centric edge to our predictive model.
