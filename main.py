import pandas as pd
from sklearn.ensemble import RandomForestClassifier 
from sklearn.preprocessing import LabelEncoder
from joblib import dump 

#load the dataset
tele_cust = pd.read_csv('Telco_Customer_Churn.csv')

#data preprocessing
#fill misiing values in 'Total_Charges' & Convert to numeric
tele_cust['TotalCharges'] = pd.to_numeric(tele_cust['TotalCharges'], errors='coerce')
tele_cust['TotalCharges'].fillna(0, inplace=True)

#convert 'churn' to binary labels
label_encoder = LabelEncoder()
tele_cust['Churn'] = label_encoder.fit_transform(tele_cust['Churn'])

#use label encoding for " internetservice " & 'contract '
tele_cust['InternetService'] = label_encoder.fit_transform(tele_cust['InternetService'])
tele_cust['Contract'] = label_encoder.fit_transform(tele_cust['Contract'])

# Select Features 
Sel_features= ['tenure', 'InternetService', 'Contract', 'MonthlyCharges', 'TotalCharges']
X= tele_cust[Sel_features]
y= tele_cust['Churn']

#train the random forest model
model = RandomForestClassifier(n_estimators=100, random_state=101)
model.fit(X,y)

#save the trwined model to a file
dump(model, 'random_forest_model.joblib')
