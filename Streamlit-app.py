import streamlit as st
import numpy as np
import pandas as pd
import os
import joblib
# from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns


# Page layout
## Page expands to full width
st.set_page_config(layout="wide")
#---------------------------------#
# Title

st.title('Bank Loan Defaulter Classifier App')
st.markdown("""
This app predicts weather a person will be defaulter or not on the basis of Historical Data!

""")
image = os.path.join(os.path.dirname(__file__), 'loandefaulter.jpg')
st.image(image, use_column_width=True)
#---------------------------------#
# About
expander_bar = st.expander("About")
expander_bar.markdown("""
* **Python libraries:** numpy, pandas, streamlit, scikit-learn, xgboost
* **Ease of Access:** This have made this assignment on streamlit so that you could easily play with inputs and check weather the program is working properly or not.
""")

# Construct the file path to xgboost_model.pkl
pickle_file_path = os.path.join(os.path.dirname(__file__), 'model.pkl')
xgb_model = joblib.load(pickle_file_path)

# Construct the file path to final_df.csv
def load_data():
    file_path = os.path.join(os.path.dirname(__file__), 'cleaned_data.csv')
    return pd.read_csv(file_path)

data = load_data()


# Sidebar 

st.sidebar.header('User Input Parameters')

def user_input_features():
    Patron_Salary = st.sidebar.number_input("Enter the annual income", min_value=500, max_value=200000000, value=2500, step=500)
    Automobile_Possession = st.sidebar.selectbox('Automobile_Possession', ["Yes", "No"])
    Two_Wheeler_Ownership = st.sidebar.selectbox('Two-Wheeler Possession', ["Yes", "No"])
    Ongoing_Borrowing = st.sidebar.selectbox('Already has loans', ["Yes", "No"])
    Residence_Proprietorship = st.sidebar.selectbox('Owns a house', ["Yes", "No"])
    Offspring_Number = st.sidebar.number_input("Enter the number of children", min_value=0, max_value=20, value=0, step=1)
    Loan_Capital = st.sidebar.number_input("Enter the Loan Capital", min_value=500, max_value=200000000, value=2500, step=500)
    Borrowing_Periodic_Payment = st.sidebar.number_input("Enter the Installment amount", min_value=500, max_value=200000000, value=2500, step=500)
    Customer_Revenue_Category = st.sidebar.selectbox('Persons job category', ["Commercial", "Service", "Retired", "Govt Job", "Student", "Unemployed", "Maternity leave", "Businessman"])
    Patron_Academic_Qualification = st.sidebar.selectbox('Academic Qualification', ["Secondary", "Graduation", "Graduation dropout", "Junior secondary", "Post Grad"])
    Customer_Conjugal_State = st.sidebar.selectbox('FAMILY STATUS', ["Single", "Married", "Widow", "Divorced"])
    Patron_Sex = st.sidebar.selectbox('Gender', ["Male", "Female"])
    Borrowing_Agreement_Category = st.sidebar.selectbox('Borrowing_Agreement_Category', ["Cash Loan", "Revolving Loan"])    
    Customer_Living_Arrangement =  st.sidebar.selectbox('HOUSING TYPE', ["Home", "Family", "Office", "Municipal", "Rental", "Shared"])
    Elderliness_in_Days = st.sidebar.number_input("Enter Age (in years)", min_value=0,)
    Enlistment_Period_in_Days = st.sidebar.number_input("Enter Enlistment Period (in years)", min_value=0,)
    Employment_Phone_Operation = st.sidebar.selectbox('Wrok Phone', ["Yes", "No"])
    Patron_Kin_Count = st.sidebar.number_input("Enter the number of Family Members", min_value=0,)
    Patron_Constant_Correspondence_Marker = st.sidebar.selectbox('Patron_Constant_Correspondence_Marker', ["Yes", "No"])
    Customer_Professional_Communication_Marker = st.sidebar.selectbox('Professional Communication by customer', ["Yes", "No"])
    Sort_of_Institution = st.sidebar.selectbox('Occupation Type', ["Self-employed", "Government", "Business Entity Type 3", "Other", "Industry: type 3", "Business Entity Type 2", "Business Entity Type 1",
                                                                    "Transport: type 4", "Construction", "Trade: type 3", "Industry: type 2", "Trade: type 7", "Trade: type 2","Agriculture", "Military", "Kindergarten",
                                                                    "Housing", "Industry: type 1", "Industry: type 11", "Bank", "School", "Industry: type 9", "Medicine", "Postal", "University", "Transport: type 2",
                                                                    "Restaurant", "Electricity", "Industry: type 4", "Security Ministries", "Services", "Transport: type 3", "Police", "Mobile", "Hotel", "Security",
                                                                    "Industry: type 7", "Advertising", "Cleaning", "Realtor", "Trade: type 6", "Culture", "Industry: type 5", "Telecom", "Trade: type 1", "Industry: type 12",
                                                                    "Industry: type 8", "Insurance" "Emergency", "Legal Services", "Industry: type 10", "Trade: type 4", "Industry: type 6", "Industry: type 13", "Transport: type 1",
                                                                    "Religion", "Trade: type 5",])
    Rating_Origin_2 = st.sidebar.number_input("Enter rating_origin_2 (up to four decimal places)",
                                                min_value=0.0,
                                                format="%.4f",  # Format to display up to four decimal places
                                                step=0.0001  # Step size for the input
                                            )
    Telecommunication_Switch = st.sidebar.number_input("Enter the Telecommunication switch", min_value=0,)
    

    input_data = {
            'Patron_Salary':Patron_Salary,
            'Automobile_Possession':Automobile_Possession,
            'Two_Wheeler_Ownership':Two_Wheeler_Ownership,
            'Ongoing_Borrowing':Ongoing_Borrowing,
            'Residence_Proprietorship':Residence_Proprietorship,
            'Offspring_Number':Offspring_Number,
            'Loan_Capital':Loan_Capital, 
            'Borrowing_Periodic_Payment':Borrowing_Periodic_Payment, 
            'Customer_Revenue_Category':Customer_Revenue_Category,
            'Patron_Academic_Qualification':Patron_Academic_Qualification,
            'Customer_Conjugal_State':Customer_Conjugal_State, 
            'Patron_Sex':Patron_Sex,
            'Borrowing_Agreement_Category':Borrowing_Agreement_Category,
            'Customer_Living_Arrangement':Customer_Living_Arrangement, 
            'Elderliness_in_Days':Elderliness_in_Days,  
            'Enlistment_Period_in_Days':Enlistment_Period_in_Days, 
            'Employment_Phone_Operation':Employment_Phone_Operation,
            'Patron_Kin_Count':Patron_Kin_Count,  
            'Patron_Constant_Correspondence_Marker':Patron_Constant_Correspondence_Marker,
            'Customer_Professional_Communication_Marker':Customer_Professional_Communication_Marker,
            'Sort_of_Institution':Sort_of_Institution, 
            'Rating_Origin_2':Rating_Origin_2,
            'Telecommunication_Switch':Telecommunication_Switch,
            }
    features = pd.DataFrame(input_data, index=[0])
    return features

input_df = user_input_features()

# Displays the user input features
st.subheader('User Input parameters')
if st.button('Show Input DataFrame'):
    st.write(input_df)
    



# Labelling DataSet
Patron_Academic_Qualification_LABEL = {"Secondary":0, "Graduation":1, "Graduation dropout":2, "Junior secondary":3, "Post Grad":4}
Patron_Sex_LABEL = {"Male":1, "Female":0}
Borrowing_Agreement_Category_LABEL = {"Cash Loan":0, "Revolving Loan":1 }
Patron_Constant_Correspondence_Marker_LABEL = {"Yes":1, "No":0}
Automobile_Possession_LABEL = {"Yes":1, "No":0}
Two_Wheeler_Ownership_LABEL = {"Yes":1, "No":0}
Ongoing_Borrowing_LABEL = {"Yes":1, "No":0}
Residence_Proprietorship_LABEL = {"Yes":1, "No":0}
Employment_Phone_Operation_LABEL = {"Yes":1, "No":0}
Customer_Professional_Communication_Marker_LABEL = {"Yes":1, "No":0}
Customer_Revenue_Category_LABEL = {"Commercial":0, "Service":1, "Retired":2, "Govt Job":3, "Student":4, "Unemployed":5, "Maternity leave":6, "Businessman":7}
Customer_Conjugal_State_LABEL = {"Married":0, "Widow":1, "Single":2, "Divorced":3}
Customer_Living_Arrangement_LABEL = {"Home":0, "Family":1, "Office":2, "Municipal":3, "Rental":4, "Shared":5}
Sort_of_Institution_LABEL = {"Self-employed":0, "Government":1, "XNA":2, "Business Entity Type 3":3, "Other":4, "Industry: type 3":5, "Business Entity Type 2":6, "Business Entity Type 1":7,
 "Transport: type 4":8, "Construction":9, "Trade: type 3":10, "Industry: type 2":11, "Trade: type 7":12, "Trade: type 2":13,"Agriculture":14, "Military":15, "Kindergarten":16,
 "Housing":17, "Industry: type 1":18, "Industry: type 11":19, "Bank":20, "School":21, "Industry: type 9":22, "Medicine":23, "Postal":24, "University":25, "Transport: type 2":26,
 "Restaurant":27, "Electricity":28, "Industry: type 4":29, "Security Ministries":30, "Services":31, "Transport: type 3":32, "Police":33, "Mobile":34, "Hotel":35, "Security":36,
 "Industry: type 7":37, "Advertising":38, "Cleaning":39, "Realtor":40, "Trade: type 6":41, "Culture":42, "Industry: type 5":43, "Telecom":44, "Trade: type 1":45, "Industry: type 12":46,
 "Industry: type 8":47, "Insurance" "Emergency":48, "Legal Services":49, "Industry: type 10":50, "Trade: type 4":51, "Industry: type 6":52, "Industry: type 13":53, "Transport: type 1":54,
 "Religion":55, "Trade: type 5":56,}

# Mapping input DATASET
input_df["Patron_Academic_Qualification"] = input_df["Patron_Academic_Qualification"].map(Patron_Academic_Qualification_LABEL)
input_df["Patron_Sex"] = input_df["Patron_Sex"].map(Patron_Sex_LABEL)
input_df["Borrowing_Agreement_Category"] = input_df["Borrowing_Agreement_Category"].map(Borrowing_Agreement_Category_LABEL)
input_df["Patron_Constant_Correspondence_Marker"] = input_df["Patron_Constant_Correspondence_Marker"].map(Patron_Constant_Correspondence_Marker_LABEL)
input_df["Automobile_Possession"] = input_df["Automobile_Possession"].map(Automobile_Possession_LABEL)
input_df["Two_Wheeler_Ownership"] = input_df["Two_Wheeler_Ownership"].map(Two_Wheeler_Ownership_LABEL)
input_df["Ongoing_Borrowing"] = input_df["Ongoing_Borrowing"].map(Ongoing_Borrowing_LABEL)
input_df["Residence_Proprietorship"] = input_df["Residence_Proprietorship"].map(Residence_Proprietorship_LABEL)
input_df["Customer_Revenue_Category"] = input_df["Customer_Revenue_Category"].map(Customer_Revenue_Category_LABEL)
input_df["Customer_Conjugal_State"] = input_df["Customer_Conjugal_State"].map(Customer_Conjugal_State_LABEL)
input_df["Sort_of_Institution"] = input_df["Sort_of_Institution"].map(Sort_of_Institution_LABEL)
input_df["Employment_Phone_Operation"] = input_df["Employment_Phone_Operation"].map(Employment_Phone_Operation_LABEL)
input_df["Customer_Living_Arrangement"] = input_df["Customer_Living_Arrangement"].map(Customer_Living_Arrangement_LABEL)
input_df["Customer_Professional_Communication_Marker"] = input_df["Customer_Professional_Communication_Marker"].map(Customer_Professional_Communication_Marker_LABEL)



# Mappig the Dataset
data["Patron_Academic_Qualification"] = data["Patron_Academic_Qualification"].map(Patron_Academic_Qualification_LABEL)
data["Patron_Sex"] = data["Patron_Sex"].map(Patron_Sex_LABEL)
data["Borrowing_Agreement_Category"] = data["Borrowing_Agreement_Category"].map(Borrowing_Agreement_Category_LABEL)
data["Patron_Constant_Correspondence_Marker"] = data["Patron_Constant_Correspondence_Marker"].map(Patron_Constant_Correspondence_Marker_LABEL)
data["Automobile_Possession"] = data["Automobile_Possession"].map(Automobile_Possession_LABEL)
# data["Two_Wheeler_Ownership"] = data["Two_Wheeler_Ownership"].map(Two_Wheeler_Ownership_LABEL)
data["Ongoing_Borrowing"] = data["Ongoing_Borrowing"].map(Ongoing_Borrowing_LABEL)
data["Residence_Proprietorship"] = data["Residence_Proprietorship"].map(Residence_Proprietorship_LABEL)
data["Customer_Revenue_Category"] = data["Customer_Revenue_Category"].map(Customer_Revenue_Category_LABEL)
data["Customer_Conjugal_State"] = data["Customer_Conjugal_State"].map(Customer_Conjugal_State_LABEL)
data["Sort_of_Institution"] = data["Sort_of_Institution"].map(Sort_of_Institution_LABEL)
data["Employment_Phone_Operation"] = data["Employment_Phone_Operation"].map(Employment_Phone_Operation_LABEL)
data["Customer_Living_Arrangement"] = data["Customer_Living_Arrangement"].map(Customer_Living_Arrangement_LABEL)
data["Customer_Professional_Communication_Marker"] = data["Customer_Professional_Communication_Marker"].map(Customer_Professional_Communication_Marker_LABEL)


X = data.drop("Default", axis = 1)
y = data["Default"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=41)

# Make predictions on the test set
y_pred = xgb_model.predict(X_test)

# Print accuracy score
accuracy = accuracy_score(y_test, y_pred)
st.subheader('Accuracy')
st.write("Accuracy(%) of the model is :",accuracy*100)

prediction = xgb_model.predict(input_df)
prediction_proba = xgb_model.predict_proba(input_df)


st.subheader('Prediction')
loan_tpye = np.array(['Non-Defaulter','Defaulter'])
st.write(loan_tpye[prediction])

st.subheader('Risk Level')
st.write("The risk level(in %) is:", prediction_proba[0][1]*100)


#Making some graphs

feature_importance = xgb_model.feature_importances_
feature_names = X_train.columns

# Create a DataFrame to associate feature names with their importance scores
feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importance})

# Sort the DataFrame by importance scores (optional)
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Plot the Feature Importance graph
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
# Subplot 1: Feature Importance
sns.barplot(data=feature_importance_df, x='Importance', y='Feature', palette="RdYlGn_r", ax= ax[0])
ax[0].set_xlabel('Feature Importance')
ax[0].set_ylabel('Features')
ax[0].set_title('XGBoost Feature Importance')

# Subplot 2: Accuracy Score
colors = ['#00A36C', 'lightgray']
explode = [0.1, 0]
ax[1].pie([accuracy, 1 - accuracy], labels=['Accuracy', ''], colors=colors, explode=explode, autopct='%1.1f%%', startangle=90)
ax[1].set_title('Accuracy Score')

plt.tight_layout()
st.pyplot(fig)