import streamlit as st
import pandas as pd
import numpy as np
import joblib

import xgboost

from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder

# Title
st.markdown('### Machine learning project by Mirsaid')
st.markdown('##### Goal: to predict the mortage backed securitires')
st.markdown('###### Github: https://github.com/mirsaidl')
model = joblib.load('xgboost.pkl')
pipeline = joblib.load('pipeline_.pkl')

# Input fields
credit_score = st.number_input("Credit Score:", min_value=0, step=100)
fthb = st.radio("Are you a first time home buyer?", ['Y', 'N'])
mip = st.number_input("Mortgage Insurance Premium (MIP):", min_value=0)
units = st.number_input("Number of units in the property:", min_value=1, step=1)
occupancy = st.selectbox("Occupancy status of the property:", ['O', 'I', 'S'])
octv = st.number_input("Original Combined Loan-to-Value ratio:", min_value=0)
dti = st.number_input("Debt to Income ratio of the borrower: ", min_value=0, step=1)
origupd = st.number_input("Original unpaid principal balance of the loan:", min_value=0)
ltv = st.number_input("Loan To Value ratio at the time of loan origination:", min_value=0)
interest_rate = st.number_input("Orig interest rate:", min_value=0.0)
channel = st.selectbox("Origination Channel:", ['T', 'R', 'C', 'B'])
ppm = st.radio("Is there a penalty for early payment of the principal?", ['Y', 'N'])
state = st.selectbox("State of the property:", ['CA', 'FL', 'MI', 'IL', 'TX', 'OH', 'CO', 'GA', 'NC', 'WA', 'AZ', 'VA',
                                                 'NY', 'PA', 'NJ', 'OR', 'MA', 'IN', 'MD', 'MO', 'MN', 'UT', 'TN', 'SC',
                                                 'WI', 'AL', 'KY', 'NV', 'CT', 'ID', 'KS', 'LA', 'NM', 'IA', 'OK', 'NE',
                                                 'NH', 'AR', 'MS', 'MT', 'DE', 'VT', 'RI', 'ME', 'PR', 'HI', 'WV', 'DC',
                                                 'WY', 'SD', 'ND', 'AK', 'GU'])
type = st.selectbox("Type of the property:", ['SF', 'PU', 'CO', 'MH', 'LH', 'CP'])
purpose = st.selectbox("Purpose of the loan (Purchase/Refinance/Cashout):", ['P', 'R', 'C'])
loanterm = st.number_input("Original term of the loan in months:", min_value=1, step=1)
ServicerName = st.selectbox("Servicer Name:", ['Other servicers     ', 'COUNTRYWIDE         ', 'BANKOFAMERICANA    ',
       'WASHINGTONMUTUALBANK', 'CHASEMANHATTANMTGECO', 'ABNAMROMTGEGROUPINC',
       'WELLSFARGOHOMEMORTGA', 'GMACMTGECORP        ', 'BAMORTGAGELLC      ',
       'CHASEMTGECO         ', 'NATLCITYMTGECO      ', 'WELLSFARGOBANKNA   ',
       'JPMORGANCHASEBANKNA', 'FTMTGESERVICESINC  ', 'SUNTRUSTMORTGAGEINC',
       'CITIMORTGAGEINC    ', 'PRINCIPALRESIDENTIAL', 'CHASEHOMEFINANCELLC ',
       'HOMESIDELENDINGINC ', 'FIFTHTHIRDBANK      '])
MonthsDelinquent = st.number_input("Number of months delinquent:",step=1)
MonthsRepayment = st.number_input("Number of months in repayment:",step=1)
Duration = st.number_input("Duration between first payment date and maturity date:", min_value=0, step=1)

# Submit button
if st.button("Predict"):
    if any([value is None or value == '' for value in [credit_score, fthb, mip, units, occupancy, octv, origupd,
                                                       ltv, interest_rate, channel, ppm, state, type, purpose,
                                                       loanterm, ServicerName, MonthsDelinquent, MonthsRepayment,
                                                       Duration]]):
        st.error("Please fill out all fields.")
    else:
        data = {
            'CreditScore': [credit_score],
            'FirstTimeHomebuyer': [fthb],
            'MIP': [mip],
            'Units': [units],
            'Occupancy': [occupancy],
            'OCLTV': [octv],
            'DTI': [dti],
            'OrigUPB': [origupd],
            'LTV': [ltv],
            'OrigInterestRate': [interest_rate],
            'Channel': [channel],
            'PPM': [ppm],
            'PropertyState': [state],
            'PropertyType': [type],
            'LoanPurpose': [purpose],
            'OrigLoanTerm': [loanterm],
            'ServicerName': [ServicerName],
            'MonthsDelinquent': [MonthsDelinquent],
            'MonthsInRepayment': [MonthsRepayment],
            'Duration': [Duration]
        }
        data = pd.DataFrame(data)
        df_display = data.reset_index(drop=True)
        st.table(df_display.T)
        x = pipeline.transform(data)
        y = model.predict(x)
        if y[0] == 1:
            st.success('Ever delinquent (high risk)')
        else:
            st.success('Not delinquent (low risk)')
