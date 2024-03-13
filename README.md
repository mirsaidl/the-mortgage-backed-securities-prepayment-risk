# DELINQUENCY
In this project, the goal is to predict the delinquency status of mortgage-backed securities. Delinquency refers to when a borrower fails to make payments on a debt, such as a mortgage. For mortgage-backed securities, which are investment products backed by a pool of mortgages, predicting delinquency is crucial for investors and financial institutions.

The "Delinquent" column in the dataset serves as the target variable for prediction. It is a binary variable where:
- **1**: Indicates that the loan is currently delinquent, meaning the borrower has failed to make payments on time.
- **0**: Indicates that the loan is not delinquent, meaning the borrower is up to date with their payments.

The aim of the machine learning models in this project is to accurately predict whether a mortgage-backed security is at risk of becoming delinquent. This prediction can help investors and financial institutions make informed decisions about managing their investment portfolios. By identifying loans that are likely to become delinquent, they can take proactive measures to mitigate risks and minimize potential losses.

In summary, the project aims to use historical data on various features related to mortgage loans to build predictive models that can forecast whether a mortgage-backed security is likely to experience delinquency in the future. This prediction can provide valuable insights for risk management and investment decision-making in the mortgage industry. 

The dataset used in this project contains 2,914,952 data points and consists of 22 columns. Below is a description of the columns in the dataset:

1. **CreditScore**: The credit score of the borrower.
2. **FirstTimeHomebuyer**: Whether the borrower is a first-time homebuyer (Y/N).
3. **MIP**: Mortgage Insurance Premium.
4. **Units**: Number of units in the property.
5. **Occupancy**: Occupancy status of the property (O - Owner-occupied, I - Investment, S - Second Home).
6. **OCLTV**: Original Combined Loan-to-Value ratio.
7. **DTI**: Debt-to-Income ratio of the borrower.
8. **OrigUPB**: Original unpaid principal balance of the loan.
9. **LTV**: Loan-to-Value ratio at the time of loan origination.
10. **OrigInterestRate**: Original interest rate.
11. **Channel**: Origination Channel (T - Retail, R - Broker, C - Correspondent, B - Direct).
12. **PPM**: Penalty for early payment of the principal (Y/N).
13. **PropertyState**: State of the property.
14. **PropertyType**: Type of the property (SF - Single Family, PU - PUD, CO - Condo, MH - Manufactured Housing, LH - Leasehold, CP - Co-op).
15. **LoanPurpose**: Purpose of the loan (P - Purchase, R - Refinance, C - Cashout).
16. **OrigLoanTerm**: Original term of the loan in months.
17. **ServicerName**: Name of the servicer.
18. **MonthsDelinquent**: Number of months delinquent.
19. **MonthsInRepayment**: Number of months in repayment.
20. **Duration**: Duration between the first payment date and maturity date of the loan.
21. **EverDelinquent**: Whether the loan has ever been delinquent (1 - Yes, 0 - No).
22. **Delinquent**: The target variable, indicating if the loan is currently delinquent (1 - Yes, 0 - No).

This dataset is used to predict the prepayment risk of mortgage-backed securities using machine learning models. The target variable "Delinquent" is the label we want to predict based on the other features in the dataset. The aim is to build a model that can accurately predict whether a loan will become delinquent, which is crucial for risk assessment in the mortgage industry.
