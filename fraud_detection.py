# Load pkgs
import numpy as np
import pandas as pd
import streamlit as st
import boto3
from sklearn.ensemble import  RandomForestClassifier

# full dataset https://www.kaggle.com/ntnu-testimon/paysim1
# Load the dataset
df = pd.read_csv(r"~/PycharmProjects/fraud_detection/dataset.csv")
df.head(3)
df_small = df[0:30]
df_usage = df[0:500000]

# Streamlit
st.write("""
# Fraud Detection App
This app predicts whether a transaction is fraudulent or not !
""")
st.subheader('Intial Data')
if st.checkbox('view data'):
    st.write(dd_small)

st.sidebar.header('User Input Parameters')
st.sidebar.subheader('When chosing the type, note! payment =0, transfer = 1, cash_out = 2, debit =3, cash_in = 4')
st.sidebar.subheader('As well as the first character of nameOrig as 0 is C and nameDest as 1 is M')

name1 = ['01231006815', '01666544295', '01305486145', '0840083671','02048537720', '090045638', '0154988899', '01912850431','01265012928', '0712410124', '01900366749', '0249177573','01648232591', '01716932897', '01026483832', '0905080434','0761750706', '01237762639', '02033524545', '01670993182', '020804602',
 '01566511282', '01959239586', '0504336483','01984094095', '01043358826', '01671590089', '01053967012','01632497828', '0764826684', '02103763750', '0215078753','0840514538', '01768242710', '0247113419', '01238616099','01608633989', '0923341586', '01470868839', '0711197015','01481594086', '01466917878',
 '0768216420', '0260084831', '0598357562', '01440738283', '0484199463', '01570470538','0512549200', '01615801298', '0460570271', '02072313080','0816944408', '0912966811', '01458621573', '046941357','0343345308', '0104716441', '01976401987', '0867288517','01528834618', '0280615803', '0166694583', '0885910946',
 '0811207775', '01161148117', '01131592118', '01262609629','01955990522', '069673470', '0527211736', '01533123860','01718906711', '071802912', '0686349795', '01423768154','01987977423', '0807322507', '0283039401', '0207471778',
'01243171897', '01376151044', '0873175411', '01443967876','01449772539', '0926859124', '01603696865', '012905860','0412788346', '01520267010', '0908084672', '0288306765','01556867940', '01839168128', '01495608502', '0835773569','0843299092', '0605982374', '01412322831', '01305004711']

name2 = ['11979787155', '12044282225', '0553264065', '038997010',
       '11230701703', '1573487274', '1408069119', '1633326333',
       '11176932104', '0195600860', '0997608398', '12096539129',
       '1972865270', '1801569151', '11635378213', '0476402209',
       '11731217984', '11877062907', '1473053293', '01100439041',
       '11344519051', '01973538135', '0515132998', '11404932042',
       '0932583850', '11558079303', '158488213', '1295304806',
       '133419717', '11940055334', '1335107734', '11757317128',
       '11804441305', '11971783162', '1151442075', '170695990',
       '11615617512', '1107994825', '11426725223', '11384454980',
       '11569435561', '01297685781', '01509514333', '1267814113',
       '11593224710', '11849015357', '12008106788', '0824009085',
       '0248609774', '1490391704', '11653361344', '02001112025',
       '1909132503', '11792384402', '11658980982', '11152606315',
       '11714688478', '11506951181', '01937962514', '0242131142',
       '0476800120', '01254526270', '01129670968', '11860591867',
       '01971489295', '1516875052', '1589987187', '1587180314',
       '01330106945', '11082411691', '02096057945', '0766572210',
       '0977993101', '12134271532', '11831010686', '1404222443',
       '161073295', '1396485834', '01761291320', '0783286238',
       '01749186397', '0392292416', '01590550415', '0665576141',
       '01359044626', '01286084959', '01225616405', '11651262695',
       '1494077446']


def user_input_features():
    step = st.sidebar.slider('step', 0, 800)
    #step = st.number_input(label="Input step", min_value=0.0, max_value=800.0)
    type = st.sidebar.slider('type', 0, 4)
    amount = st.sidebar.slider('amount', 0.0, 10000000.0)
    nameOrig = st.sidebar.selectbox('nameOrig', list(name1))
    oldbalanceOrg = st.sidebar.slider('oldbalanceOrg', 0.0, 40000000.0)
    newbalanceOrig = st.sidebar.slider('newbalanceOrig', 0.0, 40000000.0)
    nameDest = st.sidebar.selectbox('nameDest', list(name2))
    oldbalanceDest = st.sidebar.slider('oldbalanceDest',0.0, 43000000.0)
    newbalanceDest = st.sidebar.slider('newbalanceDest',0.0, 43000000.0)
    data = {'step': step,
            'type': type,
            'amount': amount,
            'nameOrig': nameOrig,
            'oldbalanceOrg': oldbalanceOrg,
            'newbalanceOrig': newbalanceOrig,
            'nameDest': nameDest,
            'oldbalanceDest': oldbalanceDest,
            'newbalanceDest': newbalanceDest}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()
st.subheader('User Input parameters')
st.write(df)

df_updated = dd_usage.replace(to_replace ='C', value = '0', regex = True)
df_updated1 = df_updated.replace(to_replace ='M', value = '1', regex = True)

s = pd.Series(df_updated1['type'])
s.replace(['0ASH_OUT'], 0, inplace= True)
sa = pd.Series(df_updated1['type'])
sa.replace(['CASH_OUT'], 0, inplace= True)
ss = pd.Series(df_updated1['type'])
ss.replace(['PAY1ENT'], 1, inplace= True)
sss = pd.Series(df_updated1['type'])
sss.replace(['0ASH_IN'], 2, inplace= True)
sb = pd.Series(df_updated1['type'])
sb.replace(['CASH_IN'], 0, inplace= True)
ssss = pd.Series(df_updated1['type'])
ssss.replace(['TRANSFER'], 3, inplace= True)
sssss = pd.Series(df_updated1['type'])
sssss.replace(['DEBIT'], 4, inplace= True)
target_names = dd.isFraud.drop_duplicates()
y = df_updated1.isFraud
x = df_updated1.drop(['isFraud', 'isFlaggedFraud'],axis=1)

#Building the Model and predicting the model. 
clf = RandomForestClassifier(class_weight={0:1,1:12},
                             criterion='entropy',
                             max_depth=12,
                             max_features='log2',
                             min_samples_leaf=10,
                             n_estimators=50,
                             n_jobs=-1,
                             random_state=42)
clf.fit(x, y)

prediction = clf.predict(df)
prediction_proba = clf.predict_proba(df)

# st.subheader('Class labels and their corresponding index number')
# st.write(df_updated1.target_names)

st.subheader('Prediction')
#st.write(df_updated1.target_names[prediction])
st.write(prediction)

st.subheader('Prediction Probability')
st.write(prediction_proba)


# def get_model_results(X_train: np.ndarray, y_train: np.ndarray,
#                       X_test: np.ndarray, y_test: np.ndarray, model):
#     """
#     model: sklearn model (e.g. RandomForestClassifier)
#     """
#     # Fit your training model to your training set
#     model.fit(X_train, y_train)
#
#     # Obtain the predicted values and probabilities from the model
#     predicted = model.predict(X_test)
#
#     try:
#         probs = model.predict_proba(X_test)
#         print('ROC Score:')
#         print(roc_auc_score(y_test, probs[:, 1]))
#     except AttributeError:
#         pass
#
#     # Print the ROC curve, classification report and confusion matrix
#     print('\nClassification Report:')
#     print(classification_report(y_test, predicted))
#     print('\nConfusion Matrix:')
#     print(confusion_matrix(y_test, predicted))
# # Input the optimal parameters in the model
# model = RandomForestClassifier(class_weight={0:1,1:12},
#                                criterion='entropy',
#                                max_depth=12,
#                                max_features='log2',
#                                min_samples_leaf=10,
#                                n_estimators=100,
#                                n_jobs=-1,
#                                random_state=42)
#
# # Get results from your model
# get_model_results(X_train, y_train, X_test, y_test, model)
#
# predicted = model.predict(X_test)
# prediction_proba = model.predict_proba(X_test)

