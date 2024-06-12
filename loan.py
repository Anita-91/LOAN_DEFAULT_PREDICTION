
import streamlit as st
from PIL import Image
import pandas as pd
import base64
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_curve,roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from imblearn.over_sampling import SMOTE #conda install -c conda-forge imbalanced-learn
from xgboost import XGBClassifier #conda install -c conda-forge xgboost

st.set_page_config(page_title="LOAN PREDICTION",
                   page_icon=":shark:",
                   layout="wide",
                   initial_sidebar_state="expanded",
                   )
st.markdown("<h1 style='text-align: center; color: white;background-color:green;border-radius:15px;margin: -1px;'>LOAN DEFAULT PREDICTION</h1>",
            unsafe_allow_html=True)

button_styles = """
    <style>
        .stButton>button {
              background-color: #33BCFF; /* Green */
              border: none;
              color: black;
              padding: 10px 30px;
              text-align: center;
              text-decoration: none;
              display: inline-block;
              font-size: 16px bold;
              hover-color:#390ADD;
              font-weight: bold;
            
        }
        
    </style>
"""
st.markdown(button_styles, unsafe_allow_html=True)


tab1, tab2, tab3, tab4 = st.tabs(["HOME", "PREDICTION", "EVALUATION", "ABOUT"])


#===== HOME
with tab1:
    col1, col2 = st.columns(2)
    col3, col4 = st.columns(2)
    with col1:
        col1.image(Image.open("C:/Users/vikra/Learningpythons/GUVI_cours/Capstone_Proj/loan_project/logo.jpg"), width=600)
    with col2:
        st.write("#### This project is to develop a predictive model that can assess the risk of loan default for loan applicants.")
        st.write("#### The objective is to help financial institutions make informed decisions about whether to approve or reject loan applications.")
        st.write("#### Data Preprocessing, Model Building, Analysis on Data, Model Evaluation, Metrics Report.")
        st.write("#### Model metrics help assess how well the model predicts loan defaults and non-defaults.")
    with col3:
        st.write("<h4 style='color:red;font-size:25px'>STEPS TO PROCEED:</h4>", unsafe_allow_html=True)
        st.markdown("##### 1.PREDICTION ->  To Predict the Loan Status for new feature/input.")
        st.markdown("##### 2.ANALYSIS/METRICS ->  Data Insights and Visualization. Evaluating Models using Metrics.")
               
    with col4:
        col4.image(Image.open("C:/Users/vikra/Learningpythons/GUVI_cours/Capstone_Proj/loan_project/l.jpg"), width=600)
   
    
#PREDICTION   
with tab2:
    df_s = pd.read_csv('C:/Users/vikra/Learningpythons/GUVI_cours/Capstone_Proj/loan_project/loan_data.csv')

    # 1. DATA PREPROCESSING
    df_1 = df_s[df_s.Income < df_s.Income.mean() + 3*df_s.Income.std()]
    clean_df = df_1.copy()
    #st.table(clean_df.head(5))
    #---
    clean_df['Loan_Status'] = clean_df['Loan_Status'].map({'Non-Default': 0, 'Default': 1})
    clean_df['Employment_Status'].fillna(clean_df['Employment_Status'].mode()[0], inplace=True)
    clean_df['Employment_Status'] = clean_df['Employment_Status'].map({'Unemployed': 0, 'Employed': 1}).astype(str).str.split('.').str[0].astype(int)
    clean_df['Location'] = clean_df['Location'].map({'Rural': 0, 'Urban': 1,'Suburban': 2})
    clean_df.drop('Gender', axis=1, inplace=True) #[990 rows x 11 columns]  #oversampling , 
    #---
    X = clean_df.drop('Loan_Status', axis=1)
    y = clean_df['Loan_Status']
    
    # 2. TRAINING:
        
    smote = SMOTE(sampling_strategy='minority')
    X_sm, y_sm = smote.fit_resample(X, y)

    
    X_train, X_test, y_train, y_test=train_test_split(X_sm, y_sm, test_size=0.2, random_state=15, stratify=y_sm)
    xgb = XGBClassifier(n_estimators=500, learning_rate=0.049999, max_depth=20, subsample=0.7)
    xgb.fit(X_train, y_train)
    
    
    # 3. PREDICT:
    col1,col2 = st.columns([1,1],gap="large")

    with col1:
        Age = st.text_input(label="Age")
        Income = st.text_input(label="Income",help='Enter Amount eg:71266.10518')
        Credit_Score = st.text_input(label="Credit_Score",help='Enter Score eg:750')
        Debt_to_Income_Ratio = st.text_input(label="Debt_to_Income_Ratio")
        Existing_Loan_Balance = st.text_input(label="Existing_Loan_Balance")
    with col2:
        Loan_Amount = st.text_input(label="Loan_Amount")
        Interest_Rate = st.text_input(label="Interest_Rate",help='Enter Amount eg:18.89158')
        Loan_Duration_Months = st.text_input(label="Loan_Duration_Months")
        Employment_Status = st.text_input(label="Employment_Status",help='Enter 0 (Employed) or 1 (Unemployed)')
        Location = st.text_input(label="Location",help='Enter 0 (Rural) or 1 (Urban) or 2 (Suburban)')
   
    if st.button("PREDICT"):
        pred = xgb.predict_proba([[int(Age), float(Income), float(Credit_Score), float(Debt_to_Income_Ratio), float(Existing_Loan_Balance), float(Loan_Amount),
                                    float(Interest_Rate), int(Loan_Duration_Months), int(Employment_Status), int(Location)]])
        s = pd.DataFrame(
            pred, columns=["NON_DEFAULT_PROBABILITY", "DEFAULT_PROBABILITY"])
        
    
# -- Evaluate
with tab3:
    col3, col4 = st.columns(2)
    analy = col3.button("ANALYSIS")
    met = col4.button("METRICS")
    
    
    if analy:
        col1, col2 = st.columns(2)
        df_s = pd.read_csv('C:/Users/vikra/Learningpythons/GUVI_cours/Capstone_Proj/loan_project/loan_data.csv')
        df_1 = df_s[df_s.Income < df_s.Income.mean() + 3*df_s.Income.std()]
        clean_df = df_1.copy()
        clean_df['Loan_Status'] = clean_df['Loan_Status'].map({'Non-Default': 0, 'Default': 1})
        ## -- clean_df.dropna(axis=0,inplace=True)
        clean_df['Employment_Status'].fillna(clean_df['Employment_Status'].mode()[0], inplace=True)
        clean_df['Employment_Status'] = clean_df['Employment_Status'].map({'Unemployed': 0, 'Employed': 1}).astype(str).str.split('.').str[0].astype(int)
        clean_df['Location'] = clean_df['Location'].map({'Rural': 0, 'Urban': 1,'Suburban': 2})
        clean_df.drop('Gender', axis=1, inplace=True) #[990 rows x 11 columns]
        
        #st.markdown("<h1 style='text-align: right;font-size:20px'>" + download_csv(clean_df) + "</h1>", unsafe_allow_html=True)

        st.table(clean_df.head(10))
        with col1:
            grouped = clean_df.groupby('Location')['Loan_Amount'].sum().reset_index()

            fig = px.bar(grouped, x='Location', y='Loan_Amount',color='Location',
                         labels={'Location': 'Location', 'Loan_Amount': 'Total Loan Amount'},
                         title='TOTAL LOAN AMOUNT Vs LOCATION')
            fig.update_layout(
                title={
                    'text': 'TOTAL LOAN AMOUNT Vs LOCATION',
                    'y': 0.95,
                    'x': 0.5,
                    'xanchor': 'center',
                    'yanchor': 'top',
                    'font': {'color': 'blue', 'size': 20}
                }
                 
            )
            st.plotly_chart(fig)
        with col2:
            grouped = clean_df.groupby('Employment_Status')['Loan_Amount'].sum().reset_index()

            fig = px.bar(grouped, x='Employment_Status', y='Loan_Amount',color='Employment_Status',
                         labels={'Employment_Status': 'Employment_Status', 'Loan_Amount': 'Total Loan Amount'},
                         title='LOAN AMOUNT Vs EMPLOYMENT_STATUS')
            fig.update_layout(
                title={
                    'text': 'LOAN AMOUNT Vs EMPLOYMENT_STATUS',
                    'y': 0.95,
                    'x': 0.5,
                    'xanchor': 'center',
                    'yanchor': 'top',
                    'font': {'color': 'blue', 'size': 20}
                }
                 
            )
            st.plotly_chart(fig)
        ## 3.
        grouped = clean_df.groupby('Location')['Income'].sum().reset_index()
        fig = go.Figure(data=[go.Pie(labels=grouped['Location'], values=grouped['Income'])])
        fig.update_layout(title='Total Income by Location')
        st.plotly_chart(fig) 
        
    #-- Model Metrics
    if met:
        X = clean_df.drop('Loan_Status', axis=1)
        #st.table(X.head(5))
        Y = clean_df['Loan_Status']
        
        smote = SMOTE(sampling_strategy='minority') #sampling_strategy='not minority',random_state=10
        X_sm, y_sm = smote.fit_resample(X, Y)
        G_train, G_test, g_train, g_test = train_test_split(X_sm, y_sm, test_size=0.2, random_state=15, stratify=y_sm)
        
        
        scaler=StandardScaler()
        X_sm_scaled=scaler.fit_transform(X_sm)
        K_train,K_test,k_train,k_test=train_test_split(X_sm_scaled, y_sm, test_size=0.2, random_state=15, stratify=y_sm)
        
        #--
        X_train,X_test,y_train,y_test=train_test_split(X, Y, test_size=0.2, random_state=15)
        def evaluate_model(model, X_train, X_test, y_train, y_test):
            model.fit(X_train, y_train)
            
            # Predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)*100
            precision = precision_score(y_test, y_pred)*100
            recall = recall_score(y_test, y_pred)*100
            #rep=classification_report(y_test,pred)
            
            y_proba = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            roc_auc = roc_auc_score(y_test, y_proba)
            return accuracy, precision, recall, fpr, tpr , roc_auc


        
        XG_model = XGBClassifier(n_estimators=500, learning_rate=0.049999, max_depth=20, subsample=0.7)
        LR_model = LogisticRegression()
        KNN_model = KNeighborsClassifier(n_neighbors=11)
        DT_model = DecisionTreeClassifier(max_depth=100, min_samples_leaf=5, min_samples_split=2, criterion='entropy')
        RF_model = RandomForestClassifier(n_estimators=500, criterion='entropy', max_depth=50, min_samples_leaf=3, min_samples_split=5)
        
        
        XG_acc, XG_prec, XG_recall, XG_fpr, XG_tpr, XG_roc_auc = evaluate_model(XG_model, G_train, G_test, g_train, g_test)
        KNN_acc, KNN_prec, KNN_recall, KNN_fpr, KNN_tpr, KNN_roc_auc = evaluate_model(KNN_model, K_train,K_test,k_train,k_test)
        LR_acc, LR_prec, LR_recall, LR_fpr, LR_tpr, LR_roc_auc = evaluate_model(LR_model, X_train, X_test, y_train, y_test)
        DT_acc, DT_prec, DT_recall, DT_fpr, DT_tpr, DT_roc_auc = evaluate_model(DT_model, X_train, X_test, y_train, y_test)
        RF_acc, RF_prec, RF_recall, RF_fpr, RF_tpr, RF_roc_auc = evaluate_model(RF_model, X_train, X_test, y_train, y_test)


        models = ['XG', 'KNN', 'LR', 'DT', 'RF']
        accuracy_values = [XG_acc, KNN_acc, LR_acc, DT_acc, RF_acc]
        colors = ['green', 'blue', 'yellow', 'red', 'orange'] 
        
        
        traces = []
        for model, accuracy, color in zip(models, accuracy_values, colors):
            trace_accuracy = go.Bar(x=[model], y=[accuracy], marker_color=color, name=model,width=0.30)
            traces.append(trace_accuracy)
        
        layout = go.Layout(title='Accuracy Vs Models',
                           xaxis=dict(title='Model'),
                           yaxis=dict(title='Accuracy'))
        
        
        fig = go.Figure(data=traces, layout=layout)
        st.plotly_chart(fig)
        
        #---
        st.markdown("##")
        col1, col2 = st.columns(2)
        with col1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=XG_fpr, y=XG_tpr, mode='lines', name='ROC Curve (XG Boost )', line=dict(color='red')))
            fig.add_shape(type='line', x0=0, y0=0, x1=1, y1=1, line=dict(color='blue', dash='dash'))
            fig.update_layout(title='ROC Curve (XG Boost)',
                              xaxis_title='False Positive Rate (FPR)',
                              yaxis_title='True Positive Rate (TPR)',
                              xaxis=dict(scaleanchor="y", scaleratio=1),
                              yaxis=dict(scaleanchor="x", scaleratio=1),
                              width=700, height=500)
            fig.add_annotation(x=0.5, y=0.25,
                               text=f'AUC Score: {XG_roc_auc:.2f}',
                               showarrow=False,
                               font=dict(color='blue', size=14))
            st.plotly_chart(fig)
            
            fig = go.Figure() 
            fig.add_trace(go.Scatter(x=LR_fpr, y=LR_tpr, mode='lines', name='ROC Curve (Linear Regression)', line=dict(color='red')))
            fig.add_shape(type='line', x0=0, y0=0, x1=1, y1=1, line=dict(color='blue', dash='dash'))
            fig.update_layout(title='ROC Curve (Linear Regression)',
                              xaxis_title='False Positive Rate (FPR)',
                              yaxis_title='True Positive Rate (TPR)',
                              xaxis=dict(scaleanchor="y", scaleratio=1),
                              yaxis=dict(scaleanchor="x", scaleratio=1),
                              width=700, height=500)
            fig.add_annotation(x=0.5, y=0.25,
                               text=f'AUC Score: {LR_roc_auc:.2f}',
                               showarrow=False,
                               font=dict(color='blue', size=14))
            st.plotly_chart(fig)
            
            
        with col2:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=KNN_fpr, y=KNN_tpr, mode='lines', name='ROC Curve (KNN)', line=dict(color='Orange')))
            fig.add_shape(type='line', x0=0, y0=0, x1=1, y1=1, line=dict(color='black', dash='dash'))
            fig.update_layout(title='ROC Curve (KNN)',
                              xaxis_title='False Positive Rate (FPR)',
                              yaxis_title='True Positive Rate (TPR)',
                              xaxis=dict(scaleanchor="y", scaleratio=1),
                              yaxis=dict(scaleanchor="x", scaleratio=1),
                              width=700, height=500)
            fig.add_annotation(x=0.5, y=0.25,
                               text=f'AUC Score: {KNN_roc_auc:.2f}',
                               showarrow=False,
                               font=dict(color='black', size=14))
            st.plotly_chart(fig)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=RF_fpr, y=RF_tpr, mode='lines', name='ROC Curve (Random Forest)', line=dict(color='Orange')))
            fig.add_shape(type='line', x0=0, y0=0, x1=1, y1=1, line=dict(color='black', dash='dash'))
            fig.update_layout(title='ROC Curve (Random Forest)',
                              xaxis_title='False Positive Rate (FPR)',
                              yaxis_title='True Positive Rate (TPR)',
                              xaxis=dict(scaleanchor="y", scaleratio=1),
                              yaxis=dict(scaleanchor="x", scaleratio=1),
                              width=700, height=500)
            fig.add_annotation(x=0.5, y=0.25,
                               text=f'AUC Score: {RF_roc_auc:.2f}',
                               showarrow=False,
                               font=dict(color='black', size=14))
            st.plotly_chart(fig)
            
            
            
# ABOUT
with tab4:        
        col1,col2 = st.columns([1,1],gap="small")
        with col1:
            col1.image(Image.open("C:/Users/vikra/Learningpythons/GUVI_cours/Capstone_Proj/loan_project/about_logo.jpg"), width=600)

        with col2:
            st.markdown("##### • The primary goal of this project is to develop a predictive model that can assess the risk of loan default for loan applicants.")
            st.markdown("##### • Model evaluation done by metrics such as accuracy, precision, recall, ROC curve, and AUC. These metrics help assess how well the model predicts loan defaults and non-defaults.")
            st.markdown("##### • The goal is to predict the loan status based on feature input and also gives insights and visualization of data.")
            st.write("<span style='color: darkgreen; font-size: 24px;font-weight: bold'>[Done by] : ANITHA VIKRAM</span>", unsafe_allow_html=True)
            st.markdown("[Github -> ProjectLink](https://github.com/Anita-91/LOAN_DEFAULT_PREDICTION.git)")
            #st.markdown("[LinkedIn](link here)")
            
          
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

