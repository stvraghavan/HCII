from io import StringIO
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

dataframe = pd.DataFrame()

uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    dataframe = pd.read_csv(uploaded_file)

def home():

    tab1,tab2,tab3,tab4,tab5,tab6 = st.tabs(['First Look','Descriptive Statistics','Correlation Chart','Histogram','Missing Value info','Box Plot'])

    with tab1:
        try:
            st.write("A snap of the dataframe")
            st.write(dataframe)
        except:
            pass
    with tab2:
        try:
            st.write(dataframe.describe())
        except:
            pass
    with tab3:
        try:
            st.write("Correlation Chart")
            corr = dataframe.corr()
            fig = px.imshow(corr)
            fig.update_layout(dict1={
                'width':700,
                'height':700
            })
            st.plotly_chart(fig)
        except:
            pass
        
    with tab4:
        try:
            temp = st.selectbox("Choose a variable",options=dataframe.columns,index=0)
            fig = px.histogram(data_frame=dataframe,x=temp)
            st.plotly_chart(fig)
        except:
            pass
    with tab5:
        try:
            col1,col2 = st.columns(2)
            with col1:
                st.write(dataframe.isnull().sum().rename("Null Values"))
            with col2:
                st.write(dataframe.isna().sum().rename("N/A Values"))
        except:
            pass
    with tab6:
        try:
            temp1 = st.selectbox("Choose a variable",options=dataframe.columns,index=1)
            fig = px.box(data_frame=dataframe,y=temp1)
            st.plotly_chart(fig)
        except:
            pass
def ml():
    def regg():
        from lazypredict.Supervised import LazyRegressor
        from sklearn.utils import shuffle
        import numpy as np
        x , y = shuffle(dataframe[x_list],dataframe[y_tar],random_state=69)
        x = x.astype(np.float32)
        offset = int(x.shape[0]*0.9)
        X_train, y_train = x[:offset], y[:offset]
        X_test, y_test = x[offset:], y[offset:]
        reg = LazyRegressor(verbose=0, ignore_warnings=False, custom_metric=None)
        models, predictions = reg.fit(X_train, X_test, y_train, y_test)
        st.write(models)
    def cls():
        from lazypredict.Supervised import LazyClassifier 
        from sklearn.model_selection import train_test_split 

        X = dataframe[x_list]
        y = dataframe[y_tar] 

        X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=.5,random_state =123)

        clf = LazyClassifier(verbose=0,ignore_warnings=True, custom_metric=None)
        models,predictions = clf.fit(X_train, X_test, y_train, y_test)

        st.write(models)
        print(models)
    st.write("Second Page")
    x_list = st.multiselect("Choose the independent varaiables",options=dataframe.columns)
    target = []
    for i in dataframe.columns:
        if(i not in x_list):
            target.append(i)

    y_tar = st.selectbox("Choose your target variable",options=target)

    type_model = st.radio("Select your model",options=['Regression','Classification'])
    if(type_model == 'Regression'):
        st.write("Regression")
        regg()
    else:
        st.write("Classification")
        cls()
page_names_to_funcs = {
    "Home": home,
    "ML Models": ml
}

demo_name = st.sidebar.selectbox("Choose a demo", page_names_to_funcs.keys())
page_names_to_funcs[demo_name]()