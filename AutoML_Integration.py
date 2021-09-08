import os
import re
import h2o
import base64
import pickle
import pathlib
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime
from tpot import TPOTRegressor
from tpot import TPOTClassifier
from collections import Counter
from h2o.automl import H2OAutoML
from sklearn import preprocessing



def show_info_dataset(dataset,total,format_file,real,header):
    
    st.subheader("Info dataset uploaded")
    st.write("Format file: ",format_file)
    st.write("Header: ",header)
    st.write("Total number rows file: ",total)
    st.write("Number rows dataset uploaded: ",real)
    st.write("Number features contained:",len(dataset.columns.tolist()))
    
def show_info_dataset_inf(dataset,total,format_file,header):
    
    st.subheader("Info dataset uploaded")
    st.write("Format file: ",format_file)
    st.write("Header: ",header)
    st.write("Total number rows file: ",total)
    st.write("Number features contained:",len(dataset.columns.tolist()))
    
def show_info_dataset_fair(dataset,total,format_file,real):
    
    st.subheader("Info dataset uploaded")
    st.write("Format file: ",format_file)
    st.write("Total number rows file: ",total)
    st.write("Number rows dataset uploaded: ",real)
    st.write("Number features contained:",len(dataset.columns.tolist()))

#function which checks if type target variable is valid or not depending the task to accomplish
def check_validity_target(dataset,option,task):
    
    validity=True
    values = dataset[option].tolist()
    
    if task=="classification":
        check = [x for x in values if isinstance(x,(float,np.float64))]
    else:
        check = [x for x in values if re.search("[A-Za-z/]+",str(x))]
    
    if len(check)!=0:
            validity=False
            
    if validity==True:
        return None
    else:
        return type(values[0])
    
#function which detects if dataset is unbalanced or not
def check_imbalance_target(y,number):
    print("Number class: "+str(number))
    
    label_count = Counter(y)
    count_label = pd.DataFrame.from_dict(label_count, orient='index').reset_index()
    count_label.columns=['target','count']
    count_label['ratio']=count_label['count']/len(y)
    count_label['number_balance']=len(y)/number
    count_label['ratio_balance']=count_label['number_balance']/len(y)
    
    print(count_label)
    
    if count_label.iloc[0]['ratio']==count_label.iloc[0]['ratio_balance']:
        check_distribution='balance'
    else:
        check_distribution='unbalance'      
    
    return check_distribution, count_label[count_label['count']<5] #threshold for multi-class task  


#function which converts categorical features into integers using labelEncoder
def preproc_categorical_variables(x):
    
    encoder = preprocessing.LabelEncoder()
    features_list = x.columns.tolist()
    
    for column in features_list:
        x_check = x.iloc[0][column]
    
        if isinstance(x_check, (int, float,np.integer,np.float64))==False:
            
            if re.search("[A-Za-z]+[0-9]*",x_check):
                x[column] = encoder.fit_transform(x[column]) #encode categorical features
            else:
                
                if re.search("[-/]+",x_check): #check date
                    x[column]=x[column].apply(lambda x: re.sub("[-/]","",x))
                else:
                    x[column] = x[column].astype(float)
                    
                x[column]=x[column].astype(int)
    return x


#function which encodes type target variable if it is integer
def special_case_int_label(y):
    
    encoder = preprocessing.LabelEncoder()
    
    if isinstance(y[0],(int,np.integer))==True:
        encoder.fit(y)
        y_encode = encoder.transform(y)
        y_encode = pd.Series(y_encode,index=y.index)
        return y_encode,encoder
    else:
        return y,None


#function which performs automl using h2o library
def h2o_automl(train,target,balance_unbalance,path_dir,task):
    
    h2o.init()
    
    train_h2o = h2o.H2OFrame(train) #convert df format h2o (required by the library)
    
    if task=="classification":
        if balance_unbalance=='balance':
            print("Perform Class, balance sample")
            aml = H2OAutoML(max_runtime_secs=30,project_name="autoMl_demo_h2o",
                    exclude_algos=["XGBoost","StackedEnsemble"],seed=2021,sort_metric="AUTO",
                    balance_classes=False)
        else:
            print("Perform Class, unbalance sample")
            aml = H2OAutoML(max_runtime_secs=30,project_name="autoMl_demo_h2o",
                    exclude_algos=["XGBoost","StackedEnsemble"],seed=2021,sort_metric="AUTO",
                    balance_classes=True)
    else:
        print("Perform Regress")
        aml = H2OAutoML(max_runtime_secs=30,project_name="autoMl_demo_h2o",
                    exclude_algos=["XGBoost","StackedEnsemble"],seed=2021,sort_metric="AUTO")
            
    h2o.remove(aml)
    
    st.info("Start autoML process with h2o..wait please")
    aml.train(y=target,training_frame = train_h2o)
    
    leaderboard_h2o = aml.leaderboard #return info performance about all models trained
    leaderboard_h2o_df = leaderboard_h2o.as_data_frame()
    
    
    st.info("Finish autoML process")
    st.write("Models builded")
    st.write(leaderboard_h2o_df)
    
    best_name_model = leaderboard_h2o_df.iloc[0]["model_id"] #get name best model
    best_model = h2o.get_model(best_name_model) #get model type ModelBase (best model)
    
    st.write("The best model is ",best_name_model)
    
    path=path_dir+'/h2o'

    model_path = h2o.save_model(best_model,path=path, force=True) #save model format h2o
    saved_model =h2o.load_model(model_path)
    pickle.dump(saved_model, open('trial_h2o.pkl','wb'))
    
    return best_model, model_path


#function which performs autoML using tpot library
def tpot_automl(x_train,y_train,type_task,scoring,path_dir,task):
    if task=="classification":
        print("Perform Class, scoring "+scoring)
        aml = TPOTClassifier(generations=5,population_size=10,
                             cv=5,verbosity=2, max_time_mins=2,
                             scoring=scoring,random_state=42)
    else:
        print("Perform regress, scoring "+scoring)
        aml = TPOTRegressor(generations=5,population_size=10,
                             cv=5,verbosity=2, max_time_mins=2,
                             scoring=scoring,random_state=42)
     
    st.info("Start autoML process with tpot..wait please")
    aml.fit(x_train,y_train)
    
    st.info("Finish autoML process")
    info = aml.fitted_pipeline_.steps #return info best pipeline (best model and possibly preprocessing executed)

    st.write("The best model is ",info)
    
    path_check = os.path.join(path_dir,'tpot')
    if not os.path.isdir(path_check):
        print("Directory "+path_check+" created")
        pathlib.Path(path_check).mkdir(parents=True, exist_ok=True) #create directory use_case/<name_file>/tpot
    
    ct = datetime.now()
    if task=="classification":
        model_path = path_check + '/best_classification_pipeline_' + str(ct.year ) + '{:02d}'.format(ct.month) + '{:02d}'.format(ct.day) + '_' + '{:02d}'.format(ct.hour) + '{:02d}'.format(ct.minute) + '{:02d}'.format(ct.second) + '.pkl'
        pickle.dump(aml.fitted_pipeline_, open(model_path, 'wb'))
    else:
        model_path = path_check + '/best_regression_pipeline_' + str(ct.year )+ str(ct.month) + str(ct.day) + '_' + str(ct.hour) + str(ct.minute) + str(ct.second) + '.pkl'
        pickle.dump(aml.fitted_pipeline_, open(model_path, 'wb'))

    return aml.fitted_pipeline_, model_path
    

#function which creates links about csv prediction and best model
def filedownload(df_pred,model_trained,library_automl,name_file):
    csv = df_pred.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:file/csv;base64,{b64}" download="pred_test_'+library_automl+'_'+name_file+'.csv">Download CSV File predicitions</a>'
    
    output_model = pickle.dumps(model_trained)
    b64 = base64.b64encode(output_model).decode()
    if library_automl=='h2o':
        href_train= f'<a href="data:file/output_model;base64,{b64}" download="best_model_h2o.pkl">Download best model (pickle format)</a>'
    else:
        now = datetime.now()
        dt_string = now.strftime("%Y%m%d_%H%M%S")
        href_train= f'<a href="data:file/output_model;base64,{b64}" download="best_model_tpot_'+dt_string+'.pkl">Download best model</a>'

    return href+"&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp"+href_train

def get_latest_model(list_file):
    
    file_model=None
    
    if len(list_file)!=1:
        
        list_df = pd.DataFrame(list_file)

        list_df['date'] = 0
        list_df['time'] = 0

        list_df['split']=list_df[0].apply(lambda x: x.split('_'))

        for i in range(len(list_df)):
            if len(list_df['split'].iloc[i]) == 5:

                list_df['date'].iloc[i]=list_df['split'].iloc[i][-2]
                list_df['time'].iloc[i]=list_df['split'].iloc[i][-1].split('.')[0]

            elif len(list_df['split'].iloc[i]) == 9:

                list_df['date'].iloc[i]=list_df['split'].iloc[i][-4]
                list_df['time'].iloc[i]=list_df['split'].iloc[i][-3]

        list_df.sort_values(['date','time'],ascending=False,inplace=True)
        
        print("Content directory")
        print(list_df)
        
        file_model=list_df.iloc[0][0]
        
    else:
        file_model=list_file[0]
    return file_model


#function which creates links about csv prediction and best model
def file_download(df_pred,name_file,library):
    csv = df_pred.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:file/csv;base64,{b64}" download="pred_inference_'+library+'_'+name_file+'.csv">Download CSV File predicitions</a>'
    
    return href