import os
import h2o
import pickle
import joblib
import pathlib
import datetime
import webbrowser
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
from db_manager import *
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from streamlit.hashing import _CodeHasher
import streamlit.components.v1 as components
from topbar_integration import display_topbar
from sklearn.preprocessing import LabelEncoder
#from AutoEDA_Integration import generate_eda_report
from sklearn.model_selection import train_test_split
#from shapash.explainer.smart_explainer import SmartExplainer
#from eXplainableAI_Integration import model_wrapper, metrics_wrapper
from sklearn.metrics import accuracy_score, balanced_accuracy_score, r2_score, \
                            precision_score, recall_score
from AutoML_Integration import show_info_dataset, check_validity_target, \
                            check_imbalance_target, preproc_categorical_variables,\
                            special_case_int_label, h2o_automl, tpot_automl,\
                            filedownload, get_latest_model, file_download, \
                            show_info_dataset_inf,show_info_dataset_fair

from fairlearn.metrics import false_positive_rate, true_positive_rate, selection_rate, MetricFrame, count
from fairlearn.reductions import ExponentiatedGradient, DemographicParity, GridSearch, DemographicParity, ErrorRate, \
                                 DemographicParity, EqualizedOdds, UtilityParity, TruePositiveRateParity, FalsePositiveRateParity, \
                                 ErrorRateParity
from fairness_plots import predictions_in_accuracy_score, predictions_in_selection_rate, plot_metric, \
                           subplots_in_accuracy_score, subplots_in_selection_rate, subplots_in_metric, \
                           model_comparison_plot

st.set_page_config(page_title='Deloitte - Web app',
                   page_icon=Image.open('deloitte_image.png'),
                   layout='centered',
                   initial_sidebar_state='expanded')
np.random.seed(0)  # set seed for consistent results with ExponentiatedGradient

try:
    # Before Streamlit 0.65
    from streamlit.ReportThread import get_report_ctx
    from streamlit.server.Server import Server
except ModuleNotFoundError:
    # After Streamlit 0.65
    from streamlit.report_thread import get_report_ctx
    from streamlit.server.server import Server

# Hide the MainMenu in the side bar
st.markdown(""" <style> #MainMenu {visibility: hidden;} footer {visibility: hidden;}</style> """, unsafe_allow_html=True)

def main():
    state = _get_state()

    if state.current_user:
        user_name = state.current_user[0]
    else:
        user_name = None
    display_topbar(user_name)

    pages = {
        "User": page_user,
        "Use Case": page_usecase,
        "AutoEDA": page_eda,
        "AutoML": page_automl,
        "Inference": page_inference,
        "Fairness": page_fairness,
        "eXplainable AI": page_explainable_ai,
    }

    st.sidebar.image("deloitte_image.png", width=180)
    st.sidebar.title("Menu")
    page = st.sidebar.radio("Select page:", tuple(pages.keys()))

    # Display the selected page with the session state
    pages[page](state)

    # Mandatory to avoid rollbacks with widgets, must be called at the end of your app
    state.sync()

def page_user(state):
    st.title("Login/Sign up")
    menu = ["Login", "Sign up"]
    choice = st.selectbox("Options:", menu)

    if choice == 'Sign up':
        st.subheader("Create New Account")

    with st.form('Select an option'):
        username = st.text_input("User Name")
        password = st.text_input("Password", type='password')
        hashed_pswd = make_hashes(password)

        if choice == "Sign up":
            roles = ['Admin', 'Guest']
            role = st.selectbox("Role", roles)
            create_account_button = st.form_submit_button("Create account")

            if create_account_button:
                current_datetime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                new_user = add_userdata(username, make_hashes(password), role, current_datetime)
                if new_user is False:
                    st.warning("This Username already exists!")
                else:
                    st.success("You have successfully created a valid Account")
                    st.info("Select login now")
                    st.balloons()

        elif choice == 'Login':
            login_button = st.form_submit_button("Login")

            if login_button:
                if username and password:
                    state.db_login_result = login_user(username, check_hashes(password, hashed_pswd))
                    if state.db_login_result:
                        st.success("Successfully logged in as {}".format(username))
                        state.current_user = [username, password]
                        st.stop()
                    else:
                        st.warning("Incorrect Username/Password")
                        st.stop()


def page_usecase(state):
    display_usersdatabase = False
    st.title("Use Cases")

    if state.current_user:
        st.write("**{}**, please create a new use case or select an existing one!".format(state.current_user[0]))

        if display_usersdatabase:
            st.subheader("User Profiles")
            user_result = view_all_users()
            users_db = pd.DataFrame.from_dict(user_result).iloc[:, 1:]
            users_db.columns = ["Username", "Password", "Role", 'SignUp DateTime']
            st.dataframe(users_db)

        if state.db_login_result[0][0]:
            user_projects = select_user_projects(state.db_login_result[0][0])
            users_use_cases = []
            if len(user_projects) > 0:
                for project in user_projects:
                    users_use_cases.append(project["project_name"])
            else:
                users_use_cases.append("None")

            if users_use_cases[0] != "None":
                st.subheader("Select Use Case")
                state.use_case_selection = st.selectbox("Use Cases", users_use_cases)
                use_case_selection_button = st.button("Select")
                if use_case_selection_button and state.use_case_selection != "None":
                    for project in user_projects:
                        if project["project_name"] == state.use_case_selection:
                            state.selected_project_id = project["id"]
                    st.success("Use case **{}** has successfully selected!".format(state.use_case_selection))
                    st.stop()

            st.subheader("Create New Case")
            state.new_use_case_name = st.text_input("Use Case Name", "")
            use_case_creation_button = st.button("Save")
            if use_case_creation_button:
                if state.new_use_case_name not in users_use_cases:
                    insert_project(state.db_login_result[0][0], state.new_use_case_name)
                    st.success("Use case **{}** has successfully created!".format(state.new_use_case_name))
                elif state.new_use_case_name in users_use_cases:
                    st.info("The use case **{}** already exists. Please create one with another name.".format(state.new_use_case_name))
                else:
                    st.error("This use case name is not valid. Please create one with another name.")
    else:
        st.info("Please Login first")
        st.stop()


def page_eda(state):
    st.title("AutoEDA")
    if state.current_user:
        if state.use_case_selection:
            st.write("""This page will perform AutoEDA, specifically it automates the exploration of a dataset so as to facilitate the understanding of its main characteristics.""")
        else:
            st.info("Please create or select a use case first.")
            st.stop()
    else:
        st.info("Please login first.")
        st.stop()

    st.header("Dataset")
    file_formats = ["xlsx", "csv", "txt"]
    file_upload = st.file_uploader("Upload your tabular data", type=file_formats)
    st.subheader("Settings")
    separator = st.text_input("File separator (if tab please insert \\t)", ",")  # default values ,
    decimal = st.text_input("Decimal point", ".")  # default values .
    header_options = ["Yes", "No"]
    header = st.radio("Presence header", header_options)
    number_rows = int(st.number_input("Number rows to load", 100))  # default input 100

    if file_upload is not None:  # if user upload any file then do
        info_file = file_upload.name.split(".")
        name_file = info_file[0]
        format_file = info_file[1]

        dict_header={"No": None,
                     "Yes": 0
                     }

        if format_file == "xlsx":
            dataset = pd.read_excel(file_upload, header=dict_header[header])
        else:
            dataset = pd.read_csv(file_upload,
                                  header=dict_header[header],
                                  sep=separator,
                                  decimal=decimal
                                  )

        total_rows = dataset.shape[0]  # total rows file
        dataset = dataset.iloc[:number_rows]  # sample dataset requested by user
        effective_rows = dataset.shape[0]
        show_info_dataset(dataset, total_rows, format_file, effective_rows, header)
        st.write(dataset)

        with st.form(key = "eda_choice"):
            st.header("Automated EDA")
            eda_choices = ["Sweetviz", "Dataprep"]
            eda_choice = st.selectbox("EDA Option:", eda_choices)
            st.subheader("Automated Exploratory Data Analaysis with **{}**".format(eda_choice))
            analyze_button = st.form_submit_button("Analyze")

        path = os.getcwd()
        report_path = os.path.join(path, 'use_cases/' + state.use_case_selection + '/datasets/' + name_file + '/reports/' + eda_choice.lower())

        if not os.path.isdir(report_path):
            pathlib.Path(report_path).mkdir(parents=True, exist_ok=True)  # create directory use_case/<name_file>/reports/<eda_choice>

        if analyze_button:
            with st.spinner(text="Generating report..."):
                generate_eda_report(dataset, eda_choice, report_path)
                st.success("Report for AutoEDA has been opened in a new tab.")
        else:
            st.info("""Press **Analyze** to generate the EDA report.""")
    else:
        st.info("Awaiting file to be uploaded.")


def page_automl(state):
    #Setup page
    st.title("AutoML")
    if state.current_user:
        if state.use_case_selection:
            st.write("""This page will perform AutoML,
                 specifically the app will return the best model according the task to accomplish and the dataset uploaded""")
        else:
            st.info("Please create or select a use case first.")
            st.stop()
    else:
        st.info("Please login first.")
        st.stop()

    #Main section
    st.header("Dataset")
    file_upload = st.file_uploader("Upload your tabular data",type = ["xlsx","csv","txt"])

    st.subheader("Settings")
    separator = st.text_input("File separator (if tab please insert \\t)", ",") #default values ,
    decimal = st.text_input("Decimal point",".") #default values .
    header = st.radio("Presence header", ["Yes","No"])
    number_rows = st.number_input("Number rows to load", 350)  # for h2o you must have X_train above 200.

    if file_upload is not None: #if user upload any file then do

        info_file = file_upload.name.split(".")
        state.name_file = info_file[0]
        state.format_file = info_file[1]
        path = os.getcwd()
        dataset_path = os.path.join(path, 'dataset', file_upload.name)

        dict_header = {"No": None, "Yes": 0}

        if state.format_file == "xlsx":
            dataset = pd.read_excel(file_upload,header=dict_header[header])
        else:
            dataset = pd.read_csv(file_upload,header=dict_header[header], sep=separator,
                                  decimal=decimal)

        state.total_rows = dataset.shape[0] #total rows file
        dataset = dataset.iloc[:number_rows] #sample dataset requested by user
        state.effective_rows = dataset.shape[0]
        show_info_dataset(dataset, state.total_rows, state.format_file, state.effective_rows, header)
        st.write(dataset)


        st.header("Select sensitive feature")
        # sensitive feature selection
        feature = [None]
        feature = feature + dataset.columns.tolist()
        # feature = [feature[-1]] + feature[0:len(feature) - 1]  # change order feature
        sensitive_feature_option = st.selectbox("Fair modeling is an area of ​​artificial intelligence that ensures that the"
                                    " result of machine modeling is not influenced by protected attributes such as "
                                    "gender, race, religion, sexual orientation, etc.", feature)
        if sensitive_feature_option is not None:
            state.dataset_fairness = dataset.copy()
            state.sf = sensitive_feature_option
            sensitive_feature = dataset.pop(sensitive_feature_option)
            # keep dataset state


        st.subheader("Setting process")

        with st.form(key = 'procedures'):

            if sensitive_feature_option is not None:
                task = st.radio("Type task", ["classification"], key="type_task")
            else:
                task = st.radio("Type task", ["classification","regression"], key="type_task")
            feature = dataset.columns.tolist()
            feature = [feature[-1]] + feature[0:len(feature)-1] #change order feature
            state.option = st.selectbox("Variable to predict", feature)
            if sensitive_feature_option is not None:
                state.library_automl = st.radio("Library for AutoML", ["tpot"])
                # state.library_automl = st.radio("Library for AutoML", ["h2o", "tpot"])
            else:
                state.library_automl = st.radio("Library for AutoML", ["h2o","tpot"])

            split_size = st.slider("Split size (%) train", 60, 90, 80, 5)
            split_seed = st.number_input("Seed size", 42)

            submit = st.form_submit_button("Start process")

        if submit: #if button above is clicked then do
            x_train = None
            encoder=None
            error=False
            type_target = check_validity_target(dataset, state.option, task) #return possibly type target not admissible

            if type_target != None:
                st.error("Attention! column %s contains a type %s which doesn't allow to perform %s  \n Please change the target variable or the task to accomplish and then re-click the button above"
                         %(state.option, type_target, task))
            else:
                st.write("A model is being built to predict the following variable: **" + str(state.option) + "**")

                y = dataset[state.option]
                x = dataset.drop(columns=[state.option])

                state.x_train_columns_sorted = sorted(x.columns.to_list())
                state.x_train_sorted_dtypes = x[state.x_train_columns_sorted].dtypes

                x_def = x.copy()
                y_def = y.copy()

                x = preproc_categorical_variables(x)

                text = "The task is a "

                if task == "classification":
                    number_label = len(pd.Series(y).unique())
                    measure_choice, count_warn = check_imbalance_target(y, number_label)

                    if len(count_warn) != 0: #if all classes contains more than 5 occurrences then do
                        error = True
                        print("Classes which contains less than 5 occurrences")
                        print(count_warn)
                        st.error("Attention! The dataset uploaded contains some classes which contains less than 5 occurrences.  \n Please change the dataset")
                    else:

                        y, encoder = special_case_int_label(y) #return y encoded only if y is type int

                        if encoder != None:
                            print("Encode")
                            y_encode = y.apply(lambda x: x+64)  #convert integer values to character in ASCII (required by the library h2o because it performs classification only if y is character)

                            y_encode_df = pd.DataFrame(y_encode, columns=['int'])
                            y_encode_df[state.option] = y_encode_df['int'].apply(lambda x: chr(x)) #get ASCII character based encoding before
                            y = y_encode_df[state.option]

                            #y_encode_df['reverse'] = y_encode_df[state.option].apply(lambda x: ord(x)-64)
                            #y_encode_df['reverse_initial']=encoder.inverse_transform(y_encode_df['reverse'])

                        y= y.astype('category') #convert y variable into categorical
                        print("Category target variable")
                        print(type(y[0]))
                        print(y)

                        number_label = len(pd.Series(y).unique())
                        state.type_task = "multi-class" if  number_label>2 else "binary"
                        st.write(text + state.type_task + " " + task)
                        measure_choice, count_warn = check_imbalance_target(y, number_label)

                        if sensitive_feature_option is not None:
                            x_train, x_test, y_train, y_test, sf_train, sf_test = train_test_split(x, y, sensitive_feature,
                                                                                train_size=split_size / 100,
                                                                                stratify=y, random_state=split_seed)
                            state.x_train_fairness = x_train.copy()
                            state.x_test_fairness = x_test.copy()
                            state.y_train_fairness = y_train.copy()
                            state.y_test_fairness = y_test.copy()
                            state.sf_train_fairness = sf_train.copy()
                            state.sf_test_fairness = sf_test.copy()
                        else:
                            x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=split_size/100,
                                                                 stratify = y, random_state=split_seed)
                else:
                    state.type_task = "regression"
                    decimal_split = y[0].astype(str).split(decimal)
                    if len(decimal_split) == 1:
                        number_round = 0
                    else:
                        number_round=len(decimal_split[1])
                    st.write(text + state.type_task)
                    measure_choice = task

                    x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=split_size/100,
                                                         random_state=split_seed)
                if x_train is not None:
                    feature_names = list(x_train.columns)
                    feature_names = [str(elem) for elem in feature_names]
                else:
                    st.stop()
                sep = " | "
                model_features = sep.join(feature_names)

                if error == False:
                    x_test_def = x_def.loc[x_test.index]
                    y_test_def = y_def.loc[y_test.index]


                    st.subheader("Setup splitting")
                    st.write("Train set: " + str(len(x_train)) + " rows | Test set: " + str(len(x_test)) + " rows")

                    scoring_dict={'balance': 'accuracy',
                                  'unbalance': 'balanced_accuracy',
                                  'regression': 'r2'}

                    print(measure_choice)

                    path=os.getcwd() #get working directory
                    if sensitive_feature_option is not None:
                        path_dir = os.path.join(path, 'use_cases/' + state.use_case_selection + '/datasets/' + state.name_file + '/fairness/' + 'before_mitigation/')
                    else:
                        path_dir = os.path.join(path, 'use_cases/' + state.use_case_selection + '/datasets/' + state.name_file)
                    print(path_dir)
                    # path_dir = os.path.join(path, 'use_cases/' + state.use_case_selection + '/datasets/' + state.name_file)
                    # print(path_dir)

                    if not os.path.isdir(path_dir):
                        print("Directory " + path_dir + " created")
                        pathlib.Path(path_dir).mkdir(parents=True, exist_ok=True) #create directory use_cases/<state.use_case_selection>/datasets/<state.dataset>

                    if state.library_automl == 'h2o':

                        train = pd.merge(x_train, y_train, left_index=True, right_index=True)
                        test = pd.merge(x_test, y_test, left_index=True, right_index=True)

                        best_model, model_path = h2o_automl(train, state.option, measure_choice, path_dir, task)

                        x_test = h2o.H2OFrame(test) #convert df h2o format (required by the library)
                        pred = best_model.predict(x_test)

                        pred_df_h2o = pred.as_data_frame()
                        print("Prediction h2o")
                        print(pred_df_h2o)

                        pred = pred_df_h2o['predict'].tolist()
                        h2o.cluster().shutdown(prompt = False) #close h2o cluster

                    else:
                        best_model, model_path = tpot_automl(x_train, y_train, state.type_task, scoring_dict[measure_choice], path_dir, task)
                        pred = best_model.predict(x_test)
                    print("Prediction")
                    pred_df = pd.DataFrame(pred, index = x_test_def.index, columns=['predict'])

                    print(pred_df)

                    if encoder != None:

                        path_file_encoder = path_dir + '/' + state.library_automl + '/label_encoder_' + str(state.option)+ '.pkl'
                        pickle.dump(encoder, open(path_file_encoder, 'wb'))

                        pred_df['reverse'] = pred_df['predict'].apply(lambda x: ord(x)-64)
                        pred_df['predict']=encoder.inverse_transform(pred_df['reverse'])
                        pred_df.drop(columns='reverse',inplace=True)

                    if state.option not in x_test_def.columns.tolist():
                        x_test_def[state.option] = y_test_def

                    test_pred_df = pd.concat([x_test_def,pred_df],axis=1)
                    if scoring_dict[measure_choice] == "accuracy":
                        test_score = accuracy_score(test_pred_df[state.option], test_pred_df['predict'])
                    elif scoring_dict[measure_choice] == 'balanced_accuracy':
                        test_score = balanced_accuracy_score(test_pred_df[state.option], test_pred_df['predict'])
                    elif scoring_dict[measure_choice] == 'r2':
                        test_pred_df['predict']=round(test_pred_df['predict'], number_round)
                        test_score = r2_score(test_pred_df[state.option], test_pred_df['predict'])
                        st.line_chart(test_pred_df[[state.option, 'predict']].reset_index(drop=True))

                    score_to_display = str(round(test_score, 3))
                    measure_to_display = scoring_dict[measure_choice].capitalize().replace('_', ' ')
                    st.write("The **{}** on test data is: **{}**".format(measure_to_display, score_to_display))

                    current_datetime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    state.new_autoML_project = insert_autoML_project(state.selected_project_id, dataset_path,
                                                       model_features, state.option, model_path, state.library_automl, current_datetime)

                    st.markdown(filedownload(test_pred_df, best_model, state.library_automl, state.name_file), unsafe_allow_html=True)

                    del dataset, x_train, x_test, x_test_def, y_test_def, y_train, y_test, pred, pred_df, test_pred_df, best_model
                    st.stop()
    else:
        st.info("Awaiting csv to be uploaded")

def page_inference(state):
    st.title("Inference")
    if state.current_user:
        if state.selected_project_id:
            models = select_user_autoML_projects(state.selected_project_id)
        else:
            st.info("Please create or select a use case first.")
            st.stop()
    else:
        st.info("Please login first.")
        st.stop()

    if models:
        st.header("Dataset")
        file_inference = st.file_uploader('Upload your tabular data to inference', type=['csv','xlsx','txt'])
        st.subheader("Settings")
        separator = st.text_input("File separator (if tab please insert \\t)", ",")
        decimal = st.text_input("Decimal point", ".")
        header = st.radio("Presence header", ["Yes","No"])

        if file_inference is not None:
            info_file = file_inference.name.split(".")
            format_file= info_file[1]
            dict_header={"No": None,"Yes": 0}

            if format_file == "xlsx":
                dataset_inference = pd.read_excel(file_inference, header=dict_header[header])
            else:
                dataset_inference = pd.read_csv(file_inference, header=dict_header[header], sep=separator,
                              decimal=decimal)

            total_rows = dataset_inference.shape[0]
            show_info_dataset_inf(dataset_inference, total_rows, format_file,header)
            st.write(dataset_inference)

            dataset_inference_columns = dataset_inference.columns.tolist()
            dataset_inference_columns_sorted = sorted(dataset_inference_columns)

            choices = ["None", "Use last model", "Select other model"]
            inference_choice = st.selectbox("Inference options for choosing a trained AutoMl model:", choices)

            if inference_choice == "None":
                st.info("Please select an option for inference.")
                st.stop()
            elif inference_choice == "Use last model":
                if state.name_file:
                    training_dataset_name = state.name_file
                    file_format = state.format_file
                    full_name = training_dataset_name + '.' + file_format
                    task = state.type_task
                    predict = state.option
                    library = state.library_automl
                    st.write('Dataset used to train the model: ', full_name)
                    st.write('Task performed: ', task)
                    st.write('Target variable: ', predict)
                else:
                    st.error("Go to the AutoML page and train a model.")
                    st.stop()

                dataset_inference_sorted_dtypes = dataset_inference[dataset_inference_columns_sorted].dtypes
                x_train_columns_sorted = state.x_train_columns_sorted
                x_train_sorted_dtypes = state.x_train_sorted_dtypes

                if predict in dataset_inference_columns:
                    st.error("Attention! The dataset uploaded contains the target variable %s to predict.  \n Please change the dataset"% predict)
                    st.stop()
                elif x_train_columns_sorted != dataset_inference_columns_sorted and not x_train_sorted_dtypes.equals(dataset_inference_sorted_dtypes):
                    st.error("Attention! The dataset uploaded does not contain the same columns nor the same datatypes with the dataset used for training.  \n Please change the dataset")
                    st.stop()
                else:
                    st.subheader("Setting train history")
                    st.write("Inference process is ready to start!")
                    st.write("Retrieving the best model obtained using **" + library + "**")

                    path = os.getcwd()
                    st.info("Checking if any model has been builded with this library")
                    path_dir = os.path.join(path,"use_cases/" + state.use_case_selection + '/datasets/' + state.name_file + "/" + library)

            elif inference_choice == "Select other model":
                st.subheader("Trained AutoML models")
                models_name_list = ["None"]
                print(models)
                for model in models:
                    model_name = os.path.normpath(model['model_path']).split(os.path.sep)[-1]
                    library = os.path.normpath(model['model_path']).split(os.path.sep)[-2]
                    training_dataset_name = os.path.normpath(model['training_dataset_path']).split(os.path.sep)[-1]
                    predict = model["target"]
                    models_name_list.append(training_dataset_name + " : " + library + " : " + model_name + " : " + predict)

                with st.form(key='model selection'):
                    st.write("**Select a model**")
                    model_choice = st.selectbox("training dataset name: library used for training : trained model : target", sorted(models_name_list))
                    select_model_button = st.form_submit_button("Retrieve")

                print("model choice: ", model_choice)
                if select_model_button and model_choice != "None":
                    model_choice_splitted = model_choice.split(' : ')
                    predict = model_choice_splitted[3]
                    print(predict)
                    file_model = model_choice_splitted[2]
                    print(file_model)
                    library =model_choice_splitted[1]
                    print(library)
                    training_dataset_name = model_choice_splitted[0].split('.')[0]
                    print(training_dataset_name)
                    x_train_columns = [model["model_features"] for model in models if file_model in model["model_path"]]
                    x_train_columns_sorted = sorted(x_train_columns[0].split(" | "))
                    print(x_train_columns_sorted)
                    if predict in dataset_inference_columns:
                        st.error("Attention! The dataset uploaded contains the target variable %s to predict.  \n Please change the dataset"% predict)
                        st.stop()
                    elif x_train_columns_sorted != dataset_inference_columns_sorted:
                        st.error("Attention! The dataset uploaded does not contain the same columns with the dataset used for training.  \n Please change the dataset")
                        st.stop()
                    else:
                        path = os.getcwd()
                        path_dir = os.path.join(path,"use_cases/" + state.use_case_selection + '/datasets/' + training_dataset_name + "/" + library)
                        print(path_dir)
                else:
                    st.info("Awaiting model to use for inference")
                    st.stop()

            if os.path.isdir(path_dir):
                list_file = os.listdir(path_dir)

                if inference_choice == "Use last model":
                    file_model = get_latest_model(list_file)

                path_model = os.path.join(path_dir, file_model)

                dataset_final = dataset_inference.copy()
                dataset_preproc = preproc_categorical_variables(dataset_inference)

                st.info("The model %s is loaded"%file_model)
                st.info("Start inference...wait please.")

                if library == 'h2o':
                    h2o.init()
                    best_model = h2o.load_model(path_model)
                    dataset_h2o = h2o.H2OFrame(dataset_preproc)
                    pred_h2o = best_model.predict(dataset_h2o)
                    pred_h2o_df = pred_h2o.as_data_frame()
                    pred = pred_h2o_df['predict'].tolist()
                    h2o.cluster().shutdown(prompt=False) #close h2o cluster
                else:
                    with open(path_model, 'rb') as f:
                        f.seek(0)
                        best_model = pickle.load(f)
                        pred = best_model.predict(dataset_preproc)

                st.info("Inference completed")

                dataset_final['predict'] = pred

                
                path_file_encoder = os.path.join(path, "use_cases/" + state.use_case_selection + '/datasets/' + training_dataset_name +'/' + library + "/label_encoder_" + predict+".pkl")

                if os.path.isfile(path_file_encoder):
                    print("Encoding before")
                    encoder = pickle.load(open(path_file_encoder, 'rb'))
                    dataset_final['reverse'] = dataset_final['predict'].apply(lambda x: ord(x)-64)
                    dataset_final['predict'] = encoder.inverse_transform(dataset_final['reverse'])
                    dataset_final.drop(columns='reverse', inplace=True)

                st.write(file_download(dataset_final, training_dataset_name, library), unsafe_allow_html=True)

                del dataset_inference, dataset_preproc, dataset_final, pred, best_model

            else:
                st.error("Attention! It doesn't exist any model builded using the dataset %s and the library %s.  \n Please change the library or start process autoML with the library %s"
                          %(training_dataset_name, library, library))
        else:
            st.info("Awaiting file to inference")
    else:
        st.info("Please train a model in AutoML page first.")



def page_fairness(state):
    #Setup page
    st.write(" # Demo Trustworthy AI - Fairness ")
    st.write("""Fair modeling is an area of artificial intelligence that ensures that the result of
    machine modeling is not influenced by protected attributes such as gender, race, religion, sexual orientation, etc.""")

    if state.current_user:
        if state.selected_project_id:
            models = select_user_autoML_projects(state.selected_project_id)
        else:
            st.info("Please create or select a use case first.")
            st.stop()
    else:
        st.info("Please login first.")
        st.stop()

    if models:
        # Main section
        st.header("Dataset")

        show_info_dataset_fair(state.dataset_fairness,state.total_rows, state.format_file,state.effective_rows)
        st.write(state.dataset_fairness)



        choices = ["None", "Use last model"]
        fairness_choice = st.selectbox("Fairness options for choosing a trained AutoMl model:", choices)

        if fairness_choice == "None":
            state.show_plot = None
            st.info("Please select an option for fairness.")
            st.stop()

        else:
            if state.name_file:
                training_dataset_name = state.name_file
                file_format = state.format_file
                full_name = training_dataset_name + '.' + file_format
                task = state.type_task
                predict = state.option
                library = state.library_automl
                st.write('Dataset used to train the model: ', full_name)
                st.write('Task performed: ', task)
                st.write('Target variable: ', predict)
            else:
                st.error("Go to the AutoML page and train a model.")
                st.stop()

            st.subheader("Setting train history")
            st.write("Fairness process is ready to start!")
            st.write("Retrieving the best model obtained using **" + library + "**")

            path = os.getcwd()
            st.info("Checking if any model has been builded with this library")
            path_dir = os.path.join(path,
                                    "use_cases/" + state.use_case_selection + '/datasets/' + state.name_file +
                                    "/fairness/" + "before_mitigation/" + library)

            state.path_dir = path_dir


            if os.path.isdir(path_dir):
                list_file = os.listdir(path_dir)

                if fairness_choice == "Use last model":
                    file_model = get_latest_model(list_file)

                path_model = os.path.join(path_dir, file_model)

                st.info("The model %s is loaded" % file_model)

                if library == 'h2o':
                    h2o.init()
                    unmitigated_predictor = h2o.load_model(path_model)
                    dataset_h2o = h2o.H2OFrame(state.x_test_fairness)
                    pred_h2o = unmitigated_predictor.predict(dataset_h2o)
                    pred_h2o_df = pred_h2o.as_data_frame()
                    predictions = pred_h2o_df['predict'].tolist()
                    h2o.cluster().shutdown(prompt=False)  # close h2o cluster

                else:
                    with open(path_model, 'rb') as f:
                        f.seek(0)

                        unmitigated_predictor = pickle.load(f)

                        predictions = unmitigated_predictor.predict(state.x_test_fairness)

                        path_file_encoder = path_dir + '/label_encoder_' + str(state.option) + '.pkl'

                        y_true = state.y_test_fairness.copy()
                        y_train = state.y_train_fairness.copy()

                        if os.path.isfile(path_file_encoder):
                            encoder = pickle.load(open(path_file_encoder, 'rb'))
                            predictions = list(map(lambda x: ord(x) - 64, predictions))
                            y_true = list(map(lambda x: ord(x) - 64, y_true))
                            y_train = list(map(lambda x: ord(x) - 64, y_train))

                        if isinstance(predictions[0], str):
                            fairness_encoder = LabelEncoder()
                            predictions = fairness_encoder.fit_transform(predictions)
                            y_true = fairness_encoder.transform(y_true)
                            y_train = fairness_encoder.transform(y_train)

                       # plot all metrics
                        metrics = {'None': None,
                                   'accuracy': accuracy_score,
                                   'precision': precision_score,
                                   'recall': recall_score,
                                   'false positive rate': false_positive_rate,
                                   'true positive rate': true_positive_rate,
                                   'selection rate': selection_rate,
                                   'count': count}

                        st.header("Evaluation metrics")

                        metric = st.selectbox("Choose an evaluation metric:", metrics.keys())

                        if metrics[metric] is not None:
                            plot_headline = "Before mitigation"
                            plot_metric(metrics[metric], metric, y_true, predictions, state.sf_test_fairness,
                                        plot_headline)

                        # plot of predictions in accuracy
                        predictions_in_accuracy_score(y_true, predictions, state.sf_test_fairness)

                        # plot of predictions in selection_rate
                        predictions_in_selection_rate(y_true, predictions, state.sf_test_fairness)



                        with st.form(key='sweep'):

                            # choose constraint
                            constraints = ["DemographicParity", "TruePositiveRateParity", "FalsePositiveRateParity",
                                           "ErrorRateParity", "EqualizedOdds"]
                            constraint_choice = st.selectbox("Choose a constraint option:", constraints)
                            if constraint_choice == "DemographicParity":
                                constraint = DemographicParity()
                            # elif constraint_choice == "UtilityParity":
                            #     constraint = UtilityParity()
                            elif constraint_choice == "TruePositiveRateParity":
                                constraint = TruePositiveRateParity()
                            elif constraint_choice == "FalsePositiveRateParity":
                                constraint = FalsePositiveRateParity()
                            elif constraint_choice == "ErrorRateParity":
                                constraint = ErrorRateParity()
                            else:
                                constraint = EqualizedOdds()

                            disparity_choices = ["Disparity in Accuracy", "Disparity in Predictions"]
                            disparity_metric = st.radio('Choose a disparity metric', disparity_choices)

                            state.disparity_metric = disparity_metric

                            grid_size = st.slider("Grid size", 10, 100, 20, 5)

                            submit1 = st.form_submit_button("Start process")

                        if submit1:
                            state.show_plot = 1
                            classifier = unmitigated_predictor
                            parameters = unmitigated_predictor.get_params()

                            mod_name = []
                            for i in parameters["steps"]:
                                mod_name.append(i[0])

                            for i in range(len(mod_name)):
                                mod_name[i] = mod_name[i] + '__sample_weight'

                            try:
                                sweep = GridSearch(classifier,
                                                   constraints=constraint,
                                                   grid_size=grid_size,
                                                   sample_weight_name=mod_name[-1]
                                                   )

                                sweep.fit(state.x_train_fairness, y_train, sensitive_features=state.sf_train_fairness)
                            except TypeError:
                                st.error("Fairlearn library does not support the model you have chosen.  \n Please repeat the autoML procedure.")
                                st.stop()


                            predictors = sweep.predictors_

                            errors, disparities = [], []
                            for m in predictors:
                                def classifier(X): return m.predict(X)

                                disparity = constraint
                                # disparity.load_data(state.x_train_fairness, pd.Series(y_train), sensitive_features=state.sf_train_fairness)

                                error = ErrorRate()
                                error.load_data(state.x_train_fairness, pd.Series(y_train),
                                                    sensitive_features=state.sf_train_fairness)

                                errors.append(error.gamma(classifier)[0])
                                disparities.append(disparity.gamma(classifier).max())

                            all_results = pd.DataFrame({"predictor": predictors, "error": errors, "disparity": disparities})

                            non_dominated = []
                            for row in all_results.itertuples():
                                errors_for_lower_or_eq_disparity = all_results["error"][
                                    all_results["disparity"] <= row.disparity]
                                if row.error <= errors_for_lower_or_eq_disparity.min():
                                    non_dominated.append(row.predictor)

                            state.non_dominated = non_dominated
                            dashboard_predicted = {}
                            for i in range(len(non_dominated)):
                                key = "dominant_model_{0}".format(i)
                                value = non_dominated[i].predict(state.x_test_fairness)
                                dashboard_predicted[key] = value

                            pred_array = np.array(predictions)
                            dashboard_predicted["unmitigated"] = pred_array

                            state.dashboard_predicted = dashboard_predicted.copy()

                            metrics_mitigated = []
                            accuracy_mitigated = []
                            metrics_mitigated_acc = []
                            if disparity_metric =='Disparity in Accuracy':
                                metrics_unmitigated = MetricFrame(accuracy_score, y_true, predictions,
                                                                  sensitive_features=state.sf_test_fairness)
                                accuracy_unmitigated = metrics_unmitigated.overall
                                for i in range(len(non_dominated)):
                                    key = "dominant_model_{0}".format(i)
                                    metrics_mitigated.append(MetricFrame(accuracy_score, y_true, dashboard_predicted[key],
                                                                sensitive_features=state.sf_test_fairness))
                                    accuracy_mitigated.append(metrics_mitigated[i].overall)


                            else:
                                metrics_unmitigated = MetricFrame(selection_rate, y_true, predictions,
                                                                  sensitive_features=state.sf_test_fairness)
                                metrics_unmitigated_accuracy = MetricFrame(accuracy_score, y_true, predictions,
                                                                  sensitive_features=state.sf_test_fairness)
                                accuracy_unmitigated = metrics_unmitigated_accuracy.overall

                                for i in range(len(non_dominated)):
                                    key = "dominant_model_{0}".format(i)
                                    metrics_mitigated.append(MetricFrame(selection_rate, y_true, dashboard_predicted[key],
                                                                    sensitive_features=state.sf_test_fairness))
                                    metrics_mitigated_acc.append(
                                        MetricFrame(accuracy_score, y_true, dashboard_predicted[key],
                                                    sensitive_features=state.sf_test_fairness))
                                    accuracy_mitigated.append(metrics_mitigated_acc[i].overall)

                            state.metrics_unmitigated = metrics_unmitigated
                            state.metrics_mitigated = metrics_mitigated
                            state.accuracy_mitigated = accuracy_mitigated
                            state.accuracy_unmitigated = accuracy_unmitigated

                        if state.show_plot is not None:
                            model_comparison_plot(state.metrics_mitigated,state.accuracy_mitigated,
                                                  state.metrics_unmitigated,
                                                  state.accuracy_unmitigated,
                                                  disparity_metric)

                        with st.form(key='model choice'):
                            if state.dashboard_predicted is not None:
                                mitigated_model_choices = list(state.dashboard_predicted.keys())[:-1]
                            else:
                                mitigated_model_choices = [None]

                            mitigated_model_choice = st.selectbox("Choose a mitigated model:",
                                                                      mitigated_model_choices)

                            submit2 = st.form_submit_button("Start process")


                        if submit2:
                            model_num = mitigated_model_choice
                            model_num = int(model_num[-1])

                            database_metrics_list = []
                            for i in range(len(state.metrics_unmitigated.by_group)):
                                mitigation_metrics_dict = {}
                                mitigation_metrics_dict["Disparity Metric"] = state.disparity_metric
                                mitigation_metrics_dict["Phase"] = "Before Mitigation"
                                mitigation_metrics_dict["Group"] = state.metrics_unmitigated.by_group.index[i]
                                # mitigation_metrics_dict["Value"] = str(round(state.metrics_unmitigated.by_group[i] * 100,2)) + ' %'
                                mitigation_metrics_dict["Value"] = state.metrics_unmitigated.by_group[i]
                                mitigation_metrics_dict["Path"] = state.path_dir
                                database_metrics_list.append(mitigation_metrics_dict)

                            mitigated_path = os.getcwd()
                            mitigated_path_dir = os.path.join(mitigated_path,'use_cases/' + state.use_case_selection + '/datasets/' + state.name_file + '/fairness/' + 'after_mitigation')

                            for i in range(len(state.metrics_mitigated[model_num].by_group)):
                                mitigation_metrics_dict = {}
                                mitigation_metrics_dict["Disparity Metric"] = state.disparity_metric
                                mitigation_metrics_dict["Phase"] = "After Mitigation"
                                mitigation_metrics_dict["Group"] = state.metrics_mitigated[model_num].by_group.index[i]
                                # mitigation_metrics_dict["Value"] = str(round(state.metrics_mitigated[model_num].by_group[i] * 100,2)) + ' %'
                                mitigation_metrics_dict["Value"] = state.metrics_mitigated[model_num].by_group[i]
                                mitigation_metrics_dict["Path"] = mitigated_path_dir
                                database_metrics_list.append(mitigation_metrics_dict)

                            mitigated_predictor = "dominant_model_" + str(model_num)

                            if disparity_metric == 'Disparity in Accuracy':
                                subplots_in_accuracy_score(y_true, predictions, state.dashboard_predicted[mitigated_predictor], state.sf_test_fairness)
                                subplots_in_selection_rate(y_true, predictions, state.dashboard_predicted[mitigated_predictor], state.sf_test_fairness)
                            else:
                                subplots_in_selection_rate(y_true, predictions, state.dashboard_predicted[mitigated_predictor], state.sf_test_fairness)
                                subplots_in_accuracy_score(y_true, predictions, state.dashboard_predicted[mitigated_predictor], state.sf_test_fairness)

                            # plot performance in selected metric
                            if metrics[metric] is not None:
                                plot_headline = "Model comparison in {}".format(metric)
                                subplots_in_metric(metrics[metric], metric, y_true, predictions, state.dashboard_predicted[mitigated_predictor], state.sf_test_fairness, plot_headline)

                            # save mitigated model
                            if not os.path.isdir(mitigated_path_dir):
                                pathlib.Path(mitigated_path_dir).mkdir(parents=True, exist_ok=True)
                            path_file_mitigated = mitigated_path_dir + '/mitigated_model_' + str(state.sf) + '.pkl'
                            pickle.dump(state.non_dominated[model_num], open(path_file_mitigated, 'wb'))

                            for elem in database_metrics_list:
                                current_datetime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                                new_fairness_assessment = insert_fairness_analysis(state.selected_project_id, state.new_autoML_project, elem["Disparity Metric"],
                                                                                   elem["Phase"], elem["Group"], float(elem["Value"]), current_datetime, elem["Path"])

                            del y_true, predictions
                            st.stop()

    else:
        st.info("Awaiting dataset to be uploaded")


def page_explainable_ai(state):
    st.title("eXplainable AI")
    st.header("Dataset")
    file_formats = ["xlsx", "csv", "txt"]
    file_upload = st.file_uploader("Upload your tabular data", type=file_formats)

    st.subheader("Settings")
    separator = st.text_input("File separator (if tab please insert \\t)", ",") # default values ,
    decimal = st.text_input("Decimal point", ".") # default values .
    header_options = ["Yes", "No"]
    header = st.radio("Presence header", header_options)
    number_rows = int(st.number_input("Number rows to load", 100)) # default input 100

    if file_upload is not None: # if user upload any file then do

        info_file = file_upload.name.split(".")
        name_file = info_file[0]
        format_file= info_file[1]

        dict_header={"No": None,
                     "Yes": 0
                     }

        if format_file == "xlsx":
            dataset = pd.read_excel(file_upload, header=dict_header[header])
        else:
            dataset = pd.read_csv(file_upload,
                                  header=dict_header[header],
                                  sep=separator,
                                  decimal=decimal
                                  )

        total_rows = int(dataset.shape[0]) # total rows file
        if total_rows != number_rows:
            dataset = dataset.iloc[:number_rows] # sample dataset requested by user
        effective_rows = dataset.shape[0]
        show_info_dataset(dataset, total_rows, format_file, effective_rows, header)
        st.write(dataset)

        st.subheader("Setting process")

        with st.form(key='procedures'):
            split_size = st.slider("Split size (%) train", 60, 90, 80, 5)
            split_seed = st.number_input("Seed size", 42)

            task_options = ["classification", "regression"]
            task = st.radio("Type task", task_options, key="type_task")

            feature = dataset.columns.tolist()
            feature = [feature[-1]] + feature[0:len(feature)-1] # change order feature
            target_variable = st.selectbox("Variable to predict", feature)

            model_options = ["XGBoost", "CatBoost", "Random Forest"]
            model_option = st.radio("Model selection for eXplainable AI", model_options)

            submit = st.form_submit_button("Start process")

        if submit: # if button above is clicked then do
            encoder = None
            error = False
            type_target = check_validity_target(dataset, target_variable, task) # return possibly type target not admissible

            if type_target != None:
                st.error("Attention! column %s contains a type %s which doesn't allow to perform %s  \n Please change the target variable or the task to accomplish and then re-click the button above"
                         %(target_variable, type_target, task))
            else:
                st.write("A model is being built to predict the following variable: **" + str(target_variable) + "**")

                y = dataset[target_variable]
                X = dataset.drop(columns=[target_variable])

                X = preproc_categorical_variables(X)

                text ="The task is a "

                if task == "classification":
                    number_label = len(pd.Series(y).unique())
                    measure_choice, count_warn = check_imbalance_target(y, number_label)

                    if len(count_warn) != 0: # if all classes contains more than 5 occurrences then do
                        error = True
                        st.error("Attention! The dataset uploaded contains some classes which contains less than 5 occurrences.  \n Please change the dataset")
                    else:
                        number_label = len(pd.Series(y).unique())
                        type_task = "multi-class" if  number_label > 2 else "binary"
                        st.write(text + type_task + " " + task)
                        measure_choice, count_warn = check_imbalance_target(y, number_label)

                        label_encoder = LabelEncoder().fit(y)
                        y = label_encoder.transform(y)

                        X_train, X_test, y_train, y_test = train_test_split(X,
                                                                            y,
                                                                            train_size=split_size / 100,
                                                                            stratify=y,
                                                                            random_state=split_seed)
                else:
                    type_task="regression"
                    decimal_split = y[0].astype(str).split(decimal)
                    if len(decimal_split) == 1:
                        number_round = 0
                    else:
                        number_round = len(decimal_split[1])
                    st.write(text + type_task)
                    measure_choice = task

                    X_train, X_test, y_train, y_test = train_test_split(X,
                                                                        y,
                                                                        train_size=split_size / 100,
                                                                        random_state=split_seed)

                if error == False:
                    st.subheader("Setup splitting")
                    st.write("Train set: " + str(len(X_train)) + " rows | Test set: " + str(len(X_test)) + " rows")

                    st.subheader('Interpretation of the model')
                    with st.spinner(text="Generating report..."):
                        model = model_wrapper(model_option, task)
                        model.fit(X_train, y_train)
                        y_pred = pd.DataFrame(model.predict(X_test), columns=['pred'], index=X_test.index)

                        xpl = SmartExplainer()
                        xpl.compile(x=X_test,
                                    model=model,
                                    y_pred=y_pred,
                                    )
                        xpl.generate_report(output_file='shapash_report.html',
                                            project_info_file='shapash_report.yml',
                                            x_train=X_train,
                                            y_train=pd.DataFrame(y_train),
                                            y_test=pd.DataFrame(y_test),
                                            title_story="Report",
                                            title_description="""This document is a data science report used for model explainability.
                                                It was generated using the Shapash library.""",
                                            metrics= metrics_wrapper(task)
                                            )

                        current_dir = os.getcwd()
                        webbrowser.open_new_tab(current_dir + "/shapash_report.html")

                        st.success('Report for eXplainable AI has been opened in a new tab.')
    else:
        st.info("Awaiting file to be uploaded")


def display_state_values(state):
    st.write("Input state:", state.input)
    st.write("Slider state:", state.slider)
    st.write("Radio state:", state.radio)
    st.write("Checkbox state:", state.checkbox)
    st.write("Selectbox state:", state.selectbox)
    st.write("Multiselect state:", state.multiselect)

    for i in range(3):
        st.write(f"Value {i}:", state[f"State value {i}"])

    if st.button("Clear state"):
        state.clear()


class _SessionState:

    def __init__(self, session, hash_funcs):
        """Initialize SessionState instance."""
        self.__dict__["_state"] = {
            "data": {},
            "hash": None,
            "hasher": _CodeHasher(hash_funcs),
            "is_rerun": False,
            "session": session,
        }

    def __call__(self, **kwargs):
        """Initialize state data once."""
        for item, value in kwargs.items():
            if item not in self._state["data"]:
                self._state["data"][item] = value

    def __getitem__(self, item):
        """Return a saved state value, None if item is undefined."""
        return self._state["data"].get(item, None)

    def __getattr__(self, item):
        """Return a saved state value, None if item is undefined."""
        return self._state["data"].get(item, None)

    def __setitem__(self, item, value):
        """Set state value."""
        self._state["data"][item] = value

    def __setattr__(self, item, value):
        """Set state value."""
        self._state["data"][item] = value

    def clear(self):
        """Clear session state and request a rerun."""
        self._state["data"].clear()
        self._state["session"].request_rerun()

    def sync(self):
        """Rerun the app with all state values up to date from the beginning to fix rollbacks."""

        # Ensure to rerun only once to avoid infinite loops
        # caused by a constantly changing state value at each run.
        #
        # Example: state.value += 1
        if self._state["is_rerun"]:
            self._state["is_rerun"] = False

        elif self._state["hash"] is not None:
            if self._state["hash"] != self._state["hasher"].to_bytes(self._state["data"], None):
                self._state["is_rerun"] = True
                self._state["session"].request_rerun()

        self._state["hash"] = self._state["hasher"].to_bytes(self._state["data"], None)


def _get_session():
    session_id = get_report_ctx().session_id
    session_info = Server.get_current()._get_session_info(session_id)

    if session_info is None:
        raise RuntimeError("Couldn't get your Streamlit Session object.")

    return session_info.session


def _get_state(hash_funcs=None):
    session = _get_session()

    if not hasattr(session, "_custom_session_state"):
        session._custom_session_state = _SessionState(session, hash_funcs)

    return session._custom_session_state



if __name__ == "__main__":
    main()