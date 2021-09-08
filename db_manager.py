import hashlib
import datetime
from mysql import connector


# hashing password
def make_hashes(password):
    return hashlib.sha256(str.encode(password)).hexdigest()


# check hashed password
def check_hashes(password, hashed_text):
    if make_hashes(password) == hashed_text:
        return hashed_text
    return False


# connecting to mysql db
def connect_to_db():
    db_conn = connector.connect(
        host="localhost",
        user="root",
        password="my123_sql456",
        database="demo_app_db"
    )
    return db_conn


# stores in db table users user's username, password, role and date of enrollment
# and returns the last inserted user's id
def add_userdata(username, password, role, datetime):
    db_conn = connect_to_db()
    cur = db_conn.cursor()
    all_users = view_all_users()
    usernames_list = [user["username"] for user in all_users]
    if username in usernames_list:
        return False
    cur.execute('INSERT INTO users (username, password, role, enrollment_datetime) VALUES (%s, %s, %s, %s)',
                  (username, password, role, datetime))
    db_conn.commit()
    print('Insertion Done')
    last_user_id = cur.lastrowid
    cur.close()
    db_conn.close()
    return last_user_id


# retrieves user's info
def login_user(username, password):
    db_conn = connect_to_db()
    cur = db_conn.cursor()
    cur.execute('SELECT * FROM users WHERE username = %s AND password = %s', (username, password))
    data = cur.fetchall()
    cur.close()
    db_conn.close()
    return data


# retrieves all users and returns them in a list of dictionaries
def view_all_users():
    db_conn = connect_to_db()
    cur = db_conn.cursor()
    cur.execute('SELECT * FROM users')
    data = cur.fetchall()
    result = []
    for row in data:
      row_dict = {"id":row[0], "username":row[1], "password":row[2], "role":row[3], "enrollment_datetime":row[4].strftime("%b %d %Y %H:%M:%S")}
      result.append(row_dict)
    cur.close()
    db_conn.close()
    return result


# updates users table
def update_users(user_id, username, password, role, datetime):
    db_conn = connect_to_db()
    cur = db_conn.cursor()
    query = """UPDATE users
                SET username = %s, password = %s, role = %s, enrollment_datetime = %s
                WHERE id = %s"""
    values = (username, password, role, datetime, user_id)
    cur.execute(query, values)
    db_conn.commit()
    print('Update completed')
    cur.close()
    db_conn.close()
    return user_id


# deletes a users table row
def delete_users(user_id):
    db_conn = connect_to_db()
    cur = db_conn.cursor()
    cur.execute('DELETE FROM users WHERE id = %s', (user_id,))
    db_conn.commit()
    print('Deletion completed')
    cur.close()
    db_conn.close()
    return user_id


# stores in db table projects project's name and user's id and returns the last inserted project's id
def insert_project(user_id, project_name):
    db_conn = connect_to_db()
    cur = db_conn.cursor()
    cur.execute('INSERT INTO projects (user_id, project_name) VALUES (%s, %s)', (user_id, project_name))
    db_conn.commit()
    print('Insertion Done')
    last_project_id = cur.lastrowid
    cur.close()
    db_conn.close()
    return last_project_id


# selects stored user's projects and returns them in a list of dictionaries
def select_user_projects(user_id):
    db_conn = connect_to_db()
    cur = db_conn.cursor()
    cur.execute('SELECT * FROM projects WHERE user_id = %s', (user_id,))
    data = cur.fetchall()
    result = []
    for row in data:
      row_dict = {"id":row[0], "user_id":row[1], "project_name":row[2]}
      result.append(row_dict)
    cur.close()
    db_conn.close()
    return result


# updates project_name in projects table
def update_user_projects(project_id, new_project_name):
    db_conn = connect_to_db()
    cur = db_conn.cursor()
    query = """UPDATE projects
                SET project_name = %s
                WHERE id = %s"""
    values = (new_project_name, project_id)
    cur.execute(query, values)
    db_conn.commit()
    print('Update completed')
    cur.close()
    db_conn.close()
    return project_id


# stores in db table auto_ml_projects and returns the last inserted AutoML project's id
def insert_autoML_project(project_id, training_dataset_path, model_features, target,
                          model_path, training_lib, login_datetime):
    db_conn = connect_to_db()
    cur = db_conn.cursor()
    cur.execute('INSERT INTO auto_ml_projects (project_id, training_dataset_path, model_features, target,'
                'model_path, training_lib, login_datetime) VALUES (%s, %s, %s, %s, %s, %s, %s)',
                  (project_id, training_dataset_path, model_features, target, model_path, training_lib, login_datetime))
    db_conn.commit()
    print('Insertion Done')
    last_automl_id = cur.lastrowid
    cur.close()
    db_conn.close()
    return last_automl_id


# selects stored user's autoML projects and returns them in a list of dictionaries
def select_user_autoML_projects(project_id):
    db_conn = connect_to_db()
    cur = db_conn.cursor()
    query = """
    SELECT  projects.project_name, auto_ml_projects.id, auto_ml_projects.project_id,
                auto_ml_projects.training_dataset_path,
                auto_ml_projects.model_features,
                auto_ml_projects.target,
                auto_ml_projects.model_path,
                auto_ml_projects.training_lib,
                auto_ml_projects.login_datetime
                FROM auto_ml_projects
                LEFT JOIN projects ON auto_ml_projects.project_id = projects.id
                WHERE project_id = %s

    """
    cur.execute(query, (project_id,))
    data = cur.fetchall()
    result = []
    for row in data:
      row_dict = {"project_name":row[0], "id":row[1], "project_id":row[2], "training_dataset_path":row[3],
                  "model_features":row[4], "target":row[5], "model_path":row[6], "training_lib":row[7], "login_datetime":row[8].strftime("%b %d %Y %H:%M:%S")}
      result.append(row_dict)
    cur.close()
    db_conn.close()
    return result


# stores in db table fairness_assessments and returns the last inserted fairness analysis' id
def insert_fairness_analysis(project_id, auto_ml_id, disparity_metric,
                          process_phase, sensitive_feature_group, metric_value, login_datetime, model_path):
    db_conn = connect_to_db()
    cur = db_conn.cursor()
    cur.execute('INSERT INTO fairness_assessments (project_id, auto_ml_id, disparity_metric,'
                'process_phase, sensitive_feature_group, metric_value, login_datetime, model_path) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)',
                (project_id, auto_ml_id, disparity_metric, process_phase, sensitive_feature_group, metric_value, login_datetime, model_path))
    db_conn.commit()
    print('Insertion Done')
    last_fairness_id = cur.lastrowid
    cur.close()
    db_conn.close()
    return last_fairness_id


# selects stored user's fairness analyses and returns them in a list of dictionaries
def select_user_fair_analyses(user_id):
    db_conn = connect_to_db()
    cur = db_conn.cursor()
    query = """
    SELECT  projects.project_name, fairness_assessments.id, fairness_assessments.project_id,
                fairness_assessments.auto_ml_id,
                fairness_assessments.disparity_metric,
                fairness_assessments.process_phase,
                fairness_assessments.sensitive_feature,
                fairness_assessments.metric_value,
                fairness_assessments.login_datetime,
                fairness_assessments.model_path
                FROM fairness_assessments
                LEFT JOIN projects ON fairness_assessments.project_id = projects.id
                WHERE user_id = %s

    """
    cur.execute(query, (user_id,))
    data = cur.fetchall()
    result = []
    for row in data:
      row_dict = {"project_name": row[0], "id": row[1], "project_id": row[2], "auto_ml_id": row[3],
                  "disparity_metric": row[4], "process_phase": row[5], "sensitive_feature": row[6],
                  "metric_value":row[7], "login_datetime": row[8].strftime("%b %d %Y %H:%M:%S"), "model_path":row[9]}
      result.append(row_dict)
    cur.close()
    db_conn.close()
    return result


def update_fairness_analysis(fair_id, disparity_metric, process_phase, sensitive_feature, metric_value, login_datetime, model_path):
    db_conn = connect_to_db()
    cur = db_conn.cursor()
    query = """UPDATE fairness_assessments
                SET disparity_metric = %s, process_phase = %s, sensitive_feature = %s, 
                metric_value = %s, login_datetime = %s, model_path = %s
                WHERE id = %s"""
    values = (disparity_metric, process_phase, sensitive_feature, metric_value, login_datetime, model_path, fair_id)
    cur.execute(query, values)
    db_conn.commit()
    print('Update completed')
    cur.close()
    db_conn.close()
    return fair_id
