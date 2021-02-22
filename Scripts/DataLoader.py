import psycopg2
import pandas as pd
from datetime import datetime
from os import listdir


def date_converter(st_date):
    try:
        st_date = datetime.strptime(st_date, "%Y-%m-%d %H:%M:%S.%f %Z")
    except Exception as ex:
        st_date = datetime.strptime(st_date, "%Y-%m-%d %H:%M:%S %Z")
    return st_date


def data_inserter(file, connection):
    df = pd.read_csv(file, compression='gzip')
    query = """INSERT INTO stackoverflow.data.java_question_answer 
    (question_id, question_title, question_body,
    question_creation_date, answer_id, answer_title,
    answer_body, answer_creation_date,
    answer_accepted_answer_id)
    VALUES (%(question_id)s, %(question_title)s, %(question_body)s, %(question_creation_date)s, %(answer_id)s, %(answer_title)s, %(answer_body)s, %(answer_creation_date)s, %(answer_accepted_answer_id)s);"""
    cursor = connection.cursor()
    for row in df.iterrows():
        temp_dict = row[1].to_dict()
        temp_dict['answer_creation_date'] = date_converter(temp_dict['answer_creation_date'])
        temp_dict['question_creation_date'] = date_converter(temp_dict['question_creation_date'])
        cursor.execute(query, temp_dict)
    connection.commit()


if __name__ == "__main__":
    path = "/home/partha/Downloads/StackOverflow/"
    connection = psycopg2.connect(dbname="stackoverflow", user="partha", password="chakraborty", host='localhost')
    for file_name in sorted(listdir(path)):
        print("Starting File {}".format(file_name))
        data_inserter(path + file_name, connection)
        print("{} file inserted".format(file_name))
    connection.close()
# conn = psycopg2.connect(dbname="stackoverflow", user="partha", password="chakraborty", host='localhost')
# cur = conn.cursor()
# df = pd.read_csv("/home/partha/Downloads/StackOverflow/JAVA_QUESTION_ANSWER-000000000000.csv", compression='gzip')
# df['question_creation_date'] = pd.to_datetime(df['question_creation_date'], format="%Y-%m-%d %H:%M:%S %Z")
# query = "INSERT INTO stackoverflow (question_id, question_title, question_body, question_creation_date, answer_id, answer_title, answer_body, answer_creation_date, answer_accepted_answer_id) VALUES (%(question_id)s, %(question_title)s, %(question_body)s, %(question_creation_date)s, %(answer_id)s, %(answer_title)s, %(answer_body)s, %(answer_creation_date)s, %(answer_accepted_answer_id)s) "
# for row in df.iterrows():
#     temp_dict = row[1].to_dict()
#     try:
#         temp_dict['answer_creation_date'] = datetime.strptime(temp_dict['answer_creation_date'], "%Y-%m-%d %H:%M:%S.%f %Z")
#     except Exception as ex:
#         temp_dict['answer_creation_date'] = datetime.strptime(temp_dict['answer_creation_date'],
#                                                               "%Y-%m-%d %H:%M:%S %Z")
#     print(temp_dict)
#     break
# conn.close()

# print(listdir("/home/partha/Downloads/StackOverflow/"))
