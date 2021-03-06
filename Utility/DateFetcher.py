import psycopg2


class DataGenerator:

    def __get_new_connection(self):
        self.connection = psycopg2.connect(dbname="stackoverflow", user="partha", password="chakraborty",
                                           host='localhost')

    def __get_new_cursor(self):
        self.cursor = self.connection.cursor()
        self.cursor.itersize = 5000

    def __convert_to_dict(self, row, columns):
        temp = dict()
        for name, value in zip(columns, row):
            temp[name.name] = value
        return temp

    def __init__(self):
        self.connection = psycopg2.connect(dbname="stackoverflow", user="partha", password="chakraborty",
                                           host='localhost')
        self.cursor = self.connection.cursor()
        self.cursor.itersize = 5000

    def is_alive(self):
        try:
            self.cursor.execute('SELECT 1')
            return True
        except psycopg2.OperationalError:
            return False

    def get_all_data(self, limit=None):
        if limit is None:
            query = "SELECT * FROM stackoverflow.data.java_question_answer ORDER BY question_creation_date DESC;"
        else:
            query = "SELECT * FROM stackoverflow.data.java_question_answer ORDER BY question_creation_date DESC LIMIT {};".format(
                limit)
        return self.get_query_answer(query=query)

    def get_data_with_codes(self, limit=None):
        if limit is None:
            query = "SELECT * FROM stackoverflow.data.java_question_answer WHERE  answer_body like '%<code>%</code>%' ORDER BY question_creation_date DESC;"
        else:
            query = "SELECT * FROM stackoverflow.data.java_question_answer  WHERE  answer_body like '%<code>%</code>%' ORDER BY question_creation_date DESC LIMIT {};".format(
                limit)
        return self.get_query_answer(query=query)

    def get_query_answer(self, query):
        if self.is_alive():
            self.cursor.execute(query)
            column_names = self.cursor.description
            for row in self.cursor.fetchall():
                yield self.__convert_to_dict(row, column_names)
                if not self.is_alive():
                    self.__get_new_connection()
                    self.__get_new_cursor()

    def get_accepted_codes(self, limit=None):
        if limit is None:
            query = "SELECT * FROM stackoverflow.data.java_question_answer WHERE  answer_id answer_body like '%<code>%</code>%' and answer_id=accepted_answer_id ORDER BY question_creation_date DESC;"
        else:
            query = "SELECT * FROM stackoverflow.data.java_question_answer  WHERE  answer_body like '%<code>%</code>%' and answer_id=accepted_answer_id ORDER BY question_creation_date DESC LIMIT {};".format(
                limit)
        return self.get_query_answer(query=query)


if __name__ == "__main__":
    generator = DataGenerator()
    for item in generator.get_accepted_codes(limit=100):
        print(item)
