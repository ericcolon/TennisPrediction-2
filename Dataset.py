import collections
from Database import *
import sqlite3
import numpy as np
import ast
from sqlalchemy import create_engine
import pickle
from google.cloud import storage
import statistics as s
from random import random


# Helper method to parse strings into integers and floats
def parse_str(s):
    try:
        return ast.literal_eval(str(s))
    except:
        return


# Methods to convert pandas dataframe into sqlite3 database
def df2sqlite(dataframe, db_name="import.sqlite", tbl_name="import"):
    conn = sqlite3.connect(db_name)
    cur = conn.cursor()

    wildcards = ','.join(['?'] * len(dataframe.columns))
    data = [tuple(x) for x in dataframe.values]

    cur.execute("drop table if exists %s" % tbl_name)

    col_str = '"' + '","'.join(dataframe.columns) + '"'
    cur.execute("create table %s (%s)" % (tbl_name, col_str))

    cur.executemany("insert into %s values(%s)" % (tbl_name, wildcards), data)

    conn.commit()
    conn.close()


def df2sqlite_v2(dataframe, db_name):
    disk_engine = create_engine('sqlite:///' + db_name + '.db')
    dataframe.to_sql(db_name, disk_engine, if_exists='append')


def show_deadline(conn, column_list):
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    sql = "SELECT * FROM stats"
    cursor.execute(sql)
    row = cursor.fetchone()
    for col in column_list:
        print('  column:', col)
        print('    value :', row[col])
        print('    type  :', type(row[col]))
    return


def google_cloud_upload():
    storage_client = storage.Client.from_service_account_json(
        'TennisPrediction-457d9e25f643.json')
    buckets = list(storage_client.list_buckets())
    print(buckets)
    """bucket_name = 'tennismodelbucket'
    bucket = storage_client.get_bucket(bucket_name)
    source_file_name = 'Local file to upload, for example ./file.txt'
    blob = bucket.blob(os.path.basename("stats.db"))

    # Upload the local file to Cloud Storage.
    blob.upload_from_filename(source_file_name)

    print('File {} uploaded to {}.'.format(
        source_file_name,
        bucket))"""


class Dataset(object):

    def __init__(self, database):
        self.db = Database(database)
        # self.store = pd.HDFStore('storage.h5')

        self.X = []
        self.y = []
        self.tournaments = self.db.get_tournaments()
        self.matches = self.db.get_matches()

    # run this first to add court types to updated_stats dataset
    def add_court_types(self):
        stats = self.store['updated_stats']  # uncomment this line
        start_time = time.time()
        del stats['court_type']
        stats["court_type"] = ""
        court_id_none = 0
        for i in (stats.index):
            print(i)
            tournament_id = stats.at[i, "ID_T"]
            tournament = self.tournaments.loc[self.tournaments['ID_T'] == tournament_id]  # Find the tournament
            if tournament.empty:
                court_id_none = court_id_none + 1
                continue
            else:
                court_id = float(tournament['ID_C_T'])  # casting it as a float
                print(court_id)
                stats.at[i, 'court_type'] = str(court_id)

        print(court_id_none)
        print("ali")
        print("stop")
        print("Time took for adding court ids to stats took--- %s seconds ---" % (time.time() - start_time))

        self.store.put('updated_stats', stats, format='table', data_columns=True)

    def create_feature_set(self):

        common_more_than_5 = 0
        start_time = time.time()
        zero_common_opponents = 0
        X = []
        y = []
        court_dict = collections.defaultdict(dict)
        court_dict[1][1] = float(1)  # 1 is Hardcourt
        court_dict[1][2] = 0.28
        court_dict[1][3] = 0.35
        court_dict[1][4] = 0.24
        court_dict[1][5] = 0.24
        court_dict[1][6] = float(1)

        court_dict[2][1] = 0.28  # 2 is Clay
        court_dict[2][2] = float(1)
        court_dict[2][3] = 0.31
        court_dict[2][4] = 0.14
        court_dict[2][5] = 0.14
        court_dict[2][6] = 0.28
        court_dict[3][1] = 0.35  # 3 is Indoor
        court_dict[3][2] = 0.31
        court_dict[3][3] = float(1)
        court_dict[3][4] = 0.25
        court_dict[3][5] = 0.25
        court_dict[3][6] = 0.35
        court_dict[4][1] = 0.24  # 4 is carpet
        court_dict[4][2] = 0.14
        court_dict[4][3] = 0.25
        court_dict[4][4] = float(1)
        court_dict[4][5] = float(1)
        court_dict[4][6] = 0.24
        court_dict[5][1] = 0.24  # 5 is Grass
        court_dict[5][2] = 0.14
        court_dict[5][3] = 0.25
        court_dict[5][4] = float(1)
        court_dict[5][5] = float(1)
        court_dict[5][6] = 0.24
        court_dict[6][1] = float(1)  # 1 is Acyrlic
        court_dict[6][2] = 0.28
        court_dict[6][3] = 0.35
        court_dict[6][4] = 0.24
        court_dict[6][5] = 0.24
        court_dict[6][6] = float(1)

        ## Bug Testing

        # code to check types of our stats dataset columns
        # with sqlite3.connect('stats.db', detect_types=sqlite3.PARSE_DECLTYPES) as conn:
        #  show_deadline(conn, list(stats))

        # Create a new pandas dataframe from the sqlite3 database we created
        conn = sqlite3.connect('stats.db')
        dataset = pd.read_sql_query('SELECT * FROM stats', conn)  # atp players list

        # This changes all values to numeric if sqlite3 conversion gave a string
        dataset = dataset.apply(pd.to_numeric,
                                errors='coerce')
        print(len(dataset))
        print(dataset.isnull().sum(axis=0))
        print(dataset.info())
        print(dataset.dtypes)
        dataset.dropna(subset=['SERVEADV1'], inplace=True)  # drop invalid stats (63)
        print(len(dataset))
        dataset = dataset.reset_index(drop=True)  # reset indexes if any more rows are dropped
        for i in reversed(dataset.index):

            print(i)
            player1_id = dataset.at[i, "ID1"]
            player2_id = dataset.at[i, "ID2"]

            # All games that two players have played
            player1_games = dataset.loc[np.logical_or(dataset.ID1 == player1_id, dataset.ID2 == player1_id)]

            player2_games = dataset.loc[np.logical_or(dataset.ID1 == player2_id, dataset.ID2 == player2_id)]

            curr_tournament = dataset.at[i, "ID_T"]
            current_court_id = dataset.at[i, "court_type"]

            #  print(type(player1_games))

            # ondition1 = (df.col1 == 10) & (df.col2 <= 15)
            #   condition2 = (df.col3 == 7) & (df.col4 >= 4)
            #  # at this point, condition1 and condition2 are vectors of bools
            # df1 = df[condition1]"""
            # [list(x) for x in dt.T.itertuples()]

            # Games played earlier than the current tournament we are investigating
            earlier_games_of_p1 = [game for game in player1_games.itertuples() if
                                   game.ID_T < curr_tournament]

            earlier_games_of_p2 = [game for game in player2_games.itertuples() if
                                   game.ID_T < curr_tournament]
            # print(type(earlier_games_of_p1[0]))

            # print("ali")
            #    print("stop")

            opponents_of_p1 = [
                games.ID2 if (player1_id == games.ID1) else
                games.ID1 for games in earlier_games_of_p1]

            opponents_of_p2 = [
                games.ID2 if (player2_id == games.ID1) else
                games.ID1 for games in earlier_games_of_p2]

            sa = set(opponents_of_p1)
            sb = set(opponents_of_p2)

            # Find common opponents that these players have faced
            common_opponents = sa.intersection(sb)

            if len(common_opponents) > 5:
                common_more_than_5 = common_more_than_5 + 1

            if len(common_opponents) == 0:
                zero_common_opponents = zero_common_opponents + 1
                continue

            else:

                player1_games_updated = [game for opponent in common_opponents for game in earlier_games_of_p1 if
                                         (player1_id == game.ID1 and opponent == game.ID2) or (
                                                 player1_id == game.ID2 and opponent == game.ID1)]
                player2_games_updated = [game for opponent in common_opponents for game in earlier_games_of_p2 if
                                         (player2_id == game.ID1 and opponent == game.ID2) or (
                                                 player2_id == game.ID2 and opponent == game.ID1)]

                # Common opponent matches are taken. The features are calculated with their respective weights (from court surface matrix)

                # TO DO -- ACES should be divided to their per game average..

                list_of_serveadv_1 = [game.SERVEADV1 * court_dict[current_court_id][
                    game.court_type] if game.ID1 == player1_id
                                      else game.SERVEADV2 * court_dict[current_court_id][
                    game.court_type]
                                      for game in player1_games_updated]
                list_of_serveadv_2 = [game.SERVEADV1 * court_dict[current_court_id][
                    game.court_type] if game.ID1 == player2_id else game.SERVEADV2 *
                                                                    court_dict[current_court_id][
                                                                        game.court_type]
                                      for game in player2_games_updated]

                list_of_complete_1 = [game.COMPLETE1 * court_dict[current_court_id][
                    game.court_type] if game.ID1 == player1_id else game.COMPLETE2 *
                                                                    court_dict[current_court_id][
                                                                        game.court_type]
                                      for game in player1_games_updated]
                list_of_complete_2 = [game.COMPLETE1 if game.ID1 == player2_id else game.COMPLETE2
                                      for game in player2_games_updated]

                list_of_w1sp_1 = [game.W1SP1 * court_dict[current_court_id][
                    game.court_type] if game.ID1 == player1_id else game.W1SP2 *
                                                                    court_dict[current_court_id][
                                                                        game.court_type]
                                  for game in player1_games_updated]
                list_of_w1sp_2 = [game.W1SP1 * court_dict[current_court_id][
                    game.court_type] if game.ID1 == player2_id else game.W1SP2 *
                                                                    court_dict[current_court_id][
                                                                        game.court_type]
                                  for game in player2_games_updated]

                list_of_aces_1 = [(game.ACES_1) * court_dict[current_court_id][
                    ((game.court_type))] if game.ID1 == player1_id else (game.ACES_2) *
                                                                        court_dict[current_court_id][
                                                                            ((game.court_type))]
                                  for game in player1_games_updated]

                list_of_aces_2 = [game.ACES_1 * court_dict[current_court_id][
                    game.court_type] if game.ID1 == player2_id else game.ACES_2 *
                                                                    court_dict[current_court_id][
                                                                        game.court_type]
                                  for game in player2_games_updated]

                serveadv_1 = s.mean(list_of_serveadv_1)
                serveadv_2 = s.mean(list_of_serveadv_2)
                complete_1 = s.mean(list_of_complete_1)
                complete_2 = s.mean(list_of_complete_2)
                w1sp_1 = s.mean(list_of_w1sp_1)
                w1sp_2 = s.mean(list_of_w1sp_2)
                aces_1 = s.mean(list_of_aces_1)
                aces_2 = s.mean(list_of_aces_2)

                if (random() > 0.5):
                    # Player 1 has won. So we label it 1.
                    feature = np.array(
                        [serveadv_1 - serveadv_2, complete_1 - complete_2, w1sp_1 - w1sp_2, aces_1 - aces_2])
                    label = 1
                    X.append(feature)
                    y.append(label)

                else:
                    feature = np.array(
                        [serveadv_2 - serveadv_1, complete_2 - complete_1, w1sp_2 - w1sp_1, aces_2 - aces_1])
                    label = 0
                    X.append(feature)
                    y.append(label)

        print(common_more_than_5)
        print(zero_common_opponents)

        print("Time took for creating stat features for each match took--- %s seconds ---" % (time.time() - start_time))
        with open("data.txt", "wb") as fp:  # Pickling
            pickle.dump(X, fp)

        with open("label.txt", "wb") as fp:  # Pickling
            pickle.dump(y, fp)

        return [X, y]

    def train_model(self):
        pass

    def test_model(self):
        pass


DT = Dataset("db.sqlite")
# DT.add_court_types()  # This should be moved to feature extraction
# Code to convert a panda dataframe into sqlite 3 database:  # df2sqlite_v2(DT.store['updated_stats'],stats)
data_label = DT.create_feature_set()
print(len(data_label[0]))
print(len(data_label[1]))

# google_cloud_upload()
