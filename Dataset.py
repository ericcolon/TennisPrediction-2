import collections
import pickle
import statistics as s
from random import random

import numpy as np
from sklearn import preprocessing
from sklearn import svm
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sqlalchemy import create_engine
from yellowbrick.classifier import ClassificationReport

from Database import *


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


"""
def google_cloud_upload():
   storage_client = storage.Client.from_service_account_json(
       'TennisPrediction-457d9e25f643.json')
   buckets = list(storage_client.list_buckets())
   print(buckets)
   Bucket_name = 'tennismodelbucket'
    bucket = storage_client.get_bucket(bucket_name)
    source_file_name = 'Local file to upload, for example ./file.txt'
    blob = bucket.blob(os.path.basename("stats.db"))

    # Upload the local file to Cloud Storage.
    blob.upload_from_filename(source_file_name)

    print('File {} uploaded to {}.'.format(
        source_file_name,
        bucket))"""


class Dataset(object):

    def __init__(self, updated_stats):

        # Create a new pandas dataframe from the sqlite3 database we created
        conn = sqlite3.connect(updated_stats + '.db')
        dataset = pd.read_sql_query('SELECT * FROM updated_stats', conn)
        # This changes all values to numeric if sqlite3 conversion gave a string
        print(len(dataset))

        dataset = dataset.apply(pd.to_numeric,
                                errors='coerce')

        print(dataset.info())
        print(dataset.isna().sum())
        dataset.dropna(subset=['SERVEADV1'], inplace=True)  # drop invalid stats (22)
        dataset.dropna(subset=['court_type'], inplace=True)  # drop invalid stats (616)
        dataset.dropna(subset=['H21H'], inplace=True)  # drop invalid stats (7)

        print(len(dataset))

        dataset = dataset.reset_index(drop=True)  # reset indexes if any more rows are dropped
        self.dataset = dataset
        # run this first to add court types to updated_stats dataset

    def create_feature_set(self, feature_set_name, label_set_name):

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

        for i in reversed(self.dataset.index):

            print(i)
            player1_id = self.dataset.at[i, "ID1"]
            player2_id = self.dataset.at[i, "ID2"]

            # All games that two players have played
            player1_games = self.dataset.loc[
                np.logical_or(self.dataset.ID1 == player1_id, self.dataset.ID2 == player1_id)]

            player2_games = self.dataset.loc[
                np.logical_or(self.dataset.ID1 == player2_id, self.dataset.ID2 == player2_id)]

            curr_tournament = self.dataset.at[i, "ID_T"]
            current_court_id = self.dataset.at[i, "court_type"]

            # Games played earlier than the current tournament we are investigating
            earlier_games_of_p1 = [game for game in player1_games.itertuples() if
                                   game.ID_T < curr_tournament]

            earlier_games_of_p2 = [game for game in player2_games.itertuples() if
                                   game.ID_T < curr_tournament]

            # Get past opponents of both players
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
                # If they have zero common opponents, we cannot get features for this match
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
                    game.court_type] if game.ID1 == player1_id else game.SERVEADV2 * court_dict[current_court_id][
                    game.court_type] for game in player1_games_updated]

                list_of_serveadv_2 = [game.SERVEADV1 * court_dict[current_court_id][
                    game.court_type] if game.ID1 == player2_id else game.SERVEADV2 * court_dict[current_court_id][
                    game.court_type] for game in player2_games_updated]

                list_of_complete_1 = [game.COMPLETE1 * court_dict[current_court_id][
                    game.court_type] if game.ID1 == player1_id else game.COMPLETE2 * court_dict[current_court_id][
                    game.court_type] for game in player1_games_updated]

                list_of_complete_2 = [game.COMPLETE1 * court_dict[current_court_id][
                    game.court_type] if game.ID1 == player2_id else game.COMPLETE2 * court_dict[current_court_id][
                    game.court_type] for game in player2_games_updated]

                list_of_w1sp_1 = [game.W1SP1 * court_dict[current_court_id][
                    game.court_type] if game.ID1 == player1_id else game.W1SP2 * court_dict[current_court_id][
                    game.court_type] for game in player1_games_updated]

                list_of_w1sp_2 = [game.W1SP1 * court_dict[current_court_id][
                    game.court_type] if game.ID1 == player2_id else game.W1SP2 * court_dict[current_court_id][
                    game.court_type] for game in player2_games_updated]

                list_of_aces_1 = [game.ACES_1 * court_dict[current_court_id][
                    game.court_type] if game.ID1 == player1_id else (game.ACES_2) * court_dict[current_court_id][
                    game.court_type] for game in player1_games_updated]

                list_of_aces_2 = [game.ACES_1 * court_dict[current_court_id][
                    game.court_type] if game.ID1 == player2_id else game.ACES_2 * court_dict[current_court_id][
                    game.court_type] for game in player2_games_updated]

                serveadv_1 = s.mean(list_of_serveadv_1)
                serveadv_2 = s.mean(list_of_serveadv_2)
                complete_1 = s.mean(list_of_complete_1)
                complete_2 = s.mean(list_of_complete_2)
                w1sp_1 = s.mean(list_of_w1sp_1)
                w1sp_2 = s.mean(list_of_w1sp_2)
                aces_1 = s.mean(list_of_aces_1)
                aces_2 = s.mean(list_of_aces_2)

                # The first feature of our feature set is the last match on the stats dataset
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

        print("{} matches had more than 5 common opponents in the past".format(common_more_than_5))
        print("{} matches 0 common opponents in the past".format(zero_common_opponents))

        # Need to reverse the arrays !!
        # rev_X = X[::-1]
        # rev_y = Y[::-1]
        print("Time took for creating stat features for each match took--- %s seconds ---" % (time.time() - start_time))
        with open(feature_set_name, "wb") as fp:  # Pickling
            pickle.dump(X, fp)

        with open(label_set_name, "wb") as fp:  # Pickling
            pickle.dump(y, fp)

        return [X, y]

    def train_and_test_svm_model(self, model_name, dataset_name, labelset_name, dump):
        print("Training a new model {} on dataset {} and label set {}".format(model_name, dataset_name, labelset_name))

        start_time = time.time()

        pickle_in = open(dataset_name, "rb")
        X = np.asarray(pickle.load(pickle_in))
        pickle_in_2 = open(labelset_name, "rb")
        Y = np.asarray(pickle.load(pickle_in_2))

        print(X.shape)
        print(Y.shape)

        # Need the reverse the feature and labels because last match is the first feature in our array ??

        rev_X = X[::-1]
        rev_y = Y[::-1]

        X_train, X_test, y_train, y_test = train_test_split(rev_X, rev_y, test_size=0.2)

        print(len(X_train))
        print(len(X_test))

        print("Size of our first dimension is {}.".format(np.size(rev_X, 0)))
        print("Size of our second dimension is {}.".format(np.size(rev_X, 1)))

        # Split into train and test datasets. Until around tournamnet id 11364 is training
        print("The standard deviations of our features is {}.".format(np.std(rev_X, axis=0)))

        # This line standardisez a feature X by dividing it by its standard deviation.
        X_scaled = preprocessing.scale(rev_X, with_mean=False)

        # Center to the mean and component wise scale to unit variance. --> X_scaled = preprocessing.scale(rev_X)

        train_X = X_scaled[:69685]
        test_X = X_scaled[69685:]
        train_Y = rev_y[:69685]
        test_Y = rev_y[69685:]

        print(len(train_X))
        print(len(test_X))

        clf = svm.NuSVC()

        clf.fit(train_X, train_Y)
        self.test_model(clf, model_name, train_X, test_X, train_Y, test_Y, dataset_name, labelset_name)

        print("Time took for training and testing the model took--- %s seconds ---" % (time.time() - start_time))
        # now we save the model to a file if test were successfull
        if (dump):
            joblib.dump(clf, model_name)

    def test_model(self, model, modelname, X_train, X_test, y_train, y_test, dataset_name, labelset_name):

        print("Training accuracy for {} on {} and {} is: {}".format(modelname, dataset_name, labelset_name,
                                                                    model.score(X_train, y_train)))

        print("Testing accuracy for {} on {} and {} is: {}".format(modelname, dataset_name, labelset_name,
                                                                   model.score(X_test, y_test)))

    def train_test_with_visualization(self, dataset_name, labelset_name):
        pickle_in = open(dataset_name, "rb")
        X = np.asarray(pickle.load(pickle_in))
        pickle_in_2 = open(labelset_name, "rb")
        Y = np.asarray(pickle.load(pickle_in_2))

        print(X.shape)
        print(Y.shape)

        # Split into train and test datasets. Until around tournamnet id 11364 is training
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        rev_X = X[::-1]
        rev_y = Y[::-1]

        train_X = rev_X[:69685]
        test_X = rev_X[69685:]
        train_Y = rev_y[:69685]
        test_Y = rev_y[69685:]

        clf_1 = svm.NuSVC()
        visualizer = ClassificationReport(clf_1)
        visualizer.fit(train_X, train_Y)  # Fit the visualizer and the model
        visualizer.score(test_X, test_Y)  # Evaluate the model on the test data
        g = visualizer.poof()  # Draw/show/poof the data

    def test_feature_space(self):
        print("Sonuc olarak COMPLETE2 feature'ini data_v1'da yanlis yazmisiz. High success rate oradan geliyor.")
        pickle_1 = open("data_v1.txt", "rb")
        X_1 = np.asarray(pickle.load(pickle_1))
        pickle_2 = open("data_v2.txt", "rb")
        X_2 = np.asarray(pickle.load(pickle_2))
        pickle_1 = open("label_v1.txt", "rb")
        y_1 = np.asarray(pickle.load(pickle_1))
        pickle_2 = open("label_v2.txt", "rb")
        y_2 = np.asarray(pickle.load(pickle_2))

        feat1_1 = X_1[:, 0]
        feat1_2 = X_2[:, 0]

        feat2_1 = X_1[:, 1]  # Get first column of np array
        feat2_2 = X_2[:, 1]

        feat3_1 = X_1[:, 2]  # Get second column of np array
        feat3_2 = X_2[:, 2]

        feat4_1 = X_1[:, 3]
        feat4_2 = X_2[:, 3]

        print(np.absolute(feat2_1[:10]))
        print(np.absolute(feat2_2[:10]))

        assert (np.allclose(np.absolute(feat1_1), np.absolute(feat1_2)))
        print(np.allclose(np.absolute(feat2_1[:10]), np.absolute(feat2_2[:10])))
        assert (np.allclose(np.absolute(feat3_1), np.absolute(feat3_2)))
        assert (np.allclose(np.absolute(feat4_1), np.absolute(feat4_2)))

        print(np.count_nonzero(y_1))
        print(np.count_nonzero(y_2))

    def make_predictions(self, model_name, player1_id, player2_id, current_court_id, curr_tournament):
        clf = joblib.load(model_name)

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

        # All games that two players have played
        player1_games = self.dataset.loc[np.logical_or(self.dataset.ID1 == player1_id, self.dataset.ID2 == player1_id)]

        player2_games = self.dataset.loc[np.logical_or(self.dataset.ID1 == player2_id, self.dataset.ID2 == player2_id)]

        # This value should be higher than anything else curr_tournament = dataset.at[i, "ID_T"]
        earlier_games_of_p1 = [game for game in player1_games.itertuples() if
                               game.ID_T < curr_tournament]

        earlier_games_of_p2 = [game for game in player2_games.itertuples() if
                               game.ID_T < curr_tournament]

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
            print("These players have more than 5 common opponents ")

            player1_games_updated = [game for opponent in common_opponents for game in earlier_games_of_p1 if
                                     (player1_id == game.ID1 and opponent == game.ID2) or (
                                             player1_id == game.ID2 and opponent == game.ID1)]
            player2_games_updated = [game for opponent in common_opponents for game in earlier_games_of_p2 if
                                     (player2_id == game.ID1 and opponent == game.ID2) or (
                                             player2_id == game.ID2 and opponent == game.ID1)]

            list_of_serveadv_1 = [game.SERVEADV1 * court_dict[current_court_id][
                game.court_type] if game.ID1 == player1_id else game.SERVEADV2 * court_dict[current_court_id][
                game.court_type] for game in player1_games_updated]
            list_of_serveadv_2 = [game.SERVEADV1 * court_dict[current_court_id][
                game.court_type] if game.ID1 == player2_id else game.SERVEADV2 * court_dict[current_court_id][
                game.court_type] for game in player2_games_updated]

            list_of_complete_1 = [game.COMPLETE1 * court_dict[current_court_id][
                game.court_type] if game.ID1 == player1_id else game.COMPLETE2 * court_dict[current_court_id][
                game.court_type] for game in player1_games_updated]

            list_of_complete_2 = [game.COMPLETE1 * court_dict[current_court_id][
                game.court_type] if game.ID1 == player2_id else game.COMPLETE2 * court_dict[current_court_id][
                game.court_type] for game in player2_games_updated]

            list_of_w1sp_1 = [game.W1SP1 * court_dict[current_court_id][
                game.court_type] if game.ID1 == player1_id else game.W1SP2 * court_dict[current_court_id][
                game.court_type] for game in player1_games_updated]

            list_of_w1sp_2 = [game.W1SP1 * court_dict[current_court_id][
                game.court_type] if game.ID1 == player2_id else game.W1SP2 * court_dict[current_court_id][
                game.court_type] for game in player2_games_updated]

            list_of_aces_1 = [game.ACES_1 * court_dict[current_court_id][
                game.court_type] if game.ID1 == player1_id else game.ACES_2 * court_dict[current_court_id][
                game.court_type] for game in player1_games_updated]

            list_of_aces_2 = [game.ACES_1 * court_dict[current_court_id][
                game.court_type] if game.ID1 == player2_id else game.ACES_2 * court_dict[current_court_id][
                game.court_type] for game in player2_games_updated]

            serveadv_1 = s.mean(list_of_serveadv_1)
            serveadv_2 = s.mean(list_of_serveadv_2)
            complete_1 = s.mean(list_of_complete_1)
            complete_2 = s.mean(list_of_complete_2)
            w1sp_1 = s.mean(list_of_w1sp_1)
            w1sp_2 = s.mean(list_of_w1sp_2)
            aces_1 = s.mean(list_of_aces_1)
            aces_2 = s.mean(list_of_aces_2)
            feature = np.array(
                [serveadv_1 - serveadv_2, complete_1 - complete_2, w1sp_1 - w1sp_2, aces_1 - aces_2])

            print("The prediction between player {} and player {} is {}".format(player1_id, player2_id, clf.predict(
                np.array(feature).reshape(1, -1))))

        # feature_rev = np.array(
        #      [serveadv_2 - serveadv_1, complete_2 - complete_1, w1sp_2 - w1sp_1, aces_2 - aces_1])
        #  print(clf.predict(np.array(feature_rev).reshape(1, -1)))

        else:
            print("These players do not have enough common opponents to make predictions")
            return

        # DT.add_court_types()  # This should be moved to feature extraction


# Code to convert a panda dataframe into sqlite 3 database:  # df2sqlite_v2(DT.store['updated_stats'],stats)
DT = Dataset("updated_stats")

# To create the feature and label space
data_label = DT.create_feature_set('data_v3.txt', 'label_v3.txt')
print(len(data_label[0]))
print(len(data_label[1]))

# To create an SVM Model
# DT.train_and_test_svm_model("svm_model_v3.pkl", 'data_v2.txt', 'label_v2.txt', False)
# DT.train_test_with_visualization("data_v2.txt", "label_v2.txt")

# To test the model
"""DT.test_model("svm_model.pkl", "data_v1.txt", "label_v1.txt")
DT.test_model("svm_model_v2.pkl", "data_v1.txt", "label_v1.txt")
DT.test_model("svm_model.pkl", "data_v2.txt", "label_v2.txt")
DT.test_model("svm_model_v2.pkl", "data_v2.txt", "label_v2.txt")
"""

# Bug Fixing
# DT.test_feature_space()

# Visualization Part
# DT.train_test_with_visualization("data_v2.txt", "label_v2.txt")

# To make predictions on Wimbledon Day 3
"""p1_list = [19, 6101, 7459, 5837, 4061, 11704, 20193, 7806, 30470, 2123, 4454, 18094, 678, 655, 10995, 961,6465]
p2_list = [5127, 9471, 791, 9840, 1266, 9861, 22429, 12661, 26923, 5917, 1092, 27482, 685, 14177, 10828, 22087,468]

for p1, p2 in zip(p1_list, p2_list):
    DT.make_predictions("svm_model.pkl",p1, p2, 5, 15300)

"""
# To make predictions on Wimbledon Day 4
"""p1_list = [677, 875, 6465, 5992, 17829, 22434, 4067, 20847, 8806, 33502, 13447, 4009, 39309, 1113, 650, 29812]
p2_list = [9043, 29932, 468, 6458, 18017, 6081, 3833, 11003, 11522, 12043, 6648, 25543, 14727, 9521, 29939, 4010]

for p1, p2 in zip(p1_list, p2_list):
    DT.make_predictions("svm_model.pkl",p1, p2, 5, 15300)

"""
