# General Imports
import collections
import sqlite3
import statistics as s
from collections import defaultdict
from collections import Counter
from random import random
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D

import numpy as np
import time
import pickle
import pandas as pd
from sqlalchemy import create_engine
# Sklearn imports
from sklearn import preprocessing
from sklearn import svm
from sklearn import tree
from sklearn.ensemble import AdaBoostClassifier  # For Classification
from sklearn.ensemble import GradientBoostingClassifier  # For Classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_curve, auc

from sklearn.externals import joblib
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
# from sklearn.tree import DecisionTreeClassifier
# Other Classes
from OddsScraper import loads_odds_into_a_list

# For Amazon EC2 Instance
import boto3
# For R-Data conversion
import feather


def convert_dataframe_into_rdata(df):
    path = 'updated_stats.feather'
    feather.write_dataframe(df, path)


def test_final_results(result_file, odds_file, length_of_guesses):
    with open(result_file, 'rb') as handle:
        dict_of_results = pickle.load(handle)

    with open(odds_file, 'rb') as handle:
        odds = pickle.load(handle)

    print(dict_of_results)
    print(odds)
    count = 0
    correct = 0
    for match, res in dict_of_results.items():

        if (len(res) > length_of_guesses):
            most_common_result = Most_Common(res)
            print(res)
            print(res.count(most_common_result) / len(res))
            if (res.count(most_common_result) / len(res)) < 0.70:
                continue
            else:
                if match not in odds:
                    continue
                else:
                    count = count + 1
                    odd = odds[match]

                    print(
                        "Prediction for match {} was {}. The result was {}. The odds were {}.The odds we chose to bet was {}"
                            .format(match, most_common_result, odd[1], odd[3],
                                    odd[3][abs(int(most_common_result) - 1)]))
                    if most_common_result == odd[1]:
                        correct = correct + 1
    print(correct)
    print(count)
    print(correct / count)


def Most_Common(lst):
    data = Counter(lst)
    return data.most_common(1)[0][0]


# Helper function to get snapshot of basic and advanced sqlite3 databases
def sqlite3_database_extractor():
    conn = sqlite3.connect('updated_stats_v3.db')
    adv_stats = pd.read_sql_query('SELECT * FROM updated_stats_v3 LIMIT 1000', conn)
    df2sqlite_v2(adv_stats, "Adv_Stats_short")
    conn_v2 = sqlite3.connect('db.sqlite')
    basic_stats = pd.read_sql_query("SELECT * FROM stat_atp LIMIT 1000", conn_v2)
    df2sqlite_v2(basic_stats, "Basic_Stats_short")


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
    # dataframe.to_sql(db_name, disk_engine, if_exists='replace', chunksize=1000)

    """Bundan onceki !!!! Bunu unutma updated_stats V3 icin bunu yapmak daha dogru olabilir. Dont know the difference
    #     dataframe.to_sql(db_name, disk_engine ,if_exists='append')"""


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


def test_svm_model(modelname, dataset_name, labelset_name, split):
    pickle_in = open(dataset_name, "rb")
    x = np.asarray(pickle.load(pickle_in))
    pickle_in_2 = open(labelset_name, "rb")
    y = np.asarray(pickle.load(pickle_in_2))
    model = joblib.load(modelname)
    rev_X = x[::-1]
    rev_y = y[::-1]
    number_of_columns = rev_X.shape[1] - 1
    print(number_of_columns)

    # Before standardizing we want to take out the H2H column

    h2h = rev_X[:, number_of_columns]
    print(rev_X.shape)
    print(h2h.shape)

    # Delete this specific column
    rev_X = np.delete(rev_X, np.s_[-1], 1)
    print(rev_X.shape)
    print((rev_X.shape[1]))
    # Before
    # This line standardizes a feature X by dividing it by its standard deviation.
    X_scaled = preprocessing.scale(rev_X, with_mean=False)
    X_scaled = np.column_stack((X_scaled, h2h))
    print(X_scaled.shape)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, rev_y, test_size=split, shuffle=False)

    print("Training accuracy for {} on {} and {} is: {}".format(modelname, dataset_name, labelset_name,
                                                                model.score(X_train, y_train)))

    print("Testing accuracy for {} on {} and {} is: {}".format(modelname, dataset_name, labelset_name,
                                                               model.score(X_test, y_test)))


def tune_dt_parameters(x_train, y_train, x_test, y_test):
    parameter_values = [2, 4, 8, 16, 20, 32]

    train_results = []
    test_results = []
    for param in parameter_values:
        dt = tree.DecisionTreeClassifier(max_depth=param)
        dt.fit(x_train, y_train)

        train_pred = dt.predict(x_train)
        false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)
        roc_auc = auc(false_positive_rate, true_positive_rate)
        # Add auc score to previous train results
        train_results.append(roc_auc)
        y_pred = dt.predict(x_test)
        false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
        roc_auc = auc(false_positive_rate, true_positive_rate)
        # Add auc score to previous test results
        test_results.append(roc_auc)

    line1, = plt.plot(parameter_values, train_results, 'b', label="Train AUC")
    line2, = plt.plot(parameter_values, test_results, 'r', label="Test AUC")
    plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
    plt.ylabel("AUC score")
    plt.xlabel("Tree depth")
    plt.show()


def calculate_time_discount(discount_factor, current_year, game_year):
    time_since_the_match = abs(current_year - game_year)
    if time_since_the_match == 1:
        return discount_factor
    else:
        return (discount_factor ** time_since_the_match)


# Used in prediction mode of Decision Stump Model
def preprocess_features_of_predictions(features, standard_deviations):
    feat = features[~np.all(features == 0, axis=1)]  # Delete the zero np arrays
    # h2h_pred = features[:, features.shape[1] - 1]  # Save h2h feature
    features_shortened = np.delete(feat, np.s_[-1], 1)  # Delete h2h feature out from the feature set
    # features_scaled = feat / standard_deviations[None, :]  # Scale other features"
    "Uncommented this line for taking out h2h feature"
    # features_final = np.column_stack((features_scaled, h2h_pred))  # Add H2H statistics back to the mix
    return features_shortened


# Helper function to preprocess features and labels before training Decision Stump Model
def preprocess_features_before_training(features, labels):
    # 1. Reverse the feature and label set
    # 2. Scale the features to unit variance (except h2h feature). Also save the std. deviation of each feature
    # 3. Remove any duplicates (so we don't have any hard to find problems with dictionaries later
    x = features[::-1]
    y = labels[::-1]
    number_of_columns = x.shape[1] - 1

    # Before standardizing we want to take out the H2H column
    h2h = x[:, number_of_columns]

    # Delete this specific column
    x_shortened = np.delete(x, np.s_[-1], 1)

    standard_deviations = np.std(x, axis=0)

    # Center to the mean and component wise scale to unit variance.
    x_scaled = preprocessing.scale(x, with_mean=False)
    "Uncommented this line for taking out h2h feature"
    last_x = np.column_stack((x_scaled, h2h))  # Add H2H statistics back to the mix

    # We need to get rid of the duplicate values in our dataset. (Around 600 dups. Will check it later)
    x_scaled_no_duplicates, indices = np.unique(x_shortened, axis=0, return_index=True)

    y_no_duplicates = y[indices]
    return [x_scaled_no_duplicates, y_no_duplicates, standard_deviations]


class Models(object):

    # TODO try gradient boosted regression trees. link: https://www.youtube.com/watch?v=IXZKgIsZRm0&t=892s
    # TODO XGBOOST, multi layered neural network - trying to do AdaBoost gradient as well
    # TODO : Hyperparameter tuning of DT's, taking off h2h, creating feature set again, adding time discount factor,
    # TODO Increasing number of common opponents in the past. Getting some probability for predictions.
    def __init__(self, database_name):

        # Create a new pandas dataframe from the sqlite3 database we created
        conn = sqlite3.connect(database_name + '.db')

        # The name on this table should be the same as the dataframe
        dataset = pd.read_sql_query('SELECT * FROM updated_stats_v3', conn)

        # This changes all values to numeric if sqlite3 conversion gave a string
        dataset = dataset.apply(pd.to_numeric, errors='coerce')

        # Print statements to visually test some values
        dataset.dropna(subset=['SERVEADV1'], inplace=True)  # drop invalid stats (22)
        dataset.dropna(subset=['court_type'], inplace=True)  # drop invalid stats (616)
        dataset.dropna(subset=['H21H'], inplace=True)  # drop invalid stats (7)
        dataset.dropna(subset=['Number_of_games'], inplace=True)  # drop invalid stats (7)

        # Reset indexes after dropping N/A values
        dataset = dataset.reset_index(drop=True)  # reset indexes if any more rows are dropped
        dataset['year'].fillna(2018, inplace=True)
        dataset = dataset.reset_index(drop=True)  # reset indexes if any more rows are dropped
        self.dataset = dataset
        print(self.dataset.isna().sum())
        for i in (self.dataset.index):
            current_year = float(self.dataset.at[i, 'year'])
            if current_year == "":
                self.dataset.at[i, 'year'] = float(2018)
        conn_players = sqlite3.connect('atp_players.db')
        self.players = pd.read_sql_query('SELECT * FROM atp_players', conn_players)
        self.players['ID_P'] = self.players['ID_P'].apply(pd.to_numeric)

        # convert_dataframe_into_rdata(self.dataset)

        # Dictionaries for Decision Stump Model
        self.old_feature_label_dict = {}
        # A dictionary to map old_features (length 6-1D) to new_features (length 100 1-D)
        self.old_feature_to_new_feature_dictionary = defaultdict(list)
        self.new_feature_to_label_dictionary = {}
        # Only used for prediction mode
        self.predictions_old_feature_to_new_feature_dictionary = defaultdict(list)
        conn.close()
        conn_players.close()

        ec2 = boto3.resource('ec2', region_name='us-east-1')
        for instance in ec2.instances.all():
            print(instance.id, instance.state)

    def create_feature_set(self, feature_set_name, label_set_name):
        # Takes the dataset created by FeatureExtraction and calculates required features for our model.
        # As of July 17: takes 90 minutes to complete the feature creation.
        common_opponents_is_five = 0
        start_time = time.time()
        zero_common_opponents = 0
        ten_common_opponents = 0
        x = []
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
        court_dict[6][1] = float(1)  # 6 is Acyrlic
        court_dict[6][2] = 0.28
        court_dict[6][3] = 0.35
        court_dict[6][4] = 0.24
        court_dict[6][5] = 0.24
        court_dict[6][6] = float(1)

        # Bug Testing
        # code to check types of our stats dataset columns
        # with sqlite3.connect('stats.db', detect_types=sqlite3.PARSE_DECLTYPES) as conn:
        #  show_deadline(conn, list(stats))

        # Start c
        for i in reversed(self.dataset.index):

            print(i)
            player1_id = self.dataset.at[i, "ID1"]
            player2_id = self.dataset.at[i, "ID2"]
            if self.dataset.at[i, 'year'] == "":
                self.dataset.at[i, 'year'] = 2018

            # All games that two players have played
            player1_games = self.dataset.loc[
                np.logical_or(self.dataset.ID1 == player1_id, self.dataset.ID2 == player1_id)]

            player2_games = self.dataset.loc[
                np.logical_or(self.dataset.ID1 == player2_id, self.dataset.ID2 == player2_id)]

            curr_tournament = self.dataset.at[i, "ID_T"]
            current_court_id = self.dataset.at[i, "court_type"]
            current_year = float(self.dataset.at[i, 'year'])
            if current_year == "":
                current_year = float(2018)
            time_discount_factor = 0.8
            # Games played earlier than the current tournament we are investigating
            earlier_games_of_p1 = [game for game in player1_games.itertuples() if game.ID_T < curr_tournament]

            earlier_games_of_p2 = [game for game in player2_games.itertuples() if game.ID_T < curr_tournament]

            # Get past opponents of both players
            opponents_of_p1 = [games.ID2 if (player1_id == games.ID1) else games.ID1 for games in earlier_games_of_p1]
            opponents_of_p2 = [games.ID2 if (player2_id == games.ID1) else games.ID1 for games in earlier_games_of_p2]

            # We check if these two players have played before
            m = self.dataset[(self.dataset['ID1'] == player1_id) & (self.dataset['ID2'] == player2_id)].index.tolist()
            m.append(
                self.dataset[(self.dataset['ID1'] == player2_id) & (self.dataset['ID2'] == player1_id)].index.tolist())

            # If they did, we manually add player id's to their own opponent list so that intersection picks it up.
            past_match_indexes = [x for x in m if x]

            if len(past_match_indexes) > 1:
                opponents_of_p1.append(player1_id)
                opponents_of_p2.append(player2_id)
            sa = set(opponents_of_p1)
            sb = set(opponents_of_p2)

            # Find common opponents that these players have faced
            # TO DO : YOU HAVE TO ADD THE CURRENT OPPONENT IF THEY HAVE PLAYED BEFORE
            common_opponents = sa.intersection(sb)

            if len(common_opponents) < 5:
                zero_common_opponents = zero_common_opponents + 1
                # If they have zero common opponents, we cannot get features for this match
                continue

            else:
                if len(common_opponents) > 9:
                    ten_common_opponents = ten_common_opponents + 1
                common_opponents_is_five = common_opponents_is_five + 1
                # Find matches played against common opponents
                player1_games_updated = [game for opponent in common_opponents for game in earlier_games_of_p1 if
                                         (player1_id == game.ID1 and opponent == game.ID2) or (
                                                 player1_id == game.ID2 and opponent == game.ID1)]
                player2_games_updated = [game for opponent in common_opponents for game in earlier_games_of_p2 if
                                         (player2_id == game.ID1 and opponent == game.ID2) or (
                                                 player2_id == game.ID2 and opponent == game.ID1)]

                # Get the stats from those matches. Weighted by their surface matrix.
                list_of_serveadv_1 = [game.SERVEADV1 * court_dict[current_court_id][
                    game.court_type]
                                      if game.ID1 == player1_id else game.SERVEADV2 * court_dict[current_court_id][
                    game.court_type] for game in
                                      player1_games_updated]

                list_of_serveadv_2 = [game.SERVEADV1 * court_dict[current_court_id][
                    game.court_type]
                                      if game.ID1 == player2_id else game.SERVEADV2 * court_dict[current_court_id][
                    game.court_type] for game in
                                      player2_games_updated]

                list_of_complete_1 = [game.COMPLETE1 * court_dict[current_court_id][
                    game.court_type]
                                      if game.ID1 == player1_id else game.COMPLETE2 * court_dict[current_court_id][
                    game.court_type] for game in
                                      player1_games_updated]

                list_of_complete_2 = [game.COMPLETE1 * court_dict[current_court_id][
                    game.court_type]
                                      if game.ID1 == player2_id else game.COMPLETE2 * court_dict[current_court_id][
                    game.court_type] for game in
                                      player2_games_updated]

                list_of_w1sp_1 = [game.W1SP1 * court_dict[current_court_id][
                    game.court_type]
                                  if game.ID1 == player1_id else game.W1SP2 * court_dict[current_court_id][
                    game.court_type] for game in
                                  player1_games_updated]

                list_of_w1sp_2 = [game.W1SP1 * court_dict[current_court_id][
                    game.court_type]
                                  if game.ID1 == player2_id else game.W1SP2 * court_dict[current_court_id][
                    game.court_type] for game in
                                  player2_games_updated]

                list_of_breaking_points_1 = [game.BP1 * court_dict[current_court_id]
                [game.court_type]
                                             if game.ID1 == player1_id else game.BP2 * court_dict[current_court_id][
                    game.court_type] for game in
                                             player1_games_updated]

                list_of_breaking_points_2 = [game.BP1 * court_dict[current_court_id][
                    game.court_type]
                                             if game.ID1 == player2_id else game.BP2 * court_dict[current_court_id][
                    game.court_type] for game in
                                             player2_games_updated]

                # ADDED: ACES PER GAME (NOT PER MATCH)
                list_of_aces_1 = [game.ACES_1 * court_dict[current_court_id][game.court_type] / game.Number_of_games
                                  if game.ID1 == player1_id else game.ACES_2 * court_dict[current_court_id][
                    game.court_type] / game.Number_of_games
                                  for game in player1_games_updated]

                list_of_aces_2 = [game.ACES_1 * court_dict[current_court_id][game.court_type] / game.Number_of_games
                                  if game.ID1 == player2_id else game.ACES_2 * court_dict[current_court_id][
                    game.court_type] / game.Number_of_games
                                  for game in player2_games_updated]

                list_of_tpw_1 = [game.TPWP1 * court_dict[current_court_id][game.court_type] / game.Number_of_games
                                 if game.ID1 == player1_id else game.TPWP2 * court_dict[current_court_id][
                    game.court_type] / game.Number_of_games
                                 for game in player1_games_updated]

                list_of_tpw_2 = [game.TPWP1 * court_dict[current_court_id][game.court_type] / game.Number_of_games
                                 if game.ID1 == player2_id else game.TPWP2 * court_dict[current_court_id][
                    game.court_type] / game.Number_of_games
                                 for game in player2_games_updated]

                # List of head to head statistics between two players
                list_of_h2h_1 = [game.H12H * court_dict[current_court_id][game.court_type]
                                 if game.ID1 == player1_id else game.H21H * court_dict[current_court_id][
                    game.court_type] for game in
                                 player1_games_updated]

                list_of_h2h_2 = [game.H12H * court_dict[current_court_id][game.court_type]
                                 if game.ID1 == player2_id else game.H21H * court_dict[current_court_id][
                    game.court_type] for game in
                                 player2_games_updated]

                serveadv_1 = s.mean(list_of_serveadv_1)
                serveadv_2 = s.mean(list_of_serveadv_2)
                complete_1 = s.mean(list_of_complete_1)
                complete_2 = s.mean(list_of_complete_2)
                w1sp_1 = s.mean(list_of_w1sp_1)
                w1sp_2 = s.mean(list_of_w1sp_2)
                bp_1 = s.mean(list_of_breaking_points_1)
                bp_2 = s.mean(list_of_breaking_points_2)
                aces_1 = s.mean(list_of_aces_1)  # Aces per game
                aces_2 = s.mean(list_of_aces_2)
                h2h_1 = s.mean(list_of_h2h_1)
                h2h_2 = s.mean(list_of_h2h_2)
                tpw1 = s.mean(list_of_tpw_1)  # Percentage of total points won
                tpw2 = s.mean(list_of_tpw_2)

                if bp_1 == 1 or bp_1 == 0 or bp_2 == 0 or bp_2 == 1:
                    continue
                # The first feature of our feature set is the last match on the stats dataset
                if random() > 0.5:
                    # Player 1 has won. So we label it 1.
                    feature = np.array(
                        [serveadv_1, complete_1, w1sp_1, aces_1, bp_1, tpw1, serveadv_2, complete_2, w1sp_2, aces_2,
                         bp_2, tpw2, serveadv_1 - serveadv_2, complete_1 - complete_2, w1sp_1 - w1sp_2, aces_1 - aces_2,
                         bp_1 - bp_2, tpw1 - tpw2, h2h_1 - h2h_2])

                    label = 1

                    if np.any(np.isnan(feature)):
                        continue
                    else:
                        x.append(feature)
                        y.append(label)

                else:
                    # Else player 1 has lost, so we label it 0. Tht hope is that player 1 is chosen arbitrarily.
                    feature = np.array(
                        [serveadv_2, complete_2, w1sp_2, aces_2, bp_2, tpw2, serveadv_1, complete_1, w1sp_1, aces_1,
                         bp_1, tpw1, serveadv_2 - serveadv_1, complete_2 - complete_1, w1sp_2 - w1sp_1, aces_2 - aces_1,
                         bp_2 - bp_1, tpw2 - tpw1, h2h_2 - h2h_1])
                    label = 0

                    if np.any(np.isnan(feature)):
                        continue
                    else:
                        x.append(feature)
                        y.append(label)

        print("{} matches had more than 5 common opponents in the past".format(common_opponents_is_five))
        print("{} matches had more than 10 common opponents in the past".format(ten_common_opponents))
        print("{} matches 0 common opponents in the past".format(zero_common_opponents))
        print("The total number of matches in our feature set is {}".format(len(x)))
        print("Time took for creating stat features for each match took--- %s seconds ---" % (time.time() - start_time))
        with open(feature_set_name, "wb") as fp:  # Pickling
            pickle.dump(x, fp)

        with open(label_set_name, "wb") as fp:  # Pickling
            pickle.dump(y, fp)

        return [x, y]

    def train_and_test_svm_model(self, model_name, dataset_name, labelset_name, dump, split):

        print("Training a new model {} on dataset {} and label set {}".format(model_name, dataset_name, labelset_name))

        start_time = time.time()

        pickle_in = open(dataset_name, "rb")
        data = np.asarray(pickle.load(pickle_in))
        pickle_in_2 = open(labelset_name, "rb")
        label = np.asarray(pickle.load(pickle_in_2))

        print("Size of our first dimension is {}.".format(np.size(data, 0)))
        print("Size of our second dimension is {}.".format(np.size(data, 1)))
        x = data[::-1]
        y = label[::-1]
        number_of_columns = x.shape[1] - 1

        # Before standardizing we want to take out the H2H column

        h2h = x[:, number_of_columns]

        # Delete this specific column
        x = np.delete(x, np.s_[-1], 1)

        # Center to the mean and component wise scale to unit variance.
        x_scaled = preprocessing.scale(x, with_mean=False)
        # x_scaled = np.column_stack((x_scaled, h2h))
        print(x_scaled.shape)
        x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=split, shuffle=False)
        # Need the reverse the feature and labels because last match is the first feature in our array ??

        print(len(x_train))
        print(len(x_test))

        print("The standard deviations of our features is {}.".format(np.std(x_train, axis=0)))
        print("The means of our features is {}.".format(np.mean(x_train, axis=0)))

        # Create and train the model
        clf = svm.NuSVC()
        clf.fit(x_train, y_train)

        # Testing the model

        print("Training accuracy for {} on {} and {} is: {}".format(model_name, dataset_name, labelset_name,
                                                                    clf.score(x_train, y_train)))

        print("Testing accuracy for {} on {} and {} is: {}".format(model_name, dataset_name, labelset_name,
                                                                   clf.score(x_test, y_test)))

        print("Time took for training and testing the model took--- %s seconds ---" % (time.time() - start_time))

        # now we save the model to a file if test were successful
        if (dump):
            joblib.dump(clf, model_name)

    def train_adaboost_classifier(self, dataset_name, labelset_name, adaboost):
        start_time = time.time()

        pickle_in = open(dataset_name, "rb")
        features = np.asarray(pickle.load(pickle_in))
        pickle_in_2 = open(labelset_name, "rb")
        labels = np.asarray(pickle.load(pickle_in_2))
        # Preprocess the feature and label space
        x_scaled_no_duplicates, y_no_duplicates, standard_deviations = preprocess_features_before_training(features,
                                                                                                           labels)

        x_train, x_test, y_train, y_test = train_test_split(x_scaled_no_duplicates, y_no_duplicates, test_size=0.2,
                                                            shuffle=False)
        print("Size of our first dimension is {}.".format(np.size(x_scaled_no_duplicates, 0)))
        print("Size of our second dimension is {}.".format(np.size(x_scaled_no_duplicates, 1)))
        print("The number of UNIQUE features in our feature space is {}".format(len(x_scaled_no_duplicates)))
        print("New label set size must be {}.".format(len(y_no_duplicates)))
        sgd = SGDClassifier(loss="hinge", eta0=0.0001)  # create the tuned classifier

        if adaboost:
            clf = AdaBoostClassifier(n_estimators=100, base_estimator=sgd, learning_rate=1)
        else:
            clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1)
        # Above I have used decision tree as a base estimator, you can use any ML learner as base estimator if it ac# cepts sample weight
        clf.fit(x_train, y_train)

        print("AdaBoost classifier training accuracy {}.".format(clf.score(x_train, y_train)))
        print("AdaBoost classifier testing accuracy {}.".format(clf.score(x_test, y_test)))
        print("Time took to train Adaboost Classifier  was --- {} seconds ---".format(time.time() - start_time))

    def train_decision_stump_model(self, dataset_name, labelset_name, number_of_features, development_mode,
                                   prediction_mode, historical_tournament, training_mode,
                                   test_given_model, save, tournament_pickle_file_name, court_type):

        pickle_in = open(dataset_name, "rb")
        features = np.asarray(pickle.load(pickle_in))
        pickle_in_2 = open(labelset_name, "rb")
        labels = np.asarray(pickle.load(pickle_in_2))

        # Preprocess the feature and label space
        x_scaled_no_duplicates, y_no_duplicates, standard_deviations = preprocess_features_before_training(features,
                                                                                                           labels)
        print("Size of our first dimension is {}.".format(np.size(x_scaled_no_duplicates, 0)))
        print("Size of our second dimension is {}.".format(np.size(x_scaled_no_duplicates, 1)))
        print("The number of UNIQUE features in our feature space is {}".format(len(x_scaled_no_duplicates)))
        print("New label set size must be {}.".format(len(y_no_duplicates)))

        # dict of tuple (6D np array --> label)
        self.old_feature_label_dict = {tuple(x_scaled_no_duplicates[x]): y_no_duplicates[x] for x in
                                       range(len(x_scaled_no_duplicates))}

        print("Our final set includes {} features".format(len(self.old_feature_label_dict)))

        for feat in x_scaled_no_duplicates:
            assert tuple(feat) in self.old_feature_label_dict

        if development_mode:
            # This mode is used for hyperparameter optimization
            x_tr, x_test, y_tr, y_test = train_test_split(x_scaled_no_duplicates, y_no_duplicates, test_size=0.2,
                                                          shuffle=False)

            x_train, x_dev, y_train, y_dev = train_test_split(x_tr, y_tr, test_size=0.2, shuffle=False)

            print("Size of the training set is: {}.".format((len(x_train))))
            print("Size of the dev set is: {}.".format((len(x_dev))))
            print("Size of the test set is: {}.".format((len(x_test))))

            assert len(x_train) + len(x_dev) + len(x_test) == len(x_scaled_no_duplicates)

            # WARNING: BLOWING UP THE FEATURE SPACE
            self.create_100_decision_stumps(x_train, y_train, x_scaled_no_duplicates, 0.2)  # create DT stumps

            # these are the new feature and label set we will training SGD Classifier
            decision_stump_features, decision_stump_labels = self.create_new_vector_label_dataset(
                self.old_feature_to_new_feature_dictionary)

            # Testing data turned into this format
            test_data = np.asarray([self.old_feature_to_new_feature_dictionary[tuple(x)] for x in x_test])
            test_label = np.asarray([self.old_feature_label_dict[tuple(x)] for x in x_test])

            # Development data to 100 length 1d vector
            dev_data = np.asarray([self.old_feature_to_new_feature_dictionary[tuple(x)] for x in x_dev])
            dev_label = np.asarray([self.old_feature_label_dict[tuple(x)] for x in x_dev])

            tune_dt_parameters(decision_stump_features, decision_stump_labels, dev_data, dev_label)

            self.old_feature_to_new_feature_dictionary.clear()
            self.old_feature_label_dict.clear()

        if prediction_mode:
            match_to_odds_dictionary = {}
            match_to_results_dictionary = {}
            p1_list = []
            p2_list = []
            if historical_tournament:
                odds_of_wimbledon_2018 = loads_odds_into_a_list(tournament_pickle_file_name)
                # odds_of_wimbledon_2018.pop(0)  # Nadal vs del Potro UK odds are wrong
                print(odds_of_wimbledon_2018)
                self.players.ID_P = self.players.ID_P.astype(int)
                print("The initial number of games in this tournament with odds scraped is {}.".format(
                    len(odds_of_wimbledon_2018)))

                count = 0

                results = []
                for odds in reversed(odds_of_wimbledon_2018):
                    assert len(odds) == 7
                    p1 = odds[4]
                    p2 = odds[5]
                    p1_id = self.players[self.players['NAME_P'] == p1]
                    p2_id = self.players[self.players['NAME_P'] == p2]
                    result = odds[6]

                    if p1_id.empty or p2_id.empty:
                        continue
                    else:
                        count = count + 1
                        results.append(result)
                        player1_id = int(p1_id['ID_P'])
                        player2_id = int(p2_id['ID_P'])
                        p1_list.append(player1_id)
                        p2_list.append(player2_id)
                        match_to_odds_dictionary[tuple([player1_id, player2_id])] = [odds[1], odds[2]]
                        match_to_results_dictionary[tuple([player1_id, player2_id])] = result

                print("The number of matches left after scraping the ID's of players is {}.".format(count))

                for players, result in match_to_results_dictionary.copy().items():
                    p1 = list(players)[0]
                    p2 = list(players)[1]
                    p1_list.append(p2)
                    p2_list.append(p1)
                    results.append(int(abs(result - 1)))
                    match_to_results_dictionary[tuple([p2, p1])] = abs(int(abs(result - 1)))
                    odds = match_to_odds_dictionary[tuple([p1, p2])]

                    match_to_odds_dictionary[tuple([p2, p1])] = list(reversed(odds))

                assert len(p1_list) == len(p2_list) == len(results)

                print(len(match_to_results_dictionary))

                # DONT FORGET: IF PLAYERS DO NOT HAVE COMMON OPPONENTS, YOU HAVE TO DELETE THAT ENTRY FROM THE SETS

                index_games_dict_for_prediction = {
                    tuple(self.make_predictions_using_DT(p1, p2, court_type, 17000, number_of_features)): [p1, p2]
                    for
                    i, (p1, p2) in enumerate(zip(p1_list, p2_list))}

                features_from_prediction = np.asarray(
                    [np.asarray(feature) for feature in index_games_dict_for_prediction.keys()])
                print("features_from_prediction length: {}".format(len(features_from_prediction)))

                # THIS IS FOR WIMBLEDON 2018
                """
                del match_to_results_dictionary[tuple([30856, 59356])]
                del match_to_results_dictionary[tuple([10813, 63017])]
                del match_to_results_dictionary[tuple([50386, 36519])]
                del match_to_results_dictionary[tuple([9839, 63016])]
                del match_to_results_dictionary[tuple([59356, 30856])]
                del match_to_results_dictionary[tuple([63017, 10813])]
                del match_to_results_dictionary[tuple([36519, 50386])]
                del match_to_results_dictionary[tuple([63016, 9839])]
                """

                # THIS IS FOR US OPEN 2017

                """
                del match_to_results_dictionary[tuple([22056, 34511])]
                del match_to_results_dictionary[tuple([34511, 22056])]

                """
                """
                #This is for qujing Challengers
                del match_to_results_dictionary[tuple([17359, 35539])]
                del match_to_results_dictionary[tuple([45197, 28296])]
                del match_to_results_dictionary[tuple([35539, 17359])]
                del match_to_results_dictionary[tuple([28296, 45197])]
                """
                # THIS IS FOR US OPEN 2018
                """
                del match_to_results_dictionary[tuple([18495, 29171])]
                del match_to_results_dictionary[tuple([14606, 34861])]
                del match_to_results_dictionary[tuple([22428, 25919])]
                del match_to_results_dictionary[tuple([28586, 26381])]

                del match_to_results_dictionary[tuple([10901, 56846])]
                del match_to_results_dictionary[tuple([56846, 38911])]
                del match_to_results_dictionary[tuple([31392, 27082])]
                del match_to_results_dictionary[tuple([27082, 38911])]

                del match_to_results_dictionary[tuple([40609, 9831])]
                del match_to_results_dictionary[tuple([38911, 1092])]
                del match_to_results_dictionary[tuple([29171, 18495])]
                del match_to_results_dictionary[tuple([34861, 14606])]

                del match_to_results_dictionary[tuple([25919, 22428])]
                del match_to_results_dictionary[tuple([26381, 28586])]
                del match_to_results_dictionary[tuple([56846, 10901])]
                del match_to_results_dictionary[tuple([38911, 56846])]

                del match_to_results_dictionary[tuple([27082, 31392])]
                del match_to_results_dictionary[tuple([38911, 27082])]
                del match_to_results_dictionary[tuple([9831, 40609])]
                del match_to_results_dictionary[tuple([1092, 38911])]
                
                 """
                # For ATP DOHA 2017
                del match_to_results_dictionary[tuple([30470, 25708])]
                del match_to_results_dictionary[tuple([25708, 30470])]
                del index_games_dict_for_prediction[tuple(np.zeros([number_of_features, ]))]

            else:

                odds_of_tournament = loads_odds_into_a_list(tournament_pickle_file_name)
                print(odds_of_tournament)
                self.players.ID_P = self.players.ID_P.astype(int)
                print("The initial number of games in this tournament with odds scraped is {}.".format(
                    len(odds_of_tournament)))
                count = 0

                for odds in reversed(odds_of_tournament):
                    assert len(odds) == 6
                    p1 = odds[4]
                    p2 = odds[5]
                    p1_id = self.players[self.players['NAME_P'] == p1]
                    p2_id = self.players[self.players['NAME_P'] == p2]
                    if p1_id.empty or p2_id.empty:
                        continue
                    else:
                        count = count + 1
                        player1_id = int(p1_id['ID_P'])
                        player2_id = int(p2_id['ID_P'])
                        p1_list.append(player1_id)
                        p2_list.append(player2_id)
                        match_to_odds_dictionary[tuple([player1_id, player2_id])] = [odds[1], odds[2]]
                    print("The number of matches left after scraping the ID's of players is {}.".format(count))
                    assert len(p1_list) == len(p2_list)

                index_games_dict_for_prediction = {tuple(self.make_predictions_using_DT(p1, p2, court_type
                                                                                        , 17000, number_of_features)): [
                    p1, p2] for i, (p1, p2) in enumerate(zip(p1_list, p2_list))}

                features_from_prediction = np.asarray(
                    [np.asarray(feature) for feature in index_games_dict_for_prediction.keys()])
                print("features_from_prediction length: {}".format(len(features_from_prediction)))

            # Scale the features with the standard deviations of our dataset.
            features_from_prediction_final = preprocess_features_of_predictions(features_from_prediction,
                                                                                standard_deviations)

            print("index_games_dict_for_prediction length: {}".format(len(index_games_dict_for_prediction)))
            print("features_from_prediction_final length: {}".format(len(features_from_prediction_final)))
            print("match_to_results_dictionary length: {}".format(len(match_to_results_dictionary)))
            assert len(features_from_prediction_final) == len(match_to_results_dictionary) == len(
                index_games_dict_for_prediction)

            results = list(match_to_results_dictionary.values())
            temporary = features_from_prediction_final.tolist()
            # This dictionary ties last version of features to player id's
            features_match_dictionary = {tuple(temporary[m]): v for m, (k, v) in
                                         enumerate(index_games_dict_for_prediction.items())}

            # WARNING: BLOWING UP THE FEATURE SPACE
            self.create_100_decision_stumps_include_predictions(x_scaled_no_duplicates, y_no_duplicates,
                                                                x_scaled_no_duplicates, 0.2,
                                                                features_from_prediction_final)

            # these are the new feature and label set we will training SGD Classifier
            decision_stump_features, decision_stump_labels = self.create_new_vector_label_dataset(
                self.old_feature_to_new_feature_dictionary)
            print("Length of old_feature_to_new_feature_dictionary is {},".format(
                len(self.old_feature_to_new_feature_dictionary)))
            bet_amount = 10
            total_winnings = 0

            if test_given_model:
                linear_clf = joblib.load("DT_Model_3.pkl")
                linear_clf.fit(decision_stump_features, decision_stump_labels)

            else:

                linear_clf = tree.DecisionTreeClassifier(max_depth=8)
                # sgd = SGDClassifier(loss="hinge", eta0=0.0001)  # create the tuned classifier
                # linear_clf = AdaBoostClassifier(n_estimators=100, base_estimator=sgd, learning_rate=1)
                # linear_clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=8)

                # linear_clf = SGDClassifier(loss="hinge", eta0=0.01)  # create the tuned classifier
                linear_clf.fit(decision_stump_features, decision_stump_labels)

                print("Decision Tree classifier training accuracy {}.".format(
                    linear_clf.score(decision_stump_features, decision_stump_labels)))
                if historical_tournament:
                    games = np.asarray(
                        [np.asarray(feature) for feature in
                         self.predictions_old_feature_to_new_feature_dictionary.values()])
                    actual_labels = np.asarray(results)

                    print("Wimbledon 2018 Accuracy is {}.".format(linear_clf.score(games, actual_labels)))
                else:
                    pass
            if historical_tournament:
                # print(linear_clf.classes_)
                predictions = {}
                for i, (old_test_vector, new_test_vector) in enumerate(
                        self.predictions_old_feature_to_new_feature_dictionary.items()):
                    match = features_match_dictionary[tuple(old_test_vector)]
                    prediction = linear_clf.predict(np.asarray(new_test_vector).reshape(1, -1))
                    prediction_probability = linear_clf.predict_proba(np.asarray(new_test_vector).reshape(1, -1))
                    odds = match_to_odds_dictionary[tuple(match)]
                    result = results[i]

                    predictions[tuple(match)] = [prediction[0], result, np.asarray(prediction_probability), odds,
                                                 odds[abs(int(prediction) - 1)]]
                    # print(predictions)
                    print("Prediction for match {} was {}. The result was {}. The prediction probability is {}."
                          "The odds were {}.The odds we chose to bet was {}".format(match, prediction, result,
                                                                                    prediction_probability
                                                                                    , odds,
                                                                                    odds[abs(int(prediction) - 1)]))

                    if result == prediction:
                        total_winnings = total_winnings + (bet_amount * float(odds[abs(int(prediction) - 1)]))
                    else:
                        total_winnings = total_winnings - bet_amount

                    # print("Our total winnings so far is {}".format(total_winnings))

                    print(total_winnings)

                print("Total amount of bets we made is: {}".format(bet_amount * len(results)))
                print(total_winnings)
                ROI = (total_winnings - (bet_amount * len(results))) / (bet_amount * len(results)) * 100
                print("Our ROI for Wimbledon 2018 was: {}.".format(ROI))

                correct = 0
                count = 0
                winnings = 0
                result_dict = {}
                for match, data in predictions.items():
                    match = list(match)

                    id1 = match[0]
                    id2 = match[1]
                    pred = data[0]
                    result = data[1]
                    odds = data[3]
                    selected_odd = data[4]
                    probability = data[2]
                    probability = [url for l in probability for url in l]

                    match2 = predictions[tuple([id2, id1])]

                    pred2 = match2[0]
                    result2 = match2[1]
                    odds2 = match2[3]
                    selected_odd2 = match2[4]
                    probability2 = match2[2]

                    reverse_prob_2 = np.fliplr(probability2)

                    average_probability = np.mean([probability, reverse_prob_2], axis=0).tolist()

                    average_probability = [url for l in average_probability for url in l]
                    # print("The average probability for match {} was {}".format(match, average_probability))
                    higher_prob = max(average_probability)
                    if higher_prob < 0.70:
                        continue
                    else:

                        prediction = average_probability.index(higher_prob)
                        if float(odds[abs(int(prediction) - 1)]) < 1.49:
                            continue
                        else:
                            count = count + 1
                            print("Prediction for match {} was {}. The result was {}. The prediction probability is {}."
                                  "The odds were {}.The odds we chose to bet was {}".format(match, prediction, result,
                                                                                            average_probability
                                                                                            , odds,
                                                                                            odds[abs(
                                                                                                int(prediction) - 1)]))

                            if prediction == result:
                                correct = correct + 1
                                result_dict[tuple(match)] = [prediction, result, average_probability, odds]
                                winnings = winnings + (bet_amount * float(odds[abs(int(prediction) - 1)]))

                            else:
                                winnings = winnings - bet_amount
                                result_dict[tuple(match)] = [prediction, result, average_probability, odds]

                print(correct)
                print(count)
                print(correct / count)
                ROI = (winnings - (bet_amount * (count))) / (bet_amount * (count)) * 100
                print("Our ROI was: {}.".format(ROI))
                return predictions, result_dict
            if save:
                joblib.dump(linear_clf, 'DT_Model_3.pkl')

        if training_mode:

            # We want to train our model and test its training and testing accuracy
            x_train, x_test, y_train, y_test = train_test_split(x_scaled_no_duplicates, y_no_duplicates, test_size=0.2,
                                                                shuffle=False)
            print("Size of the training set is: {}.".format((len(x_train))))
            print("Size of the test set is: {}.".format((len(x_test))))

            assert len(x_train) + len(x_test) == len(x_scaled_no_duplicates)

            # WARNING: BLOWING UP THE FEATURE SPACE
            self.create_100_decision_stumps(x_train, y_train, x_scaled_no_duplicates, 0.2)  # create DT stumps

            # these are the new feature and label set we will training SGD Classifier
            decision_stump_features, decision_stump_labels = self.create_new_vector_label_dataset(
                self.old_feature_to_new_feature_dictionary)

            # creating 100 1d vectors from our test dataset
            test_data = np.asarray([self.old_feature_to_new_feature_dictionary[tuple(x)] for x in x_test])
            test_label = np.asarray([self.old_feature_label_dict[tuple(x)] for x in x_test])

            linear_clf = tree.DecisionTreeClassifier(max_depth=16)
            linear_clf.fit(decision_stump_features, decision_stump_labels)

            print("Linear SGD classifier training accuracy {}.".format(
                linear_clf.score(decision_stump_features, decision_stump_labels)))
            print("Linear SGD classifier testing accuracy {}.".format(linear_clf.score(test_data, test_label)))
            y_pred = linear_clf.predict(test_data)
            false_positive_rate, true_positive_rate, thresholds = roc_curve(test_label, y_pred)
            roc_auc = auc(false_positive_rate, true_positive_rate)
            print("ROC SCORE: {}".format(roc_auc))
            self.old_feature_to_new_feature_dictionary.clear()
            self.old_feature_label_dict.clear()
            if save:
                joblib.dump(linear_clf, 'DT_Model_99.pkl')

    def create_100_decision_stumps_include_predictions(self, x, y, whole_training_set, test_size, predictions):
        for i in range(100):
            start_time = time.time()

            data_train, data_test, labels_train, labels_test = train_test_split(x, y, test_size=test_size, shuffle=True)
            # train a Decision Stump - Classifier

            clf = tree.DecisionTreeClassifier(max_depth=16)
            # clf = KNeighborsClassifier()

            clf.fit(data_train, labels_train)
            print(clf.feature_importances_)

            for data_point in whole_training_set:
                # for each data point in the whole set, predict its label
                predicted_label = clf.predict(data_point.reshape(1, -1))

                # add the predicted label to the list which is mapped to its data_point
                # dict: tuple of 6D np array to 100D np array
                self.old_feature_to_new_feature_dictionary[tuple(data_point)].append(predicted_label[0])

            for prediction in predictions:
                predicted_label = clf.predict(prediction.reshape(1, -1))
                self.predictions_old_feature_to_new_feature_dictionary[tuple(prediction)].append(predicted_label[0])

            print("Time took to train decision stump number "
                  "{} and make predictions was --- {} seconds ---".format(i, time.time() - start_time))

    def create_100_decision_stumps(self, x, y, whole_training_set, test_size):
        # Now we create and train 100 Decision Stumps.
        # Then predict label of each data point in four fold files (560 data points) and store them in a {vector: label list} dictionary
        for i in range(100):
            start_time = time.time()

            data_train, data_test, labels_train, labels_test = train_test_split(x, y, test_size=test_size, shuffle=True)
            # train a Decision Stump - Classifier

            clf = tree.DecisionTreeClassifier(max_depth=16)
            clf.fit(data_train, labels_train)

            for data_point in whole_training_set:
                # for each data point in the whole set, predict its label
                predicted_label = clf.predict(data_point.reshape(1, -1))

                # add the predicted label to the list which is mapped to its data_point
                # dict: tuple of 6D np array to 100D np array
                self.old_feature_to_new_feature_dictionary[tuple(data_point)].append(predicted_label[0])
            print("Time took to train decision stump number {} and make predictions was --- {} seconds ---".format(i,
                                                                                                                   time.time() - start_time))

    def create_new_vector_label_dataset(self, old_new_feature_dict):

        # Functions associates new 100-1D vectors with their correct labels
        X = []
        y = []
        for old_feature, new_feature in old_new_feature_dict.items():
            X.append(list(new_feature))
            y.append(self.old_feature_label_dict[old_feature])

        X = np.array(X)
        y = np.array(y)
        return [X, y]

    def get_average_features(self, player1_id, player2_id, earlier_games_of_p1, earlier_games_of_p2, court_dict,
                             current_court_id,
                             time_discount_factor, current_year, common_opponents):
        if len(common_opponents) > 5:
            player1_games_updated = [game for opponent in common_opponents for game in earlier_games_of_p1 if
                                     (player1_id == game.ID1 and opponent == game.ID2) or (
                                             player1_id == game.ID2 and opponent == game.ID1)]
            player2_games_updated = [game for opponent in common_opponents for game in earlier_games_of_p2 if
                                     (player2_id == game.ID1 and opponent == game.ID2) or (
                                             player2_id == game.ID2 and opponent == game.ID1)]

            # Get the stats from those matches. Weighted by their surface matrix.
            list_of_serveadv_1 = [game.SERVEADV1 * court_dict[current_court_id][
                game.court_type]
                                  if game.ID1 == player1_id else game.SERVEADV2 * court_dict[current_court_id][
                game.court_type] for game in
                                  player1_games_updated]

            list_of_serveadv_2 = [game.SERVEADV1 * court_dict[current_court_id][
                game.court_type]
                                  if game.ID1 == player2_id else game.SERVEADV2 * court_dict[current_court_id][
                game.court_type] for game in
                                  player2_games_updated]

            list_of_complete_1 = [game.COMPLETE1 * court_dict[current_court_id][
                game.court_type]
                                  if game.ID1 == player1_id else game.COMPLETE2 * court_dict[current_court_id][
                game.court_type] for game in
                                  player1_games_updated]

            list_of_complete_2 = [game.COMPLETE1 * court_dict[current_court_id][
                game.court_type]
                                  if game.ID1 == player2_id else game.COMPLETE2 * court_dict[current_court_id][
                game.court_type] for game in
                                  player2_games_updated]

            list_of_w1sp_1 = [game.W1SP1 * court_dict[current_court_id][
                game.court_type]
                              if game.ID1 == player1_id else game.W1SP2 * court_dict[current_court_id][
                game.court_type] for game in
                              player1_games_updated]

            list_of_w1sp_2 = [game.W1SP1 * court_dict[current_court_id][
                game.court_type]
                              if game.ID1 == player2_id else game.W1SP2 * court_dict[current_court_id][
                game.court_type] for game in
                              player2_games_updated]

            list_of_breaking_points_1 = [game.BP1 * court_dict[current_court_id]
            [game.court_type]
                                         if game.ID1 == player1_id else game.BP2 * court_dict[current_court_id][
                game.court_type] for game in
                                         player1_games_updated]

            list_of_breaking_points_2 = [game.BP1 * court_dict[current_court_id][
                game.court_type]
                                         if game.ID1 == player2_id else game.BP2 * court_dict[current_court_id][
                game.court_type] for game in
                                         player2_games_updated]

            # ADDED: ACES PER GAME (NOT PER MATCH)
            list_of_aces_1 = [game.ACES_1 * court_dict[current_court_id][game.court_type] / game.Number_of_games
                              if game.ID1 == player1_id else game.ACES_2 * court_dict[current_court_id][
                game.court_type] / game.Number_of_games
                              for game in player1_games_updated]

            list_of_aces_2 = [game.ACES_1 * court_dict[current_court_id][game.court_type] / game.Number_of_games
                              if game.ID1 == player2_id else game.ACES_2 * court_dict[current_court_id][
                game.court_type] / game.Number_of_games
                              for game in player2_games_updated]

            list_of_tpw_1 = [game.TPWP1 * court_dict[current_court_id][game.court_type] / game.Number_of_games
                             if game.ID1 == player1_id else game.TPWP2 * court_dict[current_court_id][
                game.court_type] / game.Number_of_games
                             for game in player1_games_updated]

            list_of_tpw_2 = [game.TPWP1 * court_dict[current_court_id][game.court_type] / game.Number_of_games
                             if game.ID1 == player2_id else game.TPWP2 * court_dict[current_court_id][
                game.court_type] / game.Number_of_games
                             for game in player2_games_updated]

            # List of head to head statistics between two players
            list_of_h2h_1 = [game.H12H * court_dict[current_court_id][game.court_type]
                             if game.ID1 == player1_id else game.H21H * court_dict[current_court_id][
                game.court_type] for game in
                             player1_games_updated]

            list_of_h2h_2 = [game.H12H * court_dict[current_court_id][game.court_type]
                             if game.ID1 == player2_id else game.H21H * court_dict[current_court_id][
                game.court_type] for game in
                             player2_games_updated]

            serveadv_1 = s.mean(list_of_serveadv_1)
            serveadv_2 = s.mean(list_of_serveadv_2)
            complete_1 = s.mean(list_of_complete_1)
            complete_2 = s.mean(list_of_complete_2)
            w1sp_1 = s.mean(list_of_w1sp_1)
            w1sp_2 = s.mean(list_of_w1sp_2)
            bp_1 = s.mean(list_of_breaking_points_1)
            bp_2 = s.mean(list_of_breaking_points_2)
            aces_1 = s.mean(list_of_aces_1)
            aces_2 = s.mean(list_of_aces_2)
            h2h_1 = s.mean(list_of_h2h_1)
            h2h_2 = s.mean(list_of_h2h_2)
            tpw1 = s.mean(list_of_tpw_1)  # Percentage of total points won
            tpw2 = s.mean(list_of_tpw_2)
            return [serveadv_1, serveadv_2, complete_1, complete_2, w1sp_1, w1sp_2, bp_1, bp_2, aces_1, aces_2, h2h_1,
                    h2h_2, tpw1, tpw2]

    def make_predictions_using_DT(self, player1_id, player2_id, current_court_id, curr_tournament, number_of_features):

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

        current_year = 2018
        time_discount_factor = 0.8
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

        # We check if these two players have played before
        m = self.dataset[(self.dataset['ID1'] == player1_id) & (self.dataset['ID2'] == player2_id)].index.tolist()
        m.append(
            self.dataset[(self.dataset['ID1'] == player2_id) & (self.dataset['ID2'] == player1_id)].index.tolist())

        # If they did, we manually add player id's to their own opponent list so that intersection picks it up.
        past_match_indexes = [x for x in m if x]

        if len(past_match_indexes) > 1:
            opponents_of_p1.append(player1_id)
            opponents_of_p2.append(player2_id)
        sa = set(opponents_of_p1)
        sb = set(opponents_of_p2)

        # Find common opponents that these players have faced
        common_opponents = sa.intersection(sb)

        if len(common_opponents) > 5:

            player1_games_updated = [game for opponent in common_opponents for game in earlier_games_of_p1 if
                                     (player1_id == game.ID1 and opponent == game.ID2) or (
                                             player1_id == game.ID2 and opponent == game.ID1)]
            player2_games_updated = [game for opponent in common_opponents for game in earlier_games_of_p2 if
                                     (player2_id == game.ID1 and opponent == game.ID2) or (
                                             player2_id == game.ID2 and opponent == game.ID1)]

            # Get the stats from those matches. Weighted by their surface matrix.
            list_of_serveadv_1 = [game.SERVEADV1 * court_dict[current_court_id][
                game.court_type]
                                  if game.ID1 == player1_id else game.SERVEADV2 * court_dict[current_court_id][
                game.court_type] for game in
                                  player1_games_updated]

            list_of_serveadv_2 = [game.SERVEADV1 * court_dict[current_court_id][
                game.court_type]
                                  if game.ID1 == player2_id else game.SERVEADV2 * court_dict[current_court_id][
                game.court_type] for game in
                                  player2_games_updated]

            list_of_complete_1 = [game.COMPLETE1 * court_dict[current_court_id][
                game.court_type]
                                  if game.ID1 == player1_id else game.COMPLETE2 * court_dict[current_court_id][
                game.court_type] for game in
                                  player1_games_updated]

            list_of_complete_2 = [game.COMPLETE1 * court_dict[current_court_id][
                game.court_type]
                                  if game.ID1 == player2_id else game.COMPLETE2 * court_dict[current_court_id][
                game.court_type] for game in
                                  player2_games_updated]

            list_of_w1sp_1 = [game.W1SP1 * court_dict[current_court_id][
                game.court_type]
                              if game.ID1 == player1_id else game.W1SP2 * court_dict[current_court_id][
                game.court_type] for game in
                              player1_games_updated]

            list_of_w1sp_2 = [game.W1SP1 * court_dict[current_court_id][
                game.court_type]
                              if game.ID1 == player2_id else game.W1SP2 * court_dict[current_court_id][
                game.court_type] for game in
                              player2_games_updated]

            list_of_breaking_points_1 = [game.BP1 * court_dict[current_court_id]
            [game.court_type]
                                         if game.ID1 == player1_id else game.BP2 * court_dict[current_court_id][
                game.court_type] for game in
                                         player1_games_updated]

            list_of_breaking_points_2 = [game.BP1 * court_dict[current_court_id][
                game.court_type]
                                         if game.ID1 == player2_id else game.BP2 * court_dict[current_court_id][
                game.court_type] for game in
                                         player2_games_updated]

            # ADDED: ACES PER GAME (NOT PER MATCH)
            list_of_aces_1 = [game.ACES_1 * court_dict[current_court_id][game.court_type] / game.Number_of_games
                              if game.ID1 == player1_id else game.ACES_2 * court_dict[current_court_id][
                game.court_type] / game.Number_of_games
                              for game in player1_games_updated]

            list_of_aces_2 = [game.ACES_1 * court_dict[current_court_id][game.court_type] / game.Number_of_games
                              if game.ID1 == player2_id else game.ACES_2 * court_dict[current_court_id][
                game.court_type] / game.Number_of_games
                              for game in player2_games_updated]

            list_of_tpw_1 = [game.TPWP1 * court_dict[current_court_id][game.court_type] / game.Number_of_games
                             if game.ID1 == player1_id else game.TPWP2 * court_dict[current_court_id][
                game.court_type] / game.Number_of_games
                             for game in player1_games_updated]

            list_of_tpw_2 = [game.TPWP1 * court_dict[current_court_id][game.court_type] / game.Number_of_games
                             if game.ID1 == player2_id else game.TPWP2 * court_dict[current_court_id][
                game.court_type] / game.Number_of_games
                             for game in player2_games_updated]

            # List of head to head statistics between two players
            list_of_h2h_1 = [game.H12H * court_dict[current_court_id][game.court_type]
                             if game.ID1 == player1_id else game.H21H * court_dict[current_court_id][
                game.court_type] for game in
                             player1_games_updated]

            list_of_h2h_2 = [game.H12H * court_dict[current_court_id][game.court_type]
                             if game.ID1 == player2_id else game.H21H * court_dict[current_court_id][
                game.court_type] for game in
                             player2_games_updated]

            serveadv_1 = s.mean(list_of_serveadv_1)
            serveadv_2 = s.mean(list_of_serveadv_2)
            complete_1 = s.mean(list_of_complete_1)
            complete_2 = s.mean(list_of_complete_2)
            w1sp_1 = s.mean(list_of_w1sp_1)
            w1sp_2 = s.mean(list_of_w1sp_2)
            bp_1 = s.mean(list_of_breaking_points_1)
            bp_2 = s.mean(list_of_breaking_points_2)
            aces_1 = s.mean(list_of_aces_1)  # Aces per game
            aces_2 = s.mean(list_of_aces_2)
            h2h_1 = s.mean(list_of_h2h_1)
            h2h_2 = s.mean(list_of_h2h_2)
            tpw1 = s.mean(list_of_tpw_1)  # Percentage of total points won
            tpw2 = s.mean(list_of_tpw_2)

            if bp_1 == 1 or bp_1 == 0 or bp_2 == 0 or bp_2 == 1:
                print("After averaging breaking point conversion was 0 or 1.")
                return np.zeros([number_of_features, ])  # you have to change this number as you keep adding features!
            else:

                feature = np.array(
                    [serveadv_1, complete_1, w1sp_1, aces_1, bp_1, tpw1, serveadv_2, complete_2, w1sp_2, aces_2,
                     bp_2, tpw2, serveadv_1 - serveadv_2, complete_1 - complete_2, w1sp_1 - w1sp_2, aces_1 - aces_2,
                     bp_1 - bp_2, tpw1 - tpw2, h2h_1 - h2h_2])
                return feature
        else:
            print("The players {} and {} do not have enough common opponents to make predictions.".format(player1_id,
                                                                                                          player2_id))
            return np.zeros([number_of_features, ])


DT = Models("updated_stats_v3")  # Initalize the model class with our sqlite3 advanced stats database

# To create the feature and label space
data_label = DT.create_feature_set('data_v11.txt', 'label_v11.txt')
print(len(data_label[0]))
print(len(data_label[1]))

# To create an SVM Model
# DT.train_and_test_svm_model("svm_model_tpw_no_h2h.pkl", 'data_tpw_h2h.txt', 'label_tpw_h2h.txt', True, 0.2)
# To test the model
# test_model("svm_model_v4_h2h.pkl", "data_with_h2h.txt", "label_with_h2h.txt", 0.2)
# test_model("svm_model_v3.pkl", "data_v3.txt", "label_v3.txt", 0.2)

# To train an AdaBoost Classifier
# To train and make predictions on Decision Stump Model
# US OPEN 2018-17
"""
predictions, result_dict = DT.train_decision_stump_model('data_v10_long.txt', 'label_v10_long.txt',
                                                         number_of_features=19,
                                                         development_mode=False,
                                                         prediction_mode=True, historical_tournament=True,
                                                         save=False,
                                                         training_mode=False,
                                                         test_given_model=False,
                                                         tournament_pickle_file_name='atp_doha_2017_odds_v2.pkl',
                                                         court_type=1)
"""
# WIMBLEDON 2018
"""
predictions, result_dict = DT.train_decision_stump_model('data_v10_long.txt', 'label_v10_long.txt',
                                                         number_of_features=19,
                                                         development_mode=False,
                                                         prediction_mode=True, historical_tournament=True,
                                                         save=False,
                                                         training_mode=False,
                                                         test_given_model=False,
                                                         tournament_pickle_file_name='wimbledon_2018_odds_v2.pkl',
                                                         court_type=5)

"""
"""
# To train a model and get training and testing accuracy 
DT.train_decision_stump_model('data_v10_long.txt', 'label_v10_long.txt',
                              number_of_features=19,
                              development_mode=False,
                              prediction_mode=False, historical_tournament=True,
                              save=False,
                              training_mode=True,
                              test_given_model=False,
                              tournament_pickle_file_name='wimbledon_2018_odds_v2.pkl',
                              court_type=5)
"""
"""
start_time = time.time()
dct = defaultdict(list)

for i in range(10):
    predictions, result_dict = DT.train_decision_stump_model('data_v10_long.txt', 'label_v10_long.txt', number_of_features=19,
                                                             development_mode=False,
                                                             prediction_mode=True, historical_tournament=True,
                                                             save=False,
                                                             training_mode=False,
                                                             test_given_model=False,
                                                             tournament_pickle_file_name='tokyo_2018_oods.pkl',
                                                             court_type=1)
    if i == 5:
        with open('tokyo_2018_predictions.pickle', 'wb') as handle:
            pickle.dump(predictions, handle, protocol=pickle.HIGHEST_PROTOCOL)

    for k, v in result_dict.items():
        dct[k].append(v[0])

with open('tokyo_2018_results.pickle', 'wb') as handle:
    pickle.dump(dct, handle, protocol=pickle.HIGHEST_PROTOCOL)

print("Time took to run this whole thing was --- {} seconds ---".format(time.time() - start_time))

test_final_results('beijing_2018_results.pickle', 'beijing_2018_predictions.pickle', length_of_guesses=5)
"""
# test_final_results('result_wimbledon_2018_odds.pickle', 'odds_wimbledon_2018_odds.pickle', length_of_guesses=7)
"""
DT.train_decision_stump_model('data_v6.txt', 'label_v6.txt', number_of_features=7, development_mode=False,
                              prediction_mode=True, historical_tournament=True, save=False, training_mode=False,
                              test_given_model=False, tournament_pickle_file_name='wimbledon_2018_odds_v2.pkl',
                              court_type=5)                   
"""

# test_final_results('result_us_open.pickle', 'odds_us_open.pickle', length_of_guesses=4)
# test_final_results('result_wimbledon_2018_odds.pickle', 'odds_wimbledon_2018_odds.pickle', length_of_guesses=7)
