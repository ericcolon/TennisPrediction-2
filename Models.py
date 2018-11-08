# General Imports
import collections
import pickle
import sqlite3
import statistics as s
import time
import sys
from collections import Counter
from collections import defaultdict, OrderedDict
from random import random
import math
import numpy as np
import matplotlib.pyplot as plt
# import boto3  # For Amazon EC2 Instance
import feather  # For R-Data conversion
import pandas as pd
from matplotlib.legend_handler import HandlerLine2D
# Sklearn imports
import sklearn
from sklearn import preprocessing, tree
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import ExtraTreesClassifier  # For Classification
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.model_selection import train_test_split, cross_val_score
from sqlalchemy import create_engine

# Other Classes
from NeuralNets import make_nn_predictions, NeuralNetModel


# Data viz
# from mlens.visualization import corr_X_y, corrmat

# Model evaluation
# from mlens.metrics import make_scorer
# from mlens.model_selection import Evaluator

# Ensemble
# from mlens.ensemble import SuperLearner

def read_oddsportal_data(data_file, players_file, scraping_mode, odds_length, p1_index, p2_index, result_index):
    match_to_odds_dictionary = collections.OrderedDict()
    match_to_results_dictionary = collections.OrderedDict()
    p1_list = []
    p2_list = []
    results = []
    count = 0
    if scraping_mode == 1:
        match_to_initial_odds_dictionary = collections.OrderedDict()
        for odds in reversed(data_file):
            assert len(odds) == odds_length
            p1 = odds[p1_index]
            p2 = odds[p2_index]
            p1_id = players_file[players_file['NAME_P'] == p1]
            p2_id = players_file[players_file['NAME_P'] == p2]
            result = odds[result_index]

            if p1_id.empty or p2_id.empty:
                continue
            else:
                count = count + 1
                player1_id = int(p1_id['ID_P'])
                player2_id = int(p2_id['ID_P'])
                p1_list.append(player1_id)
                p2_list.append(player2_id)
                results.append(result)
                match_to_odds_dictionary[tuple([player1_id, player2_id])] = [odds[1], odds[2]]
                match_to_initial_odds_dictionary[tuple([player1_id, player2_id])] = [odds[4], odds[5]]

                match_to_results_dictionary[tuple([player1_id, player2_id])] = result

        return [p1_list, p2_list, results, match_to_results_dictionary, match_to_odds_dictionary,
                match_to_initial_odds_dictionary]

    elif scraping_mode == 2:
        for odds in reversed(data_file):
            assert len(odds) == odds_length
            p1 = odds[p1_index]
            p2 = odds[p2_index]
            p1_id = players_file[players_file['NAME_P'] == p1]
            p2_id = players_file[players_file['NAME_P'] == p2]
            result = odds[result_index]

            if p1_id.empty or p2_id.empty:
                continue
            else:
                count = count + 1
                player1_id = int(p1_id['ID_P'])
                player2_id = int(p2_id['ID_P'])
                p1_list.append(player1_id)
                p2_list.append(player2_id)
                results.append(result)
                match_to_odds_dictionary[tuple([player1_id, player2_id])] = [odds[1], odds[2]]

                match_to_results_dictionary[tuple([player1_id, player2_id])] = result
                return [p1_list, p2_list, results, match_to_results_dictionary, match_to_odds_dictionary]
    elif scraping_mode == 3:
        for odds in reversed(data_file):
            assert len(odds) == odds_length
            p1 = odds[p1_index]
            p2 = odds[p2_index]
            p1_id = players_file[players_file['NAME_P'] == p1]
            p2_id = players_file[players_file['NAME_P'] == p2]

            if p1_id.empty or p2_id.empty:
                continue
            else:
                count = count + 1
                player1_id = int(p1_id['ID_P'])
                player2_id = int(p2_id['ID_P'])
                p1_list.append(player1_id)
                p2_list.append(player2_id)

                match_to_odds_dictionary[tuple([player1_id, player2_id])] = [odds[1], odds[2]]

                return [p1_list, p2_list, match_to_odds_dictionary]


def create_database_based_on_set(database_name, match_type, new_db_name):
    conn = sqlite3.connect(database_name + '.db')

    # The name on this table should be the same as the dataframe
    dataset = pd.read_sql_query('SELECT * FROM updated_stats_v5', conn)

    # This changes all values to numeric if sqlite3 conversion gave a string
    dataset = dataset.apply(pd.to_numeric, errors='coerce')

    # Print statements to visually test some values
    dataset.dropna(subset=['SERVEADV1'], inplace=True)  # drop invalid stats (22)
    dataset.dropna(subset=['court_type'], inplace=True)  # drop invalid stats (616)
    dataset.dropna(subset=['H21H'], inplace=True)  # drop invalid stats (7)
    dataset.dropna(subset=['Number_of_games'], inplace=True)  # drop invalid stats (7)
    dataset.dropna(subset=['Game_Spread'], inplace=True)  # drop invalid stats (7)

    # Reset indexes after dropping N/A values
    dataset = dataset.reset_index(drop=True)  # reset indexes if any more rows are dropped
    dataset['year'].fillna(2018, inplace=True)
    dataset = dataset.reset_index(drop=True)  # reset indexes if any more rows are dropped

    for i in dataset.index:
        print("We are at index {}".format(i))
        current_year = float(dataset.at[i, 'year'])
        if current_year == "":
            dataset.at[i, 'year'] = float(2018)

    # Drop 3 set matches or 5 set amtches based on match_type argument
    if match_type == 3:
        dataset = dataset.loc[dataset['2_set_wins_tournament'] == 1]
    else:
        dataset = dataset.loc[dataset['2_set_wins_tournament'] == 0]

    dataset = dataset.reset_index(drop=True)  # reset indexes if any more rows are dropped

    # Convert it into sqlite 3 database
    df2sqlite_v2(dataset, new_db_name)


def convert_dataframe_into_rdata(df, name):
    path = name
    feather.write_dataframe(df, path)


def most_common(lst):
    data = Counter(lst)
    return data.most_common(1)[0][0]


# Function to loads info from file to a list
def loads_odds_into_a_list(odds_file):
    with open(odds_file, 'rb') as f:
        data = pickle.load(f)

    updated_data = [d for d in data if 'bwin' in d]
    # for d in updated_data:
    # print(d)
    return updated_data


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
    # dataframe.to_sql(db_name, disk_engine, if_exists='append')
    dataframe.to_sql(db_name, disk_engine, if_exists='replace', chunksize=1000)

    """Bundan onceki !!!! Bunu unutma updated_stats V3 icin bunu yapmak daha dogru olabilir. Dont know the difference
    #     dataframe.to_sql(db_name, disk_engine ,if_exists='append')"""


# Check Sqlite3 database column names
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


# Tunes parameters of 100 Decision Stumps
def tune_dt_stumps_features(x, y, x_test, y_test):
    parameter_values = [0.2, 0.4, 0.5, 0.6, 0.8]

    train_results = []
    test_results = []
    dt = sklearn.tree.DecisionTreeClassifier(max_depth=4)
    for param in parameter_values:
        test_pred = []
        train_pred = []
        start_time = time.time()
        print("The parameter value is: {}".format(param))
        for i in range(100):
            data_train, data_test, labels_train, labels_test = train_test_split(x, y, test_size=param, shuffle=True)
            clf = sklearn.tree.DecisionTreeClassifier(max_depth=4)
            clf.fit(data_train, labels_train)  # train the model
            train_pred.append((clf.predict(x)))  # make predictions on train set
            test_pred.append(clf.predict(x_test))  # make predictions on test set

        new_train_pred = list(
            map(list, zip(*train_pred)))  # transpose list of lists to have (num(features),num(predictions)) shape
        new_test_pred = list(map(list, zip(*test_pred)))

        print("Train set shape after getting 100 predictions is {}".format(np.array(new_train_pred).shape))
        print("Train set shape after getting 100 predictions is {}".format(np.array(new_test_pred).shape))
        print("Time took to train decision stumps and make predictions was --- {} seconds ---".format(
            time.time() - start_time))
        dt_x_train = np.array(new_train_pred)
        dt_x_test = np.array(new_test_pred)
        dt.fit(dt_x_train, y)
        train_pred = dt.predict(dt_x_train)
        false_positive_rate, true_positive_rate, thresholds = roc_curve(y, train_pred)
        roc_auc = auc(false_positive_rate, true_positive_rate)
        # Add auc score to previous train results
        train_results.append(roc_auc)
        y_pred = dt.predict(dt_x_test)
        false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
        roc_auc = auc(false_positive_rate, true_positive_rate)
        # Add auc score to previous test results
        test_results.append(roc_auc)
    line1, = plt.plot(parameter_values, train_results, 'b', label="Decision Stumps Train AUC")
    line2, = plt.plot(parameter_values, test_results, 'r', label="Decision Stumps Test AUC")
    plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
    plt.ylabel("AUC score")
    plt.xlabel("Tree depth")
    plt.show()


# Tune max depth parameter
def tune_dt_max_depth(x_train, y_train, x_test, y_test):
    parameter_values = [2, 4, 8, 16, 20, 32]

    train_results = []
    test_results = []
    for param in parameter_values:
        dt = sklearn.tree.DecisionTreeClassifier(max_depth=param)
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


# Tune min_samples_split parameter
def tune_dt_min_samples_split(x_train, y_train, x_test, y_test):
    train_results = []
    test_results = []
    min_samples_splits = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 2, 3, 4]
    for min_samples_split in min_samples_splits:
        dt = sklearn.tree.DecisionTreeClassifier(max_depth=4, min_samples_split=min_samples_split)
        dt.fit(x_train, y_train)
        train_pred = dt.predict(x_train)
        false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)
        roc_auc = auc(false_positive_rate, true_positive_rate)
        train_results.append(roc_auc)
        y_pred = dt.predict(x_test)
        false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
        roc_auc = auc(false_positive_rate, true_positive_rate)
        test_results.append(roc_auc)
    line1, = plt.plot(min_samples_splits, train_results, 'b', label='Train AUC')
    line2, = plt.plot(min_samples_splits, test_results, 'r', label='Test AUC')
    plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
    plt.ylabel('AUC score')
    plt.xlabel('min samples split')
    plt.show()


# Tune min_samples_leaf parameter
def tune_dt_min_samples_leaf(x_train, y_train, x_test, y_test):
    min_samples_leafs = [0.1, 0.2, 0.3, 0.4, 1]
    train_results = []
    test_results = []
    for min_samples_leaf in min_samples_leafs:
        dt = sklearn.tree.DecisionTreeClassifier(max_depth=4, min_samples_leaf=min_samples_leaf)
        dt.fit(x_train, y_train)
        train_pred = dt.predict(x_train)
        false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)
        roc_auc = auc(false_positive_rate, true_positive_rate)
        train_results.append(roc_auc)
        y_pred = dt.predict(x_test)
        false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
        roc_auc = auc(false_positive_rate, true_positive_rate)
        test_results.append(roc_auc)

    line1, = plt.plot(min_samples_leafs, train_results, 'b', label='Train AUC')
    line2, = plt.plot(min_samples_leafs, test_results, 'r', label='Test AUC')
    plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
    plt.ylabel('AUC score')
    plt.xlabel('min samples leaf')
    plt.show()


# Tune max_features parameter
def tune_dt_max_features(x_train, y_train, x_test, y_test):
    max_features = list(range(1, 18))
    train_results = []
    test_results = []
    for max_feature in max_features:
        dt = sklearn.tree.DecisionTreeClassifier(max_depth=4, max_features=max_feature)
        dt.fit(x_train, y_train)
        train_pred = dt.predict(x_train)
        false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)
        roc_auc = auc(false_positive_rate, true_positive_rate)
        train_results.append(roc_auc)
        y_pred = dt.predict(x_test)
        false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
        roc_auc = auc(false_positive_rate, true_positive_rate)
        test_results.append(roc_auc)

    line1, = plt.plot(max_features, train_results, 'b', label='Train AUC')
    line2, = plt.plot(max_features, test_results, 'r', label='Test AUC')
    plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
    plt.ylabel('AUC score')
    plt.xlabel('max features')
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
    # features_shortened = np.delete(feat, np.s_[-1], 1)  # Delete h2h feature out from the feature set
    features_scaled = feat / standard_deviations[None, :]  # Scale other features"
    "Uncommented this line for taking out h2h feature"
    # features_final = np.column_stack((features_scaled, h2h_pred))  # Add H2H statistics back to the mix
    return feat


# Helper function to preprocess features and labels before training Decision Stump Model
def preprocess_features_before_training(features, labels):
    # 1. Reverse the feature and label set
    # 2. Scale the features to unit variance (except h2h feature). Also save the std. deviation of each feature
    # 3. Remove any duplicates (so we don't have any hard to find problems with dictionaries later
    x = features[::-1]
    y = labels[::-1]
    # number_of_columns = x.shape[1] - 1

    # Before standardizing we want to take out the H2H column
    # h2h = x[:, number_of_columns]

    # Delete this specific column
    x_shortened = np.delete(x, np.s_[-1], 1)

    standard_deviations = np.std(x, axis=0)

    # Center to the mean and component wise scale to unit variance.
    x_scaled = preprocessing.scale(x, with_mean=False)
    "Uncommented this line for taking out h2h feature"
    # last_x = np.column_stack((x_scaled, h2h))  # Add H2H statistics back to the mix

    # We need to get rid of the duplicate values in our dataset. (Around 600 dups. Will check it later)
    x_scaled_no_duplicates, indices = np.unique(x, axis=0, return_index=True)

    y_no_duplicates = y[indices]
    return [x_scaled_no_duplicates, y_no_duplicates, standard_deviations]


# Helper function to do n-fold CV Stacked Generalization
def Stacking(model, X, y, test, n_fold):
    seed = 1
    skf = StratifiedKFold(n_splits=n_fold, random_state=seed)
    # test_pred = np.empty((test.shape[0], 1), float)

    train_pred = np.empty((0, 1), float)
    for train_index, test_index in skf.split(X, y):
        # print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_val = X[train_index], X[test_index]
        y_train, y_val = y[train_index], y[test_index]
        sample_x_train, sample_x_test, sample_y_train, sample_y_test = train_test_split(
            X, y, test_size=0.5, shuffle=True)
        # print(len(X_train))
        model.fit(X=sample_x_train, y=sample_y_train)
        train_pred = np.append(train_pred, model.predict(X_val))
    sample_x_train, sample_x_test, sample_y_train, sample_y_test = train_test_split(
        X, y, test_size=0.5, shuffle=True)
    model.fit(X=sample_x_train, y=sample_y_train)
    test_pred = model.predict(test)
    # print(test_pred.shape)
    # print(train_pred.shape)

    return test_pred.reshape(-1, 1), train_pred


# Helper function to do n-fold CV Stacked Generalization with class probability distributions
def Stacking_with_probability(model, X, y, test, n_fold):
    seed = 7
    skf = StratifiedKFold(n_splits=n_fold, random_state=seed)

    train_pred = []

    for train_index, test_index in skf.split(X, y):
        # print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_val = X[train_index], X[test_index]  # training set = x_train, predictions made on x_val
        y_train, y_val = y[train_index], y[test_index]

        model.fit(X=X_train, y=y_train)
        train_pred.append((model.predict_proba(X_val)))  # get probability distribution of each predictions

    model.fit(X=X, y=y)  # fit the data on entire train set to get prob. distributions from test set
    test_pred = model.predict_proba(test)
    train_pred = [url for l in train_pred for url in l]

    print("Train set shape  after stacking class probability distributions is {}".format(np.array(train_pred).shape))
    print("Test set shape  after stacking class probability distributions is {}".format(test_pred.shape))

    return test_pred, np.array(train_pred)


# Helper function to train a Bagged Decision Tree Model
def Bagged_Decision_Trees(split, X, y, num_trees):
    seed = 7
    kfold = KFold(n_splits=split, random_state=seed)
    cart = sklearn.tree.DecisionTreeClassifier()
    model = BaggingClassifier(base_estimator=cart, n_estimators=num_trees, random_state=seed)
    results = cross_val_score(model, X, y, cv=kfold)
    print("Result of Bagging {} Decision Trees is {}".format(num_trees, results.mean()))


def resize_prediction_arrays(y_pred_torch, y):
    #   prob = F.softmax(y_pred_torch, dim=1)
    #  print(prob)
    y_pred_np = y_pred_torch.data.numpy()
    pred_np = np.argmax(y_pred_np, axis=1)
    pred_np = np.reshape(pred_np, len(pred_np), order='F').reshape((len(pred_np), 1))
    label_np = y.reshape(len(y), 1)
    return pred_np, label_np


class Models(object):

    def __init__(self, database_name):

        # Create a new pandas dataframe from the sqlite3 database we created
        conn = sqlite3.connect(database_name + '.db')

        # The name on this table should be the same as the dataframe
        dataset = pd.read_sql_query('SELECT * FROM updated_stats_v6', conn)

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
        # print(self.dataset.isna().sum())
        for i in self.dataset.index:
            current_year = float(self.dataset.at[i, 'year'])
            if current_year == "":
                self.dataset.at[i, 'year'] = float(2018)
        conn_players = sqlite3.connect('atp_players.db')
        self.players = pd.read_sql_query('SELECT * FROM atp_players', conn_players)
        self.players['ID_P'] = self.players['ID_P'].apply(pd.to_numeric)

        conn.close()
        conn_players.close()

        # ec2 = boto3.resource('ec2', region_name='us-east-1')
        # for instance in ec2.instances.all():
        #   print(instance.id, instance.state)

    # Creates the feature stats from advanced stats database
    def create_feature_set(self, feature_set_name, label_set_name, labeling_method):
        # Takes the dataset created by FeatureExtraction and calculates required features for our model.
        # As of October 8: takes 40-45 minutes to complete the feature creation.
        common_opponents_exist = 0
        start_time = time.time()
        zero_common_opponents = 0
        ten_common_opponents = 0
        feature_uncertainty_dict = OrderedDict()

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

        # Start
        for i in reversed(self.dataset.index):

            print("We are on index {}".format(i))
            player1_id = self.dataset.at[i, "ID1"]
            player2_id = self.dataset.at[i, "ID2"]
            if self.dataset.at[i, 'year'] == "":
                self.dataset.at[i, 'year'] = 2018

            # All games that two players have played
            player1_games = self.dataset.loc[
                np.logical_or(self.dataset.ID1 == player1_id, self.dataset.ID2 == player1_id)]

            player2_games = self.dataset.loc[
                np.logical_or(self.dataset.ID1 == player2_id, self.dataset.ID2 == player2_id)]

            # Get required information from current column
            curr_tournament = self.dataset.at[i, "ID_T"]
            current_court_id = self.dataset.at[i, "court_type"]
            current_year = float(self.dataset.at[i, 'year'])
            if current_year == "":
                current_year = float(2018)
            time_discount_factor = 0.8

            total_number_of_games = self.dataset.at[i, "Number_of_games"]
            total_number_of_sets = self.dataset.at[i, "Number_of_sets"]
            game_spread = self.dataset.at[i, "Number_of_sets"]

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
            common_opponents = sa.intersection(sb)

            if len(common_opponents) == 0:
                zero_common_opponents = zero_common_opponents + 1
                # If they have zero common opponents, we cannot get features for this match
                continue

            else:
                if len(common_opponents) > 9:
                    ten_common_opponents = ten_common_opponents + 1
                common_opponents_exist = common_opponents_exist + 1
                # Find matches played against common opponents
                player1_games_updated = [game for opponent in common_opponents for game in earlier_games_of_p1 if
                                         (player1_id == game.ID1 and opponent == game.ID2) or (
                                                 player1_id == game.ID2 and opponent == game.ID1)]
                player2_games_updated = [game for opponent in common_opponents for game in earlier_games_of_p2 if
                                         (player2_id == game.ID1 and opponent == game.ID2) or (
                                                 player2_id == game.ID2 and opponent == game.ID1)]

                weights_of_p1 = defaultdict(list)
                weights_of_p2 = defaultdict(list)
                # The below calculations are used to devise an uncertainty value for this match.
                for opponent in common_opponents:
                    for game in player1_games_updated:
                        if opponent == game.ID2 or opponent == game.ID1:
                            #   print("Current Opponent {}".format(opponent))
                            #   print("Current Court Id {}".format(current_court_id))
                            #   print("Court Id of the game Id {}".format(game.court_type))
                            #   print("Current Year {}".format(current_year))
                            #   print("Game Year {}".format(game.year))
                            #   print("Surface matrix weight {}".format(court_dict[current_court_id][game.court_type]))
                            #  print("Time Discount {}".format(
                            #    calculate_time_discount(time_discount_factor, current_year, game.year)))
                            weights_of_p1[opponent].append(
                                court_dict[current_court_id][game.court_type] * calculate_time_discount(
                                    time_discount_factor, current_year, game.year))
                for opponent in common_opponents:
                    for game in player2_games_updated:
                        if opponent == game.ID2 or opponent == game.ID1:
                            weights_of_p2[opponent].append(court_dict[current_court_id][game.court_type]
                                                           * calculate_time_discount(time_discount_factor, current_year,
                                                                                     game.year))
                sum_of_weights = 0
                for key in weights_of_p1.keys() & weights_of_p2.keys():
                    sum_of_weights = sum_of_weights + (sum(weights_of_p1[key]) * sum(weights_of_p2[key]))

                # Using all of the weighting, we devise a overall uncertainty value
                overall_uncertainty = 1 / sum_of_weights

                # Get the difference of weighted average for each feature.
                serveadv_1, serveadv_2, complete_1, complete_2, w1sp_1, w1sp_2, bp_1, bp_2, aces_1, aces_2, h2h_1, \
                h2h_2, tpw1, tpw2 = self.get_average_features(player1_id, player2_id, player1_games_updated,
                                                              player2_games_updated, court_dict, current_court_id,
                                                              time_discount_factor, current_year)

                if bp_1 == 1 or bp_1 == 0 or bp_2 == 0 or bp_2 == 1:
                    continue

                if random() > 0.5:
                    # Player 1 has won. So we label it 1.
                    feature = np.array(

                        [serveadv_1 - serveadv_2, complete_1 - complete_2, w1sp_1 - w1sp_2, aces_1 - aces_2,
                         bp_1 - bp_2, tpw1 - tpw2, h2h_1 - h2h_2])

                    # Different label methods. Options = outcome or game spread
                    if labeling_method == 'game_spread':
                        label = int(game_spread)
                    else:
                        label = 1

                    if np.any(np.isnan(feature)):
                        continue
                    else:
                        feature_uncertainty_dict[tuple(feature.tolist() + [label])] = overall_uncertainty
                        x.append(feature)
                        y.append(label)

                else:
                    # Player 2 has won. We switch Player 2 and Player 1 positions and label it 0 (Player 2 win, player 1 loss).
                    # This is necessary because our dataset has a specific format where first player always wins.

                    feature = np.array(
                        [serveadv_2 - serveadv_1, complete_2 - complete_1, w1sp_2 - w1sp_1, aces_2 - aces_1,
                         bp_2 - bp_1, tpw2 - tpw1, h2h_2 - h2h_1])

                    if labeling_method == 'game_spread':
                        label = -int(game_spread)
                    else:
                        label = 0

                    if np.any(np.isnan(feature)):
                        continue
                    else:

                        x.append(feature)
                        y.append(label)
                        feature_uncertainty_dict[tuple(feature.tolist() + [label])] = overall_uncertainty

        print("{} matches had common opponents in the past".format(common_opponents_exist))
        print("{} matches had more than 10 common opponents in the past".format(ten_common_opponents))
        print("{} matches 0 common opponents in the past".format(zero_common_opponents))
        print("The total number of matches in our feature set is {}".format(len(x)))
        print("Time took for creating stat features for each match took--- %s seconds ---" % (time.time() - start_time))

        with open(label_set_name, "wb") as fp:  # Pickling
            pickle.dump(y, fp)

        with open(feature_set_name, "wb") as fp:  # Pickling
            pickle.dump(feature_uncertainty_dict, fp)

        return [feature_uncertainty_dict, y]

    # Trains, develops and makes predictions for Stacked generalization ensemble and NN models
    def train_model(self, dataset_name, labelset_name, number_of_features, development_mode,
                    prediction_mode, historical_tournament, training_mode,
                    test_given_model, save, tournament_pickle_file_name, court_type,
                    uncertainty_used, neural_net_model_name, scraping_mode, using_neural_net=False):
        features = []
        labels = []
        fraction_of_matches = 0.4
        print("The current uncertainty threshold is: {}".format(fraction_of_matches))
        if uncertainty_used:
            print("We are currenty working with data set {} and label set {}".format(dataset_name, labelset_name))
            pickle_in = open(dataset_name, "rb")
            features_uncertainty_dict = pickle.load(pickle_in)

            # sort the dictionary by the uncertainty of matches
            sorted_dict = {r: features_uncertainty_dict[r] for r in
                           sorted(features_uncertainty_dict, key=features_uncertainty_dict.get, reverse=False)}

            threshold_uncertainty_key = list(sorted_dict)[int(math.floor(len(sorted_dict) * fraction_of_matches))]
            threshold_uncertainty = features_uncertainty_dict[threshold_uncertainty_key]
            print("Our threshold uncertainty will be {}".format(threshold_uncertainty))
            # get the required percentage of least uncertain matches
            for k, uncertainty in features_uncertainty_dict.items():
                if uncertainty < threshold_uncertainty:
                    features_and_labels = list(k)
                    features.append(np.asarray(features_and_labels[0:7]))
                    labels.append(np.asarray(features_and_labels[-1]))

            features = np.asarray(features)
            labels = np.asarray(labels)


        else:
            print("We are currenty working with data set {} and label set {}".format(dataset_name, labelset_name))
            pickle_in = open(dataset_name, "rb")
            features = np.asarray(pickle.load(pickle_in))
            pickle_in_2 = open(labelset_name, "rb")
            labels = np.asarray(pickle.load(pickle_in_2))

        # Preprocess the feature and label space
        x_scaled_no_duplicates, y_no_duplicates, standard_deviations = preprocess_features_before_training(features,
                                                                                                           labels)
        # Convert feature and label set into RDATA for further analysis

        # convert_dataframe_into_rdata(pd.DataFrame(x_scaled_no_duplicates), 'features.feather')
        # convert_dataframe_into_rdata(pd.DataFrame(y_no_duplicates), 'labels.feather')
        print("Size of our first dimension is {}.".format(np.size(x_scaled_no_duplicates, 0)))
        print("Size of our second dimension is {}.".format(np.size(x_scaled_no_duplicates, 1)))
        print("The number of UNIQUE features in our feature space is {}".format(len(x_scaled_no_duplicates)))
        print("New label set size must be {}.".format(len(y_no_duplicates)))
        print("Number of labels with label == 1 is {}".format(np.count_nonzero(y_no_duplicates)))
        print("Number of labels with label == 0 is {}".format(len(y_no_duplicates) - np.count_nonzero(y_no_duplicates)))

        # Create train and test sets for all 3 options.
        x_train, x_test, y_train, y_test = train_test_split(x_scaled_no_duplicates, y_no_duplicates, test_size=0.2,
                                                            shuffle=False)

        if development_mode:
            print("We are in development mode.")

            print("Size of the training set is: {}.".format((len(x_train))))
            print("Size of the test set is: {}.".format((len(x_test))))

            assert len(x_train) + len(x_test) == len(x_scaled_no_duplicates)

            # TUNE 100 DECISION STUMPS
            # tune_dt_stumps_features(x=x_train, y=y_train, x_test=x_test, y_test=y_test)

            decision_stump_x_train, decision_stump_x_test = self.get_decision_stump_predictions(x=x_train, y=y_train,
                                                                                                x_test=x_test,
                                                                                                test_size=0.5, depth=4)
            # WARNING: BLOWING UP THE FEATURE SPACE

            linear_clf = sklearn.tree.DecisionTreeClassifier(max_depth=4)
            self.calculate_accuracy_and_roc_score(linear_clf, decision_stump_x_train,
                                                  y_train,
                                                  decision_stump_x_test, y_test)
            tune_dt_max_depth(decision_stump_x_train, y_train, decision_stump_x_test,
                              y_test)
            tune_dt_min_samples_split(decision_stump_x_train, y_train, decision_stump_x_test,
                                      y_test)

            tune_dt_min_samples_leaf(decision_stump_x_train, y_train, decision_stump_x_test,
                                     y_test)
            tune_dt_max_features(decision_stump_x_train, y_train, decision_stump_x_test,
                                 y_test)

        if prediction_mode:
            print('We are in prediction mode.')

            match_uncertainty_dict = collections.OrderedDict()

            prediction_threshold = 90
            print("We are investigating {}.".format(tournament_pickle_file_name))
            odds_file = loads_odds_into_a_list(tournament_pickle_file_name)
            print(odds_file)
            self.players.ID_P = self.players.ID_P.astype(int)
            print("The initial number of games in this tournament with odds scraped is {}.".format(
                len(odds_file)))

            count = 0

            if scraping_mode == 1:  # When he have additional opening odds)
                p1_list, p2_list, results, match_to_results_dictionary, match_to_odds_dictionary, match_to_initial_odds_dictionary = read_oddsportal_data(
                    odds_file, self.players, scraping_mode, 9, 6, 7, 8)

            elif scraping_mode == 2:  # When we only have the final odds
                p1_list, p2_list, results, match_to_results_dictionary, match_to_odds_dictionary = read_oddsportal_data(
                    odds_file, self.players, scraping_mode, 7, 4, 5, 6)
                match_to_initial_odds_dictionary = OrderedDict()

            else:  # Mode where we have no results and no opening odds. Scraping Mode 3
                p1_list, p2_list, match_to_odds_dictionary = read_oddsportal_data(odds_file, self.players,
                                                                                  scraping_mode, 6, 4, 5, 0)
                match_to_initial_odds_dictionary = OrderedDict()
                match_to_results_dictionary = OrderedDict()

            print("The number of matches left after scraping the ID's of players is {}.".format(count))
            print(len(match_to_initial_odds_dictionary))
            # Get the list of [match, calculated feature, calculated uncertainty] for each match
            match_features_uncertainty_list = [
                self.create_prediction_features(p1, p2, court_type, 20000, number_of_features) for i, (p1, p2) in
                enumerate(zip(p1_list, p2_list))]
            print('When the features from prediction is first created, its size is {}'.format(
                len(match_features_uncertainty_list)))

            for i, (match, feature, uncertainty) in enumerate(match_features_uncertainty_list):
                if uncertainty > prediction_threshold:
                    del match_to_results_dictionary[match]
                    del match_to_odds_dictionary[match]
                    del match_to_initial_odds_dictionary[match]

                elif np.array_equal(feature, np.zeros([number_of_features, ])):
                    del match_to_results_dictionary[match]
                    del match_to_odds_dictionary[match]
                    del match_to_initial_odds_dictionary[match]

                else:
                    match_uncertainty_dict[match] = uncertainty
            print("Ater uncertainty threshold, remaining number of matches is {}".format(
                len(match_to_results_dictionary)))

            features_from_prediction = np.asarray([element[1] for element in match_features_uncertainty_list if
                                                   element[0] in list(match_to_results_dictionary.keys())])

            print("features_from_prediction length: {}".format(len(features_from_prediction)))

            # Scale the features with the standard deviations of our dataset.
            features_from_prediction_final = preprocess_features_of_predictions(features_from_prediction,
                                                                                standard_deviations)

            # Because we might have taken some results of the list
            if scraping_mode is not 3:
                final_results = np.asarray([result for match, result in match_to_results_dictionary.items()])
            else:
                final_results = np.zeros(5)

            print("The Final Number of features: {}".format(len(features_from_prediction_final)))
            print("match_to_results_dictionary length: {}".format(len(match_to_results_dictionary)))
            print("Number of results: {}".format(len(final_results)))
            print("match_to_odds_dictionary length: {}".format(len(match_to_odds_dictionary)))
            print("match to initial odds dictionary length {}".format(len(match_to_initial_odds_dictionary)))
            print("match_uncertainty_dict length: {}".format(len(match_uncertainty_dict)))

            # Sanity check
            if scraping_mode == 1:
                assert len(features_from_prediction_final) == len(match_to_results_dictionary) == len(
                    final_results) == len(match_to_odds_dictionary) == len(match_uncertainty_dict) == len(
                    match_to_initial_odds_dictionary)
            elif scraping_mode == 2:
                assert len(features_from_prediction_final) == len(match_to_results_dictionary) == len(
                    final_results) == len(match_to_odds_dictionary) == len(match_uncertainty_dict)
            else:
                assert len(features_from_prediction_final) == len(match_to_odds_dictionary) == len(
                    match_uncertainty_dict)

            if using_neural_net:
                linear_clf = make_nn_predictions(neural_net_model_name, tournament_pickle_file_name,
                                                 x_scaled_no_duplicates, y_no_duplicates,
                                                 features_from_prediction_final, final_results,
                                                 match_to_results_dictionary, match_to_odds_dictionary, self.players,
                                                 match_uncertainty_dict, match_to_initial_odds_dictionary,
                                                 scraping_mode)

            else:
                # WARNING: BLOWING UP THE FEATURE SPACE
                dt_x_train, dt_x_test = self.get_decision_stump_predictions(x=x_scaled_no_duplicates, y=y_no_duplicates,
                                                                            x_test=features_from_prediction_final,
                                                                            test_size=0.5, depth=4)
                linear_clf = tree.DecisionTreeClassifier(max_depth=4)
                # linear_clf = ExtraTreesClassifier(n_estimators=20)
                linear_clf.fit(dt_x_train, y_no_duplicates)

                if historical_tournament:

                    self.calculate_accuracy_and_roc_score(linear_clf, dt_x_train,
                                                          y_no_duplicates,
                                                          dt_x_test, final_results)
                else:
                    pass
                if historical_tournament:

                    bet_amount = 10
                    total_winnings = 0
                    count = 0
                    correct = 0
                    predictions_dict = {}
                    match_to_results_list = list(match_to_results_dictionary.items())  # get list of matches

                    # USING 100 DECISION STUMP MODEL
                    for i, (feature, result) in enumerate(zip(dt_x_test, final_results)):

                        prediction = linear_clf.predict(feature.reshape(1, -1))
                        prediction_probability = linear_clf.predict_proba(feature.reshape(1, -1))

                        match = match_to_results_list[i][0]  # can do this because everything is ordered
                        odds = match_to_odds_dictionary[tuple(match)]

                        predictions_dict[tuple(match)] = [prediction[0], result, np.asarray(prediction_probability),
                                                          odds,
                                                          odds[abs(int(prediction) - 1)]]
                        # print(predictions)
                        print("Prediction for match {} was {}. The result was {}. The prediction probability is {}."
                              "The odds were {}.The odds we chose to bet was {}".format(match, prediction, result,
                                                                                        prediction_probability
                                                                                        , odds,
                                                                                        odds[abs(int(prediction) - 1)]))
                        max_probability = np.amax(prediction_probability)

                        if max_probability > 0.65:
                            print("Max probability is {}".format(max_probability))
                            if float(odds[abs(int(prediction) - 1)]) > 1.3:
                                print("The odds we selected was {}".format(odds[abs(int(prediction) - 1)]))

                                if result == prediction:

                                    if abs(float(odds[abs(int(prediction) - 1)])) < 20:
                                        correct = correct + 1
                                        count = count + 1
                                        total_winnings = total_winnings + (
                                                bet_amount * float(odds[abs(int(prediction) - 1)]))
                                else:
                                    count = count + 1
                                    total_winnings = total_winnings - bet_amount

                        print("Our total winnings so far is {}".format(total_winnings))

                    print("Total amount of bets we made is: {}".format(bet_amount * count))
                    print("Total Winnings: {}".format(total_winnings))
                    ROI = (total_winnings - (bet_amount * count)) / (bet_amount * count) * 100
                    print("Our ROI for {} was: {}.".format(tournament_pickle_file_name, ROI))
                    print("Accuracy over max probability and with odd threshold is {}.".format(correct / count))

                    result_dict = {}

                    # if prediction == result:
                    #     correct = correct + 1
                    #    result_dict[tuple(match)] = [prediction, result, average_probability, odds]
                    #    if abs(float(odds[abs(int(prediction) - 1)])) < 20:
                    #        winnings = winnings + (bet_amount * float(odds[abs(int(prediction) - 1)]))

                    #   else:
                    #     winnings = winnings - bet_amount
                    #     result_dict[tuple(match)] = [prediction, result, average_probability, odds]

                    return predictions_dict, result_dict
            if save:
                joblib.dump(linear_clf, 'DT_Model_3.pkl')

        if training_mode:
            print("We are in training mode")
            print("Neural Net Model")
            NeuralNetModel(x_scaled_no_duplicates, y_no_duplicates.reshape(-1), batchsize=128, dev_set_size=0.4,
                           threshold=str(fraction_of_matches), text_file=False)

            sys.exit()
            # Bagged_Decision_Trees(5, x_scaled_no_duplicates, y_no_duplicates, 10)

            linear_clf = sklearn.tree.DecisionTreeClassifier(max_depth=4)

            # linear_clf = ExtraTreesClassifier(n_estimators=10)

            seed = 7

            # We want to train our model and test its training and testing accuracy
            print("Size of the training set is: {}.".format((len(x_train))))
            print("Size of the test set is: {}.".format((len(x_test))))

            # Try the Decision Stump with class probabilities as features. Get num_iterations * 2 class probability features  for each item training trees by randomly sampling %50

            # WARNING: BLOWING UP THE FEATURE SPACE
            dt_prob_x_train, dt_prob_x_test = self.get_decision_stump_class_probabilities(x_train=x_train,
                                                                                          y_train=y_train,
                                                                                          x_test=x_test, test_size=0.5,
                                                                                          depth=4,
                                                                                          num_iterations=20)

            assert (len(dt_prob_x_train) == len(y_train))
            assert (len(dt_prob_x_test) == len(x_test))

            self.calculate_accuracy_and_roc_score(linear_clf, dt_prob_x_train,
                                                  y_train,
                                                  dt_prob_x_test, y_test)
            # Try the Decision Stump Model. Get 100 predictions for each item training trees by randomly sampling %50

            dt_x_train, dt_x_test = self.get_decision_stump_predictions(x=x_train, y=y_train,
                                                                        x_test=x_test, test_size=0.5, depth=4)

            # Check if x and y overlap 1-1
            assert (len(dt_x_train) == len(y_train))
            assert (len(dt_x_test) == len(x_test))

            # Get training and test accuracy and ROC Score
            self.calculate_accuracy_and_roc_score(linear_clf, dt_x_train,
                                                  y_train,
                                                  dt_x_test, y_test)

            # self.calculate_accuracy_over_threshold(linear_clf, dt_x_train,
            #                                      y_train,
            #                                     dt_x_test, y_test)

            # Adding class probabilities from different algorithms to our feature set ( LEVEL 0)
            x_train_df, x_test_df = self.add_class_probabilities_to_features(
                ExtraTreesClassifier(n_estimators=20),
                x_train, x_test, y_train)

            x_train_df, x_test_df = self.add_class_probabilities_to_features(
                KNeighborsClassifier(n_neighbors=5), x_train_df.values, x_test_df.values, y_train)

            x_train_df, x_test_df = self.add_class_probabilities_to_features(
                BaggingClassifier(base_estimator=tree.DecisionTreeClassifier(max_depth=4), n_estimators=20,
                                  random_state=seed), x_train_df.values, x_test_df.values, y_train)
            nn_train = pd.concat([x_train_df, x_test_df], axis=0)
            nn_labels = pd.concat([pd.DataFrame(y_train), pd.DataFrame(y_test)], axis=0)

            print(nn_train.values.shape)
            print(nn_labels.values.shape)

            # We stack them and run another model on top of it
            self.calculate_accuracy_and_roc_score(LogisticRegression(random_state=0, solver='lbfgs'), x_train_df.values,
                                                  y_train,
                                                  x_test_df.values, y_test)
            #

            # corr_X_y(x_train, y_train, figsize=(16, 10), label_rotation=80, hspace=1, fontsize=14)

            #  ensemble = SuperLearner()
            # ensemble.add(tree.DecisionTreeClassifier(max_depth=8),
            #              BaggingClassifier(base_estimator=tree.DecisionTreeClassifier(), n_estimators=100,
            #                               random_state=1))
            # ensemble.add_meta(ExtraTreesClassifier(n_estimators=100))
            # y_pred = ensemble.fit(x_scaled_no_duplicates, y_no_duplicates).predict(x_scaled_no_duplicates)
            # print("Prediction score: %.3f" % accuracy_score(y_pred, y_no_duplicates))

            if save:
                joblib.dump(linear_clf, 'DT_Model_99.pkl')

    # adds class probabilities to original feature set
    def add_class_probabilities_to_features(self, model, x_train, x_test, y_train):

        x_train_df = pd.DataFrame(x_train)
        x_test_df = pd.DataFrame(x_test)
        test_pred1, train_pred1 = Stacking_with_probability(model, n_fold=10, X=x_train,
                                                            test=x_test,
                                                            y=y_train)

        x_train_df = pd.concat([x_train_df, pd.DataFrame(train_pred1[:, 0])], axis=1)
        x_test_df = pd.concat([x_test_df, pd.DataFrame(test_pred1[:, 0])], axis=1)

        # print("Train dataframe shape after stacking is {}.".format(x_train_df.shape))
        # print("Test dataframe shape after stacking is {}.".format(x_test_df.shape))

        x_train_df = pd.concat([x_train_df, pd.DataFrame(train_pred1[:, 1])], axis=1)
        x_test_df = pd.concat([x_test_df, pd.DataFrame(test_pred1[:, 1])], axis=1)

        #  print("Train dataframe shape after stacking is {}.".format(x_train_df.shape))
        # print("Test dataframe shape after stacking is {}.".format(x_test_df.shape))
        return [x_train_df, x_test_df]

    # Calculates train and test accuracy with ROC scores
    def calculate_accuracy_and_roc_score(self, linear_clf, train_pred, y_train, test_pred, y_test):
        print("These scores are for model {}.".format(linear_clf))
        linear_clf.fit(train_pred, y_train)

        assert len(train_pred) == len(y_train)
        assert len(y_test) == len(test_pred)

        # print("Feature Importances {}".format(linear_clf.feature_importances_))
        print("Training Accuracy {}".format(linear_clf.score(train_pred, y_train)))
        print("Testing Accuracy {}".format(linear_clf.score(test_pred, y_test)))

        y_train_pred = linear_clf.predict(train_pred)
        false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, y_train_pred)
        roc_auc = auc(false_positive_rate, true_positive_rate)
        print("ROC SCORE for training is : {}".format(roc_auc))

        y_test_pred = linear_clf.predict(test_pred)
        false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_test_pred)
        roc_auc = auc(false_positive_rate, true_positive_rate)
        print("ROC SCORE for testing is : {}".format(roc_auc))

    # Calculates accuracies but only takes account over a certain probability threshold
    def calculate_accuracy_over_threshold(self, linear_clf, train_pred, y_train, test_pred, y_test):
        linear_clf.fit(train_pred, y_train)

        for test_point, label in zip(test_pred, y_test):
            print("Our test point is {} and the label is {}".format(test_point, label))
            predicted_label = linear_clf.predict(test_point.reshape(1, -1))
            prediction_probability = linear_clf.predict_proba(test_point.reshape(1, -1))
            print("Predicted label was {}".format(predicted_label[0]))
            print("Prediction probability is was {}".format(prediction_probability))

    # Creates the meta level class probability distribution features
    def get_decision_stump_class_probabilities(self, x_train, y_train, x_test, test_size, depth, num_iterations):

        x_train_df = pd.DataFrame(np.zeros((len(x_train), num_iterations * 2)), dtype=float)
        x_test_df = pd.DataFrame(np.zeros((len(x_test), num_iterations * 2)), dtype=float)

        print("100 decision stump depth is {}".format(depth))
        col = 0  # column number to fill in dataframes
        start_time = time.time()
        for i in range(num_iterations):
            data_train, data_test, labels_train, labels_test = train_test_split(x_train, y_train, test_size=test_size,
                                                                                shuffle=True)
            clf = sklearn.tree.DecisionTreeClassifier(max_depth=depth)
            clf.fit(data_train, labels_train)  # train the model
            train_pred = []
            test_pred = []
            train_pred.append(clf.predict_proba(x_train))  # get probability distribution of each prediction
            test_pred.append(clf.predict_proba(x_test))
            train_pred_np = np.array([url for l in train_pred for url in l])
            test_pred_np = np.array([url for l in test_pred for url in l])

            x_train_df[col] = train_pred_np[:, 0]  # Populate the dataframes with first probability
            x_test_df[col] = test_pred_np[:, 0]
            col = col + 1
            x_train_df[col] = train_pred_np[:, 1]  # Populate the dataframes with second probability
            x_test_df[col] = test_pred_np[:, 1]
            col = col + 1

        print("Train pred shape {}".format(x_train_df.shape))
        print("Test pred shape {}".format(x_test_df.shape))
        print("Time took to create decision stumps with class probabilities was --- {} seconds ---".format(
            time.time() - start_time))
        return [x_train_df, x_test_df]

    # Creates the meta level prediction features
    def get_decision_stump_predictions(self, x, y, x_test, test_size, depth):
        print("Running 100 iterations for Decision Stump Predictions")
        test_pred = []
        train_pred = []
        start_time = time.time()
        print("100 decision stump depth is {}".format(depth))
        for i in range(100):
            data_train, data_test, labels_train, labels_test = train_test_split(x, y, test_size=test_size, shuffle=True)
            clf = sklearn.tree.DecisionTreeClassifier(max_depth=depth)
            clf.fit(data_train, labels_train)  # train the model
            train_pred.append((clf.predict(x)))  # make predictions on train set
            test_pred.append(clf.predict(x_test))  # make predictions on test set

        new_train_pred = list(
            map(list, zip(*train_pred)))  # transpose list of lists to have (num(features),num(predictions)) shape
        new_test_pred = list(map(list, zip(*test_pred)))

        print("Train set shape after getting 100 predictions is {}".format(np.array(new_train_pred).shape))
        print("Train set shape after getting 100 predictions is {}".format(np.array(new_test_pred).shape))
        print("Time took to train decision stumps and make predictions was --- {} seconds ---".format(
            time.time() - start_time))
        return [np.array(new_train_pred), np.array(new_test_pred)]

    def get_average_features(self, player1_id, player2_id, player1_games_updated, player2_games_updated, court_dict,
                             current_court_id, time_discount_factor, current_year):
        list_of_serveadv_1 = [game.SERVEADV1 * court_dict[current_court_id][
            game.court_type] * calculate_time_discount(time_discount_factor, current_year, game.year)
                              if game.ID1 == player1_id else game.SERVEADV2 * court_dict[current_court_id][
            game.court_type] * calculate_time_discount(time_discount_factor, current_year, game.year) for game
                              in
                              player1_games_updated]

        list_of_serveadv_2 = [game.SERVEADV1 * court_dict[current_court_id][
            game.court_type] * calculate_time_discount(time_discount_factor, current_year, game.year)
                              if game.ID1 == player2_id else game.SERVEADV2 * court_dict[current_court_id][
            game.court_type] * calculate_time_discount(time_discount_factor, current_year, game.year) for game
                              in
                              player2_games_updated]

        list_of_complete_1 = [game.COMPLETE1 * court_dict[current_court_id][
            game.court_type] * calculate_time_discount(time_discount_factor, current_year, game.year)
                              if game.ID1 == player1_id else game.COMPLETE2 * court_dict[current_court_id][
            game.court_type] * calculate_time_discount(time_discount_factor, current_year, game.year) for game
                              in
                              player1_games_updated]

        list_of_complete_2 = [game.COMPLETE1 * court_dict[current_court_id][
            game.court_type] * calculate_time_discount(time_discount_factor, current_year, game.year)
                              if game.ID1 == player2_id else game.COMPLETE2 * court_dict[current_court_id][
            game.court_type] * calculate_time_discount(time_discount_factor, current_year, game.year) for game
                              in
                              player2_games_updated]

        list_of_w1sp_1 = [game.W1SP1 * court_dict[current_court_id][
            game.court_type] * calculate_time_discount(time_discount_factor, current_year, game.year)
                          if game.ID1 == player1_id else game.W1SP2 * court_dict[current_court_id][
            game.court_type] * calculate_time_discount(time_discount_factor, current_year, game.year) for game
                          in
                          player1_games_updated]

        list_of_w1sp_2 = [game.W1SP1 * court_dict[current_court_id][
            game.court_type] * calculate_time_discount(time_discount_factor, current_year, game.year)
                          if game.ID1 == player2_id else game.W1SP2 * court_dict[current_court_id][
            game.court_type] * calculate_time_discount(time_discount_factor, current_year, game.year) for game
                          in
                          player2_games_updated]

        list_of_breaking_points_1 = [game.BP1 * court_dict[current_court_id]
        [game.court_type] * calculate_time_discount(time_discount_factor, current_year, game.year)
                                     if game.ID1 == player1_id else game.BP2 * court_dict[current_court_id][
            game.court_type] * calculate_time_discount(time_discount_factor, current_year, game.year) for game
                                     in
                                     player1_games_updated]

        list_of_breaking_points_2 = [game.BP1 * court_dict[current_court_id][
            game.court_type] * calculate_time_discount(time_discount_factor, current_year, game.year)
                                     if game.ID1 == player2_id else game.BP2 * court_dict[current_court_id][
            game.court_type] * calculate_time_discount(time_discount_factor, current_year, game.year) for game
                                     in
                                     player2_games_updated]

        # ADDED: ACES PER GAME (NOT PER MATCH)
        list_of_aces_1 = [game.ACES_1 * court_dict[current_court_id][game.court_type]
                          * calculate_time_discount(time_discount_factor, current_year,
                                                    game.year) / game.Number_of_games
                          if game.ID1 == player1_id else game.ACES_2 * court_dict[current_court_id][
            game.court_type] * calculate_time_discount(time_discount_factor, current_year,
                                                       game.year) / game.Number_of_games
                          for game in player1_games_updated]

        list_of_aces_2 = [game.ACES_1 * court_dict[current_court_id][game.court_type]
                          * calculate_time_discount(time_discount_factor, current_year,
                                                    game.year) / game.Number_of_games
                          if game.ID1 == player2_id else game.ACES_2 * court_dict[current_court_id][
            game.court_type] * calculate_time_discount(time_discount_factor, current_year,
                                                       game.year) / game.Number_of_games
                          for game in player2_games_updated]

        list_of_tpw_1 = [game.TPWP1 * court_dict[current_court_id][game.court_type]
                         * calculate_time_discount(time_discount_factor, current_year,
                                                   game.year) / game.Number_of_games
                         if game.ID1 == player1_id else game.TPWP2 * court_dict[current_court_id][
            game.court_type] * calculate_time_discount(time_discount_factor, current_year,
                                                       game.year) / game.Number_of_games
                         for game in player1_games_updated]

        list_of_tpw_2 = [game.TPWP1 * court_dict[current_court_id][game.court_type]
                         * calculate_time_discount(time_discount_factor, current_year,
                                                   game.year) / game.Number_of_games
                         if game.ID1 == player2_id else game.TPWP2 * court_dict[current_court_id][
            game.court_type] * calculate_time_discount(time_discount_factor, current_year,
                                                       game.year) / game.Number_of_games
                         for game in player2_games_updated]

        # List of head to head statistics between two players
        list_of_h2h_1 = [game.H12H * court_dict[current_court_id][game.court_type]
                         * calculate_time_discount(time_discount_factor, current_year, game.year)
                         if game.ID1 == player1_id else game.H21H * court_dict[current_court_id][
            game.court_type] * calculate_time_discount(time_discount_factor, current_year, game.year) for game
                         in
                         player1_games_updated]

        list_of_h2h_2 = [game.H12H * court_dict[current_court_id][game.court_type]
                         * calculate_time_discount(time_discount_factor, current_year, game.year)
                         if game.ID1 == player2_id else game.H21H * court_dict[current_court_id][
            game.court_type] * calculate_time_discount(time_discount_factor, current_year, game.year) for game
                         in
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
        return [serveadv_1, serveadv_2, complete_1, complete_2, w1sp_1, w1sp_2, bp_1, bp_2, aces_1, aces_2, h2h_1,
                h2h_2, tpw1, tpw2]

    # Creates features for unseen data points
    def create_prediction_features(self, player1_id, player2_id, current_court_id, curr_tournament, number_of_features):

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

        if len(common_opponents) > 0:

            player1_games_updated = [game for opponent in common_opponents for game in earlier_games_of_p1 if
                                     (player1_id == game.ID1 and opponent == game.ID2) or (
                                             player1_id == game.ID2 and opponent == game.ID1)]
            player2_games_updated = [game for opponent in common_opponents for game in earlier_games_of_p2 if
                                     (player2_id == game.ID1 and opponent == game.ID2) or (
                                             player2_id == game.ID2 and opponent == game.ID1)]

            weights_of_p1 = defaultdict(list)
            weights_of_p2 = defaultdict(list)
            for opponent in common_opponents:
                for game in player1_games_updated:
                    if opponent == game.ID2 or opponent == game.ID1:
                        #   print("Current Opponent {}".format(opponent))
                        #   print("Current Court Id {}".format(current_court_id))
                        #   print("Court Id of the game Id {}".format(game.court_type))
                        #   print("Current Year {}".format(current_year))
                        #   print("Game Year {}".format(game.year))
                        #   print("Surface matrix weight {}".format(court_dict[current_court_id][game.court_type]))
                        #  print("Time Discount {}".format(
                        #    calculate_time_discount(time_discount_factor, current_year, game.year)))
                        weights_of_p1[opponent].append(court_dict[current_court_id][
                                                           game.court_type] * calculate_time_discount(
                            time_discount_factor, current_year, game.year))
            for opponent in common_opponents:
                for game in player2_games_updated:
                    if opponent == game.ID2 or opponent == game.ID1:
                        #   print("Current Opponent {}".format(opponent))
                        #   print("Current Court Id {}".format(current_court_id))
                        #   print("Court Id of the game Id {}".format(game.court_type))
                        #   print("Current Year {}".format(current_year))
                        #   print("Game Year {}".format(game.year))
                        #   print("Surface matrix weight {}".format(court_dict[current_court_id][game.court_type]))
                        #  print("Time Discount {}".format(
                        #    calculate_time_discount(time_discount_factor, current_year, game.year)))
                        weights_of_p2[opponent].append(court_dict[current_court_id][
                                                           game.court_type] * calculate_time_discount(
                            time_discount_factor, current_year, game.year))
            sum_of_weights = 0
            for key in weights_of_p1.keys() & weights_of_p2.keys():
                # print(key, weights_of_p1[key], weights_of_p2[key])
                sum_of_weights = sum_of_weights + (sum(weights_of_p1[key]) * sum(weights_of_p2[key]))
            overall_uncertainty = 1 / sum_of_weights

            serveadv_1, serveadv_2, complete_1, complete_2, w1sp_1, w1sp_2, bp_1, bp_2, aces_1, aces_2, h2h_1, \
            h2h_2, tpw1, tpw2 = self.get_average_features(player1_id, player2_id, player1_games_updated,
                                                          player2_games_updated, court_dict, current_court_id,
                                                          time_discount_factor, current_year)
            # Get the stats from those matches. Weighted by their surface matrix.
            # Get the stats from those matches. Weighted by their surface matrix.

            if bp_1 == 1 or bp_1 == 0 or bp_2 == 0 or bp_2 == 1:
                print("After averaging breaking point conversion was 0 or 1.")

                # Element 1 = player ids
                # Element 2 = calculated feature
                # Element 3 = overall uncertainty
                return [tuple([player1_id, player2_id]), np.zeros([number_of_features, ]), overall_uncertainty]


            else:

                feature = np.array(
                    [serveadv_1 - serveadv_2, complete_1 - complete_2, w1sp_1 - w1sp_2, aces_1 - aces_2,
                     bp_1 - bp_2, tpw1 - tpw2, h2h_1 - h2h_2])
                return [tuple([player1_id, player2_id]), feature, overall_uncertainty]
        else:
            print("The players {} and {} do not have enough common opponents to make predictions.".format(player1_id,
                                                                                                          player2_id))
            return [tuple([player1_id, player2_id]), np.zeros([number_of_features, ]), 10000]


DT = Models("updated_stats_v6")  # Initalize the model class with our sqlite3 advanced stats database

# To create the feature and label space

# data_label = DT.create_feature_set('uncertainty_dict_v15_h2h_update_game_spread.txt', 'label_v15_h2h_update_game_spread.txt',
#                                   labeling_method="game_spread")


DT.train_model('uncertainty_dict_14.txt', 'label_v12_short.txt',
               number_of_features=7,
               development_mode=False,
               prediction_mode=True, historical_tournament=True,
               save=False,
               training_mode=True,
               test_given_model=False,
               tournament_pickle_file_name='atp_finals_nov11-12.pkl',
               court_type=1, uncertainty_used=True, neural_net_model_name='ckpt.pth04adam05.tar', scraping_mode=3,
               using_neural_net=True)

# WIMBLEDON 2018
"""
predictions, result_dict = DT.train_decision_stump_model('data_v12_short.txt', 'label_v12_short.txt',
                                                         number_of_features=7,
                                                         development_mode=False,
                                                         prediction_mode=False, historical_tournament=True,
                                                         save=False,
                                                         training_mode=True,
                                                         test_given_model=False,
                                                         tournament_pickle_file_name='us_open_2018_odds.pkl',
                                                         court_type=1)
"""
"""
# To train a model and get training and testing accuracy 
DT.train_decision_stump_model('data_v12.txt', 'label_v12.txt',
                              number_of_features=19,
                              development_mode=False,
                              prediction_mode=True, historical_tournament=True,
                              save=False,
                              training_mode=False,
                              test_given_model=False,
                              tournament_pickle_file_name='us_open_2017_odds_v2.pkl',
                              court_type=1)


"""
"""
# Hyperparameter Tuning for DT Model
DT.train_decision_stump_model('data_v12.txt', 'label_v12.txt',
                              number_of_features=19,
                              development_mode=False,
                              prediction_mode=True, historical_tournament=True,
                              save=False,
                              training_mode=False,
                              test_given_model=False,
                              tournament_pickle_file_name='us_open_2018_odds.pkl',
                              court_type=1)
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


# test_final_results('result_us_open.pickle', 'odds_us_open.pickle', length_of_guesses=4)
# test_final_results('result_wimbledon_2018_odds.pickle', 'odds_wimbledon_2018_odds.pickle', length_of_guesses=7)
