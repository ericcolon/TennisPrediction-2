import pandas as pd
import numpy as np
import time


class Dataset(object):

    def __init__(self, database):
        # feature_extraction = FeatureExtraction(database)
        # feature_extraction.check_hdfstorage()
        # hdfstore = feature_extraction.return_hdf_storage()

        self.store = pd.HDFStore('storage.h5', mode='r')
        self.X = []
        self.y = []

    def print_results(self):

        print(self.store['stats'].head(20))

    def create_feature_set(self):
        stats = self.store['stats']  # uncomment this line
        common_more_than_10 = 0
        start_time = time.time()

        for i in reversed(stats.index):

            print(i)
            player1_id = stats.at[i, "ID1"]
            player1_games = stats.loc[np.logical_or(stats.ID1 == player1_id, stats.ID2 == player1_id)]

            curr_tournament = float(stats.at[i, "ID_T"])

            earlier_games_of_p1 = [game for index, game in player1_games.iterrows() if
                                   (float(game['ID_T']) < curr_tournament)]

            player2_id = stats.at[i, "ID2"]
            player2_games = stats.loc[np.logical_or(stats.ID1 == player2_id, stats.ID2 == player2_id)]

            earlier_games_of_p2 = [game for index, game in player2_games.iterrows() if
                                   (float(game['ID_T']) < curr_tournament)]

            opponents_of_p1 = [
                games['ID2'] if (player1_id == games['ID1']) else
                games['ID1'] for games in earlier_games_of_p1]

            opponents_of_p2 = [
                games['ID2'] if (player1_id == games['ID1']) else
                games['ID1'] for games in earlier_games_of_p2]

            sa = set(opponents_of_p1)
            sb = set(opponents_of_p2)
            c = sa.intersection(sb)

            if (len(c) > 10):
                common_more_than_10 = common_more_than_10 + 1

        print(common_more_than_10)
        print("Time took for creating stat features for each match took--- %s seconds ---" % (time.time() - start_time))


DT = Dataset("db.sqlite")
# DT.create_feature_set()
DT.create_feature_set()
