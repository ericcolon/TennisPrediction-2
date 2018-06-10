import pandas as pd
import numpy as np


class Dataset(object):

    def __init__(self, database):
        # feature_extraction = FeatureExtraction(database)
        # feature_extraction.check_hdfstorage()
        # hdfstore = feature_extraction.return_hdf_storage()

        self.store = pd.HDFStore('storage.h5', mode='r')
        self.X = []
        self.y = []

    def find_common_opponents(self):
        stats = self.store['stats']  # uncomment this line
        common_more_than_10 = 0
        for i in stats.index:
            opponents_of_p1 = []
            opponents_of_p2 = []
            player1_id = stats.at[i, "ID1"]
            player1_games = stats.loc[np.logical_or(stats.ID1 == player1_id, stats.ID2 == player1_id)]
            player2_id = stats.at[i, "ID2"]
            player2_games = stats.loc[np.logical_or(stats.ID1 == player2_id, stats.ID2 == player2_id)]

            for j in player1_games.index:
                if player1_id == player1_games.at[j, "ID1"]:
                    opponents_of_p1.append(player1_games.at[j, "ID2"])
                else:
                    opponents_of_p1.append(player1_games.at[j, "ID1"])

            for k in player2_games.index:
                if player2_id == player2_games.at[k, "ID1"]:
                    opponents_of_p2.append(player2_games.at[k, "ID2"])
                else:
                    opponents_of_p2.append(player2_games.at[k, "ID1"])

            sa = set(opponents_of_p1)
            sb = set(opponents_of_p2)
            c = sa.intersection(sb)
            if (len(c) > 10):
                common_more_than_10 = common_more_than_10 + 1

        print(common_more_than_10)
            #s1 = pd.merge(dfA, dfB, how='inner', on=['S', 'T']) # you can do this by comparing pandadataframes where two players have ID1 and you want to get same ID2 players



DT = Dataset("db.sqlite")
DT.find_common_opponents()
