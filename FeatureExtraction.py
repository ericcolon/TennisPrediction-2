from Database import *
import time
import pandas as pd


class FeatureExtraction(object):

    def __init__(self, database):

        self.db = Database(database)

        self.matches = self.db.get_matches()
        self.stats = self.db.get_stats()

        self.hdf = pd.HDFStore('storage.h5')
        self.players = self.db.get_players()

        p1 = self.players.loc[self.players['ID_P'] == 4344]
        p2 = self.players.loc[self.players['ID_P'] == 14588]


    # Create results for each game (can extract other features like (number of games played vs)
    def create_results(self):
        self.matches["Result"] = ""
        start_time = time.time()
        # print(self.matches.index)
        for i in self.matches.index:
            result = self.matches.at[i, 'RESULT_G']
            sets = result.split()

            if 'w/o' in sets or 'ret.' in sets or 'def.' in sets or 'n/p' in sets:

                self.matches = self.matches.drop(i)

            else:
                sets_p1 = 0
                sets_p2 = 0
                for s in sets:
                    games = s.split('-')

                    p1_games = int(games[0][:1])
                    p2_games = int(games[1][:1])
                    if p1_games > p2_games:
                        sets_p1 = sets_p1 + 1
                    else:
                        sets_p2 = sets_p2 + 1

                if sets_p1 > sets_p2:
                    self.matches.at[i, 'Result'] = str(1)
                else:
                    self.matches.at[i, 'Result'] = str(0)

        print("Time took for creating results for each match--- %s seconds ---" % (time.time() - start_time))

        # self.matches.to_pickle("/Users/aysekozlu/pycharmprojects/tennisprediction/matches.pkl")
        self.hdf.put('matches', self.matches, format='table', data_columns=True)

    def create_new_stats(self):
        start_time = time.time()
        self.stats["FSP1"] = ""
        self.stats["W1SP1"] = ""
        self.stats["W2SP1"] = ""
        self.stats["WRP1"] = ""
        self.stats["FSP2"] = ""
        self.stats["W1SP2"] = ""
        self.stats["W2SP2"] = ""
        self.stats["WRP2"] = ""
        print("Number of matches with statistics is: {}".format(len(self.stats)))
        dropped_games = 0
        for i in self.stats.index:

            oncourt_stats = [int(self.stats.at[i, 'FS_1']), int(self.stats.at[i, 'FSOF_1']),
            int(self.stats.at[i, 'FS_2']), int(self.stats.at[i, 'FSOF_2']),
            int(self.stats.at[i, 'W1S_1']), int(self.stats.at[i, 'W1SOF_1']),
            int(self.stats.at[i, 'W1S_2']), int(self.stats.at[i, 'W1SOF_2']),
                 int(self.stats.at[i, 'W2S_1']), int(self.stats.at[i, 'W2SOF_1']),
                 int(self.stats.at[i, 'W2S_2']), int(self.stats.at[i, 'W2SOF_2']),
                 int(self.stats.at[i, 'RPW_1']), int(self.stats.at[i, 'RPWOF_1']),
                 int(self.stats.at[i, 'RPW_2']), int(self.stats.at[i, 'RPWOF_2'])]

            if (0 in oncourt_stats):
                print("This game was invalid because a given stat was equal to 0 ")
                dropped_games = dropped_games + 1
                continue

            fs_percentage_p1 = float(int(self.stats.at[i, 'FS_1']) / int(self.stats.at[i, 'FSOF_1']))
            fs_percentage_p2 = float(int(self.stats.at[i, 'FS_2']) / int(self.stats.at[i, 'FSOF_2']))
            w1sp1 = float(int(self.stats.at[i, 'W1S_1']) / int(self.stats.at[i, 'W1SOF_1']))
            w1sp2 = float(int(self.stats.at[i, 'W1S_2']) / int(self.stats.at[i, 'W1SOF_2']))
            w2sp1 = float(int(self.stats.at[i, 'W2S_1']) / int(self.stats.at[i, 'W2SOF_1']))
            w2sp2 = float(int(self.stats.at[i, 'W2S_2']) / int(self.stats.at[i, 'W2SOF_2']))
            wrp1 = float(int(self.stats.at[i, 'RPW_1']) / int(self.stats.at[i, 'RPWOF_1']))
            wrp2 = float(int(self.stats.at[i, 'RPW_2']) / int(self.stats.at[i, 'RPWOF_2']))
            stat_feats = [fs_percentage_p1, fs_percentage_p2, w1sp1, w1sp2, w2sp1, w2sp2, wrp1, wrp2]

            if all(f > 0 and f < 1 for f in stat_feats):
                self.stats.at[i, "FSP1"] = str(fs_percentage_p1)
                self.stats.at[i, "FSP2"] = str(fs_percentage_p2)
                self.stats.at[i, "W1SP1"] = str(w1sp1)
                self.stats.at[i, "W1SP2"] = str(w1sp2)
                self.stats.at[i, "W2SP1"] = str(w2sp1)
                self.stats.at[i, "W2SP2"] = str(w2sp2)
                self.stats.at[i, "WRP1"] = str(wrp1)
                self.stats.at[i, "WRP2"] = str(wrp2)

            else:
                self.stats = self.stats.drop(i)
                print("this game was invalid")
                dropped_games = dropped_games + 1

        print("Number of matches dropped because of invalid stats were: {}".format(dropped_games))
        print('The number of remaining games in our stats dataset is {}:'.format(len(self.stats)))
        print("Time took for creating stat features for each match took--- %s seconds ---" % (time.time() - start_time))

        # self.matches.to_pickle("/Users/aysekozlu/pycharmprojects/tennisprediction/matches.pkl")
        self.hdf.put('stats', self.stats, format='table', data_columns=True)

    def check_hdfstorage(self):
        print(self.hdf)
        print(self.hdf['matches'].shape)
        print(self.hdf['stats'].shape)
        df = self.hdf.select('matches')
        for i in df.index:
            assert (df.at[i, 'Result'] == str(1) or df.at[i, 'Result'] == str(0))

        

    def get_pkl_matches(self):

        df = pd.read_pickle("/Users/aysekozlu/pycharmprojects/tennisprediction/matches.pkl")  # to get the pickle file

        print(df.shape)
        for i in df.index:
            assert (df.at[i, 'Result'] == 1 or df.at[i, 'Result'] == 0)


feature_extraction = FeatureExtraction("db.sqlite")
# db.get_pkl_matches()
# db.create_results() # We call this one time to create results, eliminate some unfinished games and store the new table in HTF Storage
feature_extraction.create_new_stats()
feature_extraction.check_hdfstorage()
