from Database import *
import time
import pandas as pd


class FeatureExtraction(object):

    def __init__(self, database):

        self.db = Database(database)
        self.hdf = pd.HDFStore('storage.h5')

        self.matches = self.db.get_matches()
        self.stats = self.db.get_stats()
        self.players = self.db.get_players()
        self.tournaments = self.db.get_tournaments()
        self.player_surface_dict = {}

        self.X = []
        self.y = []

    def create_results(self):
        # Create results for each game (can extract other features like (number of games played vs)

        # Create a new column to store the result of the game
        self.matches["Result"] = ""
        start_time = time.time()

        # For each match
        for i in self.matches.index:
            result = self.matches.at[i, 'RESULT_G']
            sets = result.split()

            # These are unfinished matches. We ignore them since they might be noisy datapoints.
            if 'w/o' in sets or 'ret.' in sets or 'def.' in sets or 'n/p' in sets:

                self.matches = self.matches.drop(i)

            else:
                sets_p1 = 0
                sets_p2 = 0
                # For each set
                for s in sets:
                    games = s.split('-')

                    p1_games = int(games[0][:1])
                    p2_games = int(games[1][:1])
                    # If player 1 has won the set, increase the number of sets won by one
                    if p1_games > p2_games:
                        sets_p1 = sets_p1 + 1
                    else:
                        sets_p2 = sets_p2 + 1
                # Result == 1 if Player 1 has won, 0 otherwise.
                if sets_p1 > sets_p2:
                    self.matches.at[i, 'Result'] = str(1)
                else:
                    self.matches.at[i, 'Result'] = str(0)

        print("Time took for creating results for each match--- %s seconds ---" % (time.time() - start_time))

        # Store our updated DataFrame in HDF Storage.
        self.hdf.put('matches', self.matches, format='table', data_columns=True)

    def create_new_stats(self):
        # We will extract new features from stats dataset

        start_time = time.time()
        self.stats["FSP1"] = ""
        self.stats["W1SP1"] = ""
        self.stats["W2SP1"] = ""
        self.stats["WRP1"] = ""
        self.stats["FSP2"] = ""
        self.stats["W1SP2"] = ""
        self.stats["W2SP2"] = ""
        self.stats["WRP2"] = ""  # winning on return percentage
        self.stats['WSP1'] = ""
        self.stats['WSP2'] = ""  # Winning on serve percentage
        self.stats['TPWP1'] = ""  # Percentage of all points won
        self.stats['TPWP2'] = ""  # Percentage of all points won (Total points won percentage)

        print("Number of matches with statistics is: {}".format(len(self.stats)))
        dropped_games = 0
        for i in self.stats.index:

            oncourt_stats = [int(self.stats.at[i, 'FSOF_1']),
                             int(self.stats.at[i, 'FSOF_2']),
                             int(self.stats.at[i, 'W1SOF_1']),
                             int(self.stats.at[i, 'W1SOF_2']),
                             int(self.stats.at[i, 'W2SOF_1']),
                             int(self.stats.at[i, 'W2SOF_2']),
                             int(self.stats.at[i, 'RPWOF_1']),
                             int(self.stats.at[i, 'RPWOF_2']),
                             int(self.stats.at[i, 'TPW_1']), int(self.stats.at[i, 'TPW_2'])]
            # Check if any of the values are 0, if so drop that match from our dataset.

            if (0 in oncourt_stats):
                print("This game was invalid because a given stat was equal to 0 ")
                dropped_games = dropped_games + 1
                continue

            # Calculating new features
            fs_percentage_p1 = float(int(self.stats.at[i, 'FS_1']) / int(self.stats.at[i, 'FSOF_1']))
            fs_percentage_p2 = float(int(self.stats.at[i, 'FS_2']) / int(self.stats.at[i, 'FSOF_2']))
            w1sp1 = float(int(self.stats.at[i, 'W1S_1']) / int(self.stats.at[i, 'W1SOF_1']))
            w1sp2 = float(int(self.stats.at[i, 'W1S_2']) / int(self.stats.at[i, 'W1SOF_2']))
            w2sp1 = float(int(self.stats.at[i, 'W2S_1']) / int(self.stats.at[i, 'W2SOF_1']))
            w2sp2 = float(int(self.stats.at[i, 'W2S_2']) / int(self.stats.at[i, 'W2SOF_2']))
            wrp1 = float(int(self.stats.at[i, 'RPW_1']) / int(self.stats.at[i, 'RPWOF_1']))
            wrp2 = float(int(self.stats.at[i, 'RPW_2']) / int(self.stats.at[i, 'RPWOF_2']))
            tpwp1 = float(
                int(self.stats.at[i, 'TPW_1']) / (int(self.stats.at[i, 'TPW_1']) + int(self.stats.at[i, 'TPW_2'])))
            tpwp2 = float(
                int(self.stats.at[i, 'TPW_2']) / (int(self.stats.at[i, 'TPW_1']) + int(self.stats.at[i, 'TPW_2'])))
            stat_feats = [fs_percentage_p1, fs_percentage_p2, w1sp1, w1sp2, w2sp1, w2sp2, wrp1, wrp2, tpwp1, tpwp2]

            if all(f > 0 and f < 1 for f in stat_feats):  # Remove if result is not in range [0,1]

                wsp1 = float((w1sp1 * fs_percentage_p1) + (w2sp1 * (
                        1 - fs_percentage_p1)))  # overall serve winning percentage of Player 1
                wsp2 = float((w1sp2 * fs_percentage_p2) + (w2sp2 * (
                        1 - fs_percentage_p2)))  # overall serve winning percentage of Player 1

                # Update the match with new features
                self.stats.at[i, "FSP1"] = str(fs_percentage_p1)
                self.stats.at[i, "FSP2"] = str(fs_percentage_p2)
                self.stats.at[i, "W1SP1"] = str(w1sp1)
                self.stats.at[i, "W1SP2"] = str(w1sp2)
                self.stats.at[i, "W2SP1"] = str(w2sp1)
                self.stats.at[i, "W2SP2"] = str(w2sp2)
                self.stats.at[i, "WRP1"] = str(wrp1)
                self.stats.at[i, "WRP2"] = str(wrp2)  # player 2's percentage of points on return
                self.stats.at[i, 'WSP1'] = str(wsp1)  # players' overall winning on serve percentages
                self.stats.at[i, 'WSP2'] = str(wsp2)
                self.stats.at[i, 'TPWP1'] = str(tpwp1)  # percentage of total points won
                self.stats.at[i, 'TPWP2'] = str(tpwp2)

            else:
                self.stats = self.stats.drop(i)
                print("this game was invalid")
                dropped_games = dropped_games + 1
        # We can now drop these columns that we used to calculate our updated stats dataset
        del self.stats['FS_1']
        del self.stats['FS_2']
        del self.stats['FSOF_1']
        del self.stats['FSOF_2']
        del self.stats['W1S_1']
        del self.stats['W1SOF_1']
        del self.stats['W1S_2']
        del self.stats['W1SOF_2']
        del self.stats['W2S_1']
        del self.stats['W2SOF_1']
        del self.stats['W2S_2']
        del self.stats['W2SOF_2']
        del self.stats['RPW_1']
        del self.stats['RPWOF_1']
        del self.stats['RPW_2']
        del self.stats['RPWOF_2']

        print("Number of matches dropped because of invalid stats were: {}".format(dropped_games))
        print('The number of remaining games in our stats dataset is {}:'.format(len(self.stats)))
        print("Time took for creating stat features for each match took--- %s seconds ---" % (time.time() - start_time))

        #Order the dataset by Tournament ID and then by Round ID. This will order our matches according to date played.
        self.stats = self.stats.sort_values(['ID_T', 'ID_R'])  # , ascending=[True, False]).sort_index()

        # Store the updated stats dataset in HDF Store
        self.hdf.put('stats', self.stats, format='table', data_columns=True)

    def check_hdfstorage(self):
        # Check everything is stored in their latest version.
        print(self.hdf)
        print(self.hdf.keys())
        print(type(self.hdf['matches']))
        print(type(self.hdf['stats']))

        print(self.hdf['stats'].shape)
        print(self.hdf['matches'].shape)

        df = self.hdf.select('matches')
        for i in df.index:
            assert (df.at[i, 'Result'] == str(1) or df.at[i, 'Result'] == str(0))

    def return_hdf_storage(self):
        return self.hdf

    def create_surface_matrix(self):
        print(self.hdf.keys())
        grass_percentage = []
        hard_percentage = []
        indoor_percentage = []
        clay_percentage = []
        for i in self.players.index:
            player_id = self.players.at[i, "ID_P"]
            # hdf = self.return_hdf_storage()
            matches = self.hdf['matches']
            # matches_of_player = self.matches.loc[self.matches['ID1_G'] == some_value]
            player_games = matches.loc[np.logical_or(matches.ID1_G == player_id, matches.ID2_G == player_id)]
            total_wins = len(player_games.loc[player_games['Result'] == str(1)])
            grass_wins = 0
            hard_wins = 0
            indoor_wins = 0
            clay_wins = 0

            for i in player_games.index:

                print(total_wins)
                tournament_id = player_games.at[i, "ID_T_G"]
                tournament = self.tournaments.loc[self.tournaments['ID_T'] == tournament_id]
                court_id = tournament["ID_C_T"]

                if court_id == 1:
                    hard_wins = hard_wins + 1
                elif court_id == 2:
                    clay_wins = clay_wins + 1

                elif court_id == 3:
                    indoor_wins = indoor_wins + 1

                elif court_id == 4:
                    grass_wins = grass_wins + 1

                    # for each win go its tournament id get what floor it was on and increase the number of wins in that floor

            print("ali")
            print("stop")
        # For each player --> Create a dictionary that maps surfaces to number of wins
        # mean percentage of matches won on surface a is calculated as getting percentage of matches won on that surface for every player
        # put that into np array and calculate mean and std.



# Running it first time, you have to run create_results() and create new stats one time to create their table in HDF Store.

feature_extraction = FeatureExtraction("db.sqlite")

' Functions to call one time only'
# db.create_results() # We call this one time to create results, eliminate some unfinished games and store the new table in HTF Storage
# feature_extraction.create_new_stats() # this takes 244 minutes when run first time
# hdf = feature_extraction.check_hdfstorage()

feature_extraction.order_dataset()
# feature_extraction.create_surface_matrix()
