from Database import *
import time
import pandas as pd
import statistics as s
import heapq
import numpy as np


class FeatureExtraction(object):

    def __init__(self, database):

        self.db = Database(database)
        self.hdf = pd.HDFStore('storage.h5')
        self.matches = self.db.get_matches()
        self.unfiltered_matches = self.db.get_unfiltered_matches()
        self.unfiltered_tournaments = self.db.get_unfiltered_tournaments()
        self.stats = self.db.get_stats()
        self.players = self.db.get_players()
        self.tournaments = self.db.get_tournaments()
        self.player_surface_dict = {}
        self.X = []
        self.y = []

    def check_hdfstorage(self):
        # Check everything is stored in their latest version. Their types and lengths

        print(self.hdf.keys())
        # print(type(self.hdf['matches']))
        print(type(self.hdf['updated_stats']))
        print(self.hdf['updated_stats'].head())

        # print(type(self.hdf['unfiltered_matches']))

    def create_results(self, table, hdf_store_name):
        # Create results for each game (can extract other features like (number of games played vs)

        # Create a new column to store the result of the game
        # table["Result"] = "" uncomment this line
        start_time = time.time()
        table["Number_of_games"] = ""
        table["Number_of_sets"] = ""
        # For each match
        for i in table.index:
            result = table.at[i, 'RESULT_G']
            sets = result.split()
            print(i)
            # These are unfinished matches. We ignore them since they might be noisy datapoints.
            if 'w/o' in sets or 'ret.' in sets or 'def.' in sets or 'n/p' in sets:
                table = table.drop(i)

            else:

                total_number_of_games = 0
                total_number_of_sets = len(sets)

                for match_set in sets:
                    games = match_set.split('-')

                    p1_games = int(games[0])
                    if "(" in games[1]:
                        p2_games = int(games[1][0])
                    else:

                        p2_games = int(games[1])
                    total_number_of_games = total_number_of_games + p1_games + p2_games
                table["Number_of_games"] = str(total_number_of_games)
                table["Number_of_sets"] = str(total_number_of_sets)

        print("Time took for creating results for each match--- %s seconds ---" % (time.time() - start_time))
        # Store our updated DataFrame in HDF Storage.
        self.hdf.put(hdf_store_name, table, format='table', data_columns=True)

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
        self.stats['SERVEADV1'] = ""
        self.stats['SERVEADV2'] = ""
        self.stats['COMPLETE1'] = ""
        self.stats['COMPLETE2'] = ""

        print("Number of matches with statistics is: {}".format(len(self.stats)))
        dropped_games = 0

        # add breaking point conversions
        for i in self.stats.index:
            print(i)
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
                serveadv1 = wsp1 - wrp2
                serveadv2 = wsp2 - wrp1
                complete1 = wsp1 * wrp1
                complete2 = wsp2 * wrp2
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
                self.stats.at[i, 'SERVEADV1'] = str(serveadv1)
                self.stats.at[i, 'SERVEADV2'] = str(serveadv2)
                self.stats.at[i, 'COMPLETE1'] = str(complete1)
                self.stats.at[i, 'COMPLETE2'] = str(complete2)

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

        # Order the dataset by Tournament ID and then by Round ID. This will order our matches according to date played.
        self.stats = self.stats.sort_values(['ID_T', 'ID_R'])  # , ascending=[True, False]).sort_index()
        # self.stats = self.stats.reset_index(drop=True)
        print("V5 of stats dataset includes {} datapoints.".format(len(self.stats)))
        # Store the updated stats dataset in HDF Store
        self.hdf.put('updated_stats', self.stats, format='table', data_columns=True)

    def get_head_to_head_statistics(self):
        stats = self.hdf['updated_stats']
        matches = self.hdf['unfiltered_matches']
        start_time = time.time()
        invalid = 0

        stats["H12H"] = ""
        stats["H21H"] = ""
        for i in stats.index:
            print(i)
            player1 = stats.at[i, "ID1"]
            player2 = stats.at[i, "ID2"]
            # Head to head games that Player 1 has won
            head_to_head_1 = matches.loc[np.logical_and(matches['ID1_G'] == player1, matches['ID2_G'] == player2)]

            # Head to head Games that Player 2 has won
            head_to_head_2 = matches.loc[np.logical_and(matches['ID1_G'] == player2, matches['ID2_G'] == player1)]

            player_1_wins = len(head_to_head_1)
            player_2_wins = len(head_to_head_2)
            if player_1_wins == 0 and player_2_wins == 0:
                print(player1)
                print(player2)
                invalid = invalid + 1
                continue
            h12h = player_1_wins / (player_2_wins + player_1_wins)
            h21h = player_2_wins / (player_1_wins + player_2_wins)
            stats.at[i, "H12H"] = str(h12h)
            stats.at[i, "H21H"] = str(h21h)

        print("Number of invalid matches is {}".format(invalid))
        print("Time took for creating head to head features for each match took--- %s seconds ---" % (
                time.time() - start_time))

        self.hdf.put('updated_stats', stats, format='table', data_columns=True)

    def reset_indexes_of_dataframe(self):
        stats = self.hdf['updated_stats']
        stats = stats.reset_index(drop=True)
        self.hdf.put('updated_stats', stats, format='table', data_columns=True)

    def calculate_surface_matrix(self, mean_grass, std_grass, mean_clay, std_clay, mean_indoor, std_indoor, mean_hard,
                                 std_hard, number_of_players):
        start_time = time.time()

        p_clay_hard = 0
        p_clay_grass = 0
        p_clay_indoor = 0
        p_hard_grass = 0
        p_hard_indoor = 0
        p_grass_indoor = 0

        matches = self.hdf['unfiltered_matches']  # uncomment this line

        print("We are investigating {} players".format(len(matches.ID1_G.unique())))

        for id in matches.ID1_G.unique():
            player_id = id

            # get all the matches of a particular player in our dataset

            player_wins = matches.loc[(matches.ID1_G == player_id)]

            if player_wins.empty:
                continue

            total_wins = len(player_wins)

            grass_wins = 0
            hard_wins = 0
            indoor_wins = 0
            clay_wins = 0
            print(player_id)

            # Iterate through wins of this player
            for j in player_wins.index:

                tournament_id = player_wins.at[j, "ID_T_G"]  # Get the tournament id of the player's game

                tournament = self.unfiltered_tournaments.loc[
                    self.unfiltered_tournaments['ID_T'] == tournament_id]  # Find the tournament

                if tournament.empty:
                    continue

                # court_id = tournament.iloc[0]['ID_C_T']  # Find which court that game was played on
                court_id = float(tournament['ID_C_T'])  # casting it as a float

                if court_id == 1:

                    hard_wins = hard_wins + 1

                elif court_id == 2:
                    clay_wins = clay_wins + 1

                elif court_id == 3:
                    indoor_wins = indoor_wins + 1

                elif court_id == 5:

                    grass_wins = grass_wins + 1

                # for each win go its tournament id get what floor it was on and increase the number of wins in that floor

            grass_percentage = float(grass_wins / total_wins)
            hard_percentage = float(hard_wins / total_wins)
            indoor_percentage = float(indoor_wins / total_wins)
            clay_percentage = float(clay_wins / total_wins)
            p_clay_hard = p_clay_hard + ((clay_percentage - mean_clay) * (hard_percentage * mean_hard))
            p_clay_grass = p_clay_grass + ((clay_percentage - mean_clay) * (grass_percentage * mean_grass))
            p_clay_indoor = p_clay_indoor + ((clay_percentage - mean_clay) * (indoor_percentage * mean_indoor))
            p_hard_grass = p_hard_grass + ((hard_percentage - mean_hard) * (grass_percentage * mean_grass))
            p_grass_indoor = p_grass_indoor + ((grass_percentage - mean_grass) * (indoor_percentage * mean_indoor))
            p_hard_indoor = p_hard_indoor + ((hard_percentage - mean_hard) * (indoor_percentage * mean_indoor))

        clay_hard_matrix_value = float(p_clay_hard / (number_of_players * std_clay * std_hard))
        clay_grass_matrix_value = float(p_clay_grass / (number_of_players * std_grass * std_clay))
        clay_indoor_matrix_value = float(p_clay_indoor / (number_of_players * std_indoor * std_clay))
        hard_grass_matrix_value = float(p_hard_grass / (number_of_players * std_hard * std_grass))
        hard_indoor_matrix_value = float(p_hard_indoor / (number_of_players * std_hard * std_indoor))
        grass_indoor_matrix_value = float(p_grass_indoor / (number_of_players * std_grass * std_indoor))

        print("clay_hard_matrix_value is {}:".format(clay_hard_matrix_value))
        print("clay_grass_matrix_value is {}:".format(clay_grass_matrix_value))
        print("clay_indoor_matrix_value is {}:".format(clay_indoor_matrix_value))
        print("hard_grass_matrix_value is {}:".format(hard_grass_matrix_value))
        print("hard_indoor_matrix_value is {}:".format(hard_indoor_matrix_value))
        print("grass_indoor_matrix_value is {}:".format(grass_indoor_matrix_value))

        print("Investigated total number of {} players".format(number_of_players))
        print("Time took for creating surface matrix--- %s seconds ---" % (time.time() - start_time))

    # For each player --> Create a dictionary that maps surfaces to number of wins
    # mean percentage of matches won on surface a is calculated as getting percentage of matches won on that surface for every player
    # put that into np array and calculate mean and std.

    def create_surface_matrix(self):
        # FOR EACH SURFACE X = (CLAY,GRASS,INDOOR,HARD) this function returns two things:

        # Mean percentage of matches won on surface X accross all players
        # Standard deviation of percentage of matches won on surface b

        grass_percentage = []
        hard_percentage = []
        indoor_percentage = []
        clay_percentage = []
        number_of_players = 0
        matches = self.hdf['unfiltered_matches']  # uncomment this line
        print("Number of matches that we will look at: {}.".format(len(matches)))
        index = 0
        players_with_no_wins = 0
        # Across all players in our database
        print("We are investigating {} players".format(len(matches.ID1_G.unique())))

        for id in matches.ID1_G.unique():
            player_id = id

            # get all the matches of a particular player in our dataset

            player_wins = matches.loc[(matches.ID1_G == player_id)]

            if player_wins.empty:
                players_with_no_wins = players_with_no_wins + 1

                continue

            index = index + 1
            number_of_players = index

            total_wins = len(player_wins)
            grass_wins = 0
            hard_wins = 0
            indoor_wins = 0
            clay_wins = 0
            print(player_id)

            # Iterate through wins of this player
            for j in player_wins.index:

                tournament_id = player_wins.at[j, "ID_T_G"]  # Get the tournament id of the player's game

                tournament = self.unfiltered_tournaments.loc[
                    self.unfiltered_tournaments['ID_T'] == tournament_id]  # Find the tournament

                if tournament.empty:
                    continue

                # court_id = tournament.iloc[0]['ID_C_T']  # Find which court that game was played on
                court_id = float(tournament['ID_C_T'])  # casting it as a float

                if court_id == 1:

                    hard_wins = hard_wins + 1

                elif court_id == 2:
                    clay_wins = clay_wins + 1

                elif court_id == 3:
                    indoor_wins = indoor_wins + 1

                elif court_id == 5:

                    grass_wins = grass_wins + 1

                # for each win go its tournament id get what floor it was on and increase the number of wins in that floor
            grass_percentage.append(float(grass_wins / total_wins))
            hard_percentage.append(float(hard_wins / total_wins))
            indoor_percentage.append(float(indoor_wins / total_wins))
            clay_percentage.append(float(clay_wins / total_wins))

        largest_grass = heapq.nlargest(1, grass_percentage)[0]
        largest_clay = heapq.nlargest(1, clay_percentage)[0]
        largest_hard = heapq.nlargest(1, hard_percentage)[0]
        largest_indoor = heapq.nlargest(1, indoor_percentage)[0]
        # m_g = s.mean(grass_percentage)
        mean_grass = s.mean(grass_percentage) * 100  # / largest_grass
        mean_clay = s.mean(clay_percentage) * 100  # / largest_clay
        mean_hard = s.mean(hard_percentage) * 100  # / largest_hard
        mean_indoor = s.mean(indoor_percentage) * 100  # / largest_indoor
        std_grass = s.stdev(grass_percentage) * 100
        std_clay = s.stdev(clay_percentage) * 100
        std_hard = s.stdev(hard_percentage) * 100
        std_indoor = s.stdev(indoor_percentage) * 100
        print("There are {} players with no wins".format(players_with_no_wins))
        print("Investigated total number of {} players".format(number_of_players))
        print(mean_grass)
        print(mean_clay)
        print(mean_hard)
        print(mean_indoor)
        print(std_grass)
        print(std_clay)
        print(std_hard)
        print(std_indoor)
        return [mean_grass, std_grass, mean_clay, std_clay, mean_indoor, std_indoor, mean_hard, std_hard,
                number_of_players]

    def get_filtered_matches(self):
        return self.matches

    def get_unfiltered_matches(self):
        return self.unfiltered_matches

    # Get the tournament id of the player's game

    # Running it first time, you have to run create_results() and create new stats one time to create their table in HDF Store.


feature_extraction = FeatureExtraction("db.sqlite")
# feature_extraction.check_hdfstorage()

"""Functions to call one time only
matches = feature_extraction.get_filtered_matches()
unfiltered_matches = feature_extraction.get_unfiltered_matches()
feature_extraction.create_results(matches, 'matches')
feature_extraction.create_results(unfiltered_matches, 'unfiltered_matches')
feature_extraction.check_hdfstorage()"""
# feature_extraction.create_new_stats() # this takes 244 minutes when run first time or 7 minutes lol
# matches = feature_extraction.get_filtered_matches()
# feature_extraction.create_results(matches, 'matches')
# feature_extraction.create_new_stats() sonra reset indexes, add aces, add court ids  ve get head to heads
feature_extraction.reset_indexes_of_dataframe()
feature_extraction.get_head_to_head_statistics()
feature_extraction.check_hdfstorage()

""" This code is to create and calculate a surface matrix 

means_and_stds = feature_extraction.create_surface_matrix()
print(means_and_stds)
feature_extraction.calculate_surface_matrix(*means_and_stds)
"""
