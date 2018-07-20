from DataExtraction import *
import time
from sqlalchemy import create_engine
import numpy as np

from bs4 import BeautifulSoup
import urllib.request
from urllib.request import urlopen


def df2sqlite_v2(dataframe, db_name):
    disk_engine = create_engine('sqlite:///' + db_name + '.db')
    dataframe.to_sql(db_name, disk_engine, if_exists='append')


def historical_odds_database(csv_file):
    dataframe = pd.read_csv(csv_file, low_memory=False)
    print("Dataframe conversion finished. Converting it into sqlite3 database")
    df2sqlite_v2(dataframe, "odds")
    print("process finished")

def url_is_alive(url):
    """Checks that a given URL is reachable."""
    request = urllib.request.Request(url)
    request.get_method = lambda: 'HEAD'

    try:
        urllib.request.urlopen(request)
        return True
    except urllib.request.HTTPError:
        return False


def odds_checker(url):
    if url_is_alive(url):
        odds_page = urlopen(url)
        content = odds_page.read()  # get content from webpage
        page_soup = BeautifulSoup(content, 'html.parser')
        cont = page_soup.find('div', class_='table-container')
        active_rows = page_soup.find_all('tr', style='display: table-row;')
        print(active_rows)
        print(cont)
        # soup = BeautifulSoup(content, 'lxml')
        # print(soup)


class FeatureExtraction(object):

    def __init__(self, database):

        self.db = DataExtraction(database)
        self.matches = self.db.get_matches()
        self.unfiltered_matches = self.db.get_unfiltered_matches()
        self.unfiltered_tournaments = self.db.get_unfiltered_tournaments()
        self.stats = self.db.get_stats()
        self.players = self.db.get_players()
        self.tournaments = self.db.get_tournaments()
        self.player_surface_dict = {}

    # For Time discount factor purposes

    def create_results(self, table):

        # Find number of games and number of sets for each game in stats dataset
        print("Before starting create_results function, the length of our database was {}".format(len(table)))
        start_time = time.time()
        table["Number_of_games"] = ""  # Create a new column to store the number of games in  a match
        table["Number_of_sets"] = ""
        empty_matches = 0
        unfinished_matches = 0
        # For each match in stats dataset
        for i in table.index:
            print(i)
            # get player ID's and tournament ID from stats dataset
            player1_id = table.at[i, 'ID1']
            player2_id = table.at[i, 'ID2']
            tour_id = table.at[i, 'ID_T']

            # our_match = self.matches[(self.matches.ID1_G == player1_id) & (self.matches.ID2_G == player2_id) & (
            #       self.matches.ID_T_G == tour_id)]

            # Find the match in matches dataset.
            our_match = self.matches.loc[
                np.logical_and(np.logical_and(self.matches['ID1_G'] == player1_id, self.matches['ID2_G'] == player2_id),
                               self.matches.ID_T_G == tour_id)]

            # If we cannot find the match in our matches dataset, record and continue.
            if our_match.empty:
                print("We were not able to find this match in the matches dataset")
                print(player1_id)
                print(player2_id)
                print(tour_id)
                empty_matches = empty_matches + 1
                continue

            else:
                # get the result and sets
                result = our_match.iloc[0]['RESULT_G']
                sets = result.split()

                # These are unfinished matches. We ignore them since they might be noisy datapoints.
                if 'w/o' in sets or 'ret.' in sets or 'def.' in sets or 'n/p' in sets:
                    table = table.drop(i)
                    unfinished_matches = unfinished_matches + 1

                else:

                    total_number_of_games = 0  # variables
                    total_number_of_sets = len(sets)

                    for match_set in sets:
                        games = match_set.split('-')
                        p1_games = int(games[0])  # number of games player 1 has won in that set

                        if "(" in games[1]:
                            p2_games = int(games[1][0])  # number of games player 2 has won in that set
                        else:
                            p2_games = int(games[1])

                        total_number_of_games = total_number_of_games + p1_games + p2_games

                    # Update the table entries
                    table.at[i, "Number_of_games"] = str(total_number_of_games)
                    table.at[i, "Number_of_sets"] = str(total_number_of_sets)

        print("After deleting unfinished games, the length of our database is {}".format(len(table)))
        print("The matches we were unable to find {}.".format(empty_matches))
        print("The number of unfinished matches in stats dataset was  {}.".format(unfinished_matches))

        print("Time took for creating results for each match--- %s seconds ---" % (time.time() - start_time))

        stats_final = table.reset_index(drop=True)  # reset indexes if any more rows are dropped

        return stats_final

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
            if 0 in oncourt_stats:
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

            if all(f > 0 and f < 1 for f in stat_feats):
                #  if required features are  in range [0,1] calculate the following
                wsp1 = float((w1sp1 * fs_percentage_p1) + (w2sp1 * (
                        1 - fs_percentage_p1)))  # overall serve winning percentage of Player 1
                wsp2 = float((w1sp2 * fs_percentage_p2) + (w2sp2 * (
                        1 - fs_percentage_p2)))  # overall serve winning percentage of Player 1
                serveadv1 = wsp1 - wrp2
                serveadv2 = wsp2 - wrp1
                complete1 = wsp1 * wrp1
                complete2 = wsp2 * wrp2
                # Update the match with new features
                self.stats.at[i, "FSP1"] = fs_percentage_p1
                self.stats.at[i, "FSP2"] = fs_percentage_p2
                self.stats.at[i, "W1SP1"] = w1sp1
                self.stats.at[i, "W1SP2"] = w1sp2
                self.stats.at[i, "W2SP1"] = w2sp1
                self.stats.at[i, "W2SP2"] = w2sp2
                self.stats.at[i, "WRP1"] = wrp1
                self.stats.at[i, "WRP2"] = wrp2  # player 2's percentage of points on return
                self.stats.at[i, 'WSP1'] = wsp1  # players' overall winning on serve percentages
                self.stats.at[i, 'WSP2'] = wsp2
                self.stats.at[i, 'TPWP1'] = tpwp1  # percentage of total points won
                self.stats.at[i, 'TPWP2'] = tpwp2
                self.stats.at[i, 'SERVEADV1'] = serveadv1
                self.stats.at[i, 'SERVEADV2'] = serveadv2
                self.stats.at[i, 'COMPLETE1'] = complete1
                self.stats.at[i, 'COMPLETE2'] = complete2

            else:
                #  if the required features are not in range [0,1] drop it from the dataset
                self.stats = self.stats.drop(i)
                print("this game was invalid because a required stat was not in range [0,1]")
                dropped_games = dropped_games + 1

        # We can now drop these columns that we used to calculate our updated features
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

        print("Number of matches dropped because of invalid stats were: {}.".format(dropped_games))
        print('The number of remaining games in our stats dataset is {}:'.format(len(self.stats)))
        print("Time took for creating stat features for each match took--- %s seconds ---" % (time.time() - start_time))

        # Order the dataset by Tournament ID and then by Round ID. This will order our matches according to date played.
        self.stats = self.stats.sort_values(['ID_T', 'ID_R'])
        stats_final = self.stats.reset_index(drop=True)

        print("V6 of stats dataset includes {} datapoints.".format(len(stats_final)))

        return stats_final

    def get_head_to_head_statistics(self, stats):

        matches = self.unfiltered_matches
        start_time = time.time()
        invalid = 0
        stats["H12H"] = ""
        stats["H21H"] = ""
        for i in stats.index:
            print(i)
            player1 = stats.at[i, "ID1"]
            player2 = stats.at[i, "ID2"]
            # Head to head games that Player 1 has won
            # head_to_head_1 = matches[(matches.ID1_G == player1) & (self.matches.ID2_G == player2)]

            # Matches that player 1 has won
            head_to_head_1 = matches.loc[np.logical_and(matches['ID1_G'] == player1, matches['ID2_G'] == player2)]

            # Head to head Games that Player 2 has won
            # head_to_head_2 = matches[(matches.ID1_G == player2) & (self.matches.ID2_G == player1)]
            # Matches that player 2 has won.
            head_to_head_2 = matches.loc[np.logical_and(matches['ID1_G'] == player2, matches['ID2_G'] == player1)]

            player_1_wins = len(head_to_head_1)
            player_2_wins = len(head_to_head_2)
            if player_1_wins == 0 and player_2_wins == 0:
                invalid = invalid + 1
                continue
            h12h = player_1_wins / (player_2_wins + player_1_wins)
            h21h = player_2_wins / (player_1_wins + player_2_wins)
            stats.at[i, "H12H"] = str(h12h)
            stats.at[i, "H21H"] = str(h21h)

        print("Number of invalid matches is {}.".format(invalid))
        print("Time took for creating head to head features for each match took--- %s seconds ---" % (
                time.time() - start_time))
        stats_final = stats.reset_index(drop=True)  # reset indexes if any more rows are dropped

        return stats_final

    def add_court_types(self, stats):
        # Adds court types to updated_stats dataset

        start_time = time.time()

        stats["court_type"] = ""
        court_id_none = 0
        for i in stats.index:
            print(i)
            tournament_id = stats.at[i, "ID_T"]
            tournament = self.tournaments.loc[self.tournaments['ID_T'] == tournament_id]  # Find the tournament

            if tournament.empty:
                court_id_none = court_id_none + 1
                continue
            else:
                court_id = float(tournament['ID_C_T'])  # casting it as a float

                stats.at[i, 'court_type'] = str(court_id)

        print("Number of games with no court id is {}.".format(court_id_none))
        stats_final = stats.reset_index(drop=True)  # reset indexes if any more rows are dropped

        print("Time took for adding court ids to stats took--- %s seconds ---" % (time.time() - start_time))

        return stats_final

    def create_tournament_year_database(self, table):
        print(len(self.tournaments))
        self.tournaments["DATE_T"] = pd.to_datetime(self.tournaments["DATE_T"])
        tours = self.tournaments["DATE_T"].dt.year.to_frame()
        tours.reset_index(level=0, inplace=True)

        table['year'] = ""
        for i in table.index:
            print(i)
            tour_id = table.at[i, "ID_T"]
            tournament = tours.loc[tours["index"] == tour_id]

            if not tournament.empty:
                table.at[i, 'year'] = int(tournament["DATE_T"])

        table['year'].fillna(2018, inplace=True)
        final_dataset = table.reset_index(drop=True)  # reset indexes if any more rows are dropped
        return final_dataset

    def get_filtered_matches(self):
        return self.matches

    def get_unfiltered_matches(self):
        return self.unfiltered_matches


"""
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
# mean percentage of matches won on surface a is calculated as getting percentage of matches won on that surface 
for every player
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

"""
""" 
# Code to create the Sqlite stats database with all the required information to create features
feature_extraction = FeatureExtraction("db.sqlite")
new_stats = feature_extraction.create_new_stats()
new_stats_v1 = feature_extraction.create_results(new_stats)
new_stats_v2 = feature_extraction.get_head_to_head_statistics(new_stats_v1)
new_stats_v3 = feature_extraction.add_court_types(new_stats_v2)
new_stats_v4 = feature_extraction.create_tournament_year_database(new_stats_v3)
print(new_stats.info())
print(new_stats_v1.info())
print(new_stats_v2.info())
print(new_stats_v3.info())
print(new_stats_v4.info())
df2sqlite_v2(new_stats_v4, 'updated_stats_v2')
"""
""" This code is to create and calculate a surface matrix 

means_and_stds = feature_extraction.create_surface_matrix()
print(means_and_stds)
feature_extraction.calculate_surface_matrix(*means_and_stds)
"""

# historical_odds_database("world_tennis_odds.csv")
odds_checker("http://www.oddsportal.com/tennis/united-kingdom/atp-wimbledon/results/")
