import collections
from Database import *
import statistics as s
from scipy import stats
import ast


# Helper method to parse strings into integers and floats
def parse_str(s):
    try:
        return ast.literal_eval(str(s))
    except:
        return


class Dataset(object):

    def __init__(self, database):
        self.db = Database(database)
        self.store = pd.HDFStore('storage.h5',mode = 'r')
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
        stats = self.store['updated_stats']  # uncomment this line
        common_more_than_5 = 0
        start_time = time.time()
        zero_common_opponents = 0
        X = []
        court_dict = collections.defaultdict(dict)
        court_dict[1][1] = 1  # 1 is Hardcourt
        court_dict[1][2] = 0.28
        court_dict[1][3] = 0.35
        court_dict[1][4] = 0.24
        court_dict[2][1] = 0.28  # 2 is Clay
        court_dict[2][2] = 1
        court_dict[2][3] = 0.31
        court_dict[2][4] = 0.14
        court_dict[3][1] = 0.35  # 3 is Indoor
        court_dict[3][2] = 0.31
        court_dict[3][3] = 1
        court_dict[3][4] = 0.25
        court_dict[4][1] = 0.24  # 4 is Grass
        court_dict[4][2] = 0.14
        court_dict[4][3] = 0.25
        court_dict[4][4] = 1
        print((stats.loc[170438]))
        print(len(stats.loc[stats['court_type'] == ""])) # We have 624 instances where court type is empty
        print(stats.at[170438, "COMPLETE1"])
        print(stats.at[170438, "COMPLETE2"])
        print(type(stats.at[170438, "COMPLETE1"]))
        print(type(float(stats.at[170438, "COMPLETE1"])))
        print("ali")
        print("ali")
        print("stop")

        ## Bug Testing here

        for i in reversed(stats.index):

            print(i)
            player1_id = stats.at[i, "ID1"]
            player2_id = stats.at[i, "ID2"]

            # All games that two players have played
            player1_games = stats.loc[np.logical_or(stats.ID1 == player1_id, stats.ID2 == player1_id)]


            player2_games = stats.loc[np.logical_or(stats.ID1 == player2_id, stats.ID2 == player2_id)]
            curr_tournament = float(stats.at[i, "ID_T"])
            # current_court_id = int(stats.at[i, "court_type"])
            # Games played earlier than the current tournament we are investigating
            earlier_games_of_p1 = [game for index, game in player1_games.iterrows() if
                                   (float(game['ID_T']) < curr_tournament)]

            earlier_games_of_p2 = [game for index, game in player2_games.iterrows() if
                                   (float(game['ID_T']) < curr_tournament)]

            opponents_of_p1 = [
                games['ID2'] if (player1_id == games['ID1']) else
                games['ID1'] for games in earlier_games_of_p1]

            opponents_of_p2 = [
                games['ID2'] if (player2_id == games['ID1']) else
                games['ID1'] for games in earlier_games_of_p2]

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
                                         (player1_id == game['ID1'] and opponent == game['ID2']) or (
                                                 player1_id == game['ID2'] and opponent == game['ID1'])]
                player2_games_updated = [game for opponent in common_opponents for game in earlier_games_of_p2 if
                                         (player2_id == game['ID1'] and opponent == game['ID2']) or (
                                                 player2_id == game['ID2'] and opponent == game['ID1'])]

                serveadv_1 = [parse_str(game.SERVEADV1) if game.ID1 == player1_id else parse_str(game.SERVEADV2) for
                              game in
                              player1_games_updated]
                serveadv_2 = [parse_str(game.SERVEADV1) if game.ID1 == player2_id else parse_str(game.SERVEADV2) for
                              game in
                              player2_games_updated]

                complete_1 = [parse_str(game.COMPLETE1) if game.ID1 == player1_id else parse_str(game.COMPLETE2) for
                              game in
                              player1_games_updated]
                complete_2 = [parse_str(game.COMPLETE1) if game.ID1 == player2_id else parse_str(game.COMPLETE2) for
                              game in
                              player2_games_updated]

                w1sp_1 = [parse_str(game.W1SP1) if game.ID1 == player1_id else parse_str(game.W1SP2) for game in
                          player1_games_updated]
                w1sp_2 = [parse_str(game.W1SP1) if game.ID1 == player2_id else parse_str(game.W1SP2) for game in
                          player2_games_updated]

                aces_1 = [parse_str(game.ACES_1) if game.ID1 == player1_id else parse_str(game.ACES_2) for game in
                          player1_games_updated]

                aces_2 = [parse_str(game.ACES_1) if game.ID1 == player2_id else parse_str(game.ACES_2) for game in
                          player2_games_updated]

               # print("ali")
               # print("stop")

        print(common_more_than_5)
        print(zero_common_opponents)

        print("Time took for creating stat features for each match took--- %s seconds ---" % (time.time() - start_time))

    def train_model(self):
        pass

    def test_model(self):
        pass


DT = Dataset("db.sqlite")
# DT.add_court_types()  # This should be moved to feature extraction
DT.create_feature_set()
