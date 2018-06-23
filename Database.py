import sqlite3
import pandas as pd
import time


class Database(object):
    def __init__(self, database):
        self.conn = sqlite3.connect(database)
        start_time = time.time()


        self.court_dict = {}
        self.create_court_dict()

        self.players = pd.read_sql_query("select ID_P,NAME_P from players_atp where NAME_P not like '%/%';",
                                         self.conn)  # atp players list
        print("Number of players in our dataset is: {}".format(len(self.players)))
        self.single_players = pd.read_sql_query("select ID_P,NAME_P from players_atp where DATE_P is not NULL",
                                                self.conn)
        print("Number of single players in our dataset is: {}".format(len(self.single_players)))

        self.tournaments = pd.read_sql_query("SELECT ID_T, NAME_T,DATE_T,ID_C_T FROM tours_atp WHERE RANK_T not like 6",
                                             self.conn)  # tournament list where RANK_T not like 0 (Should I include challenger tournaments ?? )

        self.tournaments['DATE_T'] = pd.to_datetime(self.tournaments.DATE_T)
        self.tournaments = self.tournaments[self.tournaments[
                                                'DATE_T'] > '2006-01-01']  # We only want to extract tournament ID's after 2006 (Total of 11033 tournaments)

        self.unfiltered_tournaments = pd.read_sql_query(
            "SELECT ID_T, NAME_T,DATE_T,ID_C_T FROM tours_atp WHERE RANK_T not like 6",
            self.conn)  # tournament list where RANK_T not like 0 (Should I include challenger tournaments ?? )

        self.unfiltered_tournaments['DATE_T'] = pd.to_datetime(self.unfiltered_tournaments.DATE_T)

        self.stats = pd.read_sql_query("select * from stat_atp where ID_T>= 3760", self.conn)

        # Drop the games with invalid stats
        print("Number of matches with statistics is: {}".format(len(self.stats)))

        self.stats = self.stats.dropna(subset=['FS_1'])
        self.stats = self.stats.dropna(subset=['FSOF_1'])
        self.stats = self.stats.dropna(subset=['FS_2'])
        self.stats = self.stats.dropna(subset=['W1S_1'])
        self.stats = self.stats.dropna(subset=['W1SOF_1'])
        self.stats = self.stats.dropna(subset=['W1S_2'])
        self.stats = self.stats.dropna(subset=['W1SOF_2'])
        self.stats = self.stats.dropna(subset=['W2S_1'])
        self.stats = self.stats.dropna(subset=['W2SOF_1'])
        self.stats = self.stats.dropna(subset=['W2S_2'])
        self.stats = self.stats.dropna(subset=['W2SOF_2'])
        self.stats = self.stats.dropna(subset=['RPW_1'])
        self.stats = self.stats.dropna(subset=['RPWOF_1'])
        self.stats = self.stats.dropna(subset=['RPW_2'])
        self.stats = self.stats.dropna(subset=['RPWOF_2'])
        self.stats = self.stats.dropna(subset=['TPW_1'])
        self.stats = self.stats.dropna(subset=['TPW_2'])
        self.stats = self.stats.dropna(subset=['BP_1'])
        self.stats = self.stats.dropna(subset=['BPOF_1'])
        self.stats = self.stats.dropna(subset=['BP_2'])
        self.stats = self.stats.dropna(subset=['BPOF_2'])
        self.stats = self.stats.dropna(subset=['ACES_1'])
        self.stats = self.stats.dropna(subset=['ACES_2'])

        del self.stats['UE_1']
        del self.stats['NA_1']
        del self.stats['NAOF_1']
        del self.stats['NA_2']
        del self.stats['UE_2']
        del self.stats['NAOF_2']
        del self.stats['FAST_1']
        del self.stats['FAST_2']
        del self.stats['A1S_1']
        del self.stats['A1S_2']
        del self.stats['A2S_1']
        del self.stats['A2S_2']
        del self.stats['WIS_1']
        del self.stats['WIS_2']

        print("Number of matches with statistics after dropping invalid stats is: {}".format(len(self.stats)))

        tournaments_with_stats = self.stats.ID_T.unique()
        print("Number of tournaments with statistics is: {}".format(len(tournaments_with_stats)))

        self.matches = pd.read_sql_query(
            "select * from games_atp where ID_T_G in (select ID_T from stat_atp where ID_T>= 3760)", self.conn)

        self.unfiltered_matches = pd.read_sql_query(
            "select * from games_atp", self.conn)
        print("Number of total matches in our dataset is {}:".format(len(self.matches)))

        print("The time it took to extract Panda Dataframes from Sqlite Database was --- %s seconds ---" % (
                time.time() - start_time))


        """ self.ratings = pd.read_sql_query(
            "select * from ratings_atp", self.conn)

        self.ratings['DATE_R'] = pd.to_datetime(self.ratings.DATE_R)
        self.ratings = self.ratings[self.ratings[
                                        'DATE_R'] > '2006-01-01']  # We only want to extract tournament ID's after 2006 (Total of 11033 tournaments)
"""

    def create_court_dict(self):
        court_types = pd.read_sql_query("select ID_C,NAME_C from courts", self.conn)
        for index, row in court_types.iterrows():
            self.court_dict[row["ID_C"]] = row["NAME_C"]

    def get_players(self):
        return self.players

    def get_tournaments(self):
        return self.tournaments

    def get_stats(self):
        return self.stats

    def get_matches(self):
        return self.matches

    def get_unfiltered_matches(self):
        return self.unfiltered_matches

    def get_unfiltered_tournaments(self):
        return self.unfiltered_tournaments
