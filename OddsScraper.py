import pandas as pd
import time

# Sqlite3 - Panda converter library
from sqlalchemy import create_engine
# County ISO3 code converter
import country_converter as coco

# Beautiful Soup

from bs4 import BeautifulSoup

# Requests
import urllib.request
from urllib.request import urlopen

# Selenium inports
from selenium import webdriver
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys



def df2sqlite_v2(dataframe, db_name):
    disk_engine = create_engine('sqlite:///' + db_name + '.db')
    dataframe.to_sql(db_name, disk_engine, if_exists='replace', chunksize=1000)

    """Bundan onceki !!!! Bunu unutma updated_stats V3 icin bunu yapmak daha dogru olabilir. Dont know the difference
    #     dataframe.to_sql(db_name, disk_engine ,if_exists='append')"""


class OddsScraper(object):

    def __init__(self):
        chromedriver = '/Users/aysekozlu/PyCharmProjects/TennisModel/chromedriver'
        self.driver = webdriver.Chrome(chromedriver)

    def odds_database_search(csv_file):
        dataframe = pd.read_csv(csv_file)

        # dataframe = pd.read_csv(csv_file, engine='c', dtype={'FULL': 'str', 'COUNT': 'int'},
        #                  header=0)  # converters={'COL_A': conv, 'COL_B': conv})

        """    
        start_time = time.time()
        for i in dataframe.index:
            country = dataframe.at[i, 'country']
            country = country.replace("-", " ")
            iso2_codes = coco.convert(names=country, to='ISO3')

        print("Time using the for loop was--- %s seconds ---" % (time.time() - start_time))"""

        start_time = time.time()
        dataframe['ISO3'] = dataframe.apply(
            lambda x: coco.convert(names=str(x['country']).replace("-", " "), to='ISO3'),
            axis=1)

        print("Time using apply function was--- %s seconds ---" % (time.time() - start_time))
        print(dataframe['ISO3'])
        del dataframe['country']
        del dataframe['bethard_player_1_odd']
        del dataframe['10bet_payout']
        del dataframe['betrally_player_1_odd']
        del dataframe['bethard_player_2_odd']
        del dataframe['unibet_payout']
        del dataframe['marathonbet_payout']
        del dataframe['betolimp_payout']
        del dataframe['unibet_player_1_odd']
        del dataframe['18bet_player_2_odd']
        del dataframe['18bet_player_1_odd']
        del dataframe['betrally_payout']
        del dataframe['tempobet_player_2_odd']
        del dataframe['tempobet_player_1_odd']
        del dataframe['tempobet_payout']
        del dataframe['tonybet_payout']
        del dataframe['jetbull_payout']
        del dataframe['bethard_payout']
        del dataframe['tonybet_player_2_odd']
        del dataframe['unibet_player_2_odd']
        del dataframe['5dimes_payout']
        del dataframe['18bet_payout']
        del dataframe['pinnacle_payout']
        del dataframe['pinnacle_player_2_odd']
        del dataframe['betrally_player_2_odd']
        del dataframe['10bet_player_1_odd']
        del dataframe['matchbook_player_2_odd']
        del dataframe['5dimes_player_2_odd']
        del dataframe['bet-at-home_player_2_odd']
        del dataframe['tonybet_player_1_odd']
        del dataframe['betolimp_player_1_odd']
        del dataframe['betolimp_player_2_odd']
        del dataframe['jetbull_player_2_odd']
        del dataframe['5dimes_player_1_odd']
        del dataframe['jetbull_player_1_odd']
        del dataframe['pinnacle_player_1_odd']
        del dataframe['bet-at-home_payout']
        del dataframe['10bet_player_2_odd']
        del dataframe['marathonbet_player_2_odd']
        del dataframe['bet-at-home_player_1_odd']
        del dataframe['marathonbet_player_1_odd']
        # dataset = dataframe.apply(pd.to_numeric, errors='coerce')
        dataframe.dropna(subset=['bwin_player_2_odd'], inplace=True)  # drop invalid stats (22)
        dataframe.dropna(subset=['bwin_player_1_odd'], inplace=True)  # drop invalid stats (22)

    def historical_odds_database(csv_file):
        dataframe = pd.read_csv(csv_file)

        odds_database = dataframe[
            ['player_1_name', 'player_2_name', 'player_1_score', 'player_2_score', 'bwin_player_1_odd',
             'bwin_player_2_odd', 'doubles', 'date', 'country']].copy()
        print("Starting length: {}.".format(len(odds_database)))
        odds_database.dropna(subset=['bwin_player_2_odd'], inplace=True)
        odds_database.dropna(subset=['bwin_player_1_odd'], inplace=True)
        print("Length after dropping odds: {}.".format(len(odds_database)))
        odds_database.dropna(subset=['player_1_score'], inplace=True)
        odds_database.dropna(subset=['player_2_score'], inplace=True)
        odds_database.dropna(subset=['doubles'], inplace=True)
        odds_database.dropna(subset=['date'], inplace=True)
        print("Length after dropping scores, doubles and dates.: {}.".format(len(odds_database)))

        print(odds_database.info())
        odds_database = odds_database.loc[odds_database['doubles'] == 0]
        odds_database = odds_database.reset_index(drop=True)  # reset indexes if any more rows are dropped

        # for i in odds_database.index:
        #   full_date = odds_database.at[i, 'date']
        #   year = full_date.split()[2]
        #  odds_database.at[i, 'date'] = year

        odds_database['year'] = odds_database.apply(lambda x: x['date'].split()[2], axis=1)
        odds_database['ISO3'] = odds_database.apply(
            lambda x: coco.convert(names=str(x['country']).replace("-", " "), to='ISO3'), axis=1)

        odds_database = odds_database.loc[odds_database['ISO3'] != 'not found']
        print("Length after dropping not found nationalities.: {}.".format(len(odds_database)))

        del odds_database['date']
        del odds_database['country']
        del odds_database['doubles']

        print("Dataframe conversion finished. Converting it into sqlite3 database.")
        df2sqlite_v2(odds_database, 'odds')
        print("Process finished.")

    def url_is_alive(self, url):
        """Checks that a given URL is reachable."""
        request = urllib.request.Request(url)
        request.get_method = lambda: 'HEAD'

        try:
            urllib.request.urlopen(request)
            return True
        except urllib.request.HTTPError:
            return False

    def matchUrls(self, soup):
        """
        Parameter:
       - BeautifulSoup.soup(soup) a soup object obtained from the page DOM
        Returns:
        - A list of urls to the matches.
        """
        table = soup.find('table', class_='table-main')
        matches = table.find_all('td', class_='name table-participant')
        urls = [m.a['href'] for m in matches]
        return urls

    # Gives the url's of all matches played in the specified tournament
    def tournament_odds_scraper(self, url):
        tot_urls = []
        if self.url_is_alive(url):
            print("URL IS ALIVE.")

            # driver = webdriver.Chrome()  # webdriver.Chrome("./chromedriver/chromedriver.exe")
            self.driver.get(url)
            wait_pagination = WebDriverWait(self.driver, 100).until(
                EC.element_to_be_clickable(
                    (By.ID, 'pagination')
                ))

            soup = BeautifulSoup(self.driver.page_source, 'html.parser')
            tot_urls.append(self.matchUrls(soup))

        else:
            print("URL IS NOT VALID.")
        flat_list = [item for sublist in tot_urls for item in sublist]
        print("Number of matches: {}.".format(len(flat_list)))

        return flat_list

    def odds_scraper_for_a_match(self, urls):
        # CLASS -TABLE CONTAINER
        url = urls[0]
        print(url)
        match_url = 'http://www.oddsportal.com' + url
        self.driver.get(match_url)
        soup = BeautifulSoup(self.driver.page_source, 'html.parser')
        bookie_data = self.detectBookieData(soup)
        div = soup.find('div', {'class': 'bt-3'})

        "table-main detail-odds sortable"
        print(div)
        # print(bookie_data)

    def detectBookieData(self, soup):
        """
        Returns whether there is actually bookies odds data.
        """
        return soup.find('div', class_='table-container')

    def historical_odds_scraper(self, url):
        if self.url_is_alive(url):
            print("URL IS ALIVE.")

            # driver = webdriver.Chrome()  # webdriver.Chrome("./chromedriver/chromedriver.exe")
            self.driver.get(url)
            print(self.driver.page_source)
            soup = BeautifulSoup(self.driver.page_source, 'html.parser')
            n, seasons = self.seasonParser(soup)
            print(n)
            print(seasons)
            seasons_format = ['http://www.oddsportal.com' + url
                              for url in seasons]
            print(seasons_format[0])

            self.driver.get(seasons_format[0])
            # A wait seems to be needed as the pagination loads dynamically.
            # Here, we wait up until the pagination bar can be clicked.
            wait_pagination = WebDriverWait(self.driver, 100).until(
                EC.element_to_be_clickable(
                    (By.ID, 'pagination')
                ))

            # Get the last page of the pagination.
            last_page_number = int(
                soup.find('div',
                          id='pagination').find_all('a')[-1]['x-page'])

            tot_urls = []
            for i in range(1, last_page_number + 1):

                if i is 1:
                    # Nothing to click on, the page is displayed
                    soup = BeautifulSoup(self.driver.page_source, 'html.parser')
                    tot_urls.append(self.matchUrls(soup))

                else:
                    next_page = self.driver.find_element_by_link_text(str(i))
                    next_page.send_keys(Keys.ENTER)

                    # Waiting for the pagination is as waiting for the table
                    wait_pagination = WebDriverWait(self.driver, 100).until(
                        EC.element_to_be_clickable(
                            (By.ID, 'pagination')
                        ))
                    soup = BeautifulSoup(self.driver.page_source, 'html.parser')
                    tot_urls.append(self.matchUrls(soup))

            return tot_urls

    def seasonParser(self, soup):
        """
        Parameter:
        - BeautifulSoup.soup(soup) a soup object obtained from the page DOM
        Returns:
        - The number of seasons and links to them.
        """
        seasons_box = soup.find('div', class_='main-menu2 main-menu-gray')
        n_seasons = len(seasons_box.find_all('li'))  # This will include season 2017
        seasons_links = [li.a['href'] for li in seasons_box.find_all('li')][1:]  # We don't take the first season
        return n_seasons, seasons_links


odds_scraper = OddsScraper()
"""tot_urls = odds_scraper.historical_odds_scraper("http://www.oddsportal.com/tennis/united-kingdom/atp-wimbledon/results/")

flatten_urls_list = [url for l in tot_urls for url in l]
print(len(tot_urls))
print(len(flatten_urls_list))
print(flatten_urls_list)
"""

urls = odds_scraper.tournament_odds_scraper("http://www.oddsportal.com/tennis/united-kingdom/atp-wimbledon/results/")
print(urls)
odds_scraper.odds_scraper_for_a_match(urls)

# odds_scraper.odds_database_search("world_tennis_odds.csv")

# Run this line to create the odds database
# odds_scraper.historical_odds_database("world_tennis_odds.csv")
