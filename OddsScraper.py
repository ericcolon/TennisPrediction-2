import pandas as pd
import time
import os
# Sqlite3 - Panda converter library
from sqlalchemy import create_engine
# County ISO3 code converter
import country_converter as coco

# Beautiful Soup

from bs4 import BeautifulSoup

# Requests
import urllib.request

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


def detectBookieData(soup):
    """
    Returns whether there is actually bookies odds data.
    """
    return soup.find('div', class_='table-container')


def url_is_alive(url):
    """Checks that a given URL is reachable."""
    request = urllib.request.Request(url)
    request.get_method = lambda: 'HEAD'

    try:
        urllib.request.urlopen(request)
        return True
    except urllib.request.HTTPError:
        return False


class OddsScraper(object):

    def __init__(self):
        pass

    def odds_database_search(self, csv_file):
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

    def historical_odds_database(self, csv_file):
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

    # Gives the url's of all matches played in the specified tournament. Tournament must be in 2018
    # An example url would be: "http://www.oddsportal.com/tennis/united-kingdom/atp-wimbledon/results/"
    # This would give the odds of 241 Wimbledon 2018 matches.
    def tournament_odds_scraper(self, url):
        tot_urls = []
        if url_is_alive(url):
            print("URL IS ALIVE.")
            chromedriver = '/Users/aysekozlu/PyCharmProjects/TennisModel/chromedriver'
            driver = webdriver.Chrome(chromedriver)
            # driver = webdriver.Chrome()  # webdriver.Chrome("./chromedriver/chromedriver.exe")
            driver.get(url)
            soup = BeautifulSoup(driver.page_source, 'html.parser')
            WebDriverWait(driver, 100).until(
                EC.element_to_be_clickable(
                    (By.ID, 'pagination')
                ))

            # Get the last page of the pagination.
            last_page_number = int(
                soup.find('div',
                          id='pagination').find_all('a')[-1]['x-page'])

            for i in range(1, last_page_number + 1):

                if i is 1:
                    # Nothing to click on, the page is displayed
                    soup = BeautifulSoup(driver.page_source, 'html.parser')
                    tot_urls.append(self.matchUrls(soup))

                else:
                    next_page = driver.find_element_by_link_text(str(i))
                    next_page.send_keys(Keys.ENTER)
                    # Waiting for the pagination is as waiting for the table
                    WebDriverWait(driver, 100).until(
                        EC.element_to_be_clickable(
                            (By.ID, 'pagination')
                        ))
                    soup = BeautifulSoup(driver.page_source, 'html.parser')
                    tot_urls.append(self.matchUrls(soup))
            driver.quit()
        else:
            print("URL IS NOT VALID.")

        flat_list = [item for sublist in tot_urls for item in sublist]
        print("Number of matches: {} for tournament {}.".format(len(flat_list), url.split(os.sep)[-2]))
        return flat_list

    # Scrape the odds and player info from a match url
    def odds_scraper_for_a_match(self, match_urls):
        # CLASS -TABLE CONTAINER

        chromedriver = '/Users/aysekozlu/PyCharmProjects/TennisModel/chromedriver'
        driver = webdriver.Chrome(chromedriver)
        i = 0
        # For each match
        for url in match_urls:
            match_url = 'http://www.oddsportal.com' + url
            i = i + 1
            driver.get(match_url)
            soup = BeautifulSoup(driver.page_source, 'html.parser')  # scrape the website
            bookie_data = detectBookieData(soup)
            if bookie_data is not None:
                table = bookie_data.find('table', {'class': "table-main detail-odds sortable"})  # Find the Odds Table
                # This part is scraping a Beautiful Soup Table. Returns the odds and the bookie name for the match
                table_body = table.find('tbody')
                data = []
                rows = table_body.find_all('tr')
                for row in rows:
                    cols = row.find_all('td')
                    cols = [ele.text.strip() for ele in cols]
                    if 'bwin' in cols:
                        data.append(
                            [ele for ele in cols if ele])  # Now our list includes ['bookie',odd 1, odd 2, payout]

                # Here we start a list of operations to get player names correctly.
                player_url = url.strip().split(os.sep)[-2]
                player_names_in_reverse = player_url.split('-')[:-1]  # get names from the url (they are in reverse)
                print("These are player names I got from url {}".format(player_names_in_reverse))
                content = soup.find('div', id='col-content')
                # Get rid of the span tag.
                content.span.extract()
                # Basic match info.
                player1, player2 = content.h1.get_text().lower().split(
                    ' - ')  # These names are in format Djokovic N.-Nadal R.

                player2_last_name = player2.split()[:-1]
                print("Last name of second player {}.".format(player2_last_name))  # de minaur
                player2_last_name = ['corrina-busta']

                for i in range(len(player2_last_name)):

                    el = player2_last_name[i]
                    if '-' in el:

                        el = el.split('-')

                    print(player2_last_name)

                print("Length of player 2's last name is {}.".format(len(player2_last_name)))

                player_2_last_name_index = player_names_in_reverse.index(player2_last_name[0])
                player_2_name_reverse = player_names_in_reverse[player_2_last_name_index:]

                print("Player 2 name in reverse: {}".format(player_2_name_reverse))  # de minaur alex

                player_2_first_name_index = player_2_name_reverse.index(player2_last_name[-1])
                player_2_first_name = player_names_in_reverse[player_2_first_name_index:]

                print("First Name of second player: {}".format(player_2_first_name))  # de minaur alex

                data.append(player_names_in_reverse)
                odds_and_players = [item for sublist in data for item in sublist]
                # print(odds_and_players)  # This list includes odds + player names
            else:
                print('We were unable to find bookie data')
        driver.quit()
        print(i)

    # Scrapes all the old versions of a specified tournament
    def historical_odds_for_tournament_scraper(self, url):
        if url_is_alive(url):
            print("URL IS ALIVE.")
            chromedriver = '/Users/aysekozlu/PyCharmProjects/TennisModel/chromedriver'
            driver = webdriver.Chrome(chromedriver)
            # driver = webdriver.Chrome()  # webdriver.Chrome("./chromedriver/chromedriver.exe")
            driver.get(url)

            soup = BeautifulSoup(driver.page_source, 'html.parser')
            n, seasons = self.seasonParser(soup)
            print("There are {} seasons, which are {}".format(n, seasons))

            seasons_format = ['http://www.oddsportal.com' + url
                              for url in seasons]

            print("First season is {}".format(seasons_format[0]))
            tot_urls = []
            for season in seasons_format:
                print("Scraping match urls from {}.".format(season))
                driver.get(season)
                # A wait seems to be needed as the pagination loads dynamically.
                # Here, we wait up until the pagination bar can be clicked.
                wait_pagination = WebDriverWait(driver, 100).until(
                    EC.element_to_be_clickable(
                        (By.ID, 'pagination')
                    ))

                # Get the last page of the pagination.
                last_page_number = int(
                    soup.find('div',
                              id='pagination').find_all('a')[-1]['x-page'])

                for i in range(1, last_page_number + 1):

                    if i is 1:
                        # Nothing to click on, the page is displayed
                        soup = BeautifulSoup(driver.page_source, 'html.parser')
                        tot_urls.append(self.matchUrls(soup))

                    else:
                        next_page = driver.find_element_by_link_text(str(i))
                        next_page.send_keys(Keys.ENTER)

                        # Waiting for the pagination is as waiting for the table
                        wait_pagination = WebDriverWait(driver, 100).until(
                            EC.element_to_be_clickable(
                                (By.ID, 'pagination')
                            ))
                        soup = BeautifulSoup(driver.page_source, 'html.parser')
                        tot_urls.append(self.matchUrls(soup))
            total_pages = len(tot_urls)
            driver.quit()
            print("In total we have scraped a total of {} match urls from {} oddsportal webpages".format(len(tot_urls),
                                                                                                         total_pages))
            return tot_urls

    # Function gets all the seasons of a single tournament
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

    # This function parses the archived result page of the Odds Portal website.

    def archivedResults(self, page_source):
        """
        This function parses the archived result page of the Odds Portal website.
        """
        table_soup = BeautifulSoup(page_source, 'html.parser').find('div', id='col-content').find('table')
        tournaments_list = []

        # Looking for all the anchoring tags. Grab their name and url.
        for a_tag in table_soup.find_all('a', foo='f'):
            dict_tournament = {'name': '', 'url': '', }
            dict_tournament['name'], dict_tournament['url'] = a_tag.get_text(), 'http://www.oddsportal.com' + a_tag[
                'href']
            tournaments_list.append(dict_tournament)

        return tournaments_list


odds_scraper = OddsScraper()
"""
tot_urls = odds_scraper.historical_odds_for_tournament_scraper(
    "http://www.oddsportal.com/tennis/united-kingdom/atp-wimbledon/results/")

flatten_urls_list = [url for l in tot_urls for url in l]
print(len(flatten_urls_list))
"""

urls = odds_scraper.tournament_odds_scraper("http://www.oddsportal.com/tennis/united-kingdom/atp-wimbledon/results/")
print(len(urls))
odds_scraper.odds_scraper_for_a_match(urls)

# odds_scraper.odds_database_search("world_tennis_odds.csv")

# Run this line to create the odds database
# odds_scraper.historical_odds_database("world_tennis_odds.csv")


# Archived results page
"""
chromedriver = '/Users/aysekozlu/PyCharmProjects/TennisModel/chromedriver'
driver = webdriver.Chrome(chromedriver)
driver.get('http://www.oddsportal.com/tennis/results/')
wait = WebDriverWait(driver, 10).until(EC.visibility_of_element_located((By.CSS_SELECTOR, "table.table-main.sport")))
tournaments_list = odds_scraper.archivedResults(driver.page_source)
driver.quit()

"""
