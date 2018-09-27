import os
import pickle
import time
# Requests
import urllib.request

# County ISO3 code converter
import country_converter as coco
import pandas as pd
# Beautiful Soup
from bs4 import BeautifulSoup
# Selenium inports
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.wait import WebDriverWait
# Sqlite3 - Panda converter library
from sqlalchemy import create_engine


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


def get_player_names(player_names_in_reverse, player1, player2):
    player1_last_name_list = [name for name in player1.split() if '.' not in name]
    player2_last_name_list = [name for name in player2.split() if '.' not in name]

    if '-' in player2_last_name_list[0]:  # if last name is corrina-burata, then corrina,burata
        player2_last_name_list = player2_last_name_list[0].split('-')

    if '-' in player1_last_name_list[0]:  # if last name is corrina-burata, then corrina,burata
        player1_last_name_list = player1_last_name_list[0].split('-')
    # print("player2_last_name: {}".format(player2_last_name_list))
    #  print("player1_last_name_list: {}".format(player1_last_name_list))

    player_2_last_name_index = player_names_in_reverse.index(player2_last_name_list[0])  # smith
    #  print("player_2_last_name_index: {}".format(player_2_last_name_index))

    player_2_name_reverse = player_names_in_reverse[player_2_last_name_index:]
    # print("player_2_name_reverse: {}".format(player_2_name_reverse))

    player_2_first_name_index = player_2_name_reverse.index(player2_last_name_list[-1])
    player_2_first_name = player_2_name_reverse[player_2_first_name_index + 1:]

    player1_last_name_index = player_names_in_reverse.index(player1_last_name_list[-1])
    player1_last_name = player_names_in_reverse[:player1_last_name_index + 1]
    player1_first_name = player_names_in_reverse[player1_last_name_index + 1:player_2_last_name_index:1]

    print("Fist name of Player 1 is {}:".format(player1_first_name))
    print("Last name of Player 1 is {}:".format(player1_last_name))
    print("First name of Player 2: {}".format(player_2_first_name))  # de minaur alex
    print("Last name of Player 2 is: {}.".format(player2_last_name_list))
    player1_name = str.join(' ', player1_first_name + player1_last_name)
    player2_name = str.join(' ', player_2_first_name + player2_last_name_list)
    print("Player 1 name is {}".format(player1_name))
    print('Player 2s name is {}'.format(player2_name))
    return player1_name, player2_name


def loads_odds_into_a_list(odds_file):
    with open(odds_file, 'rb') as f:
        data = pickle.load(f)

    updated_data = [d for d in data if 'bwin' in d]
    # for d in updated_data:
    # print(d)
    return updated_data


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
        # for tx in soup.find_all('th'):
        #    table_headers.append(tx)
        # print(table_headers)
        matches = table.find_all('td', class_='name table-participant')
        links = [m.a['href'] for m in matches]
        return links

    # Gives the url's of all matches played in the specified HISTORICAL tournament.
    # An example url would be: "http://www.oddsportal.com/tennis/united-kingdom/atp-wimbledon/results/"
    # This would give the odds of 241 Wimbledon 2017 matches.
    def historical_tournament_odds_scraper(self, url,one_page):
        tot_urls = []
        if url_is_alive(url):
            print("URL IS ALIVE.")
            chromedriver = '/Users/aysekozlu/PyCharmProjects/TennisModel/chromedriver'
            driver = webdriver.Chrome(chromedriver)
            # driver = webdriver.Chrome()  # webdriver.Chrome("./chromedriver/chromedriver.exe")
            driver.get(url)

            if one_page:
                soup = BeautifulSoup(driver.page_source, 'html.parser')
                tot_urls.append(self.matchUrls(soup))
            else:
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
                        link = self.matchUrls(soup)
                        tot_urls.append(link)

            driver.quit()
        else:
            print("URL IS NOT VALID.")

        flat_list = [item for sublist in tot_urls for item in sublist]
        print("Number of matches: {}. For tournament {}.".format(len(flat_list), url.split(os.sep)[-3]))
        return flat_list

    # Scrape the odds and player info from a match url for a CURRENT tournament
    def current_tournament_odds_scraper(self, url):
        tot_urls = []
        if url_is_alive(url):
            print("URL IS ALIVE.")
            chromedriver = '/Users/aysekozlu/PyCharmProjects/TennisModel/chromedriver'
            driver = webdriver.Chrome(chromedriver)
            driver.get(url)
            soup = BeautifulSoup(driver.page_source, 'html.parser')
            tot_urls.append(self.matchUrls(soup))
            driver.quit()
        else:
            print("URL IS NOT VALID.")
        flat_list = [item for sublist in tot_urls for item in sublist]
        print("Number of matches: {}. For tournament {}.".format(len(flat_list), url.split(os.sep)[-3]))
        return flat_list

    # Scraping the odds of a future match
    def odds_scraper_for_future_match(self, match_urls, odds_database_name, save):
        # get the fuckin local path
        chromedriver = '/Users/aysekozlu/PyCharmProjects/TennisModel/chromedriver'
        driver = webdriver.Chrome(chromedriver)
        i = 0
        # For each match
        odds_and_players = []
        for url in match_urls:
            data = []
            match_url = 'http://www.oddsportal.com' + url
            i = i + 1
            driver.get(match_url)
            soup = BeautifulSoup(driver.page_source, 'html.parser')  # scrape the website

            # Selecting odds button
            expansion_button = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.XPATH, '// *[ @ id = "user-header-oddsformat-expander"]')))
            expansion_button.click()
            # Selecting EU Odds
            eu_odds_button = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.XPATH, '// *[ @ id = "user-header-oddsformat"] / li[1] / a')))
            eu_odds_button.click()

            bookie_data = detectBookieData(soup)
            if bookie_data is not None:
                table = bookie_data.find('table', {'class': "table-main detail-odds sortable"})  # Find the Odds Table
                # This part is scraping a Beautiful Soup Table. Returns the odds and the bookie name for the match
                table_body = table.find('tbody')
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
                content.span.extract()
                # all_result = content.find('p', class_='result').get_text()

                player1, player2 = content.h1.get_text().lower().split(
                    ' - ')  # These names are in format Djokovic N.-Nadal R.

                # Function call gets player names in FULL --> First Name Last Name format
                player1_name, player2_name = get_player_names(player_names_in_reverse, player1, player2)
                data = [item for sublist in data for item in sublist]
                data.append(player1_name)
                data.append(player2_name)
            else:
                print('We were unable to find bookie data')
        print(len(odds_and_players))
        if save:
            with open(odds_database_name, "wb") as fp:  # Pickling
                pickle.dump(odds_and_players, fp)

            driver.quit()

    # Scraping the odds and a result of a finished match
    def odds_scraper_for_a_match(self, match_urls, odds_database_name, save):

        # get the fuckin local path
        chromedriver = '/Users/aysekozlu/PyCharmProjects/TennisModel/chromedriver'
        driver = webdriver.Chrome(chromedriver)
        i = 0
        # For each match
        odds_and_players = []
        for url in match_urls:
            data = []
            match_url = 'http://www.oddsportal.com' + url
            i = i + 1
            driver.get(match_url)
            soup = BeautifulSoup(driver.page_source, 'html.parser')  # scrape the website

            # Selecting odds button
            expansion_button = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.XPATH, '// *[ @ id = "user-header-oddsformat-expander"]')))
            expansion_button.click()
            # Selecting EU Odds
            eu_odds_button = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.XPATH, '// *[ @ id = "user-header-oddsformat"] / li[1] / a')))
            eu_odds_button.click()

            bookie_data = detectBookieData(soup)

            final_result = soup.find('p', class_='result')

            if final_result is not None:
                final_result = final_result.strong.get_text()
            else:
                print("The final result was not accessible for this match, therefore we skip it.")
                continue

            final_result = final_result.split(':') if final_result != u'' else ['']
            print(final_result)

            if bookie_data is not None and final_result is not None and len(final_result) == 2:
                if int(final_result[0]) > int(final_result[1]):
                    result = 1
                    print(result)
                else:
                    result = 0
                    print(result)
                table = bookie_data.find('table', {'class': "table-main detail-odds sortable"})  # Find the Odds Table
                # This part is scraping a Beautiful Soup Table. Returns the odds and the bookie name for the match
                table_body = table.find('tbody')
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
                content.span.extract()
                # all_result = content.find('p', class_='result').get_text()

                player1, player2 = content.h1.get_text().lower().split(
                    ' - ')  # These names are in format Djokovic N.-Nadal R.

                # Function call gets player names in FULL --> First Name Last Name format
                player1_name, player2_name = get_player_names(player_names_in_reverse, player1, player2)
                data = [item for sublist in data for item in sublist]
                data.append(player1_name)
                data.append(player2_name)
                data.append(result)

                odds_and_players.append(data)
                # print(odds_and_players)  # This list includes odds + player names

            else:
                print('We were unable to find bookie data')
        print(len(odds_and_players))
        if save:
            with open(odds_database_name, "wb") as fp:  # Pickling
                pickle.dump(odds_and_players, fp)

            driver.quit()

    # Scrapes all the old versions of a specified tournament
    def historical_odds_for_tournament_scraper(self, url):
        if url_is_alive(url):
            print("URL IS ALIVE.")
            chromedriver = '/Users/aysekozlu/PyCharmProjects/TennisModel/chromedriver'
            driver = webdriver.Chrome(chromedriver)
            # driver = webdriver.Chrome()  # webdriver.Chrome("./chromedriver/chromedriver.exe")
            driver.get(url)

            soup = BeautifulSoup(driver.page_source, 'html.parser')
            n, seasons = self.season_parser(soup)
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
            total_pages = len(tot_urls)
            driver.quit()
            print("In total we have scraped a total of {} match urls from {} oddsportal webpages".format(len(tot_urls),
                                                                                                         total_pages))
            return tot_urls

    # Function gets all the seasons of a single tournament
    def season_parser(self, soup):
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

#US OPEN 2018
urls = odds_scraper.historical_tournament_odds_scraper("http://www.oddsportal.com/tennis/new-zealand/atp-auckland/results/",one_page=True)
print(len(urls))
odds_scraper.odds_scraper_for_a_match(urls, "auckland_open_2018_odds_v2.pkl", save=True)
"""

"""
# Loading odds of a current tournament. US Open 2018 
urls = odds_scraper.current_tournament_odds_scraper("http://www.oddsportal.com/tennis/usa/atp-us-open/")
print(len(urls))
odds_scraper.odds_scraper_for_future_match(urls, "us_open_2018_august22_odds.pkl", save=True)


"""
"""
odds_scraper = OddsScraper()

#HISTORICAL WIMBLEDON
#Loading historical odds for a tournament
tot_urls = odds_scraper.historical_odds_for_tournament_scraper(
    "http://www.oddsportal.com/tennis/united-kingdom/atp-wimbledon/results/")

flatten_urls_list = [url for l in tot_urls for url in l]
print(len(flatten_urls_list))
odds_scraper.odds_scraper_for_a_match(flatten_urls_list,'historical_wimbledon_odds', save=True)
"""

# loading wimbledon 2018 odds
"""
urls = odds_scraper.tournament_odds_scraper("http://www.oddsportal.com/tennis/united-kingdom/atp-wimbledon/results/","wimbledon_2018_odds_v3")
print(len(urls))

odds_scraper.odds_scraper_for_a_match(urls, save=True)


"""

# odds_scraper.odds_database_search("world_tennis_odds.csv")

# Run this line to create the odds database
# odds_scraper.historical_odds_database("world_tennis_odds.csv")
