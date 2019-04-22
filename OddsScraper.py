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
from selenium.webdriver.common.action_chains import ActionChains

# Sqlite3 - Panda converter library
from sqlalchemy import create_engine
import re
import urllib.request


def american_to_decimal(american_odd):
    if int(american_odd) > 0:
        american_odd = float(american_odd / 100) + 1
    else:
        american_odd = float(abs(100 / american_odd)) + 1
    return american_odd


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


# Helper function to hover over odds to capture initial odds in oddsportal.com
def hover(driver, xpath):
    element = driver.find_element_by_xpath(xpath)
    hov = ActionChains(driver).move_to_element(element)
    hov.perform()


def index_of(val, in_list):
    try:
        return in_list.index(val)
    except ValueError:
        return -1


def get_player_names(player_names_in_reverse, player1, player2):
    player1_last_name_list = [name for name in player1.split() if '.' not in name]
    player2_last_name_list = [name for name in player2.split() if '.' not in name]

    if '-' in player2_last_name_list[0]:  # if last name is corrina-burata, then corrina,burata
        player2_last_name_list = player2_last_name_list[0].split('-')

    if '-' in player1_last_name_list[0]:  # if last name is corrina-burata, then corrina,burata
        player1_last_name_list = player1_last_name_list[0].split('-')
    # print("player2_last_name: {}".format(player2_last_name_list))
    #  print("player1_last_name_list: {}".format(player1_last_name_list))
    player_2_last_name_index = index_of(player2_last_name_list[0], player_names_in_reverse)

    if player_2_last_name_index == -1:
        return 0
    #  print("player_2_last_name_index: {}".format(player_2_last_name_index))

    player_2_name_reverse = player_names_in_reverse[player_2_last_name_index:]
    # print("player_2_name_reverse: {}".format(player_2_name_reverse))

    player_2_first_name_index = player_2_name_reverse.index(player2_last_name_list[-1])
    player_2_first_name = player_2_name_reverse[player_2_first_name_index + 1:]

    player1_last_name_index = index_of(player1_last_name_list[-1], player_names_in_reverse)
    if player1_last_name_index == -1:
        return 0

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


def loads_odds_into_a_list(odds_file, bookmaker_name):
    with open(odds_file, 'rb') as f:
        data = pickle.load(f)

    updated_data = [d for d in data if bookmaker_name in d]
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

    def matchUrls(self, soup, bad_oddsportal):
        """
        Parameter:
       - BeautifulSoup.soup(soup) a soup object obtained from the page DOM
        Returns:
        - A list of urls to the matches.
        """

        table = soup.find('table', class_='table-main')
        matches = table.find_all('td', class_='name table-participant')

        if bad_oddsportal:
            links = [m.contents[2]['href'] for m in matches if len(m.contents) == 3]
        else:

            links = [m.a['href'] for m in matches]

        # use below code when oddsportal decides to write bad js code

        # links = [m.contents[2]['href'] for m in matches if len(m.contents) == 3]

        return links

    def getLinks(self, url):
        html_page = urllib.request.urlopen(url)
        soup = BeautifulSoup(html_page)
        links = []

        for link in soup.findAll('a', attrs={'href': re.compile("^http://")}):
            links.append(link.get('href'))

        return links

    def historical_tournament_odds_scraper(self, url, one_page, bad_oddsportal):
        tot_urls = []
        if url_is_alive(url):
            print("URL IS ALIVE.")
            chromedriver = '/Users/aysekozlu/PyCharmProjects/TennisModel/chromedriver'
            driver = webdriver.Chrome(chromedriver)
            # driver = webdriver.Chrome()  # webdriver.Chrome("./chromedriver/chromedriver.exe")
            driver.get(url)

            if one_page:
                soup = BeautifulSoup(driver.page_source, 'html.parser')
                tot_urls.append(self.matchUrls(soup, bad_oddsportal))
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
                        tot_urls.append(self.matchUrls(soup, bad_oddsportal))

                    else:
                        next_page = driver.find_element_by_link_text(str(i))
                        next_page.send_keys(Keys.ENTER)
                        # Waiting for the pagination is as waiting for the table
                        WebDriverWait(driver, 100).until(
                            EC.element_to_be_clickable(
                                (By.ID, 'pagination')
                            ))
                        soup = BeautifulSoup(driver.page_source, 'html.parser')
                        link = self.matchUrls(soup, bad_oddsportal)
                        tot_urls.append(link)

            driver.quit()
        else:
            print("URL IS NOT VALID.")

        flat_list = [item for sublist in tot_urls for item in sublist]
        print("Number of matches: {}. For tournament {}.".format(len(flat_list), url.split(os.sep)[-3]))
        return flat_list

    # Scrape the urls of all matches played in a currently ongoing tournament.
    def current_tournament_odds_scraper(self, url, bad_oddsportal):
        tot_urls = []
        if url_is_alive(url):
            print("URL IS ALIVE.")
            chromedriver = '/Users/aysekozlu/PyCharmProjects/TennisModel/chromedriver'
            driver = webdriver.Chrome(chromedriver)
            driver.get(url)
            soup = BeautifulSoup(driver.page_source, 'html.parser')
            tot_urls.append(self.matchUrls(soup, bad_oddsportal))
            driver.quit()
        else:
            print("URL IS NOT VALID.")
        flat_list = [item for sublist in tot_urls for item in sublist]
        print("Number of matches: {}. For tournament {}.".format(len(flat_list), url.split(os.sep)[-3]))
        return flat_list

    # Scraping the odds of a future match, Only difference is we do not have results
    def odds_scraper_for_future_match(self, match_urls, odds_database_name, initial_odds_exist, save, bookmaker_name):
        chromedriver = '/Users/aysekozlu/PyCharmProjects/TennisModel/chromedriver'
        driver = webdriver.Chrome(chromedriver)
        i = 0
        init_p1_odd = 0
        init_p2_odd = 0
        # For each match
        odds_and_players = []
        for url in match_urls:
            print('Current URL {}'.format(url))
            data = []
            match_url = 'http://www.oddsportal.com' + url
            i = i + 1
            print("We are on match {}".format(i))
            driver.get(match_url)
            soup = BeautifulSoup(driver.page_source, 'html.parser')  # scrape the website

            # Selecting odds button
            expansion_button = WebDriverWait(driver, 200).until(
                EC.element_to_be_clickable((By.XPATH, '// *[ @ id = "user-header-oddsformat-expander"]')))
            expansion_button.click()
            # Selecting EU Odds
            eu_odds_button = WebDriverWait(driver, 200).until(
                EC.element_to_be_clickable((By.XPATH, '// *[ @ id = "user-header-oddsformat"] / li[1] / a')))
            eu_odds_button.click()

            bookie_data = detectBookieData(soup)
            if bookie_data is not None:
                table = bookie_data.find('table', {'class': "table-main detail-odds sortable"})  # Find the Odds Table
                # This part is scraping a Beautiful Soup Table. Returns the odds and the bookie name for the match
                table_body = table.find('tbody')
                rows = table_body.find_all('tr')
                row_counter = 0
                for row in rows:
                    row_counter = row_counter + 1
                    cols = row.find_all('td')
                    cols_text = [ele.text.strip() for ele in cols]

                    if bookmaker_name in cols_text:
                        print("We got the odds ")
                        if initial_odds_exist:
                            print("The bwin odds is at {}".format(row_counter))
                            ignored_exceptions = [EC.NoSuchElementException, EC.StaleElementReferenceException]

                            wait = WebDriverWait(driver, 200, ignored_exceptions=ignored_exceptions)
                            wait.until(EC.presence_of_element_located(
                                (By.XPATH, "//*[@id =" + '"odds-data-table"' + "]/div[1]/table/tbody/tr[" + str(
                                    row_counter) + "]/td[2]")))
                            hover(driver, "//*[@id =" + '"odds-data-table"' + "]/div[1]/table/tbody/tr[" + str(
                                row_counter) + "]/td[2]")

                            data_p1 = driver.find_element_by_xpath("//*[@id='tooltiptext']")
                            init_p1_odd = float(data_p1.text.split()[-1])

                            wait.until(EC.presence_of_element_located(
                                (By.XPATH, "//*[@id =" + '"odds-data-table"' + "]/div[1]/table/tbody/tr[" + str(
                                    row_counter) + "]/td[3]")))

                            hover(driver, "//*[@id =" + '"odds-data-table"' + "]/div[1]/table/tbody/tr[" + str(
                                row_counter) + "]/td[3]")

                            data_p2 = driver.find_element_by_xpath("//*[@id='tooltiptext']")
                            init_p2_odd = float(data_p2.text.split()[-1])
                            print(init_p1_odd)
                            print(init_p2_odd)
                        data.append(
                            [ele for ele in cols_text if ele])
                        if initial_odds_exist:
                            data.append([init_p1_odd,
                                         init_p2_odd])

                            # Here we start a list of operations to get player names correctly.
                player_url = url.strip().split(os.sep)[-2]
                player_names_in_reverse = player_url.split('-')[:-1]  # get names from the url (they are in reverse)

                print("These are player names I got from url {}".format(player_names_in_reverse))
                content = soup.find('div', id='col-content')
                content.span.extract()

                player1, player2 = content.h1.get_text().lower().split(
                    ' - ')  # These names are in format Djokovic N.-Nadal R.

                # Function call gets player names in FULL --> First Name Last Name format

                if get_player_names(player_names_in_reverse, player1, player2) == 0:
                    print("There was a problem with players names in this match")
                    continue
                else:
                    player1_name, player2_name = get_player_names(player_names_in_reverse, player1, player2)
                    data = [item for sublist in data for item in sublist]
                    data.append(player1_name)
                    data.append(player2_name)

                    # data = ['bookie',odd 1, odd 2, payout,player1name,player2name]
                    odds_and_players.append(data)

            else:
                print('We were unable to find bookie data')
                continue
        # ['Pinnacle', '+142', '-172', '95.6%', 'taylor harry fritz', 'kei nishikori']
        # ['Piinacle', '4.14', '1.24', '95.4%', 'nicolas jarry', 'alexander zverev']
        # ['Pinnacle', '1.54', '2.51', '95.4%', 'jaume munar', 'frances tiafoe']
        odds_and_players[0] = ['Pinnacle', '+142', '-172', '95.6%', 'taylor harry fritz', 'kei nishikori']
        odds_and_players[6] = ['Piinacle', '4.14', '1.24', '95.4%', 'nicolas jarry', 'alexander zverev']
        odds_and_players[7] = ['Pinnacle', '1.54', '2.51', '95.4%', 'jaume munar', 'frances tiafoe']

        print(odds_and_players)
        # del odds_and_players[0]
        # odds_and_players.append(['Pinnacle', '1.49', '2.80', '97.2%', 'felix auger aliassime', 'juan ignacio londero'])
        print((odds_and_players))
        if save:
            with open(odds_database_name, "wb") as fp:  # Pickling
                pickle.dump(odds_and_players, fp)

            driver.quit()

    # Scraping the odds and a result of a finished match
    def odds_scraper_for_a_finished_match(self, match_urls, odds_database_name, initial_odds_exist, save):

        # get the fuckin local path
        chromedriver = '/Users/aysekozlu/PyCharmProjects/TennisModel/chromedriver'
        driver = webdriver.Chrome(chromedriver)
        i = 0
        init_p1_odd = 0
        init_p2_odd = 0
        # For each match
        odds_and_players = []
        for url in match_urls:
            print('Current URL {}'.format(url))
            data = []
            match_url = 'http://www.oddsportal.com' + url
            i = i + 1
            print("We are on match {}".format(i))
            driver.get(match_url)
            soup = BeautifulSoup(driver.page_source, 'html.parser')  # scrape the website

            # Selecting odds button
            expansion_button = WebDriverWait(driver, 200).until(
                EC.element_to_be_clickable((By.XPATH, '// *[ @ id = "user-header-oddsformat-expander"]')))
            expansion_button.click()
            # Selecting EU Odds
            eu_odds_button = WebDriverWait(driver, 200).until(
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

                else:
                    result = 0

                table = bookie_data.find('table', {'class': "table-main detail-odds sortable"})  # Find the Odds Table
                # This part is scraping a Beautiful Soup Table. Returns the odds and the bookie name for the match
                table_body = table.find('tbody')
                rows = table_body.find_all('tr')
                row_counter = 0
                for row in rows:
                    row_counter = row_counter + 1
                    cols = row.find_all('td')
                    cols_text = [ele.text.strip() for ele in cols]

                    if 'bwin' in cols_text:
                        # for event_tag in cols:
                        # if event_tag.find("div", onmouseover=True) is not None:
                        #  print(event_tag)
                        # print(event_tag.find("div", onmouseover=True))
                        if initial_odds_exist:
                            print("The bwin odds is at {}".format(row_counter))
                            ignored_exceptions = [EC.NoSuchElementException, EC.StaleElementReferenceException]

                            wait = WebDriverWait(driver, 200, ignored_exceptions=ignored_exceptions)
                            wait.until(EC.presence_of_element_located(
                                (By.XPATH, "//*[@id =" + '"odds-data-table"' + "]/div[1]/table/tbody/tr[" + str(
                                    row_counter) + "]/td[2]")))
                            hover(driver, "//*[@id =" + '"odds-data-table"' + "]/div[1]/table/tbody/tr[" + str(
                                row_counter) + "]/td[2]")
                            # wait.until(EC.visibility_of_element_located(
                            #    (By.XPATH, "//*[@id =" + '"odds-data-table"' + "]/div[1]/table/tbody/tr[8]/td[2]")))

                            data_p1 = driver.find_element_by_xpath("//*[@id='tooltiptext']")
                            init_p1_odd = float(data_p1.text.split()[-1])

                            wait.until(EC.presence_of_element_located(
                                (By.XPATH, "//*[@id =" + '"odds-data-table"' + "]/div[1]/table/tbody/tr[" + str(
                                    row_counter) + "]/td[3]")))

                            hover(driver, "//*[@id =" + '"odds-data-table"' + "]/div[1]/table/tbody/tr[" + str(
                                row_counter) + "]/td[3]")

                            data_p2 = driver.find_element_by_xpath("//*[@id='tooltiptext']")
                            init_p2_odd = float(data_p2.text.split()[-1])
                            print(init_p1_odd)
                            print(init_p2_odd)
                        data.append(
                            [ele for ele in cols_text if ele])
                        if initial_odds_exist:
                            data.append([init_p1_odd,
                                         init_p2_odd])  # ['bookie',odd 1, odd 2, payout,initial_odd1, initial_odd2]
                # print('ali')
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

                if get_player_names(player_names_in_reverse, player1, player2) == 0:
                    print("There was a problem with players names in this match")
                    continue
                else:
                    player1_name, player2_name = get_player_names(player_names_in_reverse, player1, player2)
                    data = [item for sublist in data for item in sublist]
                    data.append(player1_name)
                    data.append(player2_name)
                    data.append(result)
                    # The final format is given below
                    # data = ['bookie',odd 1, odd 2, payout,initial_odd1, initial_odd2,player1name,player2name,result]
                    odds_and_players.append(data)
                # print(odds_and_players)  # This list includes odds + player names

            else:
                print('We were unable to find bookie data')
                continue
        print(len(odds_and_players))
        if save:
            with open(odds_database_name, "wb") as fp:  # Pickling
                pickle.dump(odds_and_players, fp)

            driver.quit()

    # Scrapes all the old versions of a specified tournament
    def scrape_all_versions_of_a_tournament(self, url):
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
# Beijing 2018
"""
urls = odds_scraper.historical_tournament_odds_scraper("https://www.oddsportal.com/tennis/china/atp-beijing/results/",one_page=True)
odds_scraper.odds_scraper_for_a_finished_match(urls, "beijing_2018_odds.pkl", save=True)
"""
# Tokyo 2018
"""
urls = odds_scraper.historical_tournament_odds_scraper("https://www.oddsportal.com/tennis/japan/atp-tokyo/results/",one_page=True)
odds_scraper.odds_scraper_for_a_finished_match(urls, "tokyo_2018_odds.pkl", save=True)
"""
# Wimbledon 2018

"""
urls = odds_scraper.historical_tournament_odds_scraper(
    "https://www.oddsportal.com/tennis/united-kingdom/wta-wimbledon/results/", one_page=False)
odds_scraper.odds_scraper_for_a_finished_match(urls, "wimbledon_wta_odds.pkl", initial_odds_exist=True, save=True)
"""
"""
# Paris Open 2018
urls = odds_scraper.historical_tournament_odds_scraper(
    "https://www.oddsportal.com/tennis/france/atp-paris/results/", one_page=False)
odds_scraper.odds_scraper_for_a_finished_match(urls, "paris_open_2018_v2.pkl", initial_odds_exist=True, save=True)
"""
# Auckland 2018
"""
urls = odds_scraper.historical_tournament_odds_scraper("http://www.oddsportal.com/tennis/new-zealand/atp-auckland/results/",one_page=True)
print(len(urls))
odds
_scraper.odds_scraper_for_a_finished_match(urls, "auckland_open_2018_odds_v2.pkl", save=True)
"""
# Loading odds for US Open 2018

"""
#https://www.oddsportal.com/tennis/usa/wta-us-open/results/
urls = odds_scraper.historical_tournament_odds_scraper("https://www.oddsportal.com/tennis/usa/atp-us-open/results/",
                                                       one_page=False)
print(len(urls))
odds_scraper.odds_scraper_for_a_finished_match(urls, "us_open_2018_odds_v2.pkl", initial_odds_exist=True, save=True)
"""
'/tennis/monaco/atp-monte-carlo/basilashvili-nikoloz-fucsovics-marton-l0DsQpGa/'
'/tennis/monaco/atp-monte-carlo/bautista-agut-roberto-millman-john-dvdOxOa5/'
'/tennis/monaco/atp-monte-carlo/jaziri-malek-lajovic-dusan-GpxAW1eM/'
# Loading odds for ATP Finals Nov11-12
# Loading odds for US Open 2018

urls = odds_scraper.historical_tournament_odds_scraper(
    "https://www.oddsportal.com/tennis/spain/atp-barcelona/", bad_oddsportal=False, one_page=True)
print(urls[10:22])
updated_urls = urls[10:22]
del updated_urls[2]
odds_scraper.odds_scraper_for_future_match(updated_urls, "barcelona23april.pkl", initial_odds_exist=False, save=True,
                                           bookmaker_name='Pinnacle')
curr_list = loads_odds_into_a_list('barcelona23april.pkl', bookmaker_name='Pinnacle')
"""
APRIL 16 
urls = ['/tennis/monaco/atp-monte-carlo/auger-aliassime-felix-londero-juan-ignacio-YXlYesea/',
        '/tennis/monaco/atp-monte-carlo/mannarino-adrian-norrie-cameron-IZHoP4V5/',
        '/tennis/monaco/atp-monte-carlo/tsonga-jo-wilfried-fritz-taylor-harry-rebcoen8/',
        '/tennis/monaco/atp-monte-carlo/verdasco-fernando-herbert-pierre-hugues-dQGkOOpC/',
        '/tennis/monaco/atp-monte-carlo/cilic-marin-pella-guido-KlipguuI/',
        '/tennis/monaco/atp-monte-carlo/simon-gilles-popyrin-alexei-j9aTdNug/',
        '/tennis/monaco/atp-monte-carlo/djokovic-novak-kohlschreiber-philipp-xQoHQHBA/',
        '/tennis/monaco/atp-monte-carlo/sonego-lorenzo-khachanov-karen-AczPPjq7/',
        '/tennis/monaco/atp-monte-carlo/munar-jaume-coric-borna-40sLa3Pp/',
        '/tennis/monaco/atp-monte-carlo/wawrinka-stan-cecchinato-marco-Cp1gFT6s/']
odds_scraper = OddsScraper()

#HISTORICAL WIMBLEDON
#Loading historical odds for a tournament
tot_urls = odds_scraper.historical_odds_for_tournament_scraper(
    "http://www.oddsportal.com/tennis/united-kingdom/atp-wimbledon/results/")

flatten_urls_list = [url for l in tot_urls for url in l]
print(len(flatten_urls_list))
odds_scraper.odds_scraper_for_a_finished_match(flatten_urls_list,'historical_wimbledon_odds', save=True)
"""

# odds_scraper.odds_database_search("world_tennis_odds.csv")

# Run this line to create the odds database
# odds_scraper.historical_odds_database("world_tennis_odds.csv")
