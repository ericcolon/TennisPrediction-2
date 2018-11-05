from collections import defaultdict
import pandas as pd
import boto3
import numpy as np
import collections
p1_list  = [1,2,3]
p2_list = [4,5,6]
index_games_dict_for_prediction = {}
for i, (p1, p2) in enumerate(zip(p1_list, p2_list)):
    print (p1)
    print(p2)
    index_games_dict_for_prediction[i]=  [p1, p2]

print(index_games_dict_for_prediction)
"""
A tool for retrieving basic information from the running EC2 instances.


# Connect to EC2
ec2 = boto3.resource('ec2', region_name='us-east-1')

# Get information for all running instances
running_instances = ec2.instances.filter(Filters=[{
    'Name': 'instance-state-name',
    'Values': ['running']}])

ec2info = defaultdict()
for instance in running_instances:
    for tag in instance.tags:
        if 'Name' in tag['Key']:
            name = tag['Value']
    # Add instance info to a dictionary
    ec2info[instance.id] = {
        'Type': instance.instance_type,
        'State': instance.state['Name'],
        'Private IP': instance.private_ip_address,
        'Public IP': instance.public_ip_address,
        'Launch Time': instance.launch_time
    }

attributes = [ 'Type', 'State', 'Private IP', 'Public IP', 'Launch Time']
for instance_id, instance in ec2info.items():
    for key in attributes:
        print("{0}: {1}".format(key, instance[key]))
    print("------")
"""
# self.create_100_decision_stumps(x_train, y_train, x_test, 0.5, 4)  # create DT stumps
#
# # these are the new feature and label set we will training SGD Classifier
# decision_stump_x_train, decision_stump_y_train = self.create_new_vector_label_dataset(
#     self.old_feature_to_new_feature_dictionary)
#
# test_data, test_label = self.create_new_vector_label_dataset(
#     self.old_feature_to_new_feature_dictionary_for_testing)
# print(len(decision_stump_x_train))
# print(len(test_data))
# # creating 100 1d vectors from our test dataset
#
# linear_clf.fit(decision_stump_x_train, decision_stump_y_train)
#
# print("Decision Tree model training accuracy {}.".format(
#     linear_clf.score(decision_stump_x_train, decision_stump_y_train)))
# print("Decision Tree model testing accuracy {}.".format(linear_clf.score(test_data, test_label)))
# y_pred = linear_clf.predict(test_data)
# false_positive_rate, true_positive_rate, thresholds = roc_curve(test_label, y_pred)
# roc_auc = auc(false_positive_rate, true_positive_rate)
# print("ROC SCORE: {}".format(roc_auc))
# self.old_feature_to_new_feature_dictionary.clear()
# self.old_feature_to_new_feature_dictionary_for_testing.clear()


"""
 for players, result in match_to_results_dictionary.copy().items():
                    p1 = list(players)[0]
                    p2 = list(players)[1]
                    p1_list.append(p2)
                    p2_list.append(p1)
                    results.append(int(abs(result - 1)))
                    match_to_results_dictionary[tuple([p2, p1])] = abs(int(abs(result - 1)))
                    odds = match_to_odds_dictionary[tuple([p1, p2])]

                    match_to_odds_dictionary[tuple([p2, p1])] = list(reversed(odds))"""


"""

        matches = self.unfiltered_matches
        start_time = time.time()
        invalid = 0
        stats["H12H"] = ""
        stats["H21H"] = ""
        for i in stats.index:
            print("We are H2H stats at index {}".format(i))
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


"""