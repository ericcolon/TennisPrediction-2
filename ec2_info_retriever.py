from collections import defaultdict
import pandas as pd
import boto3
import numpy as np

train_pred = np.empty((0, 1), float)
np2 = np.array([[0.12, 0.88], [0.38, 0.62]])
np3 = np.array([[0.156, 0.56], [0.56, 0.56]])

#print(np2.reshape(-1, 1))
np1 = np.array([1, 2])
all_train_pred = pd.concat([pd.DataFrame(np2), pd.DataFrame(np1)], axis=1)

a = np.empty((2,2), float)
a = np.append(a, np2,axis = 0 )
print(a.shape)
a = np.append(a, np3)
print(a)
print(all_train_pred)

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
