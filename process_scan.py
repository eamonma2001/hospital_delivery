import rosbag
import pandas as pd

bag = rosbag.Bag('nurse_to_patient2.bag')
data = []

for topic, msg, t in bag.read_messages(topics=['/base_scan']):
    ranges = [-1 if x < msg.range_min or x > msg.range_max else x for x in msg.ranges]
    data.append([t.to_sec(), ranges])  

df = pd.DataFrame(data, columns=['time', 'data'])
df.to_csv('laser.csv')
bag.close()

