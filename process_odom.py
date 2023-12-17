import rosbag
import pandas as pd

bag = rosbag.Bag('nurse_to_patient2.bag')
data = []

for topic, msg, t in bag.read_messages(topics=['/base_odometry/odom']):
    data.append([t.to_sec(), msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z, msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.covariance, msg.twist.twist.linear.x,msg.twist.twist.linear.y, msg.twist.twist.linear.z, msg.twist.twist.angular.x, msg.twist.twist.angular.y, msg.twist.twist.angular.z, msg.twist.covariance])  # Modify msg.data based on your message type

df = pd.DataFrame(data, columns=['time', 'pose_position_x', 'pose_position_y', 'pose_position_z',"pose_orient_x","pose_orient_y","pose_orient_z", "pose_cov", "twist_linear_x","twist_linear_y","twist_linear_z", "twist_ang_x", "twist_ang_y", "twist_ang_z", "twist_cov"])
df.to_csv('odom.csv')
bag.close()

