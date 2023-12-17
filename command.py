import rospy
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import custom_object_scope
import numpy as np
import collections
import sys

first_arg = ""

if len(sys.argv) == 2:
    first_arg = sys.argv[1]
else:
    print("This script requires exactly two arguments.")

def custom_robot_navigation_loss(y_true, y_pred):
    # Splitting the true and predicted values into linear and angular components
    linear_true, angular_true = y_true[:, :3], y_true[:, 3:]
    linear_pred, angular_pred = y_pred[:, :3], y_pred[:, 3:]

    # Define weights for linear and angular components
    linear_weight = 1.5  # You can adjust this weight
    angular_weight = 1.0  # Higher weight for angular velocity

    # Calculate Mean Squared Error for linear and angular components
    linear_loss = tf.reduce_mean(tf.square(linear_true - linear_pred))
    angular_loss = tf.reduce_mean(tf.square(angular_true - angular_pred))

    # Calculate the weighted sum of the individual losses
    total_loss = (linear_weight * linear_loss) + (angular_weight * angular_loss)
    return total_loss
    
# Load trained model
with custom_object_scope({'custom_robot_navigation_loss': custom_robot_navigation_loss}):
    if first_arg.lower() == "patient":
    	model = load_model('nurse_to_patient_model.tf')
    elif first_arg.lower() == "exam":
    	model = load_model('nurse_to_exam_model.tf')
    else:
    	model = load_model('nurse_to_surgery_model.tf')
    
current_state = None
current_odom = None
current_scan = None
step_count = 0

WINDOW_SIZE = 20
if first_arg.lower() == "patient":
    WINDOW_SIZE = 50

odom_buffer = collections.deque(maxlen=WINDOW_SIZE)
scan_buffer = collections.deque(maxlen=WINDOW_SIZE)

def odom_callback(msg):
    global current_odom, step_count
    global odom_buffer
    # Extract necessary information from the message
    pose = msg.pose.pose
    twist = msg.twist.twist

    # Convert data to the format your model expects
    current_odom = np.array([step_count, pose.position.x, pose.position.y, pose.position.z,
                              pose.orientation.x, pose.orientation.y, pose.orientation.z,
                              twist.linear.x, twist.linear.y, twist.linear.z,
                              twist.angular.x, twist.angular.y, twist.angular.z
                              ])
    odom_buffer.append(current_odom)
    #print(odom_buffer)
    step_count += 1  # Increment step count
   
    
def scan_callback(msg):
    global current_scan
    global scan_buffer
    
    ranges_res = []
    
    global first_arg

    num_groups = 10 
    if first_arg == "exam":
    	num_groups = 5
    	
    ranges = [-1 if x < msg.range_min or x > msg.range_max else x for x in msg.ranges]
    group_size = len(ranges) // num_groups

    for group in range(num_groups):
    	start_idx = group * group_size
    	end_idx = start_idx + group_size
    	ranges_res.append(np.mean(ranges[start_idx:end_idx]))

    	#df[column_name] = df['data'].apply(lambda x: np.mean(x[start_idx:end_idx]) if len(x) > start_idx else 0.0)

    
    current_scan = np.array(ranges_res)
    scan_buffer.append(current_scan)
    #print(scan_buffer)

def publish_command():
    global current_odom, current_scan, odom_buffer, scan_buffer
    if len(odom_buffer) == WINDOW_SIZE and len(scan_buffer) == WINDOW_SIZE:
        # Combine odom and scan data
        
        odom_array = np.array(odom_buffer)
        scan_array = np.array(scan_buffer)
        

        # Combine and reshape data for the model
        current_state = np.concatenate([odom_array, scan_array], axis=1).reshape(1, WINDOW_SIZE, -1)        
        

        # Predict the next command using the model
        next_command = model.predict(current_state)

        # Create and publish Twist message
        cmd_msg = Twist()
        cmd_msg.linear.x = next_command[0][0]
        cmd_msg.linear.y = next_command[0][1]
        cmd_msg.linear.z = next_command[0][2]
        cmd_msg.angular.x = next_command[0][3]
        cmd_msg.angular.y = next_command[0][4]
        cmd_msg.angular.z = next_command[0][5]
        cmd_pub.publish(cmd_msg)
        
def main():
    global current_state
    rospy.init_node('imitation_learning_controller')

    # Subscribe to the odom topic
    rospy.Subscriber('/base_odometry/odom', Odometry, odom_callback)
    rospy.Subscriber('/base_scan', LaserScan, scan_callback)

    global cmd_pub
    # Publisher for sending commands
    cmd_pub = rospy.Publisher('navigation/cmd_vel', Twist, queue_size=1000)

    rate = rospy.Rate(128)  # 10 Hz
    while not rospy.is_shutdown():
        publish_command()
        rate.sleep()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass


