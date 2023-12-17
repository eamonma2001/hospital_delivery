import pandas as pd
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Flatten
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.spatial import cKDTree
import ast
from tensorflow.keras.callbacks import TensorBoard
import datetime

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

# Load odometry data
odom_df = pd.read_csv('data/nurse_to_patient1/odom.csv')

# Load laser scan data
laser_df = pd.read_csv('data/nurse_to_patient1/laser.csv')

# Convert time to numeric timestamp
odom_df['time'] = pd.to_datetime(odom_df['time'], unit='s')
laser_df['time'] = pd.to_datetime(laser_df['time'], unit='s')
# Perform the as-of merge as before
df = pd.merge_asof(odom_df, laser_df,  on='time', direction='nearest')

# Set 'time' column as index
df.set_index('time', inplace=True)

# Identify numeric columns for interpolation
numeric_cols = df.select_dtypes(include=[np.number]).columns

# Interpolate only numeric columns
df[numeric_cols] = df[numeric_cols].interpolate(method='time')

# Reset index if needed (if you need 'time' as a column)
df.reset_index(inplace=True)

df.bfill()  # Backward fill
df.ffill()  # Forward fill

# Check if there are any remaining NaNs
assert not df.isna().any().any(), "There are still NaNs in the DataFrame."

# Load odometry data
odom_df2 = pd.read_csv('data/nurse_to_patient2/odom.csv')

# Load laser scan data
laser_df2 = pd.read_csv('data/nurse_to_patient2/laser.csv')

odom_df2['time'] = pd.to_datetime(odom_df2['time'], unit='s')
laser_df2['time'] = pd.to_datetime(laser_df2['time'], unit='s')
# Perform the as-of merge as before
df2 = pd.merge_asof(odom_df2, laser_df2,  on='time', direction='nearest')

# Set 'time' column as index
df2.set_index('time', inplace=True)

# Identify numeric columns for interpolation
numeric_cols2 = df2.select_dtypes(include=[np.number]).columns

# Interpolate only numeric columns
df2[numeric_cols2] = df2[numeric_cols2].interpolate(method='time')

# Reset index if needed (if you need 'time' as a column)
df2.reset_index(inplace=True)

# Handle NaNs at the start or end
df2.bfill()  # Backward fill
df2.ffill()  # Forward fill

# Check if there are any remaining NaNs
assert not df2.isna().any().any(), "There are still NaNs in the DataFrame."

df = pd.concat([df, df2])

df['next_twist_linear_x'] = df['twist_linear_x'].shift(-1)
df['next_twist_linear_y'] = df['twist_linear_y'].shift(-1)
df['next_twist_linear_z'] = df['twist_linear_z'].shift(-1)
df['next_twist_ang_x'] = df['twist_ang_x'].shift(-1)
df['next_twist_ang_y'] = df['twist_ang_y'].shift(-1)
df['next_twist_ang_z'] = df['twist_ang_z'].shift(-1)

df = df[:-1]

df['step'] = df.apply(lambda row: row['step_x'] if row['step_y'] == 0 else row['step_y'], axis=1)

# Determine the number of range values (assuming all lists are of equal length)
sample_list = ast.literal_eval(df['data'].iloc[0])[1:-1]

df['data'] = df['data'].apply(ast.literal_eval)

ranges = len(sample_list)  # Number of elements in the range array

num_groups = 10  
group_size = ranges // num_groups

averaged_ranges = []
for group in range(num_groups):
    start_idx = group * group_size
    end_idx = start_idx + group_size
    column_name = f'range_{group+1}'
    df[column_name] = df['data'].apply(lambda x: np.mean(x[start_idx:end_idx]) if len(x) > start_idx else 0.0)
    averaged_ranges.append(column_name)


features = ['step', 'pose_position_x', 'pose_position_y', 'pose_position_z', 'pose_orient_x','pose_orient_y', 'pose_orient_z', 'twist_linear_x', 'twist_linear_y', 'twist_linear_z', 'twist_ang_x','twist_ang_y','twist_ang_z'] + averaged_ranges 

window_size = 50  # Number of previous steps to consider

# Function to create sequences
def create_sequences(data, window_size):
    sequences = []
    for i in range(len(data) - window_size):
        sequences.append(data[i:(i + window_size)])
    return np.array(sequences)

# Create sequences
X = create_sequences(df[features].values, window_size)

# Labels are now the next command columns
labels = ['next_twist_linear_x', 'next_twist_linear_y', 'next_twist_linear_z', 'next_twist_ang_x', 'next_twist_ang_y', 'next_twist_ang_z']
y = df[labels].values[window_size:]

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

# Split the data into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = np.array([scaler.fit_transform(x) for x in X_train])
X_test = np.array([scaler.fit_transform(x) for x in X_test])

# Initialize the constructor
model = Sequential()

# Add an input layer
model.add(LSTM(128, activation='relu', input_shape=(window_size,len(features))))

# Add one hidden layer
model.add(Dense(128, activation='relu'))

# Add an output layer
model.add(Dense(len(labels), activation='linear'))

model.compile(optimizer='adam', loss=custom_robot_navigation_loss)

# Fit the model
history = model.fit(X_train, y_train, epochs=32, batch_size=32, validation_split=0.1, verbose=1, callbacks=[tensorboard_callback])

# Evaluate the model
loss = model.evaluate(X_test, y_test, verbose=0)
print(f'Test loss: {loss}')

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np

# Predict using the trained model
y_pred = model.predict(X_test)

# Calculate R² score
r2 = r2_score(y_test, y_pred)
print(f'R² score: {r2}')

# Calculate MAE
mae = mean_absolute_error(y_test, y_pred)
print(f'Mean Absolute Error: {mae}')

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f'Root Mean Squared Error: {rmse}')

# Save the model
model.save('nurse_to_patient_model.tf')

import matplotlib.pyplot as plt

# Function to plot history
def plot_history(history):
    # Plot Loss
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()

    # If you have R2, MAE, and RMSE metrics in history, plot them
    if 'r2' in history.history and 'val_r2' in history.history:
        plt.subplot(1, 2, 2)
        plt.plot(history.history['r2'], label='Train R2')
        plt.plot(history.history['val_r2'], label='Validation R2')
        plt.title('R2 Score')
        plt.ylabel('R2 Score')
        plt.xlabel('Epoch')
        plt.legend()

    plt.show()

    if 'mae' in history.history and 'val_mae' in history.history:
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['mae'], label='Train MAE')
        plt.plot(history.history['val_mae'], label='Validation MAE')
        plt.title('Mean Absolute Error')
        plt.ylabel('MAE')
        plt.xlabel('Epoch')
        plt.legend()

    if 'rmse' in history.history and 'val_rmse' in history.history:
        plt.subplot(1, 2, 2)
        plt.plot(history.history['rmse'], label='Train RMSE')
        plt.plot(history.history['val_rmse'], label='Validation RMSE')
        plt.title('Root Mean Squared Error')
        plt.ylabel('RMSE')
        plt.xlabel('Epoch')
        plt.legend()

        plt.show()

# Call the function with your model's history
plot_history(history)
