import numpy as np
import cv2
import os
import time
import matplotlib.pyplot as plt
import pandas as pd
import math




# Initialize variables for ROI selection and image capture
roi_selected = True
roi_start = (509,454)  # Example coordinates for the top-left corner of ROI
roi_end = (1330, 723)    # Example coordinates for the bottom-right corner of ROI
capture_images = False
image_count = 0

#
# Function to perform erosion and dilation on the selected ROI
def process_roi(frame, roi_start, roi_end, kernel_size=(7, 7), iterations=1):
    try:
        roi = frame[roi_start[1]:roi_end[1], roi_start[0]:roi_end[0]]
        print("ROI shape:", roi.shape)  # Print ROI dimensions for debugging

        # Apply erosion
        kernel = np.ones(kernel_size, np.uint8)
        roi_eroded = cv2.erode(roi, kernel, iterations=iterations)

        # Apply dilation
        roi_processed = cv2.dilate(roi_eroded, kernel, iterations=iterations)

        # Replace the processed ROI in the original frame
        frame[roi_start[1]:roi_end[1], roi_start[0]:roi_end[0]] = roi_processed

        return frame
    except Exception as e:
        print("Error in process_roi:", e)
        return frame




# Moving average function
def moving_average(data, window_size):
    cumsum = np.cumsum(data, dtype=float)
    cumsum[window_size:] = cumsum[window_size:] - cumsum[:-window_size]
    return cumsum[window_size - 1:] / window_size

# Kalman Filter class
class KF:
    def __init__(self):
        self.P = np.array([[1, 0], [0, 1]], dtype=np.float64)
        self.Q_angle = 0.003
        self.R_angle = 0.1
        self.Q_gyro = 3E-07
        self.x = np.array([[0], [0]], dtype=np.float64)

    def start_kf(self, dotAngle, dt, angle_m, Rangle):
        self.R_angle = Rangle
        self.predict(dotAngle, dt)
        return self.update(angle_m)

    def predict(self, dotAngle, dt):
        # State transition matrix A
        A = np.array([[1, -dt], [0, 1]], dtype=np.float64)

        # Process noise covariance Q
        Q = np.array([[self.Q_angle * dt**3 / 3, self.Q_angle * dt**2 / 2],
                      [self.Q_angle * dt**2 / 2, self.Q_gyro * dt]], dtype=np.float64)

        # Predict state and covariance
        self.x = A.dot(self.x) + np.array([[dt * dotAngle], [0]], dtype=np.float64)
        self.P = A.dot(self.P).dot(A.T) + Q

    def update(self, angle_m):
        # Measurement matrix H
        H = np.array([[1, 0]], dtype=np.float64)

        # Measurement noise covariance R
        R = np.array([[self.R_angle]], dtype=np.float64)

        # Calculate Kalman gain
        S = H.dot(self.P).dot(H.T) + R
        K = self.P.dot(H.T) / S[0, 0]

        # Update state and covariance
        y = angle_m - H.dot(self.x)[0, 0]
        self.x = self.x + K * y
        self.P = self.P - K * H.dot(self.P)

        # Get the filtered angle estimate and apply moving average
        filtered_angle = self.x[0, 0]
        smoothed_angle = moving_average(np.array([filtered_angle]), window_size=5)  # Adjust window_size as needed

        # Check if smoothed_angle has elements before accessing the first element
        if len(smoothed_angle) > 0:
            return smoothed_angle[0]
        else:
            return filtered_angle  # Fallback to unsmoothed value or handle appropriately

# Example usage:
kf = KF()
dotAngle = 0.1
dt = 0.01
angle_m = 0.3
Rangle = 0.05

result = kf.start_kf(dotAngle, dt, angle_m, Rangle)
print("Filtered and Smoothed Angle:", result)



# Suppress numpy rank warnings
import warnings
warnings.filterwarnings("ignore", category=np.RankWarning)

# Function for trackbar
def nothing(x):
    pass

# Create the "Frame" window and the trackbar
cv2.namedWindow("Frame")
cv2.createTrackbar("quality", "Frame", 1, 100, nothing)
cv2.setTrackbarPos("quality", "Frame", 10)  # Set initial quality to 50

# Initialize variables for ROI selection and image capture
#roi_selected = True
#roi_start = (768, 622)  # Example coordinates for the top-left corner of ROI
#roi_end = (1084, 677)    # Example coordinates for the bottom-right corner of ROI
#capture_images = False
#image_count = 0

# Video file path
video_file = "C:\\Users\\rajab\\Pictures\\Camera Roll\\0.6 origional.mp4"

# Initialize the video capture object
camera = cv2.VideoCapture(video_file)

# Output directory for saving images
output_directory = r"C:\Users\rajab\Pictures\Camera Roll\a"
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Time duration for capturing images (4 minutes and 12 seconds)
capture_duration = 252  # 252 seconds

# Lists to store stress values and corresponding time points
stress_values = []
time_points = []

# Initialize Excel File and DataFrame for data logging
excel_file = os.path.join(output_directory, "stress_data.xlsx")
if not os.path.exists(excel_file):
    df = pd.DataFrame(columns=['Time', 'stress_MPa'])
    df.to_excel(excel_file, index=False)

# Initialize a figure for the real-time graph
plt.ion()
fig, ax = plt.subplots()
ax.set_xlabel('Time (seconds)')
ax.set_ylabel('Internal Stress (Mpa)')
plt.title('Internal Stress vs. Time')
line, = ax.plot([], [])





# Initialize Kalman filter
kf = KF()

# Stoney Formula Constants
E_substrate = 120655  # Modulus of elasticity of the substrate in kg/cm²
T_substrate = 0.05077  # Thickness of the substrate in millimeters
L_substrate = 76.2  # Length of substrate in millimeters
t_deposit = 0.002538  # Deposit average thickness in millimeters
M = 1.714  # Correction factor for modulus of elasticity difference

# Number of points to use for the moving average filter
moving_avg_window = 3000

# Function to calculate distance between two points
def calculate_distance(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

# Create a timer to track the capture duration
start_time = time.time()
elapsed_time = 0  # Initialize elapsed time

# Main loop to capture frames and process data
while True:
    ret, frame = camera.read()

    if not ret:
        print("End of video file reached.")
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply dilation to the ROI for enhancing visibility
    if roi_selected:
        gray_roi = gray[roi_start[1]:roi_end[1], roi_start[0]:roi_end[0]]
        kernel = np.ones((7, 7), np.uint8)  # Define a 5x5 dilation kernel
        dilated_roi = cv2.dilate(gray_roi, kernel, iterations=1)

    # Get the quality from the trackbar
    quality = cv2.getTrackbarPos("quality", "Frame")
    quality = quality / 100.0 if quality > 0 else 0.01

    # Detect corners using the Shi-Tomasi corner detection algorithm
    if roi_selected:
        gray_roi = gray[roi_start[1]:roi_end[1], roi_start[0]:roi_end[0]]
        corners = cv2.goodFeaturesToTrack(gray_roi, 100, quality , 20)
        if corners is not None:
            corners = np.intp(corners)
            # Draw circles at detected corners within the ROI
            for corner in corners:
                x, y = corner.ravel() + roi_start
                cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)
            # Mark the leftmost and rightmost corners with red circles within the ROI
            leftmost_corner = corners[corners[:, :, 0].argmin()][0] + roi_start
            rightmost_corner = corners[corners[:, :, 0].argmax()][0] + roi_start
            cv2.circle(frame, tuple(leftmost_corner), 3, (255, 0, 0), -1)
            cv2.circle(frame, tuple(rightmost_corner), 3, (255, 0, 0), -1)
            # Add text indicating leftmost and rightmost corners
            cv2.putText(frame, "Leftmost Corner", (leftmost_corner[0] - 100, leftmost_corner[1] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.putText(frame, "Rightmost Corner", (rightmost_corner[0] + 20, rightmost_corner[1] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            # Calculate distances and stress
            distance_pixels = calculate_distance(leftmost_corner, rightmost_corner)
            distance_cm = (distance_pixels / 100)
            distance_inches = distance_cm / 2.54  # Convert cm to inches
            distance_mm = distance_inches / 2 * 25.4  # Convert inches to mm

            stress_MPa = (4*E_substrate * (T_substrate ** 2) * M * distance_mm / 2) / (3 * (L_substrate ** 2) * t_deposit)
            stress_PSI = stress_MPa * 145

            # Filter the stress value with Kalman filter
            filtered_stress_MPa = kf.start_kf(0, 0.1, stress_MPa, 0.1)

            # Add text regarding distances and stress
            cv2.putText(frame,
                        f"Distance to between the Leftmost Corner and Right most Corner in  (pixels): {distance_pixels:.2f}",
                        (leftmost_corner[0], leftmost_corner[1] + 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.putText(frame,
                        f"Distance to between the Leftmost Corner and Right most Corner in   (cm): {distance_cm:.2f}",
                        (leftmost_corner[0], leftmost_corner[1] + 80), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 0, 255), 1)
            cv2.putText(frame,
                        f"Distance to between the Leftmost Corner and Right most Corner in   (mm): {distance_mm:.2f}",
                        (leftmost_corner[0], leftmost_corner[1] + 100), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 0, 255), 1)
            cv2.putText(frame, f"Deposit Stress (MPa): {stress_MPa:.2f}",
                        (leftmost_corner[0], leftmost_corner[1] + 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 0, 255), 1)
            cv2.putText(frame, f"Deposit Stress (PSI): {stress_PSI:.2f}",
                        (leftmost_corner[0], leftmost_corner[1] + 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 0, 255), 1)

            elapsed_time = int(time.time() - start_time)
            if elapsed_time <= 252:
                stress_values = np.append(stress_values, filtered_stress_MPa)
                time_points = np.append(time_points, elapsed_time)

                # Update the real-time graph with the smoothed stress values
                if len(stress_values) >= moving_avg_window:
                    smoothed_stress_values = moving_average(stress_values, moving_avg_window)

                    if len(smoothed_stress_values) > 0:
                        # Update the real-time graph with the smoothed stress values
                        line.set_xdata(time_points[-len(smoothed_stress_values):])
                        line.set_ydata(smoothed_stress_values)
                        ax.relim()
                        ax.autoscale_view()

                        # Obtain the smoothed angle value for further processing
                        smoothed_angle = smoothed_stress_values[0]
                    else:
                        smoothed_angle = 0  # Handle the case when smoothed values are empty
                else:
                    smoothed_angle = 0  # Handle the case when not enough data for moving average

                # Append data to DataFrame and Excel file only if the number of rows is less than or equal to 252
                if len(time_points) <= 252:
                    df = pd.read_excel(excel_file)
                    new_data = pd.DataFrame({'Time': [elapsed_time], 'stress_MPa': [filtered_stress_MPa]})
                    df = pd.concat([df, new_data], ignore_index=True)
                    df = df.drop_duplicates(subset=['Time'])  # Drop duplicate entries based on Time column
                    df.to_excel(excel_file, index=False)

            if roi_selected and elapsed_time >= 0 and not capture_images:
                capture_images = True

            if capture_images and image_count < capture_duration:
                if int(elapsed_time) > image_count:
                    image_count += 1
                    image_filename = os.path.join(output_directory, f"image_{image_count}.png")
                    cv2.imwrite(image_filename, frame)

            if image_count >= capture_duration:
                print(f"Image capture completed.")
                break




    # Display the frame in the "Frame" window
    cv2.imshow("Frame", frame)

    # Apply morphology operations within ROI
    #frame_processed = apply_morphology_operations(frame, roi_start, roi_end)

    # Display the processed frame
    #cv2.imshow("Processed Frame", frame_processed)

    # Handle keyboard input
    key = cv2.waitKey(1)
    if key == 27:
        break

    if elapsed_time >= 252:
        print(f"Data collection completed for 252 seconds.")
        break

camera.release()
cv2.destroyAllWindows()
plt.ioff()
plt.show()
# Draw the plot and pause briefly to update the plot window
#plt.draw()
#plt.pause(0.001)  # Adjust the pause duration as needed for desired refresh rate

