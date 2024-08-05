#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install opencv-python


# In[2]:


import cv2
import matplotlib.pyplot as plt


# In[3]:


import cv2

# Path to the video file
video_path = "C:\\Users\\user\\Downloads\\6981614-hd_1920_1080_30fps.mp4"

# Create a VideoCapture object
cap = cv2.VideoCapture(video_path)

# Check if the video was successfully opened
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Get video information
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_rate = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"Video Information:")
print(f"Width: {frame_width}")
print(f"Height: {frame_height}")
print(f"Frame Rate: {frame_rate} FPS")
print(f"Frame Count: {frame_count}")

# Read and process frames in a loop
frame_number = 0
while cap.isOpened():
    ret, frame = cap.read()
    
    if not ret:
        print("Finished reading frames.")
        break
    
    frame_number += 1
    print(f"Processing frame {frame_number}")

    # (Optional) Display the frame
    cv2.imshow('Frame', frame)

    # (Optional) Save the frame as an image file
    frame_filename = f'frame_{frame_number}.png'
    cv2.imwrite(frame_filename, frame)
    
    # Press 'q' to exit the loop early
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()


# In[4]:


#2
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Define the path to the video file
video_path = "C:\\Users\\user\\Downloads\\6981614-hd_1920_1080_30fps.mp4"

# Open the video file
cap = cv2.VideoCapture(video_path)

# Check if the video file opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Initialize counters for frame types
frame_counts = {'I': 1, 'P': 0, 'B': 0}

# Read the first frame
ret, prev_frame = cap.read()

if not ret:
    print("Error: Could not read the first frame.")
    cap.release()
    exit()

# Convert the first frame to grayscale
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

# Process the remaining frames
frame_index = 1
while True:
    # Read the next frame
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Calculate absolute difference between the current frame and the previous frame
    diff = cv2.absdiff(prev_gray, gray)
    non_zero_count = np.count_nonzero(diff)

    # Classify frame type (simplified approximation)
    if non_zero_count > 1000:  # Threshold value, adjust as needed
        frame_counts['P'] += 1
    else:
        frame_counts['B'] += 1

    # Update the previous frame
    prev_gray = gray

    frame_index += 1

# Calculate the percentage of each frame type
total_frames = sum(frame_counts.values())
percentages = {k: (v / total_frames) * 100 for k, v in frame_counts.items()}

# Print the results
print(f"Total frames: {total_frames}")
print(f"I-frames: {frame_counts['I']} ({percentages['I']:.2f}%)")
print(f"P-frames: {frame_counts['P']} ({percentages['P']:.2f}%)")
print(f"B-frames: {frame_counts['B']} ({percentages['B']:.2f}%)")

# Plot the distribution of frame types
plt.figure(figsize=(10, 5))

# Bar plot
plt.subplot(1, 2, 1)
plt.bar(frame_counts.keys(), frame_counts.values(), color=['blue', 'green', 'red'])
plt.xlabel('Frame Type')
plt.ylabel('Count')
plt.title('Frame Type Distribution (Count)')

# Pie chart
plt.subplot(1, 2, 2)
plt.pie(frame_counts.values(), labels=frame_counts.keys(), autopct='%1.1f%%', colors=['blue', 'green', 'red'])
plt.title('Frame Type Distribution (%)')

plt.tight_layout()
plt.show()

# Release the video capture object
cap.release()


# In[5]:


#Lab Task 3: Visualizing Frames
#Objective:
#Extract actual frames from the video and display them using Python.
#1. Extract Frames:
#o Use ffmpeg to extract individual I, P, and B frames from the video.
#o Save these frames as image files.

import cv2
import os

# Path to the video fil

# Directory to save the extracted frames
output_dir = 'extracted_frames'
os.makedirs(output_dir, exist_ok=True)

# Open the video file
cap = cv2.VideoCapture(video_path)

# Check if the video file was opened successfully
if not cap.isOpened():
    print("Error: Could not open video file.")
else:
    # Initialize a frame counter
    frame_count = 0
   
    # Define how frequently to save frames (e.g., every 30th frame)
    save_frequency = 30
   
    while True:
        ret, frame = cap.read()
       
        if not ret:
            break  # Exit the loop if no frames left
       
        frame_count += 1
       
        # Save every 'save_frequency' frame
        if frame_count % save_frequency == 0:
            frame_filename = f"{output_dir}/frame_{frame_count}.jpg"
            cv2.imwrite(frame_filename, frame)
            print(f"Saved {frame_filename}")
           
            # Optionally, display the saved frame
            cv2.imshow('Extracted Frame', frame)
            cv2.waitKey(500)  # Display each frame for 500 ms (adjust as needed)

    # Release the video capture object
    cap.release()

# Destroy any OpenCV windows
cv2.destroyAllWindows()


# In[11]:


import cv2
import os

# Path to the video file"C:\\Users\\user\\Downloads\\6981614-hd_1920_1080_30fps.mp4"
video_path = 
output_dir = 'extracted_frames'

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Open the video file
cap = cv2.VideoCapture(video_path)

# Check if the video file was opened successfully
if not cap.isOpened():
    print("Error: Could not open video file.")
else:
    frame_counter = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # Exit the loop if no frames are left

        frame_counter += 1
       
        # Save every 10th frame as an I-frame, every 30th as a P-frame, and every 50th as a B-frame
        if frame_counter % 50 == 0:  # B-frame
            frame_type = 'B'
        elif frame_counter % 30 == 0:  # P-frame
            frame_type = 'P'
        elif frame_counter % 10 == 0:  # I-frame
            frame_type = 'I'
        else:
            continue  # Skip frames not matching the criteria
       
        # Save frames as image files
        output_filename = os.path.join(output_dir, f"{frame_counter}_{frame_type}.png")
        cv2.imwrite(output_filename, frame)
        print(f"Saved {output_filename}")

    # Release the video capture object
    cap.release()
    print("All frames have been processed and saved.")


# In[7]:


from PIL import Image
import matplotlib.pyplot as plt

# Load and display the first few frames of each type (I, P, B)
frame_filenames = sorted(os.listdir(output_dir))

plt.figure(figsize=(15, 5))

# Display the first I-frame
i_frame_path = next((os.path.join(output_dir, f) for f in frame_filenames if 'I' in f), None)
if i_frame_path:
    img = Image.open(i_frame_path)
    plt.subplot(1, 3, 1)
    plt.imshow(img)
    plt.title("I-Frame")
    plt.axis('off')
    plt.show()

plt.figure(figsize=(15, 5))
# Display the first P-frame
p_frame_path = next((os.path.join(output_dir, f) for f in frame_filenames if 'P' in f), None)
if p_frame_path:
    img = Image.open(p_frame_path)
    plt.subplot(1, 3, 2)
    plt.imshow(img)
    plt.title("P-Frame")
    plt.axis('off')
    plt.show()

plt.figure(figsize=(15, 5))
# Display the first B-frame
b_frame_path = next((os.path.join(output_dir, f) for f in frame_filenames if 'B' in f), None)
if b_frame_path:
    img = Image.open(b_frame_path)
    plt.subplot(1, 3, 3)
    plt.imshow(img)
    plt.title("B-Frame")
    plt.axis('off')
    plt.show()


# In[9]:


import cv2
import numpy as np
import os

def extract_frames(video_path, output_dir):
    # Open video file
    video = cv2.VideoCapture(video_path)
    
    # Check if video opened successfully
    if not video.isOpened():
        raise Exception("Could not open video file.")
    
    frame_index = 0
    frame_types = {'I': 0, 'P': 0, 'B': 0}
    frame_sizes = {'I': [], 'P': [], 'B': []}
    
    while True:
        ret, frame = video.read()
        if not ret:
            break

        # Extract frame type info
        frame_type = get_frame_type(video, frame_index)
        if frame_type in frame_types:
            # Save the frame to a temporary file
            frame_filename = os.path.join(output_dir, f"frame_{frame_index}.jpg")
            cv2.imwrite(frame_filename, frame)
            
            # Calculate file size
            file_size = os.path.getsize(frame_filename)
            frame_sizes[frame_type].append(file_size)
            
            # Increment index
            frame_index += 1
    
    video.release()
    
    # Clean up temporary files
    clean_up_directory(output_dir)
    
    return frame_sizes

def get_frame_type(video, frame_index):
    # This is a placeholder. OpenCV does not provide direct information about I, P, and B frames.
    # You might need to use a different library or method to determine frame types.
    # Here we assume we could label all frames as 'I' just for demonstration.
    return 'I'  # Replace this with actual logic if available.

def calculate_average_sizes(frame_sizes):
    averages = {}
    for frame_type, sizes in frame_sizes.items():
        if sizes:
            averages[frame_type] = np.mean(sizes)
        else:
            averages[frame_type] = 0
    return averages

def clean_up_directory(directory):
    for root, dirs, files in os.walk(directory, topdown=False):
        for name in files:
            os.remove(os.path.join(root, name))
        for name in dirs:
            os.rmdir(os.path.join(root, name))
    os.rmdir(directory)

def main(video_path, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    frame_sizes = extract_frames(video_path, output_dir)
    average_sizes = calculate_average_sizes(frame_sizes)
    
    for frame_type, avg_size in average_sizes.items():
        print(f"Average size of {frame_type} frames: {avg_size} bytes")

if __name__ == "__main__":
    video_path = "C:\\Users\\user\\Downloads\\6981614-hd_1920_1080_30fps.mp4"
    output_dir = 'extracted_frames'
    main(video_path, output_dir)


# In[14]:


pip install opencv-python


# In[ ]:


import cv2
import os

def reconstruct_video_from_i_frames(i_frames_dir, output_video_path, frame_rate):
    # Check if directory exists
    if not os.path.exists(i_frames_dir):
        raise FileNotFoundError(f"Directory '{i_frames_dir}' does not exist.")
    
    # Get list of I-frames
    frame_files = [f for f in os.listdir(i_frames_dir) if f.endswith('.jpg')]
    
    if not frame_files:
        raise FileNotFoundError(f"No frames found in '{i_frames_dir}'.")
    
    frame_files.sort()  # Ensure frames are sorted in the correct order
    
    # Read the first frame to get the size
    first_frame_path = os.path.join(i_frames_dir, frame_files[0])
    first_frame = cv2.imread(first_frame_path)
    if first_frame is None:
        raise ValueError(f"Failed to read the frame '{first_frame_path}'.")
    
    height, width, _ = first_frame.shape
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (width, height))
    
    # Write frames to video
    for frame_file in frame_files:
        frame_path = os.path.join(i_frames_dir, frame_file)
        frame = cv2.imread(frame_path)
        if frame is None:
            raise ValueError(f"Failed to read the frame '{frame_path}'.")
        video_writer.write(frame)
    
    video_writer.release()

# Set the paths
i_frames_dir = 'i_frames'
output_video_path = 'reconstructed_video.mp4'
frame_rate = 1  # Reduced frame rate (e.g., 1 frame per second)
reconstruct_video_from_i_frames(i_frames_dir, output_video_path, frame_rate)


# In[ ]:


import os

def get_file_size(file_path):
    """Returns the file size in bytes."""
    if os.path.exists(file_path):
        return os.path.getsize(file_path)
    else:
        print(f"Error: File not found at {file_path}")
        return None

# Paths to your frame images
i_frame_path = "C:\\Users\\user\\Downloads\\6981614-hd_1920_1080_30fps.mp4
p_frame_path =  "C:\\Users\\user\\Downloads\\6981614-hd_1920_1080_30fps.mp4"
b_frame_path =  "C:\\Users\\user\\Downloads\\6981614-hd_1920_1080_30fps.mp4"

# Get file sizes
i_frame_size = get_file_size(i_frame_path)
p_frame_size = get_file_size(p_frame_path)
b_frame_size = get_file_size(b_frame_path)

# Print sizes
print(f"I-frame size: {i_frame_size / 1024:.2f} KB")
print(f"P-frame size: {p_frame_size / 1024:.2f} KB")
print(f"B-frame size: {b_frame_size / 1024:.2f} KB")

# Average file size for comparison (if multiple frames)
# This example assumes you have multiple frames, replace with appropriate logic if needed

