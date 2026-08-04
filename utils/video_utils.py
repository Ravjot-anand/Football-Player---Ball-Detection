import cv2

def read_video(video_path):
    """
    Reads a video file and returns the frames as a list of images.
    
    Args:
        video_path (str): Path to the video file.
    
    Returns:
        list: List of frames (images).
    """
    cap = cv2.VideoCapture(video_path)
    frames = []#images is the list of frames
    while True:
        ret, frame = cap.read()# read the next frame from the video
        if not ret:#if there are no more frames to read, break the loop
            break
        frames.append(frame)#append the frame to the list of frames
    return frames

def save_video(output_video_frames, output_path):
    '''
    Saves a list of frames as a video file.
    '''
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Define the codec
    # Create VideoWriter object
    out = cv2.VideoWriter(output_path, fourcc, 24, (output_video_frames[0].shape[1], output_video_frames[0].shape[0]))
    for frame in output_video_frames:
            out.write(frame)  # Write the frame to the output video
    out.release()  # Release the VideoWriter object
