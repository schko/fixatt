from Desktop.fixatt.preprocessing.heatmap_generator import *
import datetime
import cv2
import json
from pathlib import Path
import pickle


def process_driving_video(time_csv, et_df, time_start=10, is_event=True, max_frames=3, save_frames=True, 
                   output_path='.', vid_path=None, save_trial_info=False, save_averaged_dat=False, 
                   save_prepend=None, eye_accumulation_period=None):
    
    """ Generate fixation map within the eye_accumulation_period (premotor period in the manuscript) 
        and detect edges for frames before the detected events.

    Args:
        time_csv (df): 
            video's dataframe which links frames to timestamps
        et_df (df): 
            eye tracking coordinates dataframe
        time_start (float, optional): 
            time (in LSL time, seconds) of reference point to collect data. Defaults to 10.
        is_event (bool, optional): 
            if true, the time_start is treated as an event and max_frames as the window prior to the 
            event to save. It's the opposite of forward data collection, which if is_event == False, 
            we collect up to max_frames from time_start.
        max_frames (int, optional): 
            total number of frames to plot, set to 0 to loop through all frames. Defaults to 3.
        save_frames (bool, optional): 
            saves the frame before the time_start. Defaults to True.
        save_trial_info (json, optional): 
            associated metadata to store
        save_prepend (str, optional): 
            prepend to add to saved file
        eye_accumulation_period (float, optional): 
            time (in seconds) of the lead time to event to pool over for fixation_map generation
    """
    Path(f"{output_path}{save_prepend}").mkdir(parents=True, exist_ok=True)

    # load video based on video path provided
    cap = cv2.VideoCapture(cv2.samples.findFile(f"{vid_path}"))
    # extract video information i.e. dimension and frames per second
    width = cap.get(3)
    height = cap.get(4)
    fps = cap.get(5)

    # define the period for fixation data aggregation if parameter: eye_accumulation_period is not defined  
    if not eye_accumulation_period:
        eye_accumulation_period = max_frames/fps

    # get reference time within the video
    video_ref_time = (time_start-time_csv.lsl_timestamp.iloc[0])
    print(
        f"Reference time within video is {str(datetime.timedelta(seconds=video_ref_time))}")
    
    # 0 is the start of video but this frame number can be whatever 
    frame_idx = int(video_ref_time*fps)
    if is_event:
        frame_idx -= max_frames
    
    # capture the frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx-1)
    ret, frame1 = cap.read()

    local_frame_no = 1
    saved_image_file_name = None

    # capture 
    while (1):
        timestamp = time_csv.iloc[frame_idx].lsl_timestamp
        ret, frame = cap.read()
        orig_frame = frame.copy()
        if not ret:
            print('No frames grabbed!')
            break

        # edge detection for number of frames defined by max_frames
        img_gray = cv2.cvtColor(orig_frame, cv2.COLOR_BGR2GRAY) # Convert to graycsale
        img_blur = cv2.GaussianBlur(img_gray, (3, 3), 0) # Blur the image for better edge detection
        # mean 255/2 * .66 for min threshold, 255/2 * 1.33 for max
        edges = cv2.Canny(image=img_blur, threshold1=25, threshold2=50)

        # save identified frame from the video
        if save_frames:
            saved_image_file_name = f"{output_path}{save_prepend}/frame{local_frame_no}.jpg"
            cv2.imwrite(saved_image_file_name, frame)

        
        df_sort = et_df.iloc[(et_df['timestamp'] - timestamp).abs().argsort()[0]]
        # eye_x and eye_y are in video coordinates - convert raw gaze coordnate to video coordinate
        eye_x = (width-(df_sort['L Gaze Direction X'] * width))/2
        eye_y = (height-(df_sort['L Gaze Direction Y'] * height))/2
        # eye tracking overlay
        ey_overlay_frame = cv2.circle(frame, (int(eye_x), int(eye_y)), 10, (0, 0, 255),
                                      thickness=-1)
        
        # stop the loop when frame being processed is exceeded the max_frames
        if max_frames > 0 and local_frame_no >= max_frames:
            break
        
        frame_idx += 1
        local_frame_no += 1

    cv2.destroyAllWindows() # close the video

    # extract gaze coordinate information based on the eye_accumulation_period 
    df_sort = et_df[(et_df.timestamp >= (
        time_start-eye_accumulation_period)) & (et_df.timestamp <= time_start)]
    df_sort['duration'] = 1 # uniform across all coordinate for fixation
    # eye_x and eye_y are in video coordinates - convert raw gaze coordnate to video coordinate
    df_sort['eye_x'] = (width-(df_sort['L Gaze Direction X'] * width))/2
    df_sort['eye_y'] = (height-(df_sort['L Gaze Direction Y'] * height))/2

    # genereate the fixation map by using fixation data with in the eye_accumulation_period
    if save_frames:
        fixation_map = draw_heatmap(df_sort[['eye_x', 'eye_y', 'duration']].dropna(), (int(width), int(
            height)), saved_image_file_name, savefilename=f"{output_path}{save_prepend}/fixation_heatmap.jpg")
    else:
        fixation_map = draw_heatmap(df_sort[['eye_x', 'eye_y', 'duration']].dropna(), (int(width), int(
            height)), saved_image_file_name, savefilename=None)

    # save fixation map, detected edge for the last frame, last frame, trial information, and raw eye tracking 
    # coordinate within the eye_accumulation_period
    data_dict = {}
    if save_averaged_dat:
        saved_image_file_name = f"{output_path}{save_prepend}/avg_trial_data.p"
        data_dict = {'eye_smoothed': fixation_map, 'edges': edges, 'orig_frame': orig_frame, 
                     'save_trial_info': save_trial_info,
                     'raw_et': df_sort[['eye_x', 'eye_y', 'duration']].to_dict()}
        pickle.dump(data_dict, open(saved_image_file_name, 'wb'))

    return data_dict, df_sort


"""
gazeplotter is an open source library authored by Edwin Dalmaijer for producing different types of 
plots using eye tracking data. Plots that can be generated with this library are fixation heatmaps, 
path between fixations, and fixation locations on top of original images.

draw_heatmap() function utilizes a list of identified fixation events with their duration (fixation 
coordinate corresponding to the frame of interest) and the frame size to generate the fixation map. 
Fixation weight can be adjusted with duration. In our case, all fixations were weighted the same.

Gaussian matrix is initialized using predefined varibale. This matrix used along with the frame dimension 
to initialize/determine the fixation map size matrix. The input fixation coordiantes get re-scaled with 
parameters that define the guassian matrix. The values in the Gaussian matrix that correspond to these  
re-scaled coordinate gets extracted and sum with the existing fixation map. The value from the Gaussian
matrix is scaled by the fixation duration before summing with the existing fixation map.

"""