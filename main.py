from utils.video_utils import read_video, save_video
from Teams import TeamAssigner
from trackers import Tracker 
import cv2
import numpy as np
from Player_Ball_Assigner import PlayerBallAssigner

def main():
    #read video
    video_frames = read_video("input-videos/08fd33_4.mp4")    
    
    #initialize the tracker with the model path
    tracker = Tracker("models/best10.pt")#initialize the tracker with the model path
    tracks = tracker.track_objects(video_frames,
                                       read_from_stub=True,
                                       stub_path='stubs/track_stubs.pkl')#track objects in the video frames
    
    #Interpolate Ball positions
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])#we interpolate the ball with the help of numpy lib
    
    # Assign Player Teams
    team_assigner = TeamAssigner()
    team_assigner.assign_player_teams(video_frames, tracks['players'])

    
    """
    #Getting a cropped image of a player in frame 0
    for track_id, player in tracks["players"][0].items():
        bbox = player["bbox"]
        frame = video_frames[0]
        #crop box from the image
        cropped_player = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
        #save the cropped image
        cv2.imwrite(f"output-videos/cropped_player.jpg", cropped_player)
        
        break
        
    """
    
    # Assign Ball Aquisition
    player_assigner =PlayerBallAssigner()
    team_ball_control= []
    for frame_num, player_track in enumerate(tracks['players']):
        ball_bbox = tracks['ball'][frame_num][1]['bbox']
        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)

        if assigned_player != -1:
            tracks['players'][frame_num][assigned_player]['has_ball'] = True
            team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])
        else:
            team_ball_control.append(team_ball_control[-1])
    team_ball_control= np.array(team_ball_control)

    #draw output    
    #draw object tracks
    output_video_frames = tracker.draw_annotations(video_frames, tracks)#draw annotations on the video frames

    #save video 
    save_video(output_video_frames, "output-videos/08fd33_4_output.avi")
if __name__ == "__main__":
    main()