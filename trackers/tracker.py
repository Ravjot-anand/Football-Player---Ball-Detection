from ultralytics import YOLO
import supervision as sv
from utils import get_center_of_bbox, get_bbox_width
import cv2
import pickle
import os
import numpy as np
import pandas as pd

class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()
#this function takes a list of frames as input and returns a list of detections for each frame
    def detect_frames(self, frames):
        batch_size = 20
        detections = []
        
        for i in range(0, len(frames), batch_size):#prosess frames in batches
            #in this line we are predicting the frames in batches of 20 and storing the detections in a list 
            detections_batch = self.model.predict(frames[i:i+batch_size],conf = 0.1)
            detections += detections_batch            
             
            
        return detections
    
#this function takes a list of frames as input and returns a dictionary containing the tracks for each object in the frames
    def track_objects(self,frames,read_from_stub=False, stub_path=None):
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path,'rb') as f:
                tracks = pickle.load(f)
            return tracks        
        
        
        detections = self.detect_frames(frames)# detect objects in the frames
        
        tracks={
            "players":[],
            "referees":[],
            "ball":[]
        }
        
        
        for frame_num, detection in enumerate(detections):# loop through the detections for each frame
            cls_names = detection.names# get the class names from the detection
            cls_names_inv = {v:k for k,v in cls_names.items()}

            # Covert to supervision Detection format
            detection_supervision = sv.Detections.from_ultralytics(detection)

            # Convert GoalKeeper to player object
            for object_ind , class_id in enumerate(detection_supervision.class_id):
                if cls_names[class_id] == "goalkeeper":
                    detection_supervision.class_id[object_ind] = cls_names_inv["player"]
                
                
            # Track the objects in the frame
            detections_with_tracks = self.tracker.update_with_detections(detection_supervision)
            
            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})
            
            # Loop through the detections with tracks and store the bounding box coordinates, class id, and track id in the respective lists
            for frame_detection in detections_with_tracks:
                bbox = frame_detection[0].tolist()# get the bounding box coordinates
                class_id = frame_detection[3]# get the class id
                track_id = frame_detection[4]# get the track id
                
                # Store the bounding box coordinates, class id, and track id in the respective lists
                if class_id == cls_names_inv["player"]:
                    tracks["players"][frame_num][track_id] = {"bbox": bbox}
                if class_id == cls_names_inv["referee"]:
                    tracks["referees"][frame_num][track_id] = {"bbox": bbox}
                    
            # Loop through the detections and store the bounding box coordinates for the ball in the respective list
            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]

                if cls_id == cls_names_inv['ball']:
                    tracks["ball"][frame_num][1] = {"bbox":bbox}
            
            
        if stub_path is not None:
            with open(stub_path,'wb') as f:
                pickle.dump(tracks,f)

        return tracks
#thus function takes a frame, bounding box, color, and track id as input and draws an ellipse around the object in the frame   
    def draw_ellipse(self,frame,bbox,color,track_id=None):
        y2 = int(bbox[3])
        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)

        cv2.ellipse(
            frame,
            center=(x_center,y2),
            axes=(int(width), int(0.35*width)),
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color = color,
            thickness=2,
            lineType=cv2.LINE_4
        )

        rectangle_width = 40
        rectangle_height=20
        x1_rect = x_center - rectangle_width//2
        x2_rect = x_center + rectangle_width//2
        y1_rect = (y2- rectangle_height//2) +15
        y2_rect = (y2+ rectangle_height//2) +15

        if track_id is not None:
            cv2.rectangle(frame,
                          (int(x1_rect),int(y1_rect) ),
                          (int(x2_rect),int(y2_rect)),
                          color,
                          cv2.FILLED)
            
            x1_text = x1_rect+12
            if track_id > 99:
                x1_text -=10
            
            cv2.putText(
                frame,
                f"{track_id}",
                (int(x1_text),int(y1_rect+15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0,0,0),
                2
            )

        return frame           
#this function takes a list of frames and a dictionary of tracks as input and returns a list of frames with annotations drawn on them    
    def draw_annotations(self, video_frames, tracks):
        output_video_frames = []#output video frames with annotations
        
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()#copy the frame to avoid modifying the original frame
            # Loop through the tracks and draw the bounding boxes for each object in the frame
            player_dict = tracks["players"][frame_num]
            ball_dict = tracks["ball"][frame_num]
            referee_dict = tracks["referees"][frame_num]
        
            # Draw Players
            for track_id, player in player_dict.items():
                color = player.get("team_color", (0,0,255))
                frame = self.draw_ellipse(frame, player["bbox"],color, track_id)
                
                if player.get("has_ball", False):
                    frame = self.draw_traingle(frame, player["bbox"], (0,0,255))
            # Draw Refreee
            for _, referee in referee_dict.items():
                frame = self.draw_ellipse(frame, referee["bbox"],(0,255,255))       
            # Draw Ball
            for track_id, ball in ball_dict.items():
                frame = self.draw_traingle(frame, ball["bbox"],(0,255,0))
            
            output_video_frames.append(frame)
            
        return output_video_frames
# this function takes a frame, bounding box, and color as input and draws a triangle around the object in the frame
    def draw_traingle(self,frame,bbox,color):
        y= int(bbox[1])
        x,_ = get_center_of_bbox(bbox)

        triangle_points = np.array([
            [x,y],
            [x-10,y-20],
            [x+10,y-20],
        ])
        cv2.drawContours(frame, [triangle_points],0,color, cv2.FILLED)
        cv2.drawContours(frame, [triangle_points],0,(0,0,0), 2)

        return frame
#this function try to interpolate the ball postion with the help of previous postion      
    def interpolate_ball_positions(self,ball_positions):
        ball_positions = [x.get(1,{}).get('bbox',[]) for x in ball_positions]
        df_ball_positions = pd.DataFrame(ball_positions,columns=['x1','y1','x2','y2'])

        # Interpolate missing values
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()

        ball_positions = [{1: {"bbox":x}} for x in df_ball_positions.to_numpy().tolist()]

        return ball_positions
        