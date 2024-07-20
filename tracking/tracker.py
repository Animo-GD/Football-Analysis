from ultralytics import YOLO
import supervision as sv
import pickle
import numpy as np
import cv2
import os
import sys
sys.path.append("../")
from utils import get_box_center,get_box_width


class Tracker:
    def __init__(self,model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()
        self.names = {"ball":0,"goalkeeper":1,"player":2,"referee":3}

    def detect_frames(self,frames):
        batch_size=20
        detections=[]
        for i in range(0,len(frames),batch_size):
            detections_batch = self.model.predict(frames[i:i+batch_size],conf=0.1)
            detections += detections_batch
        
        return detections
    

    def get_object_tracks(self,frames,trainable=False,trained_model_path=None):

        if trainable and trained_model_path and os.path.exists(trained_model_path):
            with open(trained_model_path,'rb') as f:
                tracked_objects = pickle.load(f)
            return tracked_objects



        detections = self.detect_frames(frames)
        tracked_objects = {
            "players":[],
            "referees":[],
            "ball":[],
            "goalkeepers":[]
        }
        for frame_num,detection in enumerate(detections):
            cls_names = detection.names
            detection_supervision = sv.Detections.from_ultralytics(detection)

            # Track Objects
            detection_track = self.tracker.update_with_detections(detection_supervision)
            tracked_objects["players"].append({})
            tracked_objects["ball"].append({})
            tracked_objects["goalkeepers"].append({})
            tracked_objects["referees"].append({})

            # supervision result = array[[xyxy],[mask],[conf_array],[class_id],[track_id]]
            for tracks in detection_track:
                bbox = tracks[0].tolist()
                cls_id = tracks[3].tolist()
                track_id = tracks[4].tolist()

                if cls_id == self.names["player"]:
                    tracked_objects["players"][frame_num][track_id] = {"bbox":bbox}
                if cls_id == self.names["goalkeeper"]:
                    tracked_objects["goalkeepers"][frame_num][track_id] = {"bbox":bbox}
                if cls_id == self.names["referee"]:
                    tracked_objects["referees"][frame_num][track_id] = {"bbox":bbox}
                # {   "players":[{track_id:bbox},{track_id:bbox}]} ...

            
            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]

                if cls_id == self.names["ball"]:
                   tracked_objects["ball"][frame_num][1] = {"bbox":bbox}

        if trained_model_path:
            with open(trained_model_path,'wb') as f:
                pickle.dump(tracked_objects,f)
        return tracked_objects
    


    def draw_ellipse(self,frame,bbox,color,track_id=None):
        y2 = int(bbox[3])
        x_center = int(get_box_center(bbox))
        width = get_box_width(bbox)
        cv2.ellipse(
            img=frame,
            center=(x_center,y2),
            axes=(int(width),int(0.3*width)),
            angle=0.0,
            startAngle=45,
            endAngle=245,
            color=color,
            thickness=2,
            lineType=cv2.LINE_4
        )

        rect_h,rect_w = 20,40
        x1_rect = x_center - rect_w//2
        x2_rect = x_center + rect_w//2
        y1_rect = (y2-rect_h//2)+15
        y2_rect = (y2+rect_h//2)+15
        if track_id:
            cv2.rectangle(frame,(int(x1_rect),int(y1_rect)),(int(x2_rect),int(y2_rect)),color,-1)
            x1_text = x1_rect+15
            if track_id>99:
                x1_text-=10
            cv2.putText(frame,
                        f"{track_id}",
                        (int(x1_text),int(y1_rect+12)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0,0,0),
                        2)
        return frame
    

    def draw_triangle(self,frame,bbox,color):
        y = int(bbox[1])
        x = int(get_box_center(bbox))
        triangle_points = np.array(
            [[x,y],
            [x-10,y-20],
            [x+10,y-20]]
        )
        cv2.drawContours(frame,[triangle_points],0,color,-1)
        cv2.drawContours(frame,[triangle_points],0,(0,0,0),2) # Borders

        return frame
    
    def draw_annotations(self,video_frames,tracks):
        out_video_frames = []
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()
            players_dict = tracks["players"][frame_num]
            goalkeepers_dict = tracks["goalkeepers"][frame_num]
            ball_dict = tracks["ball"][frame_num]
            referees_dict = tracks["referees"][frame_num]


            # Draw Players
            for track_id,player in players_dict.items():
                frame = self.draw_ellipse(frame,player["bbox"],(0,0,255),track_id)


            # Draw Referee
            for _,referee in referees_dict.items():
                frame = self.draw_ellipse(frame,referee["bbox"],(0,255,255))

         

            # Draw goalkeeper

            for _,goalkeeper in goalkeepers_dict.items():
                frame = self.draw_ellipse(frame,goalkeeper["bbox"],(255,255,0))

            # Draw ball

            for _,ball in ball_dict.items():
                frame = self.draw_triangle(frame,ball["bbox"],(0,255,0))


            out_video_frames.append(frame)
        return out_video_frames
