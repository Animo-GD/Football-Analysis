import numpy as np
import cv2
class viewTransformer:
    def __init__(self) -> None:
        court_width = 68
        court_length = 23.32

        self.pixel_verticies = np.array([
            [110,1035],
            [265,275],
            [910,260],
            [1640,915]
            ],dtype=np.float32)
        self.target_verticies = np.array([
            [0,court_width],
            [0,0],
            [court_length,0],
            [court_length,court_width]

        ],dtype = np.float32)

        #self.pixel_verticies = self.pixel_verticies.astype(np.float32)
        #self.target_verticies = self.target_verticies.astype(np.float32)
        self.perspective_transformer = cv2.getPerspectiveTransform(self.pixel_verticies,self.target_verticies)


    def transform_position(self,pos):
        p = (int(pos[0]),int(pos[1]))
        is_inside = cv2.pointPolygonTest(self.pixel_verticies,p,False)>=0
        if not is_inside:
            return None
        
        reshaped_pos = pos.reshape(-1,1,2).astype(np.float32)

        transform_pos = cv2.perspectiveTransform(reshaped_pos,self.perspective_transformer)
        return transform_pos.reshape(-1,2)
    

    def add_transformerd_position_to_tracks(self,tracks):
        for object,object_tracks in tracks.items():
            for frame_num,track in enumerate(object_tracks):
                for track_id,track_info in track.items():
                    if "position_adjusted" not in track_info:
                        continue
                    position = track_info["position_adjusted"]
                    position = np.array(position)
                    transformed_position = self.transform_position(position)
                    if transformed_position is not None:
                        transformed_position = transformed_position.squeeze().tolist()
                    tracks[object][frame_num][track_id]["transformed_position"] = transformed_position