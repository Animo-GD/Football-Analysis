from utils import read_video,save_video,crop_player
from tracking import Tracker
from team_assigner import teamAssigner
from player_ball_assigner import playerBallAssigner
import cv2
def main():
    # Read Video
    video_frames = read_video("Input_videos/input_video.mp4")

    # Tracker

    tracker = Tracker(model_path="models/best.pt")

    tracks = tracker.get_object_tracks(video_frames,
                                       read_from_stub=True,
                                       stub_path="pretrainned/trained_tracks.pkl")
    



    
    # Interpolate the ball position
    tracks["ball"] = tracker.interpolate_ball_pos(tracks["ball"])

    # Assign ball Aquisition
    player_assigner = playerBallAssigner()
    for frame_num,player_track in enumerate(tracks["players"]):
        ball_box = tracks['ball'][frame_num][1]['bbox']
        assigned_player = player_assigner.assign_ball_to_player(player_track,ball_box)

        if assigned_player != -1:
            tracks["players"][frame_num][assigned_player]["has_ball"] = True

    # Save Cropped Image Of A Player
    #crop_player(video_frames,tracks)

    # Assign Player Teams
    team_assigner = teamAssigner()
    team_assigner.assign_team_color(video_frames[0],tracks["players"][0])


    for frame_num,player_track in enumerate(tracks["players"]):
        for player_id,track in player_track.items():
            team = team_assigner.get_player_team(video_frames[frame_num],
                                                 track["bbox"],
                                                 player_id)
            tracks["players"][frame_num][player_id]["team"] = team
            tracks["players"][frame_num][player_id]["team_color"] = team_assigner.team_colors[team]


    # Draw Output
    # Draw Tracked Objects
    output_video_frames = tracker.draw_annotations(video_frames,tracks)
    # Save Video
    save_video(output_video_frames,"Output_videos/output_video.avi")

if __name__ == "__main__":
    main()