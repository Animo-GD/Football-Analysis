from utils import read_video,save_video
from tracking import Tracker
def main():
    # Read Video
    video_frames = read_video("Input_videos/input_video.mp4")

    # Tracker

    tracker = Tracker(model_path="models/best.pt")

    tracks = tracker.get_object_tracks(video_frames,
                                       trainable=True,
                                       trained_model_path="pretrainned/trained_tracks.pkl")
    

    # Draw Output
    # Draw Tracked Objects
    output_video_frames = tracker.draw_annotations(video_frames,tracks)
    # Save Video
    save_video(output_video_frames,"Output_videos/output_video.avi")

if __name__ == "__main__":
    main()