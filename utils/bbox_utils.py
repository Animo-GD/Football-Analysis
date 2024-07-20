import cv2
def get_box_center(bbox):
    return (bbox[0] + bbox[2]) // 2
def get_box_width(bbox):
    return bbox[2]-bbox[0]


def crop_player(frames,tracks):
    for track_id,player in tracks["players"][0].items():
        bbox = list(map(int,player["bbox"]))
        frame = frames[0]
        cropped_image = frame[bbox[1]:bbox[3],bbox[0]:bbox[2]] # y1:y2 & x1:x2
        # Save Cropped Image
        cv2.imwrite(f"Output_videos/cropped_img.jpg",cropped_image)
        break