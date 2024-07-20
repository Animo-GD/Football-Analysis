import cv2

def read_video(video_path):
    vid = cv2.VideoCapture(video_path)
    frames = []
    if vid:
        print("Reading The Video....")
    while True:
        ret,frame = vid.read()
        if not ret:
            break

        frames.append(frame)
    vid.release()
    print("Done!")
    return frames

def save_video(output_video,saving_path):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(saving_path,fourcc,24,(output_video[0].shape[1],output_video[0].shape[0]))

    if out:
        print("Writing The Video....")
    for frame in output_video:
        out.write(frame)

    
    out.release()
    print("Done!")

