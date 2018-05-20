import cv2
import time
import os

input_loc = "D:\dataset_emotion\stimuli\\Neutral-Disgust"
output_loc = "D:\dataset_emotion\\disgust"

def video_to_frames(input_loc, output_loc):

    """Function to extract frames from input video file
    and save them as separate frames in an output directory.
    Args:
        input_loc: Input video file.
        output_loc: Output directory to save the frames.
    Returns:
        None
    """
    os.chdir(input_loc)
    counts = 0
    try:
        for i in os.listdir(input_loc):
            # Log the time
            time_start = time.time()
            print(i)
            # Start capturing the feed
            cap = cv2.VideoCapture(i)
            # Find the number of frames
            video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
            print("Number of frames: ", video_length)
            count = 0
            print("Converting video..\n")

            counts += 1
            # Start converting the video
            while cap.isOpened():
                # Extract the frame
                ret, frame = cap.read()
                count = count + 1

                if count > 7 :
                    cv2.imwrite((output_loc + '\\'+ str(counts) + str(count) + '.jpg'), frame)
                # Write the results back to output location.



                # If there are no more frames left
                if (count > (video_length - 1)):
                    # Log the time again
                    time_end = time.time()
                    # Release the feed
                    cap.release()
                    # Print stats
                    print("Done extracting frames.\n%d frames extracted" % count)
                    print("It took %d seconds forconversion." % (time_end - time_start))
                    break
    except Exception as e:
        print(e)


video_to_frames(input_loc, output_loc)
