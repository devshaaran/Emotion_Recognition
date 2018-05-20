import cv2
import time
import os
import random as rand

input_loc = "/media/shaaran/official/dataset_emotion/self"


def video_to_frames(input_loc):

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
    counter_s = 20
    counter_m = 30
    try:

        for i in os.listdir(input_loc):
            h = os.listdir(input_loc + '/' + i)
            print(h)
            os.chdir(input_loc + '/' + i)
            counter_s += 1
            os.mkdir(str(counter_s))
            for s in h:

                # Log the time
                time_start = time.time()
                print(i)
                print(s)
                # Start capturing the feed
                cap = cv2.VideoCapture(s)
                # Find the number of frames
                video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
                print("Number of frames: ", video_length)
                count = 0
                print("Converting video..\n")

                counts += 1
                output_loc = input_loc + '/' + i + '/' + str(counter_s)
                print(output_loc)
                # Start converting the video
                while cap.isOpened():
                    # Extract the frame
                    ret, frame = cap.read()
                    count = count + 1

                    if count > 8 :
                        cv2.imwrite((  output_loc + '/'+ str(counts) + str(count) + '.jpg'), frame)
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
            output_loc = input_loc + '/' + i + '/' + str(counter_s)
            mover = os.listdir(output_loc)
            print(mover)
            len_for = len(mover)
            print(len_for)
            len_mover = int(len_for * 20/100)
            print(len_mover)
            to_move = rand.sample(mover,k=len_mover)
            counter_m += 1
            os.mkdir(str(counter_m))
            for m in to_move:
                    os.rename(output_loc + '/' + m , input_loc + '/' + i + '/' + str(counter_m) + '/' + m)

    except Exception as e:
        print(e)


video_to_frames(input_loc)
