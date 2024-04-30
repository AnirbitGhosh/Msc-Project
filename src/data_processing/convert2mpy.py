#%%
from moviepy.editor import VideoFileClip
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import pickle
from IPython.display import HTML
import argparse
import os
import sys

#%%
def dir_path(string):
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)
    
parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data', type=dir_path, help='Pass data directory path using -d or --data flags')

#%%
def animate(i, frames):
    plt.imshow(frames[i])
    plt.axis('off')  # Turn off axes
    return []

def show_animation(video_path):
    clip = VideoFileClip(video_path)
    
    # Extract frames from the video
    frames = [frame for frame in clip.iter_frames(fps=30)]

    # Create animation
    fig = plt.figure(figsize=(8, 6))
    ani = animation.FuncAnimation(fig, animate, frames=len(frames), fargs=(frames,), interval=1000/clip.fps)
    return ani
    # HTML(ani.to_jshtml())

def dump_frames(video_path, out_name):
    clip = VideoFileClip(video_path)
    
    # Extract frames from the video
    frames = [frame for frame in clip.iter_frames(fps=30)]
    
    with open(out_name, 'wb') as f:
        pickle.dump(frames, f)
    
# %%
if __name__ == "__main__":
    if len(sys.argv) < 3:
        parser.print_usage()
        parser.print_help()
        sys.exit(1)
    
    args = parser.parse_args()
    
    data_dir = args.data
    vid_dir = os.path.join(data_dir, 'videos')
    frame_dir = os.path.join(data_dir, 'frames')
    
    if not os.path.exists(frame_dir):
        os.makedirs(frame_dir)
    
    for mp4_file in os.listdir(vid_dir):
        if mp4_file.endswith('.mp4'):
            vid_path = os.path.join(vid_dir, mp4_file)
            out_name = mp4_file[:-4]+".mpy"
            
            dump_frames(vid_path, os.path.join(frame_dir, out_name))
    