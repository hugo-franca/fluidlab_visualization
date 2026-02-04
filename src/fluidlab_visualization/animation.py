from matplotlib.figure import Figure
from .figure_layout import FluidLabFigure
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil


class FluidLabAnimation():

    def __init__(self, frames_folder_name : str  = "__frames_video__", delete_old_frames : bool = True):
        self.frames_folder_name = frames_folder_name
        self.current_frame_index = 0

        # Creating a clean folder where the frames of this animation will be saved
        if( delete_old_frames and os.path.isdir(frames_folder_name) ):
            shutil.rmtree(frames_folder_name)
        if( not os.path.isdir(frames_folder_name) ):
            os.mkdir(frames_folder_name)


    def finalize_animation(self, animation_file_name : str = "new_video.mp4", 
                                ffmpeg_folder : str = "", 
                                framerate : int = 10, 
                                delete_frames : bool = True):
        
        # Calling ffmpeg externally to put all frames together and make the video
        os.system('%sffmpeg -y -framerate %s -i %s\\frame%%04d.png -b 5000k %s' % (ffmpeg_folder, str(framerate), self.frames_folder_name, animation_file_name))

        # Deleting the folder with temporary frames
        if( delete_frames and os.path.isdir(self.frames_folder_name) ):
            shutil.rmtree(self.frames_folder_name)

    def add_frame(self, figure: FluidLabFigure | Figure, dpi : int = 72, preview : bool = False):
        figure =  figure._matplotlib_fig if isinstance(figure, FluidLabFigure) else figure

        plt.figure(figure)

        if( preview ):
            plt.show()
        plt.savefig("%s/frame%04d.png" % (self.frames_folder_name, self.current_frame_index), dpi=dpi)
        plt.close()
        self.current_frame_index += 1

