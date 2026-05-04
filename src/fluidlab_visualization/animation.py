from matplotlib.figure import Figure
from .figure_layout import FluidLabFigure
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
from pathlib import Path
import multiprocessing as mp
from functools import partial

class FluidLabAnimation():

    def __init__(self, frames_folder_name : str  = "__frames_video__", delete_old_frames : bool = True):
        self.frames_folder_name = frames_folder_name
        self.current_frame_index = 0

        # Creating a clean folder where the frames of this animation will be saved
        if( delete_old_frames and os.path.isdir(frames_folder_name) ):
            shutil.rmtree(frames_folder_name)
        if( not os.path.isdir(frames_folder_name) ):
            os.mkdir(frames_folder_name)

    def add_frame(self, figure: FluidLabFigure | Figure, dpi : int = 72, preview : bool = False):
        figure =  figure._matplotlib_fig if isinstance(figure, FluidLabFigure) else figure

        plt.figure(figure)

        if( preview ):
            plt.show()
        plt.savefig("%s/frame%04d.png" % (self.frames_folder_name, self.current_frame_index), dpi=dpi)
        plt.close()
        self.current_frame_index += 1

    def finalize_animation(self, animation_file_name : str = "new_video.mp4", 
                                ffmpeg_folder : str = "", 
                                framerate : int = 10, 
                                delete_frames : bool = True):
        
        # Calling ffmpeg externally to put all frames together and make the video
        path_images = Path(self.frames_folder_name) / "frame%04d.png"
        os.system('%sffmpeg -y -framerate %s -i %s -b 5000k %s' % (ffmpeg_folder, str(framerate), path_images, animation_file_name))

        # Deleting the folder with temporary frames
        if( delete_frames and os.path.isdir(self.frames_folder_name) ):
            shutil.rmtree(self.frames_folder_name)

    def create_animation_from_frame_function(self, function_generate_frame: callable, number_timesteps : int, args = None):

        # array_frame_numbers = list( range(0, number_timesteps, 1) )
        # func = partial(function_generate_frame, animation=self)

        with mp.Pool() as p:
            # p.map(func, array_frame_numbers)
            p.apply_async(function_generate_frame)

        # def func(aaa):
        #     print(aaa)


        # p = mp.Process(target=func, args=(self,))
        # p.start()
        # p.join()

        # p = mp.Process(target=func, args=(self,))
        # p.start()
        # p.join()




