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

    def add_frame(self, figure: FluidLabFigure | Figure, dpi : int = 72, preview : bool = False, file_format: str = "png"):
        figure =  figure._matplotlib_fig if isinstance(figure, FluidLabFigure) else figure

        plt.figure(figure)

        if( preview ):
            plt.show()
        plt.savefig("%s/frame%04d.%s" % (self.frames_folder_name, self.current_frame_index, file_format), dpi=dpi)
        plt.close()
        self.current_frame_index += 1

    def finalize_animation(self, animation_file_name : str = "new_video.mp4", 
                                    ffmpeg_path : str = "ffmpeg", 
                                    framerate : int = 10, 
                                    delete_frames : bool = True,
                                    quality: int = 35):
    
        ffmpeg_path = self.find_ffmpeg(ffmpeg_path)
        if( ffmpeg_path==None ):
            print("PROBLEM: Could not find ffmpeg in the system or in the provided path. The video file will not be created.")
            return


        # Calling ffmpeg externally to put all frames together and make the video
        path_images = Path(self.frames_folder_name) / "frame%04d.png"

        print(ffmpeg_path)
        os.system(
            '"%s" -y -framerate %s -i %s '
            '-c:v libx264 -crf %d -preset veryslow -tune animation -pix_fmt yuv420p '
            '%s'
            % (ffmpeg_path, str(framerate), path_images, quality, animation_file_name)
        )
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

    # Look for ffmpeg and see if we can find it with the given path or some default paths I normally have
    def find_ffmpeg(self, ffmpeg_path = None):

        default_tentative_paths = []
        default_tentative_paths.append( ffmpeg_path )
        default_tentative_paths.append( "ffmpeg" )
        default_tentative_paths.append("/gpfs/home6/hfranca/ffmpeg/ffmpeg-git-20240223-amd64-static/ffmpeg")
        default_tentative_paths.append("/home/hugo/VideosPython/ffmpeg-git-20240629-amd64-static/ffmpeg")

        found_path = None
        for tentative_path in default_tentative_paths:
            if( tentative_path is None ):
                continue

            path = os.path.expanduser(tentative_path)
            path = shutil.which(path)

            if path is not None:
                found_path = path
                break

        return found_path



