import cv2
import datetime
import fn
import os
from PIL import Image
from pathlib import Path
import time
import numpy as np

class Camera(object):
    """for recording images from streaming source. right now assumes only one video source will be available
    
    >> cam = Camera()
    >> im = cam.get_image()
    >> cam.save_image()
    >> cam.close()
    """    
    def __init__(self, savedir=None, camera_index=None):
        if camera_index is None:
            camera_index = get_camera_index()
        self._camera_index = camera_index
        self.cam = cv2.VideoCapture(self._camera_index)
        self.plot_id = fn.get_current_plot_id()
        if savedir == None:
            savedir = f'/home/naka/Videos/pov/{self.plot_id}'
        self.savedir = Path(savedir)
        
        if not self.savedir.exists():
            os.mkdir(self.savedir)
        
    def get_image(self):
        got_image, _image = self.cam.read()
        image = cv2.cvtColor(_image, cv2.COLOR_RGB2BGR)
        return image
    
    def close(self):
        self.cam.release()
        
    def save_image(self):
        now = get_current_timestamp()
        img = self.get_image()
        filepath = self.savedir.joinpath(f'{now}.jpg').as_posix()
        cv2.imwrite(filepath, img)
        self.most_recent_image_filepath = filepath
        
    def preview(self):
        img = self.get_image()
        return Image.fromarray(img)
    
    def video_preview(self):
        
        print('hit q to quit')
        while(True):
            frame = self.get_image()
        
            # Display the resulting frame
            cv2.imshow('frame', frame)
            
            # the 'q' button is set as the
            # quitting button you may use any
            # desired button of your choice
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Destroy all the windows
        cv2.destroyAllWindows()
        
    def get_video(
        self, 
        wait_time:float=0., # seconds
        duration:float=np.inf, # seconds
        ):
        
        print('hit q to quit')
        self.n_frames_captured = 0
        while(True):
            
            if self.n_frames_captured == 0:
                self.video_start_time = datetime.datetime.now()
                self.elapsed_time = 0
            elif self.n_frames_captured > 0:
                self.elapsed_time = (datetime.datetime.now() - self.video_start_time).total_seconds()
                
            if self.elapsed_time > duration:
                break
            
            frame = self.get_image()
            now = get_current_timestamp()
            filepath = self.savedir.joinpath(f'{now}.jpg').as_posix()
            cv2.imwrite(filepath, frame)
            # Display the resulting frame
            cv2.imshow('frame', frame)
            
            self.n_frames_captured += 1
            # the 'q' button is set as the
            # quitting button you may use any
            # desired button of your choice
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            time.sleep(wait_time)
        
    

def return_camera_indices():
#     https://stackoverflow.com/questions/8044539/listing-available-devices-in-python-opencv
    
    # checks the first 10 indices.
    index = 0
    arr = []
    i = 10
    while i > 0:
        cap = cv2.VideoCapture(index)
        if cap.read()[0]:
            arr.append(index)
            cap.release()
        index += 1
        i -= 1
    return arr


def list_ports():
    """
    Test the ports and returns a tuple with the available ports and the ones that are working.
    """
    # https://stackoverflow.com/questions/57577445/list-available-cameras-opencv-python
    non_working_ports = []
    dev_port = 0
    working_ports = []
    available_ports = []
    while len(non_working_ports) < 6: # if there are more than 5 non working ports stop the testing. 
        camera = cv2.VideoCapture(dev_port)
        if not camera.isOpened():
            non_working_ports.append(dev_port)
            print("Port %s is not working." %dev_port)
        else:
            is_reading, img = camera.read()
            w = camera.get(3)
            h = camera.get(4)
            if is_reading:
                print("Port %s is working and reads images (%s x %s)" %(dev_port,h,w))
                working_ports.append(dev_port)
            else:
                print("Port %s for camera ( %s x %s) is present but does not reads." %(dev_port,h,w))
                available_ports.append(dev_port)
        dev_port +=1

def get_camera_index():
    camera_indices = return_camera_indices()
    assert len(camera_indices) > 0, 'No camera detected'
    assert len(camera_indices) < 2, 'Multiple cameras detected and I have not yet planned for this'
    return camera_indices[0]

def get_current_timestamp():
    return datetime.datetime.now().strftime('%Y-%m-%d_T%H-%M-%S-%f')