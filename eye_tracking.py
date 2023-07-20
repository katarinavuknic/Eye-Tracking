from asyncore import write
import os
import cv2
import tkinter as tk
from tkinter import  ttk
import PIL.Image, PIL.ImageTk, PIL.ImageDraw, PIL.ImageGrab
from matplotlib import pyplot as plt
import numpy as np
import mediapipe as mp
import math
import time
import csv
import pandas as pd
from scipy import stats
from scipy.ndimage import gaussian_filter
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from config import FILE_PATH


global CAMERA_SHOW
CAMERA_SHOW = 0

global COUNT_STIMULUS
COUNT_STIMULUS = 1

global FRAME_COUNT
FRAME_COUNT = 0

global DOT_ROW
DOT_ROW = 0

global DOT_COLUMN
DOT_COLUMN = 0

global DOT_NUM
DOT_NUM = 1

RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385,384, 398]
LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]

RIGHT_IRIS = [474, 475, 476, 477]
LEFT_IRIS = [469, 470, 471, 472]

L_P_LEFT_IRIS=[471]
R_P_LEFT_IRIS=[469]
L_P_RIGHT_IRIS=[476]
R_P_RIGHT_IRIS=[474]

L_H_LEFT = [33]
L_H_RIGHT = [133]
R_H_LEFT = [362]
R_H_RIGHT = [263]
R_BOTTOM = [374]
L_BOTTOM = [145]
R_TOP = [386]
L_TOP = [159]

NOSE = [1, 4]

mp_face_mesh = mp.solutions.face_mesh

def euclidean_distance(point1, point2):
    x1, y1 = point1.ravel()
    x2, y2 = point2.ravel()
    distance = math.sqrt((x2-x1)**2 + (y2-y1)**2)
    return distance

def cosine_rule(side1, side2, side3):
    cosine_angle = (pow(side1, 2) + pow(side2, 2) - pow(side3, 2))/(2*side1*side2)
    return cosine_angle

def iris_horizontal_position(iris_center, right_point, left_point):
    iris_center_right_eye_point_distance = euclidean_distance(iris_center, right_point)
    iris_center_left_eye_point_distance = euclidean_distance(iris_center, left_point)
    horizontal_eye_parameter = euclidean_distance(right_point, left_point) 
    
    if horizontal_eye_parameter!=0 and iris_center_left_eye_point_distance!=0 and iris_center_right_eye_point_distance!=0:
        cos_angle_iris_center_left_eye_point = cosine_rule(horizontal_eye_parameter, iris_center_left_eye_point_distance, iris_center_right_eye_point_distance)
        iris_center_left_eye_point_projection_distance = cos_angle_iris_center_left_eye_point*iris_center_left_eye_point_distance
        distance_ratio = iris_center_left_eye_point_projection_distance/horizontal_eye_parameter
        return distance_ratio
    else:
        return -1

def iris_vertical_position(iris_center, top_point, bottom_point, eye_height):
    iris_center_top_eye_point_distance = euclidean_distance(iris_center, top_point)
    iris_center_bottom_eye_point_distance = euclidean_distance(iris_center, bottom_point)
    vertical_eye_parameter = euclidean_distance(top_point, bottom_point)
    
    if iris_center_top_eye_point_distance!=0 and iris_center_bottom_eye_point_distance!=0 and vertical_eye_parameter!=0:
        cos_angle_iris_center_top_eye_point = cosine_rule(iris_center_top_eye_point_distance, vertical_eye_parameter, iris_center_bottom_eye_point_distance)
        iris_center_top_eye_point_projection_distance = cos_angle_iris_center_top_eye_point*iris_center_top_eye_point_distance
        distance_ratio = iris_center_top_eye_point_projection_distance/eye_height
        return distance_ratio
    else:
        return -1

def make_directory(username):
    path = os.path.join(FILE_PATH, username)
    os.makedirs(path, exist_ok=True)
    filename = f'{path}/{username}_calibration_collected_data.csv'
    columns = ['Left eye horizontal ratio', 'Right eye horizontal ratio', 'Left eye vertical ratio', 'Right eye vertical ratio', 'Dot center x', 'Dot center y']
    with open(filename, 'a+', newline='', encoding='UTF8') as csvfile_calibration:
        csvwriter = csv.writer(csvfile_calibration)
        csvwriter.writerow(columns)

def write_calibration_parameters(username, horizontal_ratio_left_eye, horizontal_ratio_right_eye, vertical_ratio_left_eye, vertical_ratio_right_eye, dot_center_x, dot_center_y):
    columns = ['Left eye horizontal ratio', 'Right eye horizontal ratio', 'Left eye vertical ratio', 'Right eye vertical ratio', 'Dot center x', 'Dot center y']
    row = [f'{horizontal_ratio_left_eye}', f'{horizontal_ratio_right_eye}', f'{vertical_ratio_left_eye}', f'{vertical_ratio_right_eye}', f'{dot_center_x}', f'{dot_center_y}']
    filename = f"{username}/{username}_calibration_collected_data.csv"
    file_exists = os.path.isfile(filename)
    with open(filename, 'a+', newline='', encoding='UTF8') as csvfile:
        csvwriter = csv.writer(csvfile)
        if not file_exists:
            csvwriter.writerow(columns)
        csvwriter.writerow(row)

def write_stimulus_eye_gaze_metrics_in_file(username, horizontal_ratio_left_eye, horizontal_ratio_right_eye, vertical_ratio_left_eye, vertical_ratio_right_eye, left_eye_blink_check, right_eye_blink_check, time):
    columns = ['Left eye horizontal ratio', 'Right eye horizontal ratio', 'Left eye vertical ratio', 'Right eye vertical ratio', 'Left eye blink check', 'Right eye blink check', 'Time']
    row =  [f'{horizontal_ratio_left_eye}', f'{horizontal_ratio_right_eye}', f'{vertical_ratio_left_eye}', f'{vertical_ratio_right_eye}', f'{left_eye_blink_check}', f'{right_eye_blink_check}',f'{time}']
    filename = f"{username}/{username}_eye_gaze_data_stimulus_{COUNT_STIMULUS-1}.csv"
    file_exists = os.path.isfile(filename) 
    with open(filename, 'a+', newline='', encoding='UTF8') as csvfile:
        csvwriter = csv.writer(csvfile)
        if not file_exists:
            csvwriter.writerow(columns)
        csvwriter.writerow(row)

def write_predictions_in_file(username, predicted_X_screen_position, predicted_Y_screen_position, time):
    columns = ['Predicted X screen position', 'Predicted Y screen position', 'Time in ms']
    row = [f'{predicted_X_screen_position}', f'{predicted_Y_screen_position}', f'{time}'] 
    filename = f'{username}/{username}_stimulus_{COUNT_STIMULUS}_predictions.csv'
    file_exists = os.path.isfile(filename)
    with open(filename, 'a+', newline='', encoding='UTF8') as csvfile:
        csvwriter = csv.writer(csvfile)
        if not file_exists:
            csvwriter.writerow(columns)
        csvwriter.writerow(row)

def read_predictions_file(username):
    path = os.path.join(FILE_PATH, username)
    df = pd.read_csv(f'{path}/{username}_stimulus_{COUNT_STIMULUS}_predictions.csv', delimiter=',', usecols= ['Predicted X screen position', 'Predicted Y screen position', 'Time in ms'])
    return df

def write_experiment_parameters(username, event_x, event_y, prediction_x, prediction_y):
    columns = ['Event x', 'Event y', 'Prediction x', 'Prediction y']
    row = [f'{event_x}', f'{event_y}', f'{prediction_x}', f'{prediction_y}']
    filename = f'{username}/{username}_experiment_parameters.csv'
    file_exists = os.path.isfile(filename)
    with open(filename, '+a', newline='', encoding='UTF8') as csvfile:
        csvwriter = csv.writer(csvfile)
        if not file_exists:
            csvwriter.writerow(columns)
        csvwriter.writerow(row)

def read_experiment_parameters(username):
    path = os.path.join(FILE_PATH, username)
    df = pd.read_csv(f'{path}/{username}_experiment_parameters.csv', delimiter=',', usecols= ['Event x', 'Event y', 'Prediction x', 'Prediction y'])
    return df

def read_calibration_data(username):
    path = os.path.join(FILE_PATH, username)
    df = pd.read_csv(f'{path}/{username}_calibration_collected_data.csv', delimiter=',', usecols= ['Left eye horizontal ratio', 'Right eye horizontal ratio', 'Left eye vertical ratio', 'Right eye vertical ratio', 'Dot center x', 'Dot center y'])
    return df['Left eye horizontal ratio'].values.tolist(), df['Right eye horizontal ratio'].values.tolist(), df['Left eye vertical ratio'].values.tolist(), df['Right eye vertical ratio'].values.tolist(), df['Dot center x'].values.tolist(), df['Dot center y'].values.tolist()

def read_gaze_data_stimulus(username):
    path = os.path.join(FILE_PATH,username)
    df = pd.read_csv(f'{path}/{username}_eye_gaze_data_stimulus_{COUNT_STIMULUS}.csv', delimiter=',', usecols= ['Left eye horizontal ratio', 'Right eye horizontal ratio', 'Left eye vertical ratio', 'Right eye vertical ratio', 'Left eye blink check', 'Right eye blink check', 'Time'])
    return df.to_numpy()

def write_collected_training_data(username, real_x, real_y, prediction_left_eye_x, prediction_left_eye_y, prediction_right_eye_x, prediction_right_eye_y, left_parameter_h, left_parameter_v, right_parameter_h, right_parameter_v):
    columns = ['Real x', 'Real y', 'Prediction left eye x', 'Prediction left eye y', 'Prediction right eye x', 'Prediction right eye y', 'Left eye horizontal', 'Left eye vertical', 'Right eye horizontal', 'Right eye vertical']
    row = [f'{real_x}', f'{real_y}', f'{prediction_left_eye_x}', f'{prediction_left_eye_y}', f'{prediction_right_eye_x}', f'{prediction_right_eye_y}', f'{left_parameter_h}', f'{left_parameter_v}', f'{right_parameter_h}', f'{right_parameter_v}']
    filename = f'{username}/{username}_collected_training_data.csv'
    file_exists = os.path.isfile(filename)
    with open(filename, '+a', newline='', encoding='UTF8') as csvfile:
        csvwriter = csv.writer(csvfile)
        if not file_exists:
            csvwriter.writerow(columns)
        csvwriter.writerow(row)

def read_collected_training_data(username):
    path = os.path.join(FILE_PATH, username)
    df = pd.read_csv(f'{path}/{username}_collected_training_data.csv', delimiter=',', usecols=['Real x', 'Real y', 'Prediction left eye x', 'Prediction left eye y', 'Prediction right eye x', 'Prediction right eye y', 'Left eye horizontal', 'Left eye vertical', 'Right eye horizontal', 'Right eye vertical'])
    return df

def read_data_for_evaluation(username):
    path = os.path.join(FILE_PATH, username)
    df = pd.read_csv(f'{path}/random_dots_for_evaluation.csv', delimiter=',', usecols=['Real x', 'Real y', 'Prediction left eye x', 'Prediction left eye y', 'Prediction right eye x', 'Prediction right eye y', 'Left eye horizontal', 'Left eye vertical', 'Right eye horizontal', 'Right eye vertical'])
    return df

def linear_regression(list_x_variable, list_y_variable):
    slope, intercept, r, p, std_err = stats.linregress(list_x_variable, list_y_variable)
    return slope, intercept

def calculate_gaze_position(slope, intercept, x):
    return slope*x + intercept

class App(tk.Tk): 
    def __init__(self):
        super().__init__()

        self.title('Eye tracking app')
        
        self.iconphoto(False, tk.PhotoImage(file='eye_track_icon.png'))
        self.attributes('-fullscreen', True)

        container = tk.Frame(self, bg="#FFFFFF", height=self.winfo_screenheight(), width=self.winfo_screenwidth())
       
        container.pack(fill="both", expand = True)
        self.bottom_frame = tk.Frame(self)
        self.bottom_frame.pack(side=tk.BOTTOM)
        self.canvas = tk.Canvas(container, highlightthickness=0)
        self.canvas.pack(pady=120)
     
        self.label_username = tk.Label(self.bottom_frame, text="Username:")
        self.label_username.pack(side=tk.LEFT)
        self.entry_username = tk.Entry(self.bottom_frame)
        self.entry_username.pack(padx=5, side=tk.LEFT)
        self.btn_save_username = ttk.Button(self.bottom_frame, text='Save', command=self.get_input)
        self.btn_save_username.pack(padx=5, pady=5, side=tk.BOTTOM)
      
        self.open_camera_check(CAMERA_SHOW) 
        
    def open_camera_check(self, video_source):

        self.video = MyVideoCapture(CAMERA_SHOW)
        self.canvas.configure(width=self.video.width, height=self.video.height)
        self.canvas.pack()

        self.delay = 12
        self.mode = CAMERA_SHOW
        self.update()

    def get_input(self):
        print('Name: ', self.entry_username.get()) 
        user = self.entry_username.get()
        if(user != ""): 
            self.username = user
            self.bottom_frame.destroy()
            self.button_next = ttk.Button(self, text='Next', command=self.calibrate)
            self.button_next.pack(side=tk.RIGHT, padx=5, pady=5)
            make_directory(user)
        else:
            self.msg_username_empty = tk.Label(self.bottom_frame, text="Username required")
            self.msg_username_empty.pack(side=tk.BOTTOM)

    def update(self):
        global FRAME_COUNT

        ret, rgb_frame = self.video.get_frame()
    
        if ret and self.mode==0:
            self.photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(rgb_frame))
            self.image_on_canvas = self.canvas.create_image(0,0, image=self.photo, anchor=tk.NW)

        FRAME_COUNT += 1
        
        if CAMERA_SHOW==2 or CAMERA_SHOW==3 or CAMERA_SHOW==5 or CAMERA_SHOW==6:    
            current_username = self.username
            if CAMERA_SHOW == 3:
                t = time.time()
                current_time_ms = int(t*1000)
                write_stimulus_eye_gaze_metrics_in_file(current_username, self.video.eye_left_horizontal_ratio, self.video.eye_right_horizontal_ratio, self.video.left_eye_vertical_ratio, self.video.right_eye_vertical_ratio, self.video.left_eye_blink_check, self.video.right_eye_blink_check, current_time_ms)

        self.after(self.delay, self.update)

    def calibrate(self):
        global CAMERA_SHOW
        CAMERA_SHOW = 1
        self.mode = CAMERA_SHOW
        self.canvas.delete('all')      
        self.canvas.configure(height=self.winfo_screenheight(), width=self.winfo_screenwidth(), bg="#555555")
        self.canvas.pack(fill="both", expand=True, pady=0, padx=0)
        self.draw_dot()

    def dot_click_event(self, N, dot_center_x, dot_center_y, event):
        global DOT_COLUMN
        global DOT_ROW
        global DOT_NUM
       
        closest = self.canvas.find_closest(event.x, event.y)
        self.canvas.delete(closest)
                
        current_frame = self.study_frame()

        current_username = self.username
        if current_username:
            current_distance_from_screen = self.video.person_distance_from_camera
            current_left_eye_horizontal_ratio = self.video.eye_left_horizontal_ratio
            current_right_eye_horizontal_ratio = self.video.eye_right_horizontal_ratio
            current_left_eye_vertical_ratio= self.video.left_eye_vertical_ratio
            current_right_eye_vertical_ratio = self.video.right_eye_vertical_ratio

            write_calibration_parameters(current_username, current_left_eye_horizontal_ratio, current_right_eye_horizontal_ratio, current_left_eye_vertical_ratio, current_right_eye_vertical_ratio,dot_center_x, dot_center_y)
        
            # cv2.imwrite("calculating_parameters-" + f'{N}' + ".jpg", current_frame)

        if DOT_NUM < 9:
            DOT_NUM += 1
            self.draw_dot()
        else :
            print("Last dot.")
            self.get_regression_parameters()
            self.collect_training_data()

    def draw_dot(self):

        global DOT_NUM
        window_width = self.winfo_screenwidth()
        window_height = self.winfo_screenheight()

        dot_radius = 13 
        origin_x = 0
        origin_y = 0
        
        if DOT_NUM == 1:
            x0 = origin_x
            y0 = origin_y
            x1 = origin_x+2*dot_radius
            y1 = origin_y+2*dot_radius
        elif DOT_NUM ==2:
            x0 = window_width/4-dot_radius
            y0 = window_height/4-dot_radius
            x1 = window_width/4+dot_radius
            y1 = window_height/4+dot_radius
        elif DOT_NUM == 3:
            x0 = window_width*3/4-dot_radius
            y0 = window_height/4-dot_radius
            x1 = window_width*3/4+dot_radius
            y1 = window_height/4+dot_radius
        elif DOT_NUM == 4:
            x0 = window_width-2*dot_radius
            y0 = origin_y
            x1 = window_width
            y1 = origin_y+2*dot_radius
        elif DOT_NUM == 5:
            x0 = origin_x
            y0 = window_height-2*dot_radius
            x1 = origin_x+2*dot_radius
            y1 = window_height
        elif DOT_NUM == 6:
            x0 = window_width/4-dot_radius
            y0 = window_height*3/4-dot_radius
            x1 = window_width/4+dot_radius
            y1 = window_height*3/4+dot_radius
        elif DOT_NUM == 7:
            x0 = window_width/2-dot_radius
            y0 = window_height/2-dot_radius
            x1 = window_width/2+dot_radius
            y1 = window_height/2+dot_radius
        elif DOT_NUM == 8:
            x0 = window_width*3/4-dot_radius
            y0 = window_height*3/4-dot_radius
            x1 = window_width*3/4+dot_radius
            y1 = window_height*3/4+dot_radius
        elif DOT_NUM == 9:
            x0 = window_width-2*dot_radius
            y0 = window_height-2*dot_radius
            x1 = window_width
            y1 = window_height

        globals()[f'attention_dot_{DOT_NUM}'] = self.canvas.create_oval(x0, y0, x1, y1, fill='#DE4C4C', outline='#DE4C4C')
        dot_center_x = x0 + dot_radius
        dot_center_y = y0 + dot_radius
        self.canvas.pack()
        self.canvas.tag_bind(globals()[f'attention_dot_{DOT_NUM}'], '<Button-1>', lambda event, dot_num = DOT_NUM, dot_center_x = dot_center_x, dot_center_y = dot_center_y: self.dot_click_event(dot_num, dot_center_x, dot_center_y, event))

    def study_frame(self):
        ret, frame = self.video.get_frame()
        return frame 

    def stimulus_attention_tracking(self):

        global CAMERA_SHOW
        CAMERA_SHOW = 3
        self.mode = CAMERA_SHOW

        self.canvas.delete('all')      
        self.canvas.configure(height=self.winfo_screenheight(), width=self.winfo_screenwidth())
        self.canvas.pack(fill="both", expand=True, pady=0, padx=0)
        self.msg_flag = 1
        self.display_message('Observe following 3 pictures')

    def next_stimulus(self):
        global COUNT_STIMULUS
    
        if COUNT_STIMULUS == 4:
            # self.msg_flag = 2
            # self.canvas.delete('all')
            # self.canvas.configure(height=self.winfo_screenheight(), width=self.winfo_screenwidth(), bg="#494949")
            # self.canvas.pack(fill="both", expand=True, pady=0, padx=0)
            # self.display_message('Follow the dot across the screen') 
            self.save_results()

        else:
            image = PIL.Image.open(f"visual_stimuli/stimulus{COUNT_STIMULUS}.jpg")

            self.stimulus_image_width = self.canvas.winfo_screenwidth()
            self.stimulus_image_height = self.canvas.winfo_screenheight()
            image = image.resize((self.stimulus_image_width, self.stimulus_image_height), PIL.Image.Resampling.LANCZOS) #Resampling.LANCZOSResampling.LANCZOS -> ANTIALIAS
            self.current_stimulus = PIL.ImageTk.PhotoImage(image=image) 
            self.canvas.create_image(self.winfo_screenwidth()/2, self.winfo_screenheight()/2, image=self.current_stimulus, anchor=tk.CENTER)
            self.canvas.pack()
            COUNT_STIMULUS += 1
     
            self.after(10000, self.next_stimulus)

    def get_eye_parameters(self, event_x, event_y):
      
        current_horizontal_ratio_left_eye = self.video.eye_left_horizontal_ratio
        current_horizontal_ratio_right_eye = self.video.eye_right_horizontal_ratio
        current_vertical_ratio_left_eye = self.video.left_eye_vertical_ratio
        current_vertical_ratio_right_eye = self.video.right_eye_vertical_ratio

        left_eye_gaze_position_predicted_X = calculate_gaze_position(self.slope_horizontal_left_eye, self.intercept_horizontal_left_eye, current_horizontal_ratio_left_eye)
        right_eye_gaze_position_predicted_X = calculate_gaze_position(self.slope_horizontal_right_eye, self.intercept_horizontal_right_eye, current_horizontal_ratio_right_eye)

        left_eye_gaze_position_predicted_Y = calculate_gaze_position(self.slope_vertical_left_eye, self.intercept_vertical_left_eye, current_vertical_ratio_left_eye)
        right_eye_gaze_position_predicted_Y = calculate_gaze_position(self.slope_vertical_right_eye, self.intercept_vertical_right_eye, current_vertical_ratio_right_eye)

        clicked_position_predicted_X = (calculate_gaze_position(self.new_slope_horizontal_left_eye, self.new_intercept_horizontal_left_eye, left_eye_gaze_position_predicted_X) + calculate_gaze_position(self.new_slope_horizontal_right_eye, self.new_intercept_horizontal_right_eye, right_eye_gaze_position_predicted_X))/2
        clicked_position_predicted_Y = (calculate_gaze_position(self.new_slope_vertical_left_eye, self.new_intercept_vertical_left_eye, left_eye_gaze_position_predicted_Y) + calculate_gaze_position(self.new_slope_vertical_right_eye, self.new_intercept_vertical_right_eye, right_eye_gaze_position_predicted_Y))/2 
            
        write_experiment_parameters(self.username, event_x, event_y, clicked_position_predicted_X, clicked_position_predicted_Y)

    def calculate_cost_function(self):

        data = read_collected_training_data(self.username)

        random_subset = data.sample(n=15, random_state=42)
        subset_data_df = pd.DataFrame(random_subset)
        subset_data_df.to_csv(f"{self.username}/random_dots_for_evaluation.csv", index=False)

        left_eye_predicted_values_X = subset_data_df.iloc[:,2]
        right_eye_predicted_values_X = subset_data_df.iloc[:,4]
        actual_values_X = subset_data_df.iloc[:,0]
        left_eye_predicted_values_Y = subset_data_df.iloc[:,3]
        right_eye_predicted_values_Y = subset_data_df.iloc[:,5]
        actual_values_Y = subset_data_df.iloc[:,1]

        left_eye_mse_X = mean_squared_error(actual_values_X, left_eye_predicted_values_X)
        right_eye_mse_X = mean_squared_error(actual_values_X, right_eye_predicted_values_X)
        left_eye_mse_Y = mean_squared_error(actual_values_Y, left_eye_predicted_values_Y)
        right_eye_mse_Y = mean_squared_error(actual_values_Y, right_eye_predicted_values_Y)

        left_eye_Rmse_X = math.sqrt(left_eye_mse_X)
        right_eye_Rmse_X = math.sqrt(right_eye_mse_X)
        left_eye_Rmse_Y = math.sqrt(left_eye_mse_Y)
        right_eye_Rmse_Y = math.sqrt(right_eye_mse_Y)

        file = open(f"{FILE_PATH}/{self.username}/metrics.txt", "w")
        file.write("MSE before exp:\n")
        file.write(f"MSE for left eye X: {left_eye_mse_X}" + "\n")
        file.write(f"MSE for right eye X: {right_eye_mse_X}" + "\n")

        file.write(f"MSE for left eye Y: {left_eye_mse_Y}" + "\n")
        file.write(f"MSE for right eye Y: {right_eye_mse_Y}" + "\n")

        file.write(f"RMSE for left eye X: {left_eye_Rmse_X}" + "\n")
        file.write(f"RMSE for right eye X: {right_eye_Rmse_X}" + "\n")

        file.write(f"RMSE for left eye Y: {left_eye_Rmse_Y}" + "\n")
        file.write(f"RMSE for right eye Y: {right_eye_Rmse_Y}" + "\n")

        file.close()

    def remove_dot(self):
        self.canvas.delete(self.dot_evaluation)
        self.draw_evaluation_dot(self.df)

    def draw_evaluation_dot(self, df):
        dot_radius = 13
        self.df =df
        if self.row < len(df):
            dot_X = df.iloc[self.row, 0]
            dot_Y = df.iloc[self.row, 1]
            self.dot_evaluation = self.canvas.create_oval(dot_X-dot_radius, dot_Y-dot_radius, dot_X+dot_radius, dot_Y+dot_radius, fill='#DE4C4C', outline='#DE4C4C')
            self.get_eye_parameters(dot_X, dot_Y)
            self.row += 1
            self.after(1200, self.remove_dot)
        elif self.row >= len(df):
            eye_data = read_experiment_parameters(self.username)
            actual_values_X = eye_data.iloc[:-1, 0]
            actual_values_Y = eye_data.iloc[:-1, 1]
            predicted_values_X = eye_data.iloc[:-1, 2]
            predicted_values_Y = eye_data.iloc[:-1, 3]

            mse_X = mean_squared_error(actual_values_X, predicted_values_X)
            mse_Y = mean_squared_error(actual_values_Y, predicted_values_Y)
            rmse_X = math.sqrt(mse_X)
            rmse_Y = math.sqrt(mse_Y)              

            file = open(f"{FILE_PATH}/{self.username}/metrics.txt", "a")
            file.write("MSE after experiment:\n")
            file.write(f"MSE for X: {mse_X}" + "\n")
            file.write(f"MSE for Y: {mse_Y}" + "\n")
            file.write(f"RMSE for X: {rmse_X}" + "\n")
            file.write(f"RMSE for Y: {rmse_Y}" + "\n")
            file.close()
            # self.save_results()

    def calculate_cost_function_after(self):

        data = read_data_for_evaluation(self.username)
        self.row = 0
        self.draw_evaluation_dot(data)

    def support_vector_regression(self, X_values, Y_values):

        training_data = read_collected_training_data(self.username)      
        
        X = training_data.loc[:, f'{X_values}']
        y = training_data.loc[:, f'{Y_values}']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        svr = SVR(kernel='linear', C=1.0)
        svr.fit(X_train.to_numpy().reshape(-1,1), y_train.ravel())
        y_pred = svr.predict(X_test.to_numpy().reshape(-1,1))

        r2 = r2_score(y_test.ravel(), y_pred.ravel())
        new_slope, new_intercept, r_value, p_value, std_err = stats.linregress(y_pred.ravel(), y_test.ravel()) #obrni
        mse = mean_squared_error(y_test.ravel(), y_pred.ravel())
        rmse = math.sqrt(mse)

        file = open(f"{FILE_PATH}/{self.username}/metrics.txt", "a")
        file.write(f"R2 score for {X_values}: {r2}" + "\n")
        file.write(f"MSE in svr (for {X_values} and {Y_values} relationship): {mse}" + "\n")
        file.write(f"RMSE in svr (for {X_values} and {Y_values} relationship): {rmse}" + "\n")
        file.close()
        return new_slope, new_intercept

    def get_linear_regression_prediction(self):
        
        if self.username:
            current_horizontal_ratio_left_eye = self.video.eye_left_horizontal_ratio
            current_horizontal_ratio_right_eye = self.video.eye_right_horizontal_ratio
            current_vertical_ratio_left_eye = self.video.left_eye_vertical_ratio
            current_vertical_ratio_right_eye = self.video.right_eye_vertical_ratio
            current_left_eye_blink_check = self.video.left_eye_blink_check
            current_right_eye_blink_check = self.video.right_eye_blink_check

            left_eye_gaze_position_predicted_X = calculate_gaze_position(self.slope_horizontal_left_eye, self.intercept_horizontal_left_eye, current_horizontal_ratio_left_eye)
            right_eye_gaze_position_predicted_X = calculate_gaze_position(self.slope_horizontal_right_eye, self.intercept_horizontal_right_eye, current_horizontal_ratio_right_eye)

            left_eye_gaze_position_predicted_Y = calculate_gaze_position(self.slope_vertical_left_eye, self.intercept_vertical_left_eye, current_vertical_ratio_left_eye)
            right_eye_gaze_position_predicted_Y = calculate_gaze_position(self.slope_vertical_right_eye, self.intercept_vertical_right_eye, current_vertical_ratio_right_eye)
            
            return left_eye_gaze_position_predicted_X, left_eye_gaze_position_predicted_Y, right_eye_gaze_position_predicted_X, right_eye_gaze_position_predicted_Y, current_horizontal_ratio_left_eye, current_vertical_ratio_left_eye, current_horizontal_ratio_right_eye, current_vertical_ratio_right_eye, current_left_eye_blink_check, current_right_eye_blink_check

    def collect_training_data(self):

        global CAMERA_SHOW
        CAMERA_SHOW = 6 
        self.mode = CAMERA_SHOW

        self.canvas.delete('all')
        self.canvas.configure(height=self.winfo_screenheight(), width=self.winfo_screenwidth(), bg="#555555")
        self.canvas.pack(fill="both", expand=True, pady=0, padx=0)

        self.messages = ["Follow the dot across the screen", "3", "2", "1"]
        self.msg_flag = 0
        self.message_index = 0
        self.display_messages()
    
    def display_messages(self):
        if self.message_index < len(self.messages):
            message = self.messages[self.message_index]
            print("Index: ", self.message_index)
            self.display_message(message)
            self.message_index += 1
            self.after(1200, self.display_messages) 
            
    def remove_message(self):
        self.canvas.delete(self.msgId)
        if self.message_index == 3:
                self.after(1200,self.draw_animation)  
        if self.msg_flag == 1:
            self.next_stimulus()
        if self.msg_flag == 2:
             self.experiment1_calculate_error()                         
  
    def display_message(self, message):
        self.msgId = self.canvas.create_text(self.winfo_width()/2, self.winfo_height()/2, text=message, font=('Arial', 20), fill='white')
        self.after(1200, self.remove_message)

    def draw_animation(self):
        
        if self.message_index == 3:
            self.canvas.delete(self.msgId)
                
        dot_radius = 13
        origin_x = 0
        origin_y = 0

        x0 = origin_x-dot_radius 
        y0 = origin_y-dot_radius
        x1 = origin_x + dot_radius
        y1 = origin_y + dot_radius
        
        self.ball = self.canvas.create_oval(x0, y0, x1, y1, fill='#DE4C4C', outline='#DE4C4C')
        self.animate()  

    def animate(self):

        dot_radius = 13
        
        x0, y0, x1, y1 = self.canvas.coords(self.ball)
        horizontal_space = self.canvas.winfo_width()/8
        vertical_space = self.canvas.winfo_height()/8
        self.wait_time = 1200

        predicted_left_gaze_X, predicted_left_gaze_Y, predicted_right_gaze_X, predicted_right_gaze_Y, left_parameter_h, left_parameter_v, right_parameter_h, right_parameter_v, left_eye_blink, right_eye_blink = self.get_linear_regression_prediction()
    
        if not ((left_eye_blink>=0 and left_eye_blink<=4) and (right_eye_blink>=0 and right_eye_blink<=4)):
            write_collected_training_data(self.username, x1, y1, predicted_left_gaze_X, predicted_left_gaze_Y, predicted_right_gaze_X, predicted_right_gaze_Y,left_parameter_h, left_parameter_v, right_parameter_h, right_parameter_v) 

        if y1 < vertical_space+dot_radius:
            if x1 < horizontal_space+dot_radius:
                self.canvas.move(self.ball, horizontal_space, vertical_space) 
                self.after(self.wait_time, self.animate)
        elif y1 == vertical_space+dot_radius or y1 == 3*vertical_space+dot_radius or y1 == 5*vertical_space+dot_radius:
            if x1 < horizontal_space+dot_radius:
                self.canvas.move(self.ball, horizontal_space, 0) 
                self.after(self.wait_time, self.animate)
            elif x1 >= horizontal_space+dot_radius and x1 < horizontal_space*7+dot_radius:
                self.canvas.move(self.ball, horizontal_space, 0)
                self.after(self.wait_time, self.animate)
            elif x1 >= horizontal_space*7+dot_radius:
                self.canvas.move(self.ball, 0, vertical_space)
                self.after(self.wait_time, self.animate)
        elif y1 == 2*vertical_space+dot_radius or y1 == 4*vertical_space+dot_radius or y1 == 6*vertical_space+dot_radius:
            if x1 < horizontal_space+dot_radius:
                self.canvas.move(self.ball, horizontal_space, 0) 
                self.after(self.wait_time, self.animate)
            elif x1 <= horizontal_space+dot_radius:
                self.canvas.move(self.ball, 0, vertical_space) 
                self.after(self.wait_time, self.animate)
            elif x1 >= 2*horizontal_space+dot_radius and x1 < horizontal_space*8+dot_radius:
                self.canvas.move(self.ball, -horizontal_space, 0)
                self.after(self.wait_time, self.animate)
            elif x1 >= horizontal_space*7+dot_radius:
                self.canvas.move(self.ball, -horizontal_space, 0)
                self.after(self.wait_time, self.animate)
        elif y1 == 7*vertical_space+dot_radius:
            if x1 < horizontal_space+dot_radius:
                self.canvas.move(self.ball, horizontal_space, 0) 
                self.after(self.wait_time, self.animate)
            elif x1 >= horizontal_space+dot_radius and x1 < horizontal_space*7+dot_radius:
                self.canvas.move(self.ball, horizontal_space, 0)
                self.after(self.wait_time, self.animate)
            elif x1 >= horizontal_space*7+dot_radius:
                self.calculate_cost_function() 
                self.new_slope_horizontal_left_eye, self.new_intercept_horizontal_left_eye = self.support_vector_regression('Prediction left eye x', 'Real x')
                self.new_slope_horizontal_right_eye, self.new_intercept_horizontal_right_eye = self.support_vector_regression('Prediction right eye x', 'Real x')
                self.new_slope_vertical_left_eye, self.new_intercept_vertical_left_eye = self.support_vector_regression('Prediction left eye y', 'Real y')
                self.new_slope_vertical_right_eye, self.new_intercept_vertical_right_eye = self.support_vector_regression('Prediction right eye y', 'Real y')
                self.stimulus_attention_tracking()
        else:
            print("Out of screen. Exp stop.") 
            self.calculate_cost_function() 
            self.new_slope_horizontal_left_eye, self.new_intercept_horizontal_left_eye = self.support_vector_regression('Prediction left eye x', 'Real x')
            self.new_slope_horizontal_right_eye, self.new_intercept_horizontal_right_eye = self.support_vector_regression('Prediction right eye x', 'Real x')
            self.new_slope_vertical_left_eye, self.new_intercept_vertical_left_eye = self.support_vector_regression('Prediction left eye y', 'Real y')
            self.new_slope_vertical_right_eye, self.new_intercept_vertical_right_eye = self.support_vector_regression('Prediction right eye y', 'Real y')
            self.stimulus_attention_tracking()        
       
    def experiment1_calculate_error(self):  

        global CAMERA_SHOW
        CAMERA_SHOW = 5
        self.mode = CAMERA_SHOW

        self.canvas.delete('all')
        self.canvas.configure(height=self.winfo_screenheight(), width=self.winfo_screenwidth(), bg="#555555")
        self.canvas.pack(fill="both", expand=True, pady=0, padx=0)

        self.calculate_cost_function_after()

    def get_regression_parameters(self):
        left_eye_horizontal_ratios, right_eye_horizontal_ratios, left_eye_vertical_ratios, right_eye_vertical_ratios, dot_x_positions, dot_y_positions = read_calibration_data(self.username)
        slope_horizontal_left_eye, intercept_horizontal_left_eye = linear_regression(left_eye_horizontal_ratios, dot_x_positions)
        slope_vertical_left_eye, intercept_vertical_left_eye = linear_regression(left_eye_vertical_ratios, dot_y_positions)
        slope_horizontal_right_eye, intercept_horizontal_right_eye = linear_regression(right_eye_horizontal_ratios, dot_x_positions)
        slope_vertical_right_eye, intercept_vertical_right_eye = linear_regression(right_eye_vertical_ratios, dot_y_positions)
        #left eye 
        self.slope_horizontal_left_eye = slope_horizontal_left_eye
        self.intercept_horizontal_left_eye = intercept_horizontal_left_eye
        self.slope_vertical_left_eye = slope_vertical_left_eye
        self.intercept_vertical_left_eye = intercept_vertical_left_eye
        #right eye
        self.slope_horizontal_right_eye = slope_horizontal_right_eye
        self.intercept_horizontal_right_eye = intercept_horizontal_right_eye
        self.slope_vertical_right_eye = slope_vertical_right_eye
        self.intercept_vertical_right_eye = intercept_vertical_right_eye

    def save_predictions_of_stimulus_gaze(self):
        global COUNT_STIMULUS
        if self.username:
            eye_gaze_data = read_gaze_data_stimulus(self.username)

            for row in eye_gaze_data:
                if not((row[4]>=0 and row[4]<=4) and (row[5]>=0 and row[5]<=4)):
                       
                    current_horizontal_ratio_left_eye = row[0]
                    current_horizontal_ratio_right_eye = row[1]
                    current_vertical_ratio_left_eye = row[2]
                    current_vertical_ratio_right_eye = row[3]
                    gaze_time = row[6]
                        
                    left_eye_gaze_position_predicted_X = calculate_gaze_position(self.slope_horizontal_left_eye, self.intercept_horizontal_left_eye, current_horizontal_ratio_left_eye)
                    right_eye_gaze_position_predicted_X = calculate_gaze_position(self.slope_horizontal_right_eye, self.intercept_horizontal_right_eye, current_horizontal_ratio_right_eye)

                    left_eye_gaze_position_predicted_Y = calculate_gaze_position(self.slope_vertical_left_eye, self.intercept_vertical_left_eye, current_vertical_ratio_left_eye)
                    right_eye_gaze_position_predicted_Y = calculate_gaze_position(self.slope_vertical_right_eye, self.intercept_vertical_right_eye, current_vertical_ratio_right_eye)
                        
                    eyes_predicted_X = (calculate_gaze_position(self.new_slope_horizontal_left_eye, self.new_intercept_horizontal_left_eye, left_eye_gaze_position_predicted_X) + calculate_gaze_position(self.new_slope_horizontal_right_eye, self.new_intercept_horizontal_right_eye, right_eye_gaze_position_predicted_X))/2
                    eyes_predicted_Y = (calculate_gaze_position(self.new_slope_vertical_left_eye, self.new_intercept_vertical_left_eye, left_eye_gaze_position_predicted_Y) + calculate_gaze_position(self.new_slope_vertical_right_eye, self.new_intercept_vertical_right_eye, right_eye_gaze_position_predicted_Y))/2

                    write_predictions_in_file(self.username, eyes_predicted_X, eyes_predicted_Y, gaze_time)
                else:
                    print("Blink - removed!")

    def generate_gazeplot(self):
        image_path = f"{FILE_PATH}/visual_stimuli/stimulus{COUNT_STIMULUS}.jpg"
        original_image = cv2.imread(image_path)
        resized_image = cv2.resize(original_image, (self.winfo_screenwidth(), self.winfo_screenheight()))
        resized_image_rgb = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
        plt.imshow(resized_image_rgb)

        gaze_data = read_predictions_file(self.username)
        X = gaze_data['Predicted X screen position']
        Y = gaze_data['Predicted Y screen position']
        max_x = self.winfo_screenwidth()
        max_y = self.winfo_screenheight()
        timestamps_ms = gaze_data['Time in ms']
        timestamps_sec = [t / 1000 for t in timestamps_ms]
        plt.scatter(X, Y, c=timestamps_sec, cmap='viridis', marker='o')
        cbar = plt.colorbar()
        plt.xlim(0,max_x)
        plt.ylim(0,max_y)
        plt.gca().invert_yaxis()
        ax = plt.gca()
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position('top')
        
        cbar.set_label('Timestamp (seconds)')
        plt.xlabel('X coordinate')
        plt.ylabel('Y coordinate')
        plt.title("Gaze Plot")

        plt.savefig(f'{FILE_PATH}/{self.username}/gaze_plot_stimulus{COUNT_STIMULUS}.png')
        plt.clf()
    
    def generate_heatmap(self):
        eye_tracking_data = read_predictions_file(self.username)
        eye_tracking_data = eye_tracking_data.to_numpy()

        image_path = f"{FILE_PATH}/visual_stimuli/stimulus{COUNT_STIMULUS}.jpg"
        image = cv2.imread(image_path)

        canvas_width, canvas_height = (self.winfo_screenwidth(), self.winfo_screenheight()) 

        heatmap, xedges, yedges = np.histogram2d(
            eye_tracking_data[:, 0],
            eye_tracking_data[:, 1],
            bins=80
        )

        smoothed_heatmap = gaussian_filter(heatmap, sigma=2)
        normalized_heatmap = (smoothed_heatmap - np.min(smoothed_heatmap)) / (np.max(smoothed_heatmap) - np.min(smoothed_heatmap))
        resized_image = cv2.resize(image, (canvas_width, canvas_height))
        heatmap_overlay = cv2.resize(cv2.applyColorMap(np.uint8(normalized_heatmap * 255), cv2.COLORMAP_JET), (canvas_width, canvas_height))
        heatmap_overlay_image = cv2.addWeighted(resized_image, 0.7, heatmap_overlay, 0.3, 0)
        heatmap_overlay_rgb = cv2.cvtColor(heatmap_overlay_image, cv2.COLOR_BGR2RGB)

        fig = plt.figure(figsize=(canvas_width / 100, canvas_height / 100))
        plt.imshow(heatmap_overlay_rgb)
        plt.axis('off')
        plt.savefig(f'{FILE_PATH}/{self.username}/heatmap_stimulus{COUNT_STIMULUS}.png')
        plt.close(fig)

    def save_results(self):
        global COUNT_STIMULUS
        COUNT_STIMULUS = 1
        stimuli_num = 3
        for stimulus in range(stimuli_num):
            self.save_predictions_of_stimulus_gaze()
            print(f"Wait...Creating diagrams for stimulus {COUNT_STIMULUS}: ")
            self.generate_gazeplot()
            self.generate_heatmap()

            COUNT_STIMULUS += 1

class MyVideoCapture:
    def __init__(self, video_source=0):
        self.video = cv2.VideoCapture(video_source)
        
        if not self.video.isOpened():
             raise ValueError("Unable to open video source", video_source)
        
        self.width = self.video.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.video.get(cv2.CAP_PROP_FRAME_HEIGHT)

    def get_frame(self):
        
        with mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as face_mesh:
            while True:
                # if self.video.isOpened():
                ret, frame = self.video.read()
                if not ret:
                    break
                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img_h, img_w = frame.shape[:2]
                results = face_mesh.process(rgb_frame)

                if results.multi_face_landmarks:
                    mesh_points = np.array([np.multiply([p.x, p.y], [img_w, img_h]).astype(int) for p in results.multi_face_landmarks[0].landmark])
                    
                    iris_horizontal_in_mm = 11.7 #+-0.5mm
                    iris_horizontal_in_pixels = euclidean_distance(mesh_points[L_P_LEFT_IRIS][0], mesh_points[R_P_LEFT_IRIS][0])

                    focal_length = 650 #mm
                    distance_from_camera = (focal_length*iris_horizontal_in_mm)/iris_horizontal_in_pixels
                    self.person_distance_from_camera = distance_from_camera
                    
                    average_eye_height_mm = 10
                    approximated_person_to_screen_distance = 650 #mm
                    approximated_eye_height_px = (average_eye_height_mm*focal_length) / approximated_person_to_screen_distance
                         
                    (l_cx, l_cy), l_radius = cv2.minEnclosingCircle(mesh_points[LEFT_IRIS])
                    (r_cx, r_cy), r_radius = cv2.minEnclosingCircle(mesh_points[RIGHT_IRIS])   
                    self.l_cx = l_cx
                    self.l_cy = l_cy
                    self.r_cx = r_cx
                    self.r_cy = r_cy
                    center_left = np.array([l_cx, l_cy], dtype=np.int32)
                    center_right = np.array([r_cx, r_cy], dtype=np.int32)

                    right_eye_vertical = iris_vertical_position(center_right, mesh_points[RIGHT_EYE][12], mesh_points[RIGHT_EYE][4], approximated_eye_height_px)
                    left_eye_vertical = iris_vertical_position(center_left, mesh_points[LEFT_EYE][12], mesh_points[LEFT_EYE][4], approximated_eye_height_px)

                    right_eye_eyelids_distance = euclidean_distance(mesh_points[RIGHT_EYE][12], mesh_points[RIGHT_EYE][4])
                    left_eye_eyelids_distance = euclidean_distance(mesh_points[LEFT_EYE][12], mesh_points[LEFT_EYE][4])
                        
                    right_eye_horizontal = iris_horizontal_position(center_right, mesh_points[R_H_RIGHT], mesh_points[R_H_LEFT])
                    left_eye_horizontal = iris_horizontal_position(center_left, mesh_points[L_H_RIGHT], mesh_points[L_H_LEFT])

                    distance_corners = euclidean_distance(mesh_points[LEFT_EYE][8], mesh_points[RIGHT_EYE][0])
                    distance_nose_ends = euclidean_distance(mesh_points[NOSE][0], mesh_points[NOSE][1])

                    right_eye_horizontal_ratio = right_eye_horizontal/distance_corners
                    left_eye_horizontal_ratio = left_eye_horizontal/distance_corners
                    right_eye_vertical_ratio = right_eye_vertical/distance_nose_ends
                    left_eye_vertical_ratio = left_eye_vertical/distance_nose_ends

                    average_closed_eyelids_mm = 3
                    approximated_closed_eyelids_px = (average_closed_eyelids_mm*focal_length)/approximated_person_to_screen_distance

                    if (right_eye_eyelids_distance>=0 and right_eye_eyelids_distance<=approximated_closed_eyelids_px) and (left_eye_eyelids_distance>=0 and left_eye_eyelids_distance<=approximated_closed_eyelids_px):
                        print("Blinked")

                    self.right_eye_vertical_ratio = right_eye_vertical_ratio
                    self.left_eye_vertical_ratio = left_eye_vertical_ratio
                    self.eye_right_horizontal_ratio = right_eye_horizontal_ratio
                    self.eye_left_horizontal_ratio = left_eye_horizontal_ratio
                    self.left_eye_blink_check = left_eye_eyelids_distance
                    self.right_eye_blink_check = right_eye_eyelids_distance

                    cv2.circle(frame, mesh_points[RIGHT_EYE][0], 1, (255, 255, 255), -1, cv2.LINE_AA)
                    cv2.circle(frame, mesh_points[RIGHT_EYE][8], 1, (255, 255, 255), -1, cv2.LINE_AA)
                    cv2.circle(frame, mesh_points[LEFT_EYE][0], 1, (255, 255, 255), -1, cv2.LINE_AA)
                    cv2.circle(frame, mesh_points[LEFT_EYE][8], 1, (255, 255, 255), -1, cv2.LINE_AA)

                    cv2.circle(frame, mesh_points[RIGHT_EYE][12], 1, (255, 255, 255), -1, cv2.LINE_AA)
                    cv2.circle(frame, mesh_points[RIGHT_EYE][4], 1, (255, 255, 255), -1, cv2.LINE_AA)
                    cv2.circle(frame, mesh_points[LEFT_EYE][12], 1, (255, 255, 255), -1, cv2.LINE_AA)
                    cv2.circle(frame, mesh_points[LEFT_EYE][4], 1, (255, 255, 255), -1, cv2.LINE_AA)

                    cv2.circle(frame, center_left, 1, (255, 255, 255), -1, cv2.LINE_AA)
                    cv2.circle(frame, center_right, 1, (255, 255, 255), -1, cv2.LINE_AA)

                return (ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

if __name__ == "__main__":
    app = App()
    app.mainloop()   