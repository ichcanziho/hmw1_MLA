import cv2 as cv
import pandas as pd
from math import atan,pi,degrees
from pathlib import Path
import xml.etree.ElementTree as et
from tkinter import *

nearest_neighbors = 8

class utilTools():

    def __init__(self, image, path):
        self.image=image
        self.path = path
        self.bgr = cv.imread(self.image)
        self.height, self.width = self.bgr.shape[:2]
        self.center_height, self.center_widht = int(self.height / 2), int(self.width / 2)
        self.black_color,self.red_color,self.blue_color,self.green_color = (0, 0, 0),(0, 0, 255),(255,0,0),(0,255,0)
        self.origin = (self.center_widht, self.center_height)
        objects = [obj.name for obj in Path(self.path).iterdir()]
        self.extention = '.' + str(objects[0].split('.')[1])
        self.documents = [name.split('.')[0] for name in objects]
        #self.draw_bgr()
        self.position=0
        self.init_gui()
        self.totalElements=0

    def get_documents(self):
        return self.documents

    def make_data_frame(self, name):
        directory = self.path+"/" + name + self.extention
        xtree = et.parse(directory)
        xroot = xtree.getroot()
        df_cols = ["fingerprint", "minutia", "x", "y", "angle", "score_change"]
        rows = []
        for node in xroot:
            s_version = node.attrib.get("version") if node is not None else None
            s_score = node.attrib.get("score") if node is not None else None
            s_score = round(float(s_score),6)
            s_x = node.find('MissingMinutia').attrib.get('x') if node is not None else None
            s_x = int(s_x)
            s_y = node.find('MissingMinutia').attrib.get('y') if node is not None else None
            s_y = int(s_y)
            s_angle = node.find('MissingMinutia').attrib.get('angle') if node is not None else None
            s_angle = float(s_angle)
            #print(s_version, s_x, s_y, s_angle, s_score)
            rows.append({"fingerprint": name, "minutia": s_version,
                         "x": s_x, "y": s_y, 'angle': s_angle, 'score_change':s_score})

        self.out_df = pd.DataFrame(rows, columns=df_cols)

        self.totalElements = len(self.out_df)
        #print(self.out_df)

    def euclidean_distance(self,x1,y1,x2,y2):
        return ((x1-x2)**2+(y1-y2)**2)**0.5

    def get_distances_ann_abs(self,nn,levels):
        self.number_of_distances = nn
        x_vals,y_vals,minutia_vals = list(self.out_df.x),list(self.out_df.y),list(self.out_df.minutia)
        df_cols = ['minutia']
        df_cols += ['d'+str(i) for i in range(nn)]

        i = 0
        matrix=[]
        matrix_minutias=[]
        for x,y,minutia in zip(x_vals,y_vals,minutia_vals):

            x_ref = x_vals.copy()
            y_ref = y_vals.copy()
            minutia_ref = minutia_vals.copy()
            x_ref.pop(i)
            y_ref.pop(i)
            minutia_ref.pop(i)
            distances = []
            distances_with_minutias=[]

            for x_ne, y_ne,m_ne in zip(x_ref, y_ref,minutia_ref):
                dis = self.euclidean_distance(x, y, x_ne, y_ne)
                distances.append(dis)
                distances_with_minutias.append((dis,m_ne))
            distances.sort()
            distances_with_minutias.sort()
            minutia_sorted = [ m[1] for m in distances_with_minutias ]

            row = distances[:nn]
            row_minutias = minutia_sorted[:nn]
            matrix.append(row)
            matrix_minutias.append(row_minutias)
            i += 1
        ann_matrix = []
        for row in matrix:
            row_ann = []
            for level in levels:
                counter=0
                for element in row:
                    if element < level:
                        counter+=1
                row_ann.append(counter)
            ann_matrix.append(row_ann)

        # transpose of a matrix for Nearest Distances
        matrix= zip(*matrix)
        for i,row in enumerate(matrix):
            self.out_df['dA'+str(i)] = list(row)

        matrix_minutias = zip(*matrix_minutias)
        for i,row in enumerate(matrix_minutias):
            self.out_df['mA'+str(i)] = list(row)

        ann_matrix = zip(*ann_matrix)
        for level,row in zip(levels,ann_matrix):
            self.out_df['rA'+str(level)] = list(row)
        
        #print(self.out_df)

    def get_beta_data_frame(self,nearest_minutia=0):
        if nearest_minutia <= self.number_of_distances:
            col_names = self.out_df.columns
            filter_names = []
            for col in col_names:
                if 'mA' in col:
                    filter_names.append(col)

            angle = list(self.out_df.angle)

            minutia_matrix = [ list(self.out_df[data]) for data in filter_names ]

            near_angle_matrix=[]
            for minutia in minutia_matrix:
                row_new_angle=[]
                for value in minutia:
                    position = int(value[1:])
                    row_new_angle.append(angle[position])
                near_angle_matrix.append(row_new_angle)

            for i,row in enumerate( near_angle_matrix):
                row_beta=[]
                for a,b in zip(angle,row):
                    beta = self.get_beta_angle(a,b)
                    row_beta.append(beta)
                self.out_df['B'+str(i+1)] = row_beta

            #print(self.out_df)

    def get_beta_angle(self,a,b):
        if b > a:
            return b-a
        else:
            return b-a+360

    def split_list_into_matrix(self,list,split=2):

        matrix=[]
        while list != []:
            matrix.append(list[:split])
            list = list[split:]

        return matrix

    def get_alpha_data_frame(self):
        col_names = self.out_df.columns
        filter_names = []
        for col in col_names:
            if 'mA' in col:
                filter_names.append(col)

        angle_i_list = list(self.out_df.angle)
        x_i_list = list(self.out_df.x)
        y_i_list = list(self.out_df.y)
        minutia_matrix = [list(self.out_df[data]) for data in filter_names]

        angle_i_list = angle_i_list
        x_i_list = x_i_list
        y_i_list = y_i_list
        row_number = 0
        minutia_matrix_t = list(zip(*minutia_matrix))
        #print(len(minutia_matrix[0]))
        alpha_matrix=[]
        for number,m in enumerate(minutia_matrix_t):
            tuplas = self.split_list_into_matrix(list(m))

            alpha_row=[]
            for pair in tuplas:

                pj2 = int(pair[0][1:])
                pj3 = int(pair[1][1:])
                x_i = x_i_list[number]
                y_i = y_i_list[number]

                angle_j2, x_j2, y_j2 = angle_i_list[pj2], x_i_list[pj2], y_i_list[pj2]
                angle_j3, x_j3, y_j3 = angle_i_list[pj3], x_i_list[pj3], y_i_list[pj3]

                # ---------------p2-----------------------
                aux = self.get_alpha_angle((x_i - x_j2), (y_i - y_j2))
                alpha_3 = self.get_beta_angle(aux, angle_j2)
                #------
                # preguntar como se obtiene a4 es p3-p2 o p2-p3
                aux = self.get_alpha_angle((x_j3 - x_j2), (y_j3 - y_j2))
                alpha_4 = self.get_beta_angle(aux, angle_j2)
                # ---------------p3-----------------------
                aux = self.get_alpha_angle((x_i - x_j3), (y_i - y_j3))
                alpha_5 = self.get_beta_angle(aux, angle_j3)
                #------
                aux = self.get_alpha_angle((x_j2 - x_j3), (y_j2 - y_j3))
                alpha_6 = self.get_beta_angle(aux, angle_j3)

                alpha_row += [alpha_3,alpha_4,alpha_5,alpha_6]

            alpha_matrix.append(alpha_row)

        alpha_matrix = list(zip(*alpha_matrix))

        for i in range(len(alpha_matrix)):
            self.out_df['A'+str(i+1)] = alpha_matrix[i]

        #print(self.out_df)

    def clean_data_frame(self):
        col_names = list(self.out_df.columns)
        filter_names = []
        for col in col_names:
            if 'mA' in col:
                filter_names.append(col)

        col_names.remove('x')
        col_names.remove('y')
        col_names.remove('angle')
        col_names.remove('score_change')

        for m in filter_names:
            col_names.remove(m)

        col_names.append('score_change')

        self.final_output = self.out_df[col_names]

        #print("successfully!!")
        #self.final_output.to_csv('output.csv',index=False)

    def save_atrr(self):
        print("successfully!!")
        self.final_output.to_csv('output.csv',index=False)

    def get_alpha_angle(self,delta_x,delta_y):
        if delta_x > 0 and delta_y >= 0:
            return degrees( atan(delta_y/delta_x))
        elif delta_x > 0 and delta_y < 0:
            return degrees( atan(delta_y / delta_x) + 2*pi)
        elif delta_x < 0:
            return degrees( atan(delta_y / delta_x) + pi)
        elif delta_x == 0 and delta_y > 0:
            return degrees( pi/2)
        elif delta_x == 0 and delta_y < 0:
            return degrees( 3*pi/2)

    def draw_bgr(self):
        cv.line(self.bgr, (0, self.center_height), (2 * self.center_widht, self.center_height), self.black_color, 1,
                cv.LINE_AA)
        cv.line(self.bgr, (self.center_widht, 0), (self.center_widht, 2 * self.center_height), self.black_color, 1,
                cv.LINE_AA)
        cv.imshow('Draw', self.bgr)

    def convert_y(self, y):
        return self.height - y

    def draw_all_points(self,point=0):
        self.bgr = cv.imread(self.image)
        self.draw_bgr()
        for x,y in zip(list(self.out_df.x),list(self.out_df.y)):
            self.draw_point(x,y,self.red_color)

        current_x =self.out_df.x[point]
        current_y = self.out_df.y[point]
        self.draw_point(current_x, current_y, self.blue_color)
        self.draw_curves(current_x,current_y,self.green_color)

    def draw_point(self, x, y,color):
        y = self.convert_y(y)
        cv.circle(self.bgr, (x, y), 3, color, -1, cv.LINE_AA)
        cv.imshow('Draw', self.bgr)

    def draw_curves(self, x, y,color):
        y = self.convert_y(y)
        cv.circle(self.bgr, (x, y), 30, color, 1, cv.LINE_AA)
        cv.circle(self.bgr, (x, y), 60, color, 1, cv.LINE_AA)
        cv.circle(self.bgr, (x, y), 90, color, 1, cv.LINE_AA)
        cv.imshow('Draw', self.bgr)

    def init_gui(self):
        self.root = Tk()
        self.button_prev = Button(self.root, text='<<', padx=30, command=self.prev)
        self.button_prev.grid(row=0, column=0)
        self.button_next = Button(self.root, text='>>', padx=30, command=self.next)
        self.button_next.grid(row=0, column=1)

    def next(self):

        self.button_prev.config(state="normal")
        if self.position == self.totalElements-2:
            self.button_next.config(state="disabled")
            self.position += 1
        else:
            self.position += 1

        self.draw_all_points(self.position)

    def prev(self):
        self.button_next.config(state="normal")
        if self.position == 1:
            self.button_prev.config(state="disabled")
            self.position -= 1
        if self.position > 1:
            self.position -= 1
        self.draw_all_points(self.position)

    def show(self):
        self.draw_all_points()
        self.root.mainloop()
        cv.waitKey(0)
        cv.destroyAllWindows()

    def get_data_frame(self):
        self.clean_data_frame()
        return self.final_output

util = utilTools('white.png', 'data')

documents = util.get_documents()


dataFrame = pd.read_csv('empty_data.csv')

data_frame_list=[]
bad_ones=[]
for document in documents:
    print(document)
    util.make_data_frame(document)
    print(util.out_df)
    if len(util.out_df) > nearest_neighbors:
        util.get_distances_ann_abs(nearest_neighbors,[30,60,90])
        util.get_beta_data_frame()
        util.get_alpha_data_frame()
        data = util.get_data_frame()
        #print(data)
        print("---")
        data_frame_list.append(data)
    else:
        bad_ones.append(document)

for bad in bad_ones:
    print(bad)
print(len(bad_ones))


for frame in data_frame_list:
    dataFrame = pd.concat([dataFrame, frame])

dataFrame.to_csv('test.csv',index=False)

import arff
arff.dump('All_outputs.arff', dataFrame.values , relation='minutias', names=dataFrame.columns)