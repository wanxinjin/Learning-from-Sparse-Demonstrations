import os
import sys
sys.path.append(os.getcwd()+'/lib')
import time
import json
import numpy as np
from PyQt5 import QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D
from dataclasses import dataclass, field
from DemoSparse import DemoSparse
from QuadStates import QuadStates
from ObsInfo import ObsInfo


class MplWidget(QtWidgets.QWidget):
    # the lab space limit [meter] in x-axis [x_min, x_max]
    space_limit_x: list
    # the lab space limit [meter] in y-axis [y_min, y_max]
    space_limit_y: list
    # the lab space limit [meter] in z-axis [z_min, z_max]
    space_limit_y: list
    # the average speed for the quadrotor [m/s]
    quad_average_speed: float
    # number of waypoints in top-down XOY plane
    num_pts_XOY: int
    # number of waypoints in right-left XOZ plane
    num_pts_XOZ: int
    # a dataclass QuadStates for initial condition
    InitialCondition: QuadStates
    # a dataclass QuadStates for desired states
    DesiredStates: QuadStates
    # number of obstacles
    num_obs: int
    # a list of dataclasses ObsInfo
    ObsList: list
    # a 2D list for waypoints, [[x0,y0,z0], [x1,y1,z1], [x2,y2,z2]], including start and goal
    waypoints_output: list
    # a 2D list for plotting waypoints in 3D sapce, [[x0,x1,x2], [y0,y1,y2], [z0,z1,z2]], including start and goal
    waypoints_3d_plot: list
    # a 1D list to store the time-stamp for each waypoint, including the start and goal
    time_list_all: list


    def __init__(self, ObsList: list, config_data, parent=None):
        """
        constructor

        config_data:
            config_file_name = "config.json"
            json_file = open(config_file_name)
            config_data = json.load(json_file)
        """

        super(MplWidget, self).__init__(parent)

        # the lab space limit [meter] in x-axis [x_min, x_max]
        self.space_limit_x = config_data["LAB_SPACE_LIMIT"]["LIMIT_X"]
        # the lab space limit [meter] in y-axis [y_min, y_max]
        self.space_limit_y = config_data["LAB_SPACE_LIMIT"]["LIMIT_Y"]
        # the lab space limit [meter] in z-axis [z_min, z_max]
        self.space_limit_z = config_data["LAB_SPACE_LIMIT"]["LIMIT_Z"]
        # the average speed for the quadrotor [m/s]
        self.quad_average_speed = float(config_data["QUAD_AVERAGE_SPEED"])
        # a list of dataclasses ObsInfo
        self.ObsList = ObsList

        # number of waypoints in top-down XOY plane
        self.num_pts_XOY = 0
        # number of waypoints in right-left XOZ plane
        self.num_pts_XOZ = 0
        # 2D list, each sub list [x, y] is the position of a waypoint
        self.waypoints_XOY = []
        # 2D list, each sub list [x, z] is the position of a waypoint
        self.waypoints_XOZ = []

        # set the title
        self.setWindowTitle("Obtain Human Input as Sparse Demonstrations")
        # setting the geometry of window
        self.setGeometry(0, 0, 1280, 960)
        # creating a label widget to show the current mouse position
        self.label_current_mouse_position = QtWidgets.QLabel(self)
        self.label_current_mouse_position.move(600, 930)
        self.label_current_mouse_position.resize(200, 40)
        # create a canvas
        self.canvas = FigureCanvas(Figure())
        vertical_layout = QtWidgets.QVBoxLayout(self)
        vertical_layout.addWidget(self.canvas)

        # creating a label to show instructions
        self.label_instruction = QtWidgets.QLabel(self)
        # self.label_instruction.move(600, 930)
        self.label_instruction.resize(200, 200)
        instruction_str = "1:Select waypoints in XOY Plane first. 2:Then select waypoints in XOZ Plane. 3:Click Set Start. 4:Click Set Goal. 5:Click Plot 3D."
        self.label_instruction.setText(instruction_str)

        #---------------------------------------------------------------#
        # settings for top-down XOY figure
        self.canvas.axes_XOY = self.canvas.figure.add_subplot(2,2,2)
        # set axis limits
        self.canvas.axes_XOY.set_xlim([self.space_limit_x[0]-1.2, self.space_limit_x[1]+1.2])
        self.canvas.axes_XOY.set_ylim([self.space_limit_y[0]-1.2, self.space_limit_y[1]+1.2])
        self.canvas.axes_XOY.set_xlabel("x")
        self.canvas.axes_XOY.set_ylabel("y")
        self.canvas.axes_XOY.set_aspect('equal')
        # plot lab sapce boundary
        self.canvas.axes_XOY.plot(self.space_limit_x, [self.space_limit_y[0],self.space_limit_y[0]], color='black')
        self.canvas.axes_XOY.plot(self.space_limit_x, [self.space_limit_y[1],self.space_limit_y[1]], color='black')
        self.canvas.axes_XOY.plot([self.space_limit_x[0],self.space_limit_x[0]], self.space_limit_y, color='black')
        self.canvas.axes_XOY.plot([self.space_limit_x[1],self.space_limit_x[1]], self.space_limit_y, color='black')
        # set title
        self.canvas.axes_XOY.set_title("Click waypoints in top-down view (XOY Plane)")
        
        # plot obstacles
        self.num_obs = len(self.ObsList)
        if self.num_obs > 0.5:
            for i in range(0, self.num_obs):
                # in XOY plane, width means length in ObsInfo, height means width in ObsInfo
                rect = patches.Rectangle([ self.ObsList[i].center[0]-0.5*self.ObsList[i].length, self.ObsList[i].center[1]-0.5*self.ObsList[i].width ], \
                    width=self.ObsList[i].length, height=self.ObsList[i].width, linewidth=1, edgecolor='r', facecolor='r')
                # Add the patch to the Axes
                self.canvas.axes_XOY.add_patch(rect)
        # set obstacle legend
        red_patch = patches.Patch(color='red', label='Obstacles')
        # set legends
        self.canvas.axes_XOY.add_artist(self.canvas.axes_XOY.legend(handles=[red_patch]))
        self.canvas.draw()

        #---------------------------------------------------------------#
        # settings for right-left XOZ figure
        self.canvas.axes_XOZ = self.canvas.figure.add_subplot(2,2,4)
        # set axis limits
        self.canvas.axes_XOZ.set_xlim([self.space_limit_x[0]-1.2, self.space_limit_x[1]+1.2])
        self.canvas.axes_XOZ.set_ylim([self.space_limit_z[0]-0.2, self.space_limit_z[1]+1.5])
        self.canvas.axes_XOZ.set_xlabel("x")
        self.canvas.axes_XOZ.set_ylabel("z")
        self.canvas.axes_XOZ.set_aspect('equal')
        # plot lab sapce boundary
        self.canvas.axes_XOZ.plot(self.space_limit_x, [self.space_limit_z[0],self.space_limit_z[0]], color='black')
        self.canvas.axes_XOZ.plot(self.space_limit_x, [self.space_limit_z[1],self.space_limit_z[1]], color='black')
        self.canvas.axes_XOZ.plot([self.space_limit_x[0],self.space_limit_x[0]], self.space_limit_z, color='black')
        self.canvas.axes_XOZ.plot([self.space_limit_x[1],self.space_limit_x[1]], self.space_limit_z, color='black')
        # plot the ground
        self.canvas.axes_XOZ.plot([self.space_limit_x[0]-1.2,self.space_limit_x[1]+1.2], [0,0], linestyle='dashed', color='black', label='ground')
        # set title
        self.canvas.axes_XOZ.set_title("Click waypoints in right-left view (XOZ Plane), only read z values")

        # plot obstacles
        if self.num_obs > 0.5:
            for i in range(0, self.num_obs):
                # in XOZ plane, width means length in ObsInfo, height means height in ObsInfo
                rect = patches.Rectangle([ self.ObsList[i].center[0]-0.5*self.ObsList[i].length, self.ObsList[i].center[2]-0.5*self.ObsList[i].height ], \
                    width=self.ObsList[i].length, height=self.ObsList[i].height, linewidth=1, edgecolor='r', facecolor='r')
                # Add the patch to the Axes
                self.canvas.axes_XOZ.add_patch(rect)
        # set obstacle legend
        red_patch = patches.Patch(color='red', label='Obstacles')
        # set legends
        self.canvas.axes_XOZ.add_artist(self.canvas.axes_XOZ.legend(handles=[red_patch]))
        self.canvas.draw()

        #---------------------------------------------------------------#
        # settings for 3D plotting
        self.canvas.axes_3D = self.canvas.figure.add_subplot(121, projection='3d')
        self.set_axes_equal_all()

        self.canvas.axes_3D.set_xlabel("x")
        self.canvas.axes_3D.set_ylabel("y")
        self.canvas.axes_3D.set_zlabel("z")
        self.canvas.axes_3D.set_title("Waypoints in 3D space")
        # set obstacle legend
        red_patch = patches.Patch(color='red', label='Obstacles')
        # set legends
        self.canvas.axes_3D.add_artist(self.canvas.axes_3D.legend(handles=[red_patch],loc="upper left"))
        # plot obstacles
        self.plot_linear_cube()
        
        #---------------------------------------------------------------#
        self.canvas.mpl_connect("button_press_event", self.on_press)
        self.canvas.mpl_connect("motion_notify_event", self.on_move)

        # define a button
        self.button_set_start = QtWidgets.QPushButton(self)
        self.button_set_start.setStyleSheet(
            "QPushButton { color: black; }"
            "QPushButton:pressed { color: black; }")
        self.button_set_start.setText("Set Start")
        self.button_set_start.move(1000,60)
        self.button_set_start.clicked.connect(self.button_set_start_clicked)

        # define a button
        self.button_set_goal = QtWidgets.QPushButton(self)
        self.button_set_goal.setStyleSheet(
            "QPushButton { color: black; }"
            "QPushButton:pressed { color: black; }")
        self.button_set_goal.setText("Set Goal")
        self.button_set_goal.move(1000,90)
        self.button_set_goal.clicked.connect(self.button_set_goal_clicked)

        # # define a button
        self.button_plot_3d = QtWidgets.QPushButton(self)
        self.button_plot_3d.setStyleSheet(
            "QPushButton { color: black; }"
            "QPushButton:pressed { color: black; }")
        self.button_plot_3d.setText("Plot 3D")
        self.button_plot_3d.move(300,60)
        self.button_plot_3d.clicked.connect(self.button_plot_3d_clicked)


    def on_press(self, event):
        # print("event.xdata", event.xdata)
        # print("event.ydata", event.ydata)
        # print("event.inaxes", event.inaxes)
        if event.inaxes == self.canvas.axes_3D:
            # print("This point is in left 3D figure")
            pass

        elif event.inaxes == self.canvas.axes_XOY:
            # print("This point is in right top XOY figure")
            self.waypoints_XOY.append([round(event.xdata,3), round(event.ydata,3)])
            # the first waypoint is the start
            if self.num_pts_XOY < 1:
                self.canvas.axes_XOY.scatter(round(event.xdata,3), round(event.ydata,3), color='green')
                self.canvas.axes_XOY.text(round(event.xdata,3), round(event.ydata,3), "start")
            else:
                self.canvas.axes_XOY.scatter(round(event.xdata,3), round(event.ydata,3), c='C0')
                self.canvas.axes_XOY.text(round(event.xdata,3), round(event.ydata,3), str(self.num_pts_XOY))
            self.num_pts_XOY += 1
            self.canvas.draw()

        elif event.inaxes == self.canvas.axes_XOZ:
            # print("This point is in right bottom XOZ figure")
            self.waypoints_XOZ.append([round(event.xdata,3), round(event.ydata,3)])
            # the first waypoint is the start
            if self.num_pts_XOZ < 1:
                self.canvas.axes_XOZ.scatter(round(event.xdata,3), round(event.ydata,3), color='green')
                self.canvas.axes_XOZ.text(round(event.xdata,3), round(event.ydata,3), "start")
            else:
                self.canvas.axes_XOZ.scatter(round(event.xdata,3), round(event.ydata,3), c='C0')
                self.canvas.axes_XOZ.text(round(event.xdata,3), round(event.ydata,3), str(self.num_pts_XOZ))
            self.num_pts_XOZ += 1
            self.canvas.draw()

        else:
            #raise Exception("Unknown axes!")
            pass


    def on_move(self, event):
        # print("move")
        # print("event.xdata", event.xdata)
        # print("event.ydata", event.ydata)
        # print("event.inaxes", event.inaxes)
        # print("x", event.x)
        # print("y", event.y)
        try:
            if event.inaxes == self.canvas.axes_XOY:
                position_str = "(x=" + '%.3f' % event.xdata + ", y=" + '%.3f' % event.ydata + ")"
                self.label_current_mouse_position.setText(position_str)
            elif event.inaxes == self.canvas.axes_XOZ:
                position_str = "(x=" + '%.3f' % event.xdata + ", z=" + '%.3f' % event.ydata + ")"
                self.label_current_mouse_position.setText(position_str)
            else:
                position_str = "(" + '%.3f' % event.xdata + ", " + '%.3f' % event.ydata + ")"
                self.label_current_mouse_position.setText(position_str)
        except:
            self.label_current_mouse_position.setText("Mouse is out of the window.")


    def button_set_start_clicked(self):
        print("Set Start!")
        # define the initial condition
        self.InitialCondition = QuadStates()
        self.InitialCondition.position = [self.waypoints_XOY[0][0], self.waypoints_XOY[0][1], self.waypoints_XOZ[0][1]]

        print(self.InitialCondition.position)


    def button_set_goal_clicked(self):
        print("Set Goal!")
        # define the goal position
        self.DesiredStates = QuadStates()
        self.DesiredStates.position = [self.waypoints_XOY[-1][0], self.waypoints_XOY[-1][1], self.waypoints_XOZ[-1][1]]
        # plot goal
        self.canvas.axes_XOY.scatter(self.waypoints_XOY[-1][0], self.waypoints_XOY[-1][1], color='violet')
        self.canvas.axes_XOY.text(self.waypoints_XOY[-1][0], self.waypoints_XOY[-1][1], str(self.num_pts_XOY-1) + ":goal")
        # plot goal
        self.canvas.axes_XOZ.scatter(self.waypoints_XOZ[-1][0], self.waypoints_XOZ[-1][1], color='violet')
        self.canvas.axes_XOZ.text(self.waypoints_XOZ[-1][0], self.waypoints_XOZ[-1][1], str(self.num_pts_XOZ-1) + ":goal")

        self.canvas.draw()
        print(self.DesiredStates.position)


    def button_plot_3d_clicked(self):
        print("Plot in 3D space!")
        # waypoints output, including start and goal
        self.waypoints_output = []
        self.waypoints_3d_plot = [[], [], []] # for plotting, including start and goal
        for i in range(0, len(self.waypoints_XOY)):
            self.waypoints_output.append([ self.waypoints_XOY[i][0], self.waypoints_XOY[i][1], \
                self.waypoints_XOZ[i][1] ])
            print("Waypoints [x, y, z] [meter]: ", end=" ")
            print(self.waypoints_output[i])
            # for plotting
            self.waypoints_3d_plot[0].append(self.waypoints_output[i][0])
            self.waypoints_3d_plot[1].append(self.waypoints_output[i][1])
            self.waypoints_3d_plot[2].append(self.waypoints_output[i][2])

        # plot waypoints, not including start and goal
        self.canvas.axes_3D.plot3D(self.waypoints_3d_plot[0], self.waypoints_3d_plot[1], self.waypoints_3d_plot[2], 'blue', label='waypoints')
        for i in range(1, len(self.waypoints_XOY)-1):
            self.canvas.axes_3D.scatter(self.waypoints_output[i][0], self.waypoints_output[i][1], self.waypoints_output[i][2], c='C0')
        # plot start and goal
        self.canvas.axes_3D.scatter(self.InitialCondition.position[0], self.InitialCondition.position[1], self.InitialCondition.position[2], label='start', color='green')
        self.canvas.axes_3D.scatter(self.DesiredStates.position[0], self.DesiredStates.position[1], self.DesiredStates.position[2], label='goal', color='violet')
        # set legends
        self.canvas.axes_3D.legend()
        self.canvas.draw()

        # generate time-stamps for all waypoints
        self.generate_time()
        print("time_list_all")
        print(self.time_list_all)


    def generate_time(self):
        """
        Based on start, goal, and waypoints, generate the time (tau) for each point.
        """

        # self.waypoints_output including start and goal
        # a 1D list to store the time-stamp for each waypoint, including the start and goal
        self.time_list_all = [0.0]
        distance_total = 0.0
        for i in range(1, len(self.waypoints_output)):
            distance_current = np.linalg.norm( np.array(self.waypoints_output[i]) - np.array(self.waypoints_output[i-1]) )
            time_segment = round(distance_current/self.quad_average_speed, 2)
            self.time_list_all.append(time_segment+self.time_list_all[i-1])


    def plot_linear_cube(self, color='red'):
        """
        Plot obstacles in 3D space.
        """

        # plot obstacles
        num_obs = len(self.ObsList)
        if num_obs > 0.5:
            for i in range(0, num_obs):
                x = self.ObsList[i].center[0] - 0.5 * self.ObsList[i].length
                y = self.ObsList[i].center[1] - 0.5 * self.ObsList[i].width
                z = self.ObsList[i].center[2] - 0.5 * self.ObsList[i].height

                dx = self.ObsList[i].length
                dy = self.ObsList[i].width
                dz = self.ObsList[i].height

                xx = [x, x, x+dx, x+dx, x]
                yy = [y, y+dy, y+dy, y, y]
                kwargs = {'alpha': 1, 'color': color}
                self.canvas.axes_3D.plot3D(xx, yy, [z]*5, **kwargs)
                self.canvas.axes_3D.plot3D(xx, yy, [z+dz]*5, **kwargs)
                self.canvas.axes_3D.plot3D([x, x], [y, y], [z, z+dz], **kwargs)
                self.canvas.axes_3D.plot3D([x, x], [y+dy, y+dy], [z, z+dz], **kwargs)
                self.canvas.axes_3D.plot3D([x+dx, x+dx], [y+dy, y+dy], [z, z+dz], **kwargs)
                self.canvas.axes_3D.plot3D([x+dx, x+dx], [y, y], [z, z+dz], **kwargs)
                self.canvas.draw()


    def set_axes_equal_all(self):
        '''
        Make axes of 3D plot have equal scale so that spheres appear as spheres,
        cubes as cubes, etc..  This is one possible solution to Matplotlib's
        ax.set_aspect('equal') and ax.axis('equal') not working for 3D.
        Reference: https://stackoverflow.com/questions/13685386/matplotlib-equal-unit-length-with-equal-aspect-ratio-z-axis-is-not-equal-to
        '''

        x_limits = [self.space_limit_x[0], self.space_limit_x[1]]
        y_limits = [self.space_limit_y[0], self.space_limit_y[1]]
        z_limits = [self.space_limit_z[0], self.space_limit_z[1]]

        x_range = abs(x_limits[1] - x_limits[0])
        x_middle = np.mean(x_limits)
        y_range = abs(y_limits[1] - y_limits[0])
        y_middle = np.mean(y_limits)
        z_range = abs(z_limits[1] - z_limits[0])
        z_middle = np.mean(z_limits)

        # The plot bounding box is a sphere in the sense of the infinity
        # norm, hence I call half the max range the plot radius.
        plot_radius = 0.5*max([x_range, y_range, z_range])

        self.canvas.axes_3D.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
        self.canvas.axes_3D.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
        self.canvas.axes_3D.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])
        