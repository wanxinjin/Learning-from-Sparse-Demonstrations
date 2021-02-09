import os
import sys
sys.path.append(os.getcwd()+'/lib')
import time
import json
from PyQt5 import QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D


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


    def __init__(self, config_data, parent=None):
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

        # set the title
        self.setWindowTitle("Obtain Human Input as Sparse Demonstrations")

        # setting the geometry of window
        self.setGeometry(0, 0, 1280, 960)

        # creating a label widget to show the current mouse position
        self.label_current_mouse_position = QtWidgets.QLabel(self)
        self.label_current_mouse_position.move(600, 930)
        self.label_current_mouse_position.resize(200, 40)


        self.canvas = FigureCanvas(Figure())

        vertical_layout = QtWidgets.QVBoxLayout(self)
        vertical_layout.addWidget(self.canvas)




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
        self.canvas.draw()


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
        self.canvas.draw()



        # settings for 3D plotting
        self.canvas.axes_3D = self.canvas.figure.add_subplot(1,2,1)




        # number of waypoints in top-down XOY plane
        self.num_pts_XOY = 0
        # number of waypoints in right-left XOZ plane
        self.num_pts_XOZ = 0

        self.canvas.mpl_connect("button_press_event", self.on_press)
        self.canvas.mpl_connect("motion_notify_event", self.on_move)

        # 2D list, each sub list [x, y] is the position of a waypoint
        self.waypoints_XOY = []
        # 2D list, each sub list [x, z] is the position of a waypoint
        self.waypoints_XOZ = []



    def on_press(self, event):
        print("event.xdata", event.xdata)
        print("event.ydata", event.ydata)
        # print("event.inaxes", event.inaxes)
        if event.inaxes == self.canvas.axes_3D:
            print("This point is in left 3D figure")

        elif event.inaxes == self.canvas.axes_XOY:
            print("This point is in right top XOY figure")
            self.waypoints_XOY.append([round(event.xdata,3), round(event.ydata,3)])
            self.canvas.axes_XOY.scatter(round(event.xdata,3), round(event.ydata,3), c='C0')
            self.canvas.axes_XOY.text(round(event.xdata,3), round(event.ydata,3), str(self.num_pts_XOY+1))
            self.num_pts_XOY += 1
            self.canvas.draw()

        elif event.inaxes == self.canvas.axes_XOZ:
            print("This point is in right bottom XOZ figure")
            self.waypoints_XOZ.append([round(event.xdata,3), round(event.ydata,3)])
            self.canvas.axes_XOZ.scatter(round(event.xdata,3), round(event.ydata,3), c='C0')
            self.canvas.axes_XOZ.text(round(event.xdata,3), round(event.ydata,3), str(self.num_pts_XOZ+1))
            self.num_pts_XOZ += 1
            self.canvas.draw()

        else:
            raise Exception("Unknown axes!")




    def on_move(self, event):
        # print("move")
        # print("event.xdata", event.xdata)
        # print("event.ydata", event.ydata)
        # print("event.inaxes", event.inaxes)
        # print("x", event.x)
        # print("y", event.y)
        try:
            position_str = "(" + '%.3f' % event.xdata + ", " + '%.3f' % event.ydata + ")"
            self.label_current_mouse_position.setText(position_str)
        except:
            self.label_current_mouse_position.setText("Mouse is out of the window.")


if __name__ == "__main__":

    # a json configuration file
    config_file_name = "config.json"
    # Read the configuration from the json file
    json_file = open(config_file_name)
    config_data = json.load(json_file)

    app = QtWidgets.QApplication(sys.argv)
    w = MplWidget(config_data)
    w.show()
    sys.exit(app.exec_())
