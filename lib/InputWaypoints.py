#!/usr/bin/env python3
import os
import sys
import time
sys.path.append(os.getcwd()+'/lib')
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D
from dataclasses import dataclass, field
from DemoSparse import DemoSparse
from QuadStates import QuadStates
from ObsInfo import ObsInfo


class InputWaypoints(object):
    # the lab space limit [meter] in x-axis [x_min, x_max]
    space_limit_x: list
    # the lab space limit [meter] in y-axis [y_min, y_max]
    space_limit_y: list
    # the lab space limit [meter] in z-axis [z_min, z_max]
    space_limit_y: list
    # the average speed for the quadrotor [m/s]
    quad_average_speed: float


    def __init__(self, config_data):
        """
        constructor

        config_data:
            config_file_name = "config.json"
            json_file = open(config_file_name)
            config_data = json.load(json_file)
        """

        # the lab space limit [meter] in x-axis [x_min, x_max]
        self.space_limit_x = config_data["LAB_SPACE_LIMIT"]["LIMIT_X"]
        # the lab space limit [meter] in y-axis [y_min, y_max]
        self.space_limit_y = config_data["LAB_SPACE_LIMIT"]["LIMIT_Y"]
        # the lab space limit [meter] in z-axis [z_min, z_max]
        self.space_limit_z = config_data["LAB_SPACE_LIMIT"]["LIMIT_Z"]
        # the average speed for the quadrotor [m/s]
        self.quad_average_speed = float(config_data["QUAD_AVERAGE_SPEED"])


    def run(self, QuadInitialCondition: QuadStates, QuadDesiredStates: QuadStates, ObsList: list):
        """
        Run this method to obtain human inputs as sparse demonstrations.
        """

        # for the top-down view
        self.fig_top_down = plt.figure()
        self.ax_top_down = self.fig_top_down.add_subplot(1, 1, 1)
        # set axis limits
        self.ax_top_down.set_xlim([self.space_limit_x[0]-1.2, self.space_limit_x[1]+1.2])
        self.ax_top_down.set_ylim([self.space_limit_y[0]-1.2, self.space_limit_y[1]+1.2])
        self.ax_top_down.set_xlabel("x")
        self.ax_top_down.set_ylabel("y")
        self.ax_top_down.set_aspect('equal')
        # set obstacle legend
        red_patch = patches.Patch(color='red', label='Obstacles')
        # plot lab sapce boundary
        self.ax_top_down.plot(self.space_limit_x, [self.space_limit_y[0],self.space_limit_y[0]], color='black')
        self.ax_top_down.plot(self.space_limit_x, [self.space_limit_y[1],self.space_limit_y[1]], color='black')
        self.ax_top_down.plot([self.space_limit_x[0],self.space_limit_x[0]], self.space_limit_y, color='black')
        self.ax_top_down.plot([self.space_limit_x[1],self.space_limit_x[1]], self.space_limit_y, color='black')

        # plot start and goal
        self.ax_top_down.scatter(QuadInitialCondition.position[0], QuadInitialCondition.position[1], label='start', color='green')
        self.ax_top_down.scatter(QuadDesiredStates.position[0], QuadDesiredStates.position[1], label='goal', color='violet')
        
        # plot obstacles
        num_obs = len(ObsList)
        if num_obs > 0.5:
            for i in range(0, num_obs):
                # in XOY plane, width means length in ObsInfo, height means width in ObsInfo
                rect = patches.Rectangle([ ObsList[i].center[0]-0.5*ObsList[i].length, ObsList[i].center[1]-0.5*ObsList[i].width ], \
                    width=ObsList[i].length, height=ObsList[i].width, linewidth=1, edgecolor='r', facecolor='r')
                # Add the patch to the Axes
                self.ax_top_down.add_patch(rect)

        # set legends
        self.ax_top_down.add_artist(plt.legend(handles=[red_patch]))
        plt.legend(loc="upper left")
        plt.title('Select waypoints in top-down view. Middle click to terminate.', fontweight ='bold')
        print("Click your waypoints in XOY plane, order from -x to +x, -y to +y.")
        print("Middle click to terminate ginput.")
        waypoints_top_down = plt.ginput(0,0)
        print("Waypoints selection in top-down view completed. Don't close Figure 1 yet.")

        # plot waypoints
        for i in range(0, len(waypoints_top_down)):
            # print("Waypoint XOY" + str(i+1) + ": x = " + str(waypoints_top_down[i][0]) + \
            #     ", y = " + str(waypoints_top_down[i][1]))
            self.ax_top_down.scatter(waypoints_top_down[i][0], waypoints_top_down[i][1], c='C0')
        plt.draw()


        # for the right-left view
        self.fig_right_left = plt.figure()
        self.ax_right_left = self.fig_right_left.add_subplot(1, 1, 1)
        # set axis limits
        self.ax_right_left.set_xlim([self.space_limit_x[0]-1.2, self.space_limit_x[1]+1.2])
        self.ax_right_left.set_ylim([self.space_limit_z[0]-0.2, self.space_limit_z[1]+1.5])
        self.ax_right_left.set_xlabel("x")
        self.ax_right_left.set_ylabel("z")
        self.ax_right_left.set_aspect('equal')
        # set obstacle legend
        red_patch = patches.Patch(color='red', label='Obstacles')
        # plot lab sapce boundary
        self.ax_right_left.plot(self.space_limit_x, [self.space_limit_z[0],self.space_limit_z[0]], color='black')
        self.ax_right_left.plot(self.space_limit_x, [self.space_limit_z[1],self.space_limit_z[1]], color='black')
        self.ax_right_left.plot([self.space_limit_x[0],self.space_limit_x[0]], self.space_limit_z, color='black')
        self.ax_right_left.plot([self.space_limit_x[1],self.space_limit_x[1]], self.space_limit_z, color='black')
        # plot the ground
        self.ax_right_left.plot([self.space_limit_x[0]-1.2,self.space_limit_x[1]+1.2], [0,0], linestyle='dashed', color='black', label='ground')

        # plot start and goal
        self.ax_right_left.scatter(QuadInitialCondition.position[0], QuadInitialCondition.position[2], label='start', color='green')
        self.ax_right_left.scatter(QuadDesiredStates.position[0], QuadDesiredStates.position[2], label='goal', color='violet')
        
        # plot obstacles
        num_obs = len(ObsList)
        if num_obs > 0.5:
            for i in range(0, num_obs):
                # in XOZ plane, width means length in ObsInfo, height means height in ObsInfo
                rect = patches.Rectangle([ ObsList[i].center[0]-0.5*ObsList[i].length, ObsList[i].center[2]-0.5*ObsList[i].height ], \
                    width=ObsList[i].length, height=ObsList[i].height, linewidth=1, edgecolor='r', facecolor='r')
                # Add the patch to the Axes
                self.ax_right_left.add_patch(rect)
        
        # set legends
        self.ax_right_left.add_artist(plt.legend(handles=[red_patch]))
        plt.legend(loc="upper left")

        # plot waypoints
        for i in range(0, len(waypoints_top_down)):
            self.ax_right_left.scatter(waypoints_top_down[i][0], 0, c='C0')

        plt.title('Select waypoints in right-left view (only read z-axis). Middle click to terminate.', fontweight ='bold')
        print("Click your waypoints in XOZ plane, order from -x to +x, -y to +y.")
        print("The blue points are the previous waypoints you select in XOY plane.")
        print("Middle click to terminate ginput.")
        waypoints_right_left = plt.ginput(0,0)
        print("Waypoints selection in right-left view completed. Now close Figure 1 & 2.")

        # plot waypoints
        for i in range(0, len(waypoints_right_left)):
            self.ax_right_left.scatter(waypoints_right_left[i][0], waypoints_right_left[i][1], c='C0')
        plt.draw()


        # waypoints output
        waypoints_output = []
        waypoints_3d_plot = [[], [], []] # for plotting
        for i in range(0, len(waypoints_right_left)):
            waypoints_output.append([ round(waypoints_top_down[i][0],3), round(waypoints_top_down[i][1],3), \
                round(waypoints_right_left[i][1],3) ])
            print("Waypoints [x, y, z] [meter]: ", end=" ")
            print(waypoints_output[i])
            # for plotting
            waypoints_3d_plot[0].append(waypoints_output[i][0])
            waypoints_3d_plot[1].append(waypoints_output[i][1])
            waypoints_3d_plot[2].append(waypoints_output[i][2])


        # for the 3D plot
        self.fig_3d = plt.figure()
        self.ax_3d = self.fig_3d.add_subplot(111, projection='3d')

        # plot waypoints
        self.ax_3d.plot3D(waypoints_3d_plot[0], waypoints_3d_plot[1], waypoints_3d_plot[2], 'blue', label='waypoints')
        for i in range(0, len(waypoints_output)):
            self.ax_3d.scatter(waypoints_output[i][0], waypoints_output[i][1], waypoints_output[i][2], c='C0')

        # plot start and goal
        self.ax_3d.scatter(QuadInitialCondition.position[0], QuadInitialCondition.position[1], QuadInitialCondition.position[2], label='start', color='green')
        self.ax_3d.scatter(QuadDesiredStates.position[0], QuadDesiredStates.position[1], QuadDesiredStates.position[2], label='goal', color='violet')
        self.ax_3d.plot([QuadInitialCondition.position[0], waypoints_output[0][0]], [QuadInitialCondition.position[1], waypoints_output[0][1]], \
            [QuadInitialCondition.position[2], waypoints_output[0][2]], 'blue')
        self.ax_3d.plot([QuadDesiredStates.position[0], waypoints_output[-1][0]], [QuadDesiredStates.position[1], waypoints_output[-1][1]], \
            [QuadDesiredStates.position[2], waypoints_output[-1][2]], 'blue')

        # plot obstacles
        self.plot_linear_cube(ObsList, color='red')
        # set obstacle legend
        red_patch = patches.Patch(color='red', label='Obstacles')
        self.ax_3d.add_artist(plt.legend(handles=[red_patch]))

        self.ax_3d.set_xlim([self.space_limit_x[0]-1.2, self.space_limit_x[1]+1.2])
        self.ax_3d.set_ylim([self.space_limit_y[0]-1.2, self.space_limit_y[1]+1.2])
        self.ax_3d.set_zlim([self.space_limit_z[0]-0.2, self.space_limit_z[1]+1.5])
        self.ax_3d.set_xlabel("x")
        self.ax_3d.set_ylabel("y")
        self.ax_3d.set_zlabel("z")
        plt.legend(loc="upper left")
        plt.title('Waypoints in 3D. Close this window to continue.', fontweight ='bold')
        plt.show()

        # a 1D list to store the time-stamp for each waypoint, including the start and goal
        time_list_all = self.generate_time(waypoints_output, QuadInitialCondition, QuadDesiredStates)

        # define the sparse demonstration
        SparseInput = DemoSparse()
        SparseInput.waypoints = waypoints_output
        SparseInput.time_list = time_list_all[1 : -1]
        SparseInput.time_horizon = time_list_all[-1]

        return SparseInput


    def generate_time(self, waypoints_output: list, QuadInitialCondition: QuadStates, QuadDesiredStates: QuadStates):
        """
        Based on start, goal, and waypoints, generate the time (tau) for each point.
        """

        # waypoints including start and goal
        waypoints_all = [QuadInitialCondition.position] + waypoints_output + [QuadDesiredStates.position]
        # a 1D list to store the time-stamp for each waypoint, including the start and goal
        time_list_all = [0.0]
        distance_total = 0.0

        for i in range(1, len(waypoints_all)):
            distance_current = np.linalg.norm( np.array(waypoints_all[i]) - np.array(waypoints_all[i-1]) )
            time_segment = round(distance_current/self.quad_average_speed, 2)
            time_list_all.append(time_segment+time_list_all[i-1])

        return time_list_all


    def plot_linear_cube(self, ObsList: list, color='red'):
        """
        Plot obstacles in 3D space.
        """

        # plot obstacles
        num_obs = len(ObsList)
        if num_obs > 0.5:
            for i in range(0, num_obs):
                x = ObsList[i].center[0] - 0.5 * ObsList[i].length
                y = ObsList[i].center[1] - 0.5 * ObsList[i].width
                z = ObsList[i].center[2] - 0.5 * ObsList[i].height

                dx = ObsList[i].length
                dy = ObsList[i].width
                dz = ObsList[i].height

                xx = [x, x, x+dx, x+dx, x]
                yy = [y, y+dy, y+dy, y, y]
                kwargs = {'alpha': 1, 'color': color}
                self.ax_3d.plot3D(xx, yy, [z]*5, **kwargs)
                self.ax_3d.plot3D(xx, yy, [z+dz]*5, **kwargs)
                self.ax_3d.plot3D([x, x], [y, y], [z, z+dz], **kwargs)
                self.ax_3d.plot3D([x, x], [y+dy, y+dy], [z, z+dz], **kwargs)
                self.ax_3d.plot3D([x+dx, x+dx], [y+dy, y+dy], [z, z+dz], **kwargs)
                self.ax_3d.plot3D([x+dx, x+dx], [y, y], [z, z+dz], **kwargs)

