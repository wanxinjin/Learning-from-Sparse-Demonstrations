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


class InputWaypoints(object):
    # the lab space limit [meter] in x-axis [x_min, x_max]
    space_limit_x: list
    # the lab space limit [meter] in y-axis [y_min, y_max]
    space_limit_y: list
    # the lab space limit [meter] in z-axis [z_min, z_max]
    space_limit_y: list


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


    def run(self, QuadInitialCondition: QuadStates, QuadDesiredStates: QuadStates):
        fig_top_down = plt.figure()
        ax_top_down = fig_top_down.add_subplot(1, 1, 1)
        # set axis limits
        ax_top_down.set_xlim([self.space_limit_x[0]-1.2, self.space_limit_x[1]+1.2])
        ax_top_down.set_ylim([self.space_limit_y[0]-1.2, self.space_limit_y[1]+1.2])
        ax_top_down.set_xlabel("x")
        ax_top_down.set_ylabel("y")
        ax_top_down.set_aspect('equal')
        # set obstacle legend
        red_patch = patches.Patch(color='red', label='Obstacles')
        # plot lab sapce boundary
        ax_top_down.plot(self.space_limit_x, [self.space_limit_y[0],self.space_limit_y[0]], color='black')
        ax_top_down.plot(self.space_limit_x, [self.space_limit_y[1],self.space_limit_y[1]], color='black')
        ax_top_down.plot([self.space_limit_x[0],self.space_limit_x[0]], self.space_limit_y, color='black')
        ax_top_down.plot([self.space_limit_x[1],self.space_limit_x[1]], self.space_limit_y, color='black')

        # plot start and goal
        ax_top_down.scatter(QuadInitialCondition.position[0], QuadInitialCondition.position[1], label='start', color='green')
        ax_top_down.scatter(QuadDesiredStates.position[0], QuadDesiredStates.position[1], label='goal', color='violet')
        
        # set legends
        ax_top_down.add_artist(plt.legend(handles=[red_patch]))
        plt.legend(loc="upper left")
        plt.title('Select waypoints in top-down view', fontweight ='bold')
        print("Click your waypoints in XOY plane, order from -x to +x, -y to +y.")
        print("Middle click to terminate ginput.")
        waypoints_top_down = plt.ginput(0,0)
        
        # for i in range(0, len(waypoints_top_down)):
        #     print("Waypoint XOY" + str(i+1) + ": x = " + str(waypoints_top_down[i][0]) + \
        #         ", y = " + str(waypoints_top_down[i][1]))
        print("Waypoints selection in top-down view completed. Don't close Figure 1 yet.")



        # a = [10, 40, 70]
        # for idx_obs in range(0,3):
        #     rect = patches.Rectangle(a[idx_obs], width=5, height=10, linewidth=1, edgecolor='r', facecolor='r')
        #     # Add the patch to the Axes
        #     ax.add_patch(rect)




        fig_right_left = plt.figure()
        ax_right_left = fig_right_left.add_subplot(1, 1, 1)
        # set axis limits
        ax_right_left.set_xlim([self.space_limit_x[0]-1.2, self.space_limit_x[1]+1.2])
        ax_right_left.set_ylim([self.space_limit_z[0]-0.2, self.space_limit_z[1]+1.5])
        ax_right_left.set_xlabel("x")
        ax_right_left.set_ylabel("z")
        ax_right_left.set_aspect('equal')
        # set obstacle legend
        red_patch = patches.Patch(color='red', label='Obstacles')
        # plot lab sapce boundary
        ax_right_left.plot(self.space_limit_x, [self.space_limit_z[0],self.space_limit_z[0]], color='black')
        ax_right_left.plot(self.space_limit_x, [self.space_limit_z[1],self.space_limit_z[1]], color='black')
        ax_right_left.plot([self.space_limit_x[0],self.space_limit_x[0]], self.space_limit_z, color='black')
        ax_right_left.plot([self.space_limit_x[1],self.space_limit_x[1]], self.space_limit_z, color='black')
        # plot the ground
        ax_right_left.plot([self.space_limit_x[0]-1.2,self.space_limit_x[1]+1.2], [0,0], linestyle='dashed', color='black', label='ground')

        # plot start and goal
        ax_right_left.scatter(QuadInitialCondition.position[0], QuadInitialCondition.position[2], label='start', color='green')
        ax_right_left.scatter(QuadDesiredStates.position[0], QuadDesiredStates.position[2], label='goal', color='violet')
        # set legends
        ax_right_left.add_artist(plt.legend(handles=[red_patch]))
        plt.legend(loc="upper left")

        for i in range(0, len(waypoints_top_down)):
            ax_right_left.scatter(waypoints_top_down[i][0], 0, c='C0')

        plt.title('Select waypoints in right-left view (only read z values)', fontweight ='bold')
        print("Click your waypoints in XOZ plane, order from -x to +x, -y to +y.")
        print("The blue points are the previous waypoints you select in XOY plane.")
        print("Middle click to terminate ginput.")
        waypoints_right_left = plt.ginput(0,0)
        print("Waypoints selection in right-left view completed. Now close Figure 1 & 2.")


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


        fig_3d = plt.figure()
        ax_3d = fig_3d.add_subplot(111, projection='3d')

        # plot waypoints
        ax_3d.plot3D(waypoints_3d_plot[0], waypoints_3d_plot[1], waypoints_3d_plot[2], 'blue', label='waypoints')
        for i in range(0, len(waypoints_output)):
            ax_3d.scatter(waypoints_output[i][0], waypoints_output[i][1], waypoints_output[i][2], c='C0')

        # plot start and goal
        ax_3d.scatter(QuadInitialCondition.position[0], QuadInitialCondition.position[1], QuadInitialCondition.position[2], label='start', color='green')
        ax_3d.scatter(QuadDesiredStates.position[0], QuadDesiredStates.position[1], QuadDesiredStates.position[2], label='goal', color='violet')
        ax_3d.plot([QuadInitialCondition.position[0], waypoints_output[0][0]], [QuadInitialCondition.position[1], waypoints_output[0][1]], \
            [QuadInitialCondition.position[2], waypoints_output[0][2]], 'blue')
        ax_3d.plot([QuadDesiredStates.position[0], waypoints_output[-1][0]], [QuadDesiredStates.position[1], waypoints_output[-1][1]], \
            [QuadDesiredStates.position[2], waypoints_output[-1][2]], 'blue')

        ax_3d.set_xlim([self.space_limit_x[0]-1.2, self.space_limit_x[1]+1.2])
        ax_3d.set_ylim([self.space_limit_y[0]-1.2, self.space_limit_y[1]+1.2])
        ax_3d.set_zlim([self.space_limit_z[0]-0.2, self.space_limit_z[1]+1.5])
        ax_3d.set_xlabel("x")
        ax_3d.set_ylabel("y")
        ax_3d.set_zlabel("z")
        plt.legend(loc="upper left")
        plt.title('Waypoints in 3D', fontweight ='bold')


        plt.show()

        return waypoints_output


