#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Baxter RSDK Joint Trajectory Action Client
https://github.com/RethinkRobotics/baxter_examples/blob/master/scripts/joint_trajectory_client.py

You must have generated trajectories with generateTrajectoryBaxter.py. Turn on Baxter, import trajectories to the Baxter PC:

session 0:
cd ros_ws && source baxter.sh
rosrun baxter_tools tuck_arms.py -u # use -t before turning off Baxter
rosrun baxter_interface joint_trajectory_action_server.py -mode position

session 1:
cd ros_ws && source baxter.sh
rosrun baxter_examples grasp_trajectory_clients.py -l left -t /home/me/ros_ws/trajectories/sphereNoQual
"""
import argparse
import sys
#from pathlib import Path
import glob
import pickle
import numpy as np
import time
import json

from copy import copy

import rospy

import actionlib

from control_msgs.msg import (
    FollowJointTrajectoryAction,
    FollowJointTrajectoryGoal,
)
from trajectory_msgs.msg import (
    JointTrajectoryPoint,
)

import baxter_interface

from baxter_interface import CHECK_VERSION


class Trajectory(object):
    def __init__(self, limb):
        ns = 'robot/limb/' + limb + '/'
        self._client = actionlib.SimpleActionClient(
            ns + "follow_joint_trajectory",
            FollowJointTrajectoryAction,
        )
        self._goal = FollowJointTrajectoryGoal()
        self._goal_time_tolerance = rospy.Time(0.1)
        self._goal.goal_time_tolerance = self._goal_time_tolerance
        server_up = self._client.wait_for_server(timeout=rospy.Duration(10.0))
        if not server_up:
            rospy.logerr("Timed out waiting for Joint Trajectory"
                         " Action Server to connect. Start the action server"
                         " before running example.")
            rospy.signal_shutdown("Timed out waiting for Action Server")
            sys.exit(1)
        self.clear(limb)
        self._gripper = baxter_interface.Gripper(limb)
        self._gripper.calibrate()

    def add_point(self, positions, time):
        point = JointTrajectoryPoint()
        point.positions = copy(positions)
        point.time_from_start = rospy.Duration(time)
        self._goal.trajectory.points.append(point)

    def start(self):
        self._goal.trajectory.header.stamp = rospy.Time.now()
        self._client.send_goal(self._goal)
        self._gripper.open()

    def stop(self):
        self._client.cancel_goal()

    def wait(self, timeout=15.0):
        self._client.wait_for_result(timeout=rospy.Duration(timeout))

    def result(self):
        return self._client.get_result()

    def clear(self, limb):
        self._goal = FollowJointTrajectoryGoal()
        self._goal.goal_time_tolerance = self._goal_time_tolerance
        self._goal.trajectory.joint_names = [limb + '_' + joint for joint in \
            ['s0', 's1', 'e0', 'e1', 'w0', 'w1', 'w2']]

    def add_gripper(self, time, closed=True):
        self._gripper_time = time
        self._gripper_close = closed

    def wait_gripper(self, timeout=15.): # must call add_gripper before
        assert self._gripper_time<timeout, "timeout must be greater than gripper_time"
        self._client.wait_for_result(timeout=rospy.Duration(self._gripper_time))
        self._gripper.close() if self._gripper_close else self._gripper.open()
        self._client.wait_for_result(timeout=rospy.Duration(max(timeout-self._gripper_time, 0)))

# put the object at 73cm from baxter (light grey)
def main(timeTrajectory=10., repeat=None):
    """RSDK Joint Trajectory Example: Simple Action Client

    Creates a client of the Joint Trajectory Action Server
    to send commands of standard action type,
    control_msgs/FollowJointTrajectoryAction.

    Make sure to start the joint_trajectory_action_server.py
    first. Then run this example on a specified limb to
    command a short series of trajectory points for the arm
    to follow.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-t" "--trajectories", help="The directory path containing trajectories, *.npy files, it can be a sub folder")
    parser.add_argument("-l" "--limb", help="The limb to use", choices=["left", "right"], default="left")
    args = parser.parse_args(rospy.myargv()[1:])

    limb = args.limb

    print("Initializing node... ")
    rospy.init_node("rsdk_joint_trajectory_client_%s" % (limb,))
    print("Getting robot state... ")
    rs = baxter_interface.RobotEnable(CHECK_VERSION)
    print("Enabling robot... ")
    rs.enable()
    print("Running. Ctrl-c to quit")

    folder = args.trajectories
    trajectories = glob.glob(folder+"/*.npy")

    i, repeat_count = 0, -1
    add_start = 1 # slows the movement at the beginning
    timeInter = 7.
    success = dict()
    n_success, n_eval = 0, 0
    order = []
    while True:

        if repeat is None:
            x = raw_input("x: exit, n: next, g: go, current={}, next={} ".format(ind_path, None if i+1==len(trajectories) else  trajectories[i+1]))
            if x == "x":
                return
            elif x == "n":
                i += 1
                repeat_count = 0
                if i==len(trajectories): break
                else: continue
            elif x == "g":
                repeat_count += 1
                pass # run the same individual
            else:
                time.sleep(0.5) # ctrl-c
                continue
        else:
            repeat_count = (repeat_count+1) % repeat
            if repeat_count % repeat == 0 and n_eval>0:
                i += 1
            if i==len(trajectories): break

        ind_path = trajectories[i]
        with open(ind_path, "r") as f:
            ind = np.load(f)
            timeStep = timeTrajectory / len(ind)
            closeTime = np.load(f)*timeStep + timeInter + add_start
        ind_path = ind_path.split("/")[-1]
        if ind_path not in success:
            success[ind_path] = 0.
            order.append(ind_path)


        traj = Trajectory(limb)
        rospy.on_shutdown(traj.stop)
        # Command Current Joint Positions first
        limb_interface = baxter_interface.limb.Limb(limb)
        current_angles = np.array([limb_interface.joint_angle(joint) for joint in limb_interface.joint_names()])

        t = timeInter

        traj.add_point(current_angles, 0.0)

        for k,j in enumerate(ind):
            traj.add_point(j, t)
            t += timeStep
            if k==0:
                t += add_start

        traj.add_gripper(closeTime)
        traj.start()
        traj.wait_gripper(100)
        traj.stop()
        while True: # indicate whether it is a success or failure
            x = raw_input("s: success, f: fail")
            if x == "s":
                success[ind_path] = True if repeat is None else (success[ind_path]*repeat_count+1)/(repeat_count+1)
                n_success += 1
                break
            elif x == "f":
                success[ind_path] = False if repeat is None else (success[ind_path]*repeat_count+0)/(repeat_count+1)
                break
            else:
                time.sleep(0.5)
        n_eval += 1

        with open(folder+'/success.json', 'wb') as f:
            json.dump(success, f, 0) # save the success dictionary
        print(success)

    success['order'] = order
    with open(folder+'/success.json', 'wb') as f:
        json.dump(success, f, 0) # save the success dictionary
    print("n_success", n_success, "n_eval", n_eval)
    print("Exiting - Joint Trajectory Action Test Complete")

if __name__ == "__main__":
    print("start")
    main(repeat=3)
