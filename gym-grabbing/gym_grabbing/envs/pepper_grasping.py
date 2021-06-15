import pybullet as p
import qibullet as q
import gym
import os
import random
from pathlib import Path
from gym_grabbing.envs.robot_grasping import RobotGrasping
import numpy as np



class PepperGrasping(RobotGrasping):

    def __init__(
        self,
        object_position=[0, -0.05, -0.15],
        **kwargs
    ):
                 
        self.pepper = q.PepperVirtual()
        self.joints = ['HipRoll', 'LShoulderPitch', 'LShoulderRoll', 'LElbowYaw', 'LElbowRoll', 'LWristYaw', 'LHand']
        
        def pepper():
            self.pepper.loadRobot(translation=[0.2, -0.35, -1], quaternion=[0,0,1,1], physicsClientId=self.physicsClientId)
            robot_id = self.pepper.getRobotModel()
            #self.pepper.move(1,0,0) # this crashes on Mac OS
            for joint_name, joint_value in zip(self.pepper.P_STAND.getPostureJointNames(), self.pepper.P_STAND.getPostureJointValues()):
                self.p.resetJointState(robot_id, self.pepper.joint_dict[joint_name].getIndex(), joint_value)
            self.pepper.goToPosture("Stand", 1.0)
            return robot_id

        super().__init__(
            robot=pepper,
            object_position=object_position,
            table_height=0.8,
            end_effector_id=34,#self.pepper.joint_dict['LHand'].getIndex(),
            joint_ids=[4, 26, 27, 29, 30, 32, 34],#,[self.pepper.joint_dict[joint].getIndex() for joint in self.joints],
            n_control_gripper=1, # the left hand is controller with one input
            center_workspace=0,
            radius=1,
            contact_ids=list(range(36, 50)),
            allowed_collision_pair=[[37,40], [38,41], [38, 46], [38, 47], [41,49], [64, 68], [65,68], [67,70], [68,71], [71,73]],
            **kwargs
        )

        # self.joints = ['HipRoll', 'LShoulderPitch', 'LShoulderRoll', 'LElbowYaw', 'LElbowRoll', 'LWristYaw', 'LFinger21', 'LFinger22', 'LFinger23', 'LFinger11', 'LFinger12', 'LFinger13', 'LFinger41', 'LFinger42', 'LFinger43', 'LFinger31', 'LFinger32', 'LFinger33', 'LThumb1', 'LThumb2']

        
    def get_object(self, obj=None):
        # create object to grab
        if obj == 'cuboid':
            return {"shape":'cuboid', "x":0.046, "y":0.1, "z":0.216, "lateralFriction":0.5}
        elif obj == 'cube':
            return {"shape": 'cube', "unit":0.055, "lateralFriction":0.5}
        elif obj == 'sphere':
            return {"shape":'sphere', "radius":0.055, "lateralFriction":0.8}
        elif obj == 'cylinder':
            return {"shape":'cylinder', "radius":0.032, "z":0.15}
        else:
            return obj
            
    def step(self, action):
        if self.mode == 'joint positions':
            # we want one action per joint, all fingers are controlled with LHand
            assert(len(action) == self.n_actions)
            self.info['closed gripper'] = action[-1]<0
            commands = (np.array(action)+1)/2*(self.upperLimits-self.lowerLimits) + self.lowerLimits
            commands = commands.tolist()

        # apply the commands
        self.pepper.setAngles(joint_names=self.joints, joint_values=commands, percentage_speed=1)
        
        return super().step() # do not set the motors as we already dit it
