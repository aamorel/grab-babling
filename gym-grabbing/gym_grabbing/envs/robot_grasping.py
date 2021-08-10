from gym import Env, GoalEnv, spaces
from time import sleep
import pybullet_data
from pybullet_utils.bullet_client import BulletClient
from gym_grabbing.envs.utils import PDControllerStable, MLP#, suppress_output
from pathlib import Path
import weakref
import functools
import numpy as np
from collections import OrderedDict
import time
import torch as th


from typing import Dict, List, Tuple, Sequence, Callable, Any, Union, Optional
try:
    import numpy.typing as npt
    ArrayLike = npt.ArrayLike
except:
    ArrayLike = Any


class RobotGrasping(GoalEnv):
    metadata = {'render.modes': ['human', 'rgb_array', 'rgba_array']}

    def __init_subclass__(cls, *args, **kwargs): # callback trick
        super().__init_subclass__(*args, **kwargs)


    def __init__(self,
        robot: Callable[[], int], # function to load the robot, returning the id
        display: bool = False, # enable/disable display
        obj: str = 'cube', # object to load, available objects are those defined in get_object and in the folder gym-baxter-grabbing/gym_baxter_grabbing/envs/objects/
        camera: Dict[str, Union[ArrayLike, int, float]] = {'target':[0,0,0], 'distance':1, 'yaw':180, 'pitch':-40}, # initial positon of the camera
        gripper_display: bool = False, # display the ripper frame
        steps_to_roll: int = 1, # nb time to call p.stepSimulation within one step
        random_var: Optional[float] = 0, # the variance of the positon noise of the object
        delta_pos: ArrayLike = [0, 0], # position displacement of the object, it won't be reseted with reset()
        object_position: ArrayLike = [0., 0., 0.], # initial position of the object during loading, then it will fall
        object_xyzw: ArrayLike = [0,0,0,1], # initial orientation of the object during loading
        joint_positions: Optional[ArrayLike] = None, # initial joint positions
        # True: each time reset() is called, the object has a random initial position
        # False: the object has a random initial position but reset() won't randomize the position
        # None: there is no random position applied at any stage
        reset_random_initial_state: Optional[bool] = None, # if not None, it overwrites object_position and object_xyzw
        initial_state: Optional[Dict[str, ArrayLike]] = None, # set the initial state, it must be the output of get_state()
        table_height: Optional[float] = 0.7, # the height of the table
        mode: str = 'joint positions', # the control mode, either 'joint positions', 'joint velocities', 'joint torques', 'inverse kinematic'
        end_effector_id: int = -1, # link id of the end effector
        joint_ids: Optional[ArrayLike] = None, # array of int, ids of joints to control, controlable joints of the gripper must be at the end
        n_control_gripper: int = 1, # number of controllable joints belonging to the gripper
        center_workspace: Union[int, ArrayLike] = -1, # position of the center of the sphere, supposing the workspace is a sphere (robotic arm) and the robot is not moving, if int, the position of the robot link is used
        radius: float = 1, # radius of the workspace
        contact_ids: ArrayLike = [], # link id (int) of the robot gripper that can have a grasping contact
        disable_collision_pair: ArrayLike = [], # 2D array (-1,2), list of pair of link id (int) to disable collision with setCollisionFilterPair
        allowed_collision_pair: ArrayLike = [], # 2D array (-1,2), list of pair of link id (int) allowed in autocollision
        change_dynamics: Dict[int, Dict[str, Any]] = {}, # the key is the robot link id, the value is the args passed to p.changeDynamics
        #early_stopping: bool = False, # stop prematurely the episode (done=True) if a joint (excluding gripper joints) reaches a position limit in torque mode, this will disable changeDynamics
        reach = False, # the robot must reach a fictitious target placed in the reachable space, the table si not spawned
        gravity = True,
        goal = False,
        npmp_decoder=None, # neural probabilistic motor primitives decoder
    ):
        assert mode in {'joint positions', 'joint velocities', 'joint torques', 'pd stable','inverse kinematics', 'inverse dynamics', 'impedance position', 'impedance velocity', 'impedance acceleration'}, "mode must be either joint positions, joint velocities, joint torques, pd stable, inverse kinematics, inverse dynamics, impedance position, impedance velocity, impedance acceleration"
        weakref.finalize(self, self.close) # cleanup
        #with suppress_output():
        import pybullet as p
        self.p = BulletClient(connection_mode=p.GUI if display else p.DIRECT)

        self.robot = robot
        self.obj = obj.strip()
        self.table_height = table_height
        self.object_position = object_position
        self.object_xyzw = object_xyzw
        self.joint_positions = joint_positions
        self.joint_ids = joint_ids
        self.reset_random_initial_state = reset_random_initial_state
        self.initial_state = initial_state
        self.display = display
        self.gripper_display = gripper_display
        self.steps_to_roll = steps_to_roll
        self.random_var = random_var
        self.delta_pos = delta_pos

        self.physicsClientId = self.p._client
        self.end_effector_id = end_effector_id
        self.n_control_gripper = n_control_gripper
        self.mode = mode
        self.radius = radius
        self.center_workspace = center_workspace
        self.contact_ids = contact_ids
        self.disable_collision_pair = disable_collision_pair
        self.allowed_collision_pair = [set(c) for c in allowed_collision_pair]
        self.change_dynamics = change_dynamics
        self.reach = reach
        self.gravity = gravity
        self.goal = goal
        self.rng = np.random.default_rng()
        self.pd_controller = PDControllerStable(self.p)
        self.time_step = self.p.getPhysicsEngineParameters()["fixedTimeStep"]
        self.kps, self.kds = 300, 500 # pd_controller gains, these will become an array later
        self.cumreward = 0
        self.npmp_decoder = None if npmp_decoder is None else MLP.load(npmp_decoder).requires_grad_(requires_grad=False)

        self.metadata['video.frames_per_second'] = 240 / steps_to_roll
        self.camera = dict(
            width=camera.get('width', 1024),
            height=camera.get('height', 1024),
            target=camera.get('target', (0,0,0)),
            distance=camera.get('distance', 1),
            yaw=camera.get('yaw', 180),
            pitch=camera.get('pitch', 0),
        )
        self.camera.update(dict(
            # https://towardsdatascience.com/simulate-images-for-ml-in-pybullet-the-quick-easy-way-859035b2c9dd
            # http://ksimek.github.io/2013/08/13/intrinsic/
            viewMatrix=self.p.computeViewMatrixFromYawPitchRoll(
                cameraTargetPosition=camera['target'],
                distance=self.camera['distance'],
                yaw=self.camera['yaw'],
                pitch=self.camera['pitch'],
                roll=camera.get('roll', 0),
                upAxisIndex=camera.get('upAxisIndex', 2),
            ),
            #self.p.computeViewMatrix(cameraEyePosition=[0,1,0.5], cameraTargetPosition=[-0.2,0,0.3], cameraUpVector=[0,0,1]),
            projectionMatrix=self.p.computeProjectionMatrixFOV(
                fov=camera.get('fov', 90),
                aspect=self.camera['width']/self.camera['height'],
                nearVal=camera.get('nearVal', 0.1),
                farVal=camera.get('farVal', 10),
            ),# if len({'fov', 'nearVal', 'farVal'} & set(camera.keys()))>0 else self.p.getDebugVisualizerCamera()[3],
            renderer=self.p.ER_BULLET_HARDWARE_OPENGL if self.display else self.p.ER_TINY_RENDERER
        ))

        self.load_all()

        high = [
            1,1,1,                              # end effector position (robot frame)
            1,1,1,1,1,1,                        # end effector orientation (robot frame)
            np.inf,np.inf,np.inf,               # end effector linear velocity
            *[1 for i in self.joint_ids],       # joint positions
            # joint velocities can be bounded or unbouded
            *(np.ones_like(self.joint_ids) if self.mode in {'joint velocities'} else [np.inf for i in self.joint_ids]),
            *[1 for i in self.joint_ids],       # joint torque sensors Mz
            1,1,1,1,1,1,                        # object orientation, 6 coefficients of the rotation matrix
            np.inf,np.inf,np.inf,               # object linear velocity
            np.inf,np.inf,np.inf,               # object angular velocity
            1,1,1,                              # object position (robot frame)
        ]
        high = np.array(high, dtype=np.float32)
        if self.reach and self.goal: # HER
            self.observation_space = spaces.Dict(
                {
                    "achieved_goal": spaces.Box(-high[:3], high[:3], dtype='float32'),
                    "observation": spaces.Box(-high[3:-3], high[3:-3], dtype='float32'),
                    "desired_goal": spaces.Box(-high[-3:], high[-3:], dtype='float32'),
                }
            )
        else:
            self.observation_space = spaces.Box(-high, high, dtype='float32')

        self.robot_space = spaces.Box(-high[:-15], high[:-15], dtype='float32') # robot state only
        if self.npmp_decoder is not None:
            self.n_actions = self.npmp_decoder.observation_space.shape[0]-self.robot_space.shape[0]
        action_high = 1 # if self.npmp_decoder is None else 2
        self.action_space = spaces.Box(-action_high, action_high, shape=(self.n_actions,), dtype='float32')


        self.info = {
            'closed gripper': False,
            'contact object table': (),
            'contact robot table': (),
            'joint reaction forces': np.zeros(self.n_joints),
            'applied joint motor torques': np.zeros(self.n_joints),
            'joint positions': np.zeros(self.n_joints),
            'joint velocities': np.zeros(self.n_joints),
            'end effector position': np.zeros(3),
            'end effector xyzw': np.zeros(4),
            'end effector linear velocity': np.zeros(3),
            'end effector angular velocity': np.zeros(3),
        }
        self.last_action = np.zeros(self.n_actions)
        oldstep = self.step
        self.gripper = 1 # open
        def newstep(action=None): # decorate step
            assert action is None or (len(action) == self.n_actions and np.isfinite(action).all()), f"action is not valid"
            if self.npmp_decoder is not None:
                action_ = self.npmp_decoder(th.as_tensor(np.hstack((self.info['robot state'], action)), dtype=th.float)[None]).squeeze().detach().numpy()
                assert np.isfinite(action_).all(), f"decoder action is not valid"
            else:
                action_ = None if action is None else np.clip(action, -action_high, action_high)
            #self.a = action_
            self.gripper = action_[-1]
            self.info['closed gripper'] = action_[-1]<0
            out = oldstep(action_)
            self.last_action[:] = action
            return out
        self.step = newstep

        if self.reset_random_initial_state is False: # set a random position that won't change
            self.new_random_initial_state()


    def load_all(self):
        self.p.resetSimulation()
        self.p.setAdditionalSearchPath(pybullet_data.getDataPath())

        self.p.setPhysicsEngineParameter(deterministicOverlappingPairs=1)

        self.plane_id = self.p.loadURDF("plane.urdf", [0, 0, -1], useFixedBase=True) # load plane with an offset

        if self.gravity: self.p.setGravity(0., 0., -9.81) # set gravity

        self.robot_id = self.robot()
        self.reset_robot()
        self.joint_ids = np.array([i for i in range(self.p.getNumJoints(self.robot_id)) if self.p.getJointInfo(self.robot_id, i)[3]>-1] if self.joint_ids is None else self.joint_ids, dtype=int)
        self.n_joints = len(self.joint_ids)
        self.n_actions = 8 if self.mode=='inverse kinematics' else self.n_joints - self.n_control_gripper + 1
        self.kps, self.kds = np.ones(self.n_joints)*self.kps, np.ones(self.n_joints)*self.kds

        self.center_workspace_cartesian = np.array(self.p.getLinkState(self.robot_id, self.center_workspace)[0] if isinstance(self.center_workspace, int) else self.center_workspace)
        self.center_workspace_robot_frame = self.p.multiplyTransforms(*self.p.invertTransform(*self.p.getBasePositionAndOrientation(self.robot_id)), self.center_workspace_cartesian, [0,0,0,1]) # the pose of center_workspace in the robot frame

        for contact_point in self.disable_collision_pair:
            self.p.setCollisionFilterPair(self.robot_id, self.robot_id, contact_point[0], contact_point[1], enableCollision=0)

        self.lowerLimits, self.upperLimits, self.maxForce, self.maxVelocity = np.zeros(self.n_joints), np.zeros(self.n_joints), np.zeros(self.n_joints), np.zeros(self.n_joints)
        for i, id in enumerate(self.joint_ids):
            self.lowerLimits[i], self.upperLimits[i], self.maxForce[i], self.maxVelocity[i] = self.p.getJointInfo(self.robot_id, id)[8:12]
            self.p.enableJointForceTorqueSensor(self.robot_id, id)

        for id, args in self.change_dynamics.items(): # change dynamics
            if id in self.joint_ids: # update limits if needed
                index = np.nonzero(self.joint_ids==id)[0][0]
                if 'jointLowerLimit' in args and 'jointUpperLimit' in args:
                    self.lowerLimits[index] = args['jointLowerLimit']
                    self.upperLimits[index] = args['jointUpperLimit']
                if 'maxJointVelocity' in args:
                    self.maxVelocity[index] = args['maxJointVelocity']
                    if self.mode == 'joint velocities':
                        args.pop('maxJointVelocity') # do not clamp velocity if velocity control is used (actions are in -1,1)
                if 'jointLimitForce' in args:
                    self.maxForce[index] = args['jointLimitForce']
                # we got weird behaviours when using torques and changeDynamics
                if id in self.joint_ids[-self.n_control_gripper:] or self.mode not in {'joint torques', 'inverse dynamics', 'pd stable'}:
                    self.p.changeDynamics(self.robot_id, linkIndex=id, **args)

        self.maxForce = np.where(self.maxForce<=0, 100, self.maxForce)# replace bad values
        self.maxVelocity = np.where(self.maxVelocity<=0, 1, self.maxVelocity)
        self.maxAcceleration = np.ones(self.n_joints)*10# set maximum acceleration for inverse dynamics

        self.jointRanges = self.upperLimits-self.lowerLimits
        self.restPoses = [s[0] for s in self.p.getJointStates(self.robot_id, self.joint_ids)]

        self.p.setAdditionalSearchPath(pybullet_data.getDataPath())
        # table is about 62.5cm tall and the z position of the table is located at the very bottom, there is link with the ground (so it is static)
        # the top part is a box of size 1.5, 1, 0.05
        if self.table_height is not None and not self.reach:
            self.table_pos = np.array([0, 0.4, -1+(self.table_height-0.625)])
            self.table_id = self.p.loadURDF("table/table.urdf", basePosition=self.table_pos, baseOrientation=[0,0,0,1], useFixedBase=True)
        else:
            self.table_pos, self.table_id = None, None
        self.table_x_size, self.table_y_size = 1.5, 1

        self.load_object(self.get_object(self.obj), delta_pos=self.delta_pos)


        if self.display: # set the camera
            self.p.configureDebugVisualizer(self.p.COV_ENABLE_RENDERING, 1)
            self.p.configureDebugVisualizer(self.p.COV_ENABLE_GUI, 0)
            self.p.resetDebugVisualizerCamera(cameraDistance=self.camera['distance'], cameraYaw=self.camera['yaw'], cameraPitch=self.camera['pitch'], cameraTargetPosition=self.camera['target'])

            if self.gripper_display:
                self.line_width = 4
                self.lines = [self.p.addUserDebugLine([0, 0, 0], end, color, lineWidth=self.line_width, parentObjectUniqueId=self.robot_id, parentLinkIndex=self.end_effector_id) for end, color in zip(np.eye(3)*0.2, np.eye(3))]



        if self.joint_positions is not None: # set the arm, the gripper is not important
            for id, a in zip(self.joint_ids, self.joint_positions):
                self.p.resetJointState(self.robot_id, id, targetValue=a)

        for _ in range(100): self.p.stepSimulation() # let the world run for a bit

        self.target = np.zeros(3) # expressed in the cartesian world

        if self.mode in {'joint torques', 'inverse dynamics', 'pd stable'}: # disable motors to use torque control, with a small joint friction
            self.p.setJointMotorControlArray(bodyIndex=self.robot_id, jointIndices=self.joint_ids, controlMode=self.p.VELOCITY_CONTROL, forces=np.ones(self.n_joints)*1e-3)

        dynamicsInfo = self.p.getDynamicsInfo(self.obj_id, -1) # save intial friction coeficients of the object
        self.frictions = {'lateral':dynamicsInfo[1], 'rolling':dynamicsInfo[6], 'spinning':dynamicsInfo[7]}

        self.save_state = self.p.saveState()
        self.has_reset_object = False


    def step(self, action: Optional[ArrayLike]=None) -> Tuple[ArrayLike, bool, bool, Dict[str, Any]]: # actions are in [-1,1]
        if action is not None:
            la = len(action)
            if self.mode in {'joint positions', 'inverse kinematics'}:
                for id, a, v, f, u, l in zip(self.joint_ids, action, self.maxVelocity, self.maxForce, self.upperLimits, self.lowerLimits):
                    self.p.setJointMotorControl2(bodyIndex=self.robot_id, jointIndex=id, controlMode=self.p.POSITION_CONTROL, targetPosition=l+(a+1)/2*(u-l), maxVelocity=v, force=f)
                for _ in range(self.steps_to_roll): self.p.stepSimulation()
            elif self.mode in {'joint torques'}: # much harder and might not be transferable because requires very accurate URDF, center of mass, frictions...
                for _ in range(self.steps_to_roll):
                    self.p.setJointMotorControlArray(bodyIndex=self.robot_id, jointIndices=self.joint_ids[:la], controlMode=self.p.TORQUE_CONTROL, forces=action*self.maxForce[:la])
                    self.p.stepSimulation()
            elif self.mode in {'joint velocities'}:
                self.p.setJointMotorControlArray(bodyIndex=self.robot_id, jointIndices=self.joint_ids[:la], controlMode=self.p.VELOCITY_CONTROL, forces=self.maxForce[:la], targetVelocities=action*self.maxVelocity[:la])
                for _ in range(self.steps_to_roll): self.p.stepSimulation()
            elif self.mode == 'inverse dynamics': # action must contain all joints
                for _ in range(self.steps_to_roll):
                    states = self.p.getJointStates(self.robot_id, jointIndices=self.joint_ids)
                    torques = self.p.calculateInverseDynamics(self.robot_id, objPositions=[s[0] for s in states], objVelocities=[s[1] for s in states], objAccelerations=(action*self.maxAcceleration[:la]).tolist())
                    #torques = np.clip(torques, -self.maxForce, self.maxForce).tolist()
                    self.p.setJointMotorControlArray(bodyIndex=self.robot_id, jointIndices=self.joint_ids[:-self.n_control_gripper], controlMode=self.p.TORQUE_CONTROL, forces=torques[:-self.n_control_gripper])
                    self.p.stepSimulation()
            elif self.mode == 'pd stable': # action must contain all joints
                u, l = self.upperLimits[:la], self.lowerLimits[:la]
                for _ in range(self.steps_to_roll):
                    torques = self.pd_controller.computePD(bodyUniqueId=self.robot_id, jointIndices=self.joint_ids[:la], desiredPositions=l+(action+1)/2*(u-l), desiredVelocities=np.zeros(la), kps=self.kps, kds=self.kds, maxForces=self.maxForce, timeStep=self.time_step)
                    #torques[-self.n_control_gripper-1] = 0
                    self.p.setJointMotorControlArray(bodyIndex=self.robot_id, jointIndices=self.joint_ids[:-self.n_control_gripper], controlMode=self.p.TORQUE_CONTROL, forces=torques[:-self.n_control_gripper])
                    self.p.stepSimulation()
        else:
            for _ in range(self.steps_to_roll): self.p.stepSimulation()
        #time.sleep(0.001)

        self.info['contact object robot'] = self.p.getContactPoints(bodyA=self.obj_id, bodyB=self.robot_id)
        self.info['contact object plane'] = self.p.getContactPoints(bodyA=self.obj_id, bodyB=self.plane_id)
        self.info['contact robot robot'] = self.p.getContactPoints(bodyA=self.robot_id,bodyB=self.robot_id)

        if self.table_id is not None:
            self.info['contact object table'] = self.p.getContactPoints(bodyA=self.obj_id, bodyB=self.table_id)
            self.info['contact robot table'] = self.p.getContactPoints(bodyA=self.robot_id,bodyB=self.table_id)

        self.info['touch'], self.info['autocollision'], penetration = False, False, False

        for c in self.info['contact object robot']:
            penetration = penetration or c[8]<-0.005; # if contactDistance is negative, there is a penetration, this is bad
            self.info['touch'] = self.info['touch'] or c[4] in self.contact_ids # the object must touch the gripper

        for c in self.info['contact robot robot']:
            if set(c[3:5]) not in self.allowed_collision_pair:
                self.info['autocollision'] = True#; print(c[3:5])
                break

        observation = self.get_obs()

        if self.display and self.gripper_display:
            end = np.array(self.p.getMatrixFromQuaternion(self.info['end effector xyzw'])).reshape(3,3).T @ (np.eye(3)*0.2) + self.info['end effector position']
            self.lines = [self.p.addUserDebugLine(self.info['end effector position'], end, color, lineWidth=self.line_width, replaceItemUniqueId=id) for end, color, id in zip(end, np.eye(3), self.lines)]

        if self.mode == 'joint torques': # getJointState does not report the applied torques if using torque control
            self.info['applied joint motor torques'][:-self.n_control_gripper] = action*self.maxForce[:-self.n_control_gripper]
        elif self.mode in {'inverse dynamics', 'pd stable'}:
            self.info['applied joint motor torques'][:-self.n_control_gripper] = torques[:-self.n_control_gripper]

        if self.reach: # compute distance as reward
            if self.goal:
                achieved_goal, desired_goal = observation['achieved_goal'], observation['desired_goal']
            else:
                achieved_goal, desired_goal = observation[-6:-3], observation[-3:]
            reward = self.compute_reward(achieved_goal=achieved_goal, desired_goal=desired_goal, _info={**self.info})
            self.p.resetBasePositionAndOrientation(self.obj_id, self.target, (0,0,0,1))
        else: # binary reward: grasped or not
            reward = len(self.info['contact object table'] + self.info['contact object plane'] + self.info['contact robot table'])==0 and self.info['touch'] and not penetration
            self.info['is_success'] = self.cumreward > (100 / self.steps_to_roll) # success if the robot has been holding for a bit

        self.cumreward += reward
        is_out = np.logical_or(self.info['joint positions']<=self.lowerLimits, self.info['joint positions']>=self.upperLimits)
        done = False#np.all(is_out[:-self.n_control_gripper]).item() if self.early_stopping else False

        self.info['terminal_observation'] = observation # stable-baselines3
        info = {**self.info} # copy dict
        # observations are normalized, thus it is not mean to be handled by humans, check info for human readable datas
        return observation, reward, done, info

    def get_object(self, obj: Optional[str]=None):
        """ return a dict containing informations of the primitive shape or a str (urdf file) """
        return obj

    def get_state(self) -> Dict[str, ArrayLike]:
        """ return unnormalized object pose and joint positions. The returned dict can be passed to initialize another environment."""
        pos, qua = self.p.getBasePositionAndOrientation(self.obj_id)
        joint_positions = [s[0] for s in self.p.getJointStates(self.robot_id, self.joint_ids)]
        return {'object_position':pos, 'object_xyzw':qua, 'joint_positions': joint_positions}

    def reset_random_state(self) -> Tuple[ArrayLike, ArrayLike]:
        """
        Put the robot in a random configuration
        generate a random position of the object on the table, supposing the center_workspace is above the table (the robot can reach with the straight arm) and the robot is not too far from the table!
        If there is no table, positions are generated on a circle with center_workspace as the center.
        If reach is enabled, a new random position is generated in the sphere (reachable space)
        """

        for i in range(1000): # try to generate a random configuration of the robot
            for id, u, l in zip(self.joint_ids, self.upperLimits, self.lowerLimits):
                self.p.resetJointState(self.robot_id, id, targetValue=l+self.rng.random()*(u-l))
            self.p.performCollisionDetection()
            if self.table_id is not None and len(self.p.getContactPoints(bodyA=self.robot_id,bodyB=self.table_id))>0:
                continue # repeat, there is a collision with the table
            for c in self.p.getContactPoints(bodyA=self.robot_id,bodyB=self.robot_id):
                if set(c[3:5]) not in self.allowed_collision_pair:
                    break # repeat, there is an autocollision
            else: # there is no collision, stop the loop
                break
        else: # we did 1000 trials
            raise Exception('Failed 1000 times to generate a random configuration of the robot')

        if self.reach: # change the fictitious target point
            self.target = self.rng.random(3)*2-1
            while np.linalg.norm(self.target) > 1:
                self.target = self.rng.random(3)*2-1
            # scale and shift the fictitious target to get coordinates in the world
            self.target, _ = self.p.multiplyTransforms(self.center_workspace_cartesian, (0,0,0,1), self.target*self.radius, [0,0,0,1]);
            self.target = np.array(self.target)
            return # no need to reset the object

        obj_pos, obj_qua = self.p.getBasePositionAndOrientation(self.obj_id)
        rand_qua = self.rng.random(4)
        rand_qua /= np.linalg.norm(rand_qua)
        for i in range(1000): # try to generate a random pose of the object
            if self.table_id is None:
                rand_pos = self.radius*self.rng.random(size=(2,))
                rand_pos = np.hstack([rand_pos+self.center_workspace_cartesian, obj_pos[2]])
            else: # generate a position on the table
                edge = (np.array([self.table_x_size, self.table_y_size]) - self.obj_length)/2 # add the size of the object as edge margin
                rand_pos = edge * (2*self.rng.random(size=(2,)) - 1)
                rand_pos = np.hstack([rand_pos+self.table_pos[:2], obj_pos[2]])

            if np.linalg.norm(rand_pos - self.center_workspace_cartesian) < self.radius-0.2: # 20cm margin of the reachable space
                self.p.resetBasePositionAndOrientation(self.obj_id, rand_pos, rand_qua)
                return

        raise Exception('Failed 1000 times to generate a random position of the object, the robot is too far from the table or the radius is not well tuned')


    def load_object(self, obj:Optional[Union[str, dict]] = None, delta_pos: ArrayLike = [0,0]):
        pos = np.array(self.object_position)
        pos[:2] += delta_pos
        if self.random_var:
            pos[:2] += self.rng.normal(scale=self.random_var, size=2)

        # create object to grab
        if isinstance(obj, dict):
            if "shape" not in obj.keys(): raise ValueError("'shape' as a key doesn't exist in obj")
            elif obj["shape"] == 'cuboid':
                infoShape =  {"shapeType":self.p.GEOM_BOX, "halfExtents":[obj["x"]/2, obj["y"]/2, obj[z]/2]}
                obj_to_grab_id = self.p.createMultiBody(baseMass=1, baseCollisionShapeIndex=self.p.createCollisionShape(**infoShape), baseVisualShapeIndex=self.p.createVisualShape(**infoShape, rgbaColor=[1, 0, 0, 1]), useMaximalCoordinates=True)
                self.p.resetBasePositionAndOrientation(obj_to_grab_id, pos, self.object_xyzw)

            elif obj["shape"] == 'cube':
                obj_to_grab_id = self.p.loadURDF("cube_small.urdf", pos, self.object_xyzw, globalScaling=obj["unit"]/0.05, useMaximalCoordinates=True) # cube_small is a 5cm 0.1kg cube
                #self.p.changeVisualShape(obj_to_grab_id, -1, rgbaColor=[255, 0, 0])
            elif obj["shape"] == 'sphere':
                obj_to_grab_id = self.p.loadURDF("sphere_small.urdf", pos, self.object_xyzw, globalScaling=obj["radius"]/0.06, useMaximalCoordinates=True) # sphere_small is a 6cm diameter 0.1kg sphere
                self.p.changeDynamics(obj_to_grab_id, -1, rollingFriction=1e-4, spinningFriction=1e-3) # allow the sphere to roll, defaults are 1e-3, 1e-3

            elif obj["shape"] == 'cylinder':
                infoShape =  {"shapeType":self.p.GEOM_BOX, "radius": obj["radius"]}
                col_id = self.p.createCollisionShape(**infoShape, height=obj["z"])
                viz_id = self.p.createVisualShape(**infoShape, length=obj["z"], rgbaColor=[1, 0, 0, 1])
                obj_to_grab_id = self.p.createMultiBody(baseMass=1, baseCollisionShapeIndex=col_id, baseVisualShapeIndex=viz_id, useMaximalCoordinates=True)
                self.p.resetBasePositionAndOrientation(obj_to_grab_id, pos, self.object_xyzw)
            self.p.changeDynamics(obj_to_grab_id, -1, lateralFriction=obj["lateralFriction"] if "lateralFriction" in obj else 0.7) # default is 0.5
        elif isinstance(obj, str):
            urdf = Path(__file__).parent/"objects"/obj/f"{obj}.urdf"
            if not urdf.exists(): raise ValueError(str(urdf) + " doesn't exist")
            try:
                obj_to_grab_id = self.p.loadURDF(str(urdf), pos, self.object_xyzw, useMaximalCoordinates=True) # the scale is set in the urdf file
            except self.p.error as e:
                raise self.p.error(f"{e}: "+str(urdf))
        #elif obj is None:
            #pass # do not load any object
        else:
            raise ValueError("Unrecognized object: "+obj)

        #self.p.changeDynamics(obj_to_grab_id, -1, collisionMargin=0.04)
        self.obj_id = obj_to_grab_id
        aabbMin, aabbMax = self.p.getAABB(self.obj_id)
        self.obj_length = np.linalg.norm(np.array(aabbMax)- np.array(aabbMin)).item() # approximate maximum length of the object

    def reset(self, delta_pos: ArrayLike = [0,0], delta_yaw: float = 0, multiply_friction:float={}, object_position=None, object_xyzw=None, joint_positions=None, load=None):
        """
        delta_pos and self.delta_pos are relative to the initial position (during init)
        object_position, object_xyzw, joint_positions are absolute, they overwrite everything
        """
        self.cumreward = 0
        load = load or 'state'
        load = load.strip().lower()
        assert load in {'all', 'state', 'none', 'reset'}
        if load == 'all':
            self.load_all()
        elif load == 'state':
            assert not self.has_reset_object, "you can not remove/change the object and restore a state: use either reset() or reset_object(), not both"
            self.p.restoreState(self.save_state)
        elif load == 'none': # leave as it is
            pass
        elif load == 'reset':
            self.p.resetBasePositionAndOrientation(self.obj_id, self.object_position, self.object_xyzw)
            self.reset_robot()
            for _ in range(100): self.p.stepSimulation()

        if (not np.any(delta_pos)) and delta_yaw==0 and len(multiply_friction)==0  and (not self.reset_random_initial_state) and (not self.random_var) and object_position is None and object_xyzw is None and joint_positions is None:
            return self.get_obs() # do not need to change the position
        elif self.reset_random_initial_state:
            self.reset_random_state()

        pos, qua = self.p.getBasePositionAndOrientation(self.obj_id)
        pos = [pos[0]+delta_pos[0], pos[1]+delta_pos[1], pos[2]]
        if self.random_var:
            pos[0] += random.gauss(0, self.random_var)
            pos[1] += random.gauss(0, self.random_var)
        _, qua = self.p.multiplyTransforms([0,0,0], [0, 0, np.sin(delta_yaw/2), np.cos(delta_yaw/2)], [0,0,0], qua) # apply delta_yaw rotation
        pos = object_position or pos # overwrite if absolute position is given
        qua = object_xyzw or qua
        joint_pos = joint_positions or [s[0] for s in self.p.getJointStates(self.robot_id, self.joint_ids)]
        self.p.resetBasePositionAndOrientation(self.obj_id, pos, qua)
        for id, jp in zip(self.joint_ids, joint_pos): # reset the robot
            self.p.resetJointState(self.robot_id, jointIndex=id, targetValue=jp)
        new_friction = {}
        for key, value in multiply_friction.items():
            assert key in {"lateral", "rolling", "spinning"}, f"you gave {key}, allowed keys are lateral, rolling, spinning"
            new_friction[key+'Friction'] = value*self.frictions[key]
        self.p.changeDynamics(bodyUniqueId=self.obj_id, linkIndex=-1, **new_friction) # set the object friction
        #for _ in range(240): self.p.stepSimulation() # let the object stabilize

        return self.get_obs()

    def get_obs(self) -> ArrayLike:
        obj_pose = self.p.getBasePositionAndOrientation(self.obj_id)
        # we do not normalize the velocity, supposing the object is not moving that fast
        # we do not express the velocity in the robot frame, supoosing the robot is not moving
        obj_vel = self.p.getBaseVelocity(self.obj_id)
        self.info['object position'], self.info['object xyzw'] = obj_pose
        self.info['object linear velocity'], self.info['object angular velocity'] = obj_vel
        jointStates = self.p.getJointStates(self.robot_id, self.joint_ids)
        pos, vel = [0]*self.n_joints, [0]*self.n_joints
        for i, state, u, l, v in zip(range(self.n_joints), jointStates, self.upperLimits, self.lowerLimits, self.maxVelocity):
            pos[i] = 2*(state[0]-u)/(u-l) + 1 # set between -1 and 1
            vel[i] = state[1]/v # set between -1 and 1
            self.info['joint positions'][i], self.info['joint velocities'][i], jointReactionForces, self.info['applied joint motor torques'][i] = state
            self.info['joint reaction forces'][i] = jointReactionForces[-1] # get Mz

        sensor_torques = self.info['joint reaction forces']/self.maxForce # scale to [-1,1]
        absolute_center = self.p.multiplyTransforms(*self.p.getBasePositionAndOrientation(self.robot_id), *self.center_workspace_robot_frame) # the pose of center_workspace in the world
        invert = self.p.invertTransform(*absolute_center)
        obj_pos, obj_or = self.p.multiplyTransforms(*invert, *obj_pose) # the object pose in the center_workspace frame

        if self.reach: # replace the object position with the fictious target point
            obj_or = self.rng.random(4)*2-1 # random orientation
            obj_or /= np.linalg.norm(obj_or)
            obj_pos, obj_or = self.p.multiplyTransforms(*invert, self.target, obj_or) # target position in the robot frame (shoulder)
            obj_vel = (0,0,0), (0,0,0)
        obj_pos = np.array(obj_pos)/self.radius
        obj_or = self.p.getMatrixFromQuaternion(obj_or)[:6] # taking 6 parameters from the rotation matrix to let the rotation be described in a continuous representation, which is better for neural networks

        # get information on gripper
        self.info['end effector position'], self.info['end effector xyzw'], _, _, _, _, self.info['end effector linear velocity'], self.info['end effector angular velocity'] = self.p.getLinkState(self.robot_id, self.end_effector_id, computeLinkVelocity=True)

        end_pos, end_or = self.p.multiplyTransforms(*invert, self.info['end effector position'], self.info['end effector xyzw'])
        end_pos = np.array(end_pos) / self.radius
        end_or = self.p.getMatrixFromQuaternion(end_or)[:6]
        end_lin_vel, _ = self.p.multiplyTransforms(*self.p.invertTransform((0,0,0), absolute_center[1]), self.info['end effector linear velocity'], (0,0,0,1))

        self.info['robot state'] = np.hstack([end_pos, end_or, end_lin_vel, pos, vel, sensor_torques,]) # robot state without the object state
        if self.reach and self.goal:
            # the order during concatenation matters
            observation = OrderedDict([
                ("achieved_goal", end_pos),
                ("observation", np.array([end_or, end_lin_vel, pos, vel, sensor_torques, obj_or, *obj_vel])),
                ("desired_goal", obj_pos),
            ])
        else:
            observation = np.hstack((self.info['robot state'], obj_or, *obj_vel, obj_pos))
            assert np.isfinite(observation).all(), f"observation is not valid: {observation}"
        return observation

    def reset_object(self, obj=None, delta_pos=[0,0]): # TODO: delete, useless
        if obj == self.obj and not self.has_reset_object:
            self.reset(delta_pos=delta_pos)
        else:
            self.has_reset_object = True
            self.reset_robot()
            self.p.removeBody(self.obj_id)
            self.load_object(self.get_object(obj), delta_pos=delta_pos)
            for _ in range(100): self.p.stepSimulation() # let the object fall

    def reset_robot(self):
        pass


    def render(self, mode='human'):
        if mode in {'rgb_array', 'rgba_array'}: # slow !
            camera = {**self.camera}
            if mode == 'rgb_array': # if rgb, use low resolution
                camera['height'], camera['width'] = 256, 256
            img = self.p.getCameraImage(
                width=camera['width'],
                height=camera['height'],
                viewMatrix=camera['viewMatrix'],
                projectionMatrix=camera['projectionMatrix'],
                renderer=camera['renderer'],
            )[2]
            img = np.array(img, dtype=np.uint8).reshape(camera['height'], camera['width'], 4)
            return img[:,:,:3] if mode=='rgb_array' else img
        elif mode == 'human':
            pass
        else:
            super().render(mode=mode) # just raise an exception

    def close(self):
        if self.physicsClientId >=0:
            self.p.disconnect()
            self.physicsClientId = -1

    def get_joint_state(self, position=True, normalized=True):
        """ Return (un)normalized joint positions (velocities) without the gripper"""
        if position:
            u, l = self.upperLimits[:-self.n_control_gripper], self.lowerLimits[:-self.n_control_gripper]
            p = np.array([s[0] for s in self.p.getJointStates(self.robot_id, self.joint_ids[:-self.n_control_gripper])])
            if normalized:
                p = 2*(p-u)/(u-l) + 1
            return p
        else:
            v = np.array([s[1] for s in self.p.getJointStates(self.robot_id, self.joint_ids[:-self.n_control_gripper])])
            if normalized:
                v = v / self.maxVelocity[:-self.n_control_gripper]
            return v

    def compute_reward(
            self, achieved_goal: ArrayLike, desired_goal: ArrayLike, _info: Optional[Dict[str, Any]]
    ) -> np.float32:
        """ When using reaching"""
        return -np.sqrt(np.linalg.norm(np.array(desired_goal)-achieved_goal, axis=-1))

    def get_fingers(self, x):
        """Return the value of the fingers to control all finger with -1≤x≤1. Gripper opened: x=1, gripper closed: x=-1"""
        raise NotImplementedError(f'get_fingers() is not implemented in {self.__name__}.')

    def new_random_initial_state(self):
        """ generate a new random initial state and save it"""
        self.reset()
        self.reset_random_state()
        for _ in range(100): self.p.stepSimulation()
        # updates initiale state variables so load_all() will use this new configuration
        self.object_position, self.object_xyzw = self.p.getBasePositionAndOrientation(self.obj_id)
        self.joint_positions = np.array([s[0] for s in self.p.getJointStates(self.robot_id, self.joint_ids)])
        self.save_state = self.p.saveState() # save it
