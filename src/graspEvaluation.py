#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# attempt a grasp quality metric but gave up

import numpy as np
from scipy.spatial import ConvexHull, KDTree

class Object: # convert .obj to array of vertices and facets
    def __init__(self, objfile: str): # objfile is the .obj file
        self.v = np.array([]).reshape(-1,3) # vertices x,y,z in the object reference, as defined in .obj
        self.f = np.array([],dtype=int).reshape(-1,3) # facets, suppose the mesh is only made of triangles
        with open(objfile, "rt") as f:
            for line in f:
                stripped = line.strip()
                if stripped=="" or stripped[0] in ['#', '$', '!', '@']:
                    continue
                cmdargs = stripped.split()
                cmd, args = cmdargs[0], cmdargs[1:]
                if cmd=="v":
                    self.v = np.vstack([self.v, [float(args[0]), float(args[1]), float(args[2])]])
                elif cmd=="f":
                    self.f = np.vstack([self.f, [int(args[i].split("/")[0]) for i in range(3)]]) # negative values are possible
                else:
                    continue
        self.f -= 1 # in .obj files, indices start at 1
        self.centroidTree = KDTree(self.v[self.f].mean(1)) # centroids of each facet
        nv, nf = self.v.shape[0], self.f.shape[0] # nb of vertices and facets
        indexes, toSplit, _ = np.nonzero(np.arange(nv)[:,None,None]==self.f)
        indexes = np.cumsum(np.count_nonzero(indexes==np.arange(nv)[:,None], axis=-1))
        self.corners = np.split(toSplit, indexes) # list of neighbour facets for each vertex, a vertex can be a corner of 4 facets (e.g. cube) or more, so ndarray is not suitable
        self.neighbors = -np.ones((nf, 3), dtype=int) # each facet has 3 neighbors facets
        for i, facet in enumerate(self.f):
            j = 0 # counter up to 3
            for vertex in facet[:-1]: # iterating over 2 vertices is enough
                for k in self.corners[vertex]: # facet index
                    if k!=i and k not in self.neighbors[i] and ((self.f[k,0] in facet and self.f[k,1] in facet) or (self.f[k,1] in facet and self.f[k,2] in facet) or (self.f[k,0] in facet and self.f[k,2] in facet)): #  if there are 2 common vertices
                        self.neighbors[i,j] = k
                        j += 1
                        if j==3 : break
                else: continue # the magic trick: for and while loop use else at the end
                break

        expandedFacets = self.v[self.f] # (nfacets, 3(triangle), 3(xyz))
        self.n = np.cross(expandedFacets[:,1]-expandedFacets[:,0], expandedFacets[:,2]-expandedFacets[:,0]) # normal vector for each centroid
        self.n /= np.linalg.norm(self.n, axis=1)[:,None] # but we still don't know whether it is in or outward
        self.n[0] = self.normalOutward(0)
        a,b,c = self.f[0] # we do the cross product as follow: np.cross(self.v[b]-self.v[a],self.v[c]-self.v[b])
        neighborsToVisit = {n:[a,b,c] if np.dot(np.cross(self.v[b]-self.v[a],self.v[c]-self.v[b]),self.n[0])>0 else [c,b,a] for n in self.neighbors[0]} # map neighbors to be evaluated to vertices of neighbors already evaluated in the right prder
        done = np.zeros(nf, dtype=bool)
        done[0] = True
        while len(neighborsToVisit)>0: # expand the right normal to neighbors
            neighborsToVisitNext = dict()
            done[list(neighborsToVisit.keys())] = True # mark facets that have been evauated as done
            for newFacet, (a,b,c) in neighborsToVisit.items(): # facet to be evaluated facet index, and neighbor already evaluated facet index
                # let a,b,c the vertices of the facet f with the outward normal n such that cross(ab,bc) @ n >0 (with ab=b-a)
                # and f' a neighbor facet of f with vertices a,b,d therefore, the outward normal of f' is cross(ba,ad)
                sv = set(self.f[newFacet]) # the three vertices of the facet
                common = (a,b) if {a,b}<=sv else (b,c) if {b,c}<=sv else (c,a) # get the 2 vertices common to both facets
                different = next(iter(sv - set(common))) # get the vertex which is not common
                self.n[newFacet] = np.cross(self.v[common[0]]-self.v[common[1]], self.v[different]-self.v[common[0]]) # apply the rule
                for facet in self.neighbors[newFacet]: # update the next neighbor facets to evaluate
                    if not done[facet] and facet not in neighborsToVisitNext.keys():
                        neighborsToVisitNext[facet] = [common[1], common[0], different]

            neighborsToVisit = neighborsToVisitNext.copy()
        self.n /= np.linalg.norm(self.n, axis=1)[:,None]


    def getNearestNormal(self, positions): # return the indexes of outward normals given the positions (n,3)
        d, i = self.centroidTree.query(positions)
        return self.n[i]#i

    def normalOutward(self, i):# return the outward normal of facet i, expensive to compute
        c = self.centroidTree.data[i] # centroid of facet i
        n = self.n[i] # normal of facet i
        denominator = np.sum(n*self.n, axis=-1) # detect perpendicular facets
        numerator = np.sum((self.centroidTree.data-c)*self.n, axis=1) # detect intersection in the oposite direction
        valid = np.logical_and(np.abs(denominator)>1e-6, numerator*denominator>0) # keep non-perpendicular planes with intersection in the direction of the normal
        d = numerator[valid] / denominator[valid] # https://en.wikipedia.org/wiki/Line–plane_intersection
        intersections = c + n*d[:,None]
        expandedFacets = self.v[self.f[valid]]#self.v[np.delete(self.f, toDelete, axis=0)]
        v0, v1, v2 = expandedFacets[:,0], expandedFacets[:,1], expandedFacets[:,2] # http://geomalgorithms.com/a06-_intersect-2.html
        w = intersections - v0
        u = v1 - v0
        v = v2 - v0
        uu, vv, uv, wu, wv = np.sum(u*u, axis=-1), np.sum(v*v, axis=-1), np.sum(u*v, axis=-1), np.sum(w*u, axis=-1), np.sum(w*v, axis=-1) # dot products
        denominator = uv*uv - uu*vv
        s = (uv*wv - vv*wu)/denominator
        t = (uv*wu - uu*wv)/denominator
        ncross = np.count_nonzero(np.logical_and.reduce([s>0, t>0, s+t<1])) + np.count_nonzero(np.logical_or.reduce([s==0, t==0, s+t==1]))//2 # nb of time the normal of facet i cross the object
        return self.n[i]/np.linalg.norm(self.n[i])*(1 if ncross%2==0 else -1) # change the signe if inward

class Box(Object):
    def __init__(self, x,y,z):
        """self.v = np.array([[-x,-y,z],[x,-y,z],[-x,y,z],[x,y,z],[-x,y,-z],[x,y,-z],[-x,-y,-z],[x,-y,-z]])/2
        self.f = np.array([[0,1,2],[2,1,3],[2,3,4],[4,3,5],[4,5,6],[6,5,7],[6,7,0],[0,7,1],[1,7,3],[3,7,5],[6,0,4],[4,0,2]], dtype=int)
        self.n  = np.array([[0,0,1],[0,0,1],[0,1,0],[0,1,0],[0,0,-1],[0,0,-1],[0,-1,0],[0,-1,0],[1,0,0],[1,0,0],[-1,0,0],[-1,0,0]])
        self.centroidTree = KDTree(self.v[self.f].mean(1))"""
        self.x, self.y, self.z = x,y,z

    def getNearestNormal(self, positions):
        h = np.abs(positions[:,2]) - (self.z-np.minimum(self.x, self.y))/2
        minx = np.maximum(self.x-self.y, 0)/2
        miny = np.maximum(self.y-self.x, 0)/2
        east = h > positions[:,0] + minx
        north = h > positions[:,1] + miny
        west = h > -positions[:,0] - minx
        south = h > -positions[:,1] - miny
        insidePyramid = np.logical_and.reduce([east, north, west, south])
        normals = np.zeros((positions.shape[0], 3))
        normals[insidePyramid,2] = np.where(positions[insidePyramid,2]>0, 1, -1)
        xfacet = np.abs(positions[:,0])+minx > np.abs(positions[:,1])+miny
        xfacet = np.logical_and(xfacet, np.logical_not(insidePyramid))
        yfacet = np.logical_not(np.logical_or(xfacet, insidePyramid))
        normals[xfacet,0] = np.where(positions[xfacet,0]>0, 1, -1)
        normals[yfacet,1] = np.where(positions[yfacet,1]>0, 1, -1)
        return normals

class Cylinder:
    def __init__(self, r,h):
        self.r, self.h = r, h
    def getNearestNormal(self, positions):
        # if the vector is inside the cone, the nearest facet is one of the 2 flat surfaces, otherwise it is the round surface
        insideCone = np.abs(positions[:,2])-(self.h-self.r)/2 > np.sqrt(np.square(positions[:,0])+np.square(positions[:,1]))
        outsideCone = np.logical_not(insideCone)
        normals = np.zeros((positions.shape[0], 3))
        normals[insideCone,2] = np.where(positions[insideCone,2]>0, 1, -1)
        normals[outsideCone] = positions[outsideCone]
        normals[outsideCone,2] = 0
        return normals / np.linalg.norm(normals, axis=-1)[:,None]

class Sphere:
    def __init__(self, r):
        self.r = r
    def getNearestNormal(self, positions):
        return positions / np.linalg.norm(positions, axis=-1)[:,None]

from scipy import interpolate
import math
class InterpolateKeyPointsEndPauseGripAssumption(): # copy paste from https://github.com/aamorel/grab-babling/blob/master/src/controllers.py

    def __init__(self, individual, info):
        """Interpolate actions between keypoints
           Stops the movement at the end in order to make sure that the object was correctly grasped at the end.
           Only one parameter for the gripper, specifying the time at which it should close
        """
        actions = []
        gene_per_key = info['GENE_PER_KEYPOINTS'] - 1
        for i in range(info['NB_KEYPOINTS']):
            actions.append(individual[gene_per_key * i:gene_per_key * (i + 1)])

        additional_gene = individual[-1]

        n_keypoints = len(actions)
        self.pause_time = info['pause_frac'] * info['n_iter']
        interval_size = int(self.pause_time / n_keypoints)
        interp_x = [int(interval_size / 2 + i * interval_size) for i in range(n_keypoints)]
        self.action_polynome = interpolate.interp1d(interp_x, actions, kind='quadratic', axis=0,
                                                    bounds_error=False, fill_value='extrapolate')
        self.open_loop = True
        self.last_action = 0
        self.grip_time = math.floor((additional_gene / 2 + 0.5) * self.pause_time)
        self.initial_action = actions[0]
        self.initial_action = np.append(self.initial_action, 1)

    def get_action(self, i):
        if i <= self.pause_time:
            action = self.action_polynome(i)
            if i < self.grip_time:
                action = np.append(action, 1)  # gripper is open
            else:
                action = np.append(action, -1)  # gripper is closed
            self.last_action = action
        else:
            # feed last action
            action = self.last_action
        return action

import pybullet as p
import gym
import time
def simulate(metric):
    """ simulate a graspin with kuka and return the grasping quality evaluation with 'metric', an evaluator"""
    obj = Box(0.06, 0.06, 0.16)
    env = gym.make('gym_baxter_grabbing:kuka_grasping-v0', display=False, obj='cube', steps_to_roll=1)
    individual = np.load('/Users/Yakumo/Downloads/run15/type3_0.npy')
    controller_info = {'pause_frac': 0.66, 'n_iter': 2500, 'NB_KEYPOINTS': 3, 'GENE_PER_KEYPOINTS': 9}
    controller = InterpolateKeyPointsEndPauseGripAssumption(individual, controller_info)
    action = controller.initial_action
    for i in range(controller_info['n_iter']):
        #time.sleep(0.05)
        #env.render()
        observation, reward, done, info = env.step(action)
        action = controller.get_action(i)
    # c[5] is positionOnA, c[3] is linkIndexA, c[7] is contactNormalOnA
    positions = np.array([]).reshape(-1,3) # contact positions in the absolute referential
    normals = np.array([]).reshape(-1,3) # contact normal
    listContact = p.getContactPoints(bodyA=env.robot_id, bodyB=env.obj_id)
    if len(listContact)==0:
        return -np.linalg.norm(np.array(observation[0]) - np.array(observation[2])) # negative distance between gripper & object
    for c in listContact:
        if 7<c[3]<14: # if the object is touch by the gripper i.e. between the range of link ids of the gripper2 of kuka
            positions = np.vstack([positions, c[5]])
            normals = np.vstack([normals, c[7]])
    env.close()
    # position of contact points of the gripper in the grasped object reference (cartesian)
    # equivalent: posInv, orInv = p.invertTransform(observation[0], observation[1])
    # positionObjectRef = np.array([p.multiplyTransforms(posInv, orInv, c, [0,0,0,0])[0] for c in positions])
    positionObjectRef = (np.array(p.getMatrixFromQuaternion(observation[1])).reshape(3,3).T @ (positions - np.array(observation[0])).T).T
    # contact normal of the gripper pointing to the grasped object in the grasped object reference
    # equivalent: np.array([p.multiplyTransforms([0,0,0], orInv, n, [0,0,0,0])[0] for n in normals])
    normalObjectRef = (np.array(p.getMatrixFromQuaternion(observation[1])).reshape(3,3).T @ normals.T).T
    return metric(np.hstack([positionObjectRef, normalObjectRef]))

def convexGraspWrench(contacts, nedge=16, friction=0.3, scale=1):
    """
    compute the convexe hull of grasp wrenchs
    we scale the normal force component with the average distance of contact points to the centroid
    see https://www.sciencedirect.com/science/article/pii/S092188900800208X

    Parameters
    ----------
    contacts: ndarray(-1),6)
        contact.shape[0] is the nb of contact
        constact[:,:3] are all contact positions and constact[:,3:] are respective contact normals
    nedge: positive int, optional
        discretisation of the cone
    friction: positive float, optional
        friction coefficient
    scale: positive float, optional
        scale of wrenchs, the default is unit

    Return
    ------
    scipy.spatial.ConvexHull
        the convexHull of grasp wrenchs
    """
    # the reference point is the centroid of contact points in the .obj referential
    referencePoint = contacts[:,:3].mean(0)
    averageDistance = np.linalg.norm(referencePoint - contacts[:,:3], axis=-1).mean()
    rangeEdge = np.pi*2*np.arange(nedge)[:,None]/nedge
    n = contacts[:,3:]#obj.getNearestNormal(contacts[:,:3])
    x = np.cross(np.random.rand(contacts.shape[0],3), n)# tangent components
    y = np.cross(x, n)#obj.n[i]) # (ncontact, 3(xyz))
    x /= np.linalg.norm(x, axis=1)[:,None] # normalize
    y /= np.linalg.norm(y, axis=1)[:,None]
    nprimitives = contacts.shape[0]*nedge
    primitvesWrenches = np.zeros((nprimitives, 6))
    primitvesForces = n[:,None] + friction*(x[:,None]*np.cos(rangeEdge) + y[:,None]*np.sin(rangeEdge))
    primitvesForces = primitvesForces.reshape(nprimitives,3)
    primitvesForces = scale*primitvesForces/np.linalg.norm(primitvesForces, axis=-1)[:,None]
    primitvesWrenches[:,:3] = primitvesForces*averageDistance
    primitvesWrenches[:,3:] = np.cross(np.repeat(contacts[:,:3]-referencePoint, nedge, axis=0), primitvesForces)
    return ConvexHull(primitvesWrenches)

def gravityDistance(convexhull, contacts):
    # return the distance between the origin and the intersection point of the convex hull and the gravity wrench vector
    referencePoint = contacts[:,:3].mean(0)
    averageDistance = np.linalg.norm(referencePoint - contacts[:,:3], axis=-1).mean()
    normals, offsets = np.split(convexhull.equations, [-1], axis=-1)
    valid = distToSimplices(convexhull)>0 # valid facets are those which we need to go through the convex hull before reaching the them
    gravityWrench = np.zeros(6)
    gravityWrench[2] = averageDistance
    # the center of mass is 0 in the object referential because the position of the object is the position of the center of mass in pybullet
    gravityWrench[3:] = np.cross(0-referencePoint, [0,0,1])
    gravityWrench /= np.linalg.norm(gravityWrench)
    distances = -offsets[valid]/np.dot(normals[valid], gravityWrench) # https://stackoverflow.com/a/30654855
    return np.min(distances[distances>0]) if np.max(distances)>0 else np.max(distances[distances<0])
    #verticals = offsets[valid]/normals[valid,2]
    #return np.min(verticals[verticals>0]) if np.max(verticals)>0 else np.max(verticals[verticals<0])

def bflmrw(contacts, nedge=16, friction=0.3, scale=1e4, λ=0.5):
    """
    brute force largest minimum resisted wrench
    If there are less than 3 contact points, the convex hull fails, thus it returns 0
    λ is the weight to compute the balance between the grasp quality and the gravitational robustness
    the result is res=λ*quality+(1-λ)*gravityRobustness but it could be negative
    If the graps is not force closure, it returns 1/(1-result) ∈ ]0,1[ with result<0
    otherwise 1+res ∈ [1,+∞[
    """
    if contacts.shape[0]<3:
        return 0
    ch = convexGraspWrench(contacts=contacts, nedge=nedge, friction=friction, scale=scale)
    distances = distToSimplices(ch)
    minimum = np.min(distances)
    quality = minimum if minimum>0 else np.max(distances[distances<0])
    result = λ*quality+(1-λ)*gravityDistance(ch, contacts)
    return result+1 if result>0 else 1/(1-result)

def distToSimplices(ch, point=None): # point (d,) planes (n,d+1)
    d = ch.points.shape[1] # dimension
    point = point if point is not None else np.zeros(d)
    normals, offsets = np.split(ch.equations, [-1], axis=-1)
    # according to Qhull, it is negative if inside, but opposite in the paper
    dist = -(np.dot(normals, point)+offsets.squeeze()) / np.linalg.norm(normals, axis=-1) # https://math.stackexchange.com/a/1210685
    return dist

def projection(point, convexhull):
    normal, offset = np.split(convexhull.equations, [-1], axis=-1)
    return point - np.dot(normal,point)+offset / np.inner(normal,normal) * normal

def isinside(convexhull, point=None):
    if point is None:
        point = np.zeros(convexhull.points.shape[1])
    normals, offsets = np.split(convexhull.equations, [-1], axis=-1)
    return ((np.dot(normals, point)+offsets.squeeze())<0).all()

if __name__=="__main__":
    ch = ConvexHull(np.random.normal(size=(10,2)))
    print("equation", ch.equations)
    print("distance from origin to simplices",distToSimplices(ch), len(ch.equations))
    print("origin is inside of ch:",isinside(ch))
    obj = Object("/Users/Yakumo/Library/Mobile Documents/com~apple~CloudDocs/Evolution of diverse robot grasping policies/venv/lib/python3.9/site-packages/pybullet_data/cube.obj")#"data/boston_box.obj")
    print("nearest normal", obj.getNearestNormal([[0,0,1], [0,0,-1]])) # return [[0,0,1], [0,0,-1]] because cube
    print("bflmrw 4 contacts:", bflmrw(np.array([[1,0,0, -1,0,0], [-1,0,0, 1,0,0], [0,1,0, 0,-1,0], [0,-1,0, 0,1,0]])*2)) # 4 forces on each x,y facet of the cube
    print("bflmrw 3 contacts:", bflmrw(np.array([[1,0,0, -1,0,0], [-1,0,0, 1,0,0], [0,1,0, 0,-1,0]])*2)) # 3 forces, with 2 forces it raises a error
    #print("box nearest normal", Box(1,1,1).getNearestNormal(np.array([[-0.4,-0.5,-0]])))
    print("simulation", simulate(bflmrw))
