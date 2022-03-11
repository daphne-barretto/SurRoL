import numpy as np
import pybullet as pb
from direct.task import Task
import panda3d.core as p3d
from panda3d.core import (NodePath, Material, Quat, Vec3, Vec4, Mat4)
from pybullet_rendering import ShapeType
from pybullet_rendering.render.panda3d import Mesh
from pybullet_rendering.render.utils import primitive_mesh, decompose


class Scene:
    def __init__(self):
        self.world3d = NodePath('world3d')
        self.gui = NodePath('gui')

        from .application import Application
        self.app = Application.app
        self.cam = Application.app.cam
        self.taskMgr = Application.app.taskMgr
        self.loader = Application.app.loader
        self.render = Application.app.render

    def destroy(self):
        if self.world3d is not None:
            self.world3d.removeNode()
        if self.gui is not None:
            self.gui.removeNode()

        self.world3d = None
        self.gui = None

    def _initialize(self, **kwargs):
        pass


class GymEnvScene(Scene):
    def __init__(self, env_cls, env_params={}):
        super(GymEnvScene, self).__init__()

        self.env_nodes = {}

        self.env = None
        self.env_cls = env_cls
        self.env_params = env_params

        # Setup periodic simulation tasks
        self.time = 0
        self.taskMgr.add(self._step_simulation_task, "StepSimulationTask")
    
    def destroy(self):
        super(GymEnvScene, self).destroy()

        self.taskMgr.remove('StepSimulationTask')
        self.env.cid = -1

    def on_env_created(self):
        pass

    def before_simulation_step(self):
        pass

    def after_simulation_step(self):
        pass

    def _initialize(self, **kwargs):
        assert self.env is None

        if 'cid' in kwargs:
            self.env_params['cid'] = kwargs['cid']
        self.env = self.env_cls(**self.env_params)

        self.on_env_created()
    
    def _step_simulation_task(self, task):
        """Step simulation
        """

        if task.time - self.time > 1 / 240.0:
            self.before_simulation_step()

            # Step simulation
            pb.stepSimulation()

            self.after_simulation_step()

            # Call trigger update scene (if necessary) and draw methods
            pb.getCameraImage(
                width=1, height=1,
                viewMatrix=self.env._view_matrix,
                projectionMatrix=self.env._proj_matrix)
            pb.setGravity(0,0,-10.0)

            self.time = task.time

        return Task.cont

    def _update_graph(self, scene_graph, materials_only):
        """Update a scene using scene_graph description
        Arguments:
            scene_graph {SceneGraph} -- scene description
            materials_only {bool} -- update only shape materials
        """

        for k, v in scene_graph.nodes.items():
            node = self.world3d.attachNewNode(f'node_{k:04d}')
            self.env_nodes[k] = node

            for shape in v.shapes:
                # load model
                if shape.type == ShapeType.Mesh:
                    model = self.loader.load_model(shape.mesh.filename)
                else:
                    mesh = Mesh.from_trimesh(primitive_mesh(shape))
                    model = node.attach_new_node(mesh)

                if shape.material is not None:
                    # set material
                    material = Material()
                    material.setAmbient(Vec4(*shape.material.diffuse_color))
                    material.setDiffuse(Vec4(*shape.material.diffuse_color))
                    material.setSpecular(Vec3(*shape.material.specular_color))
                    material.setShininess(5.0)
                    model.setMaterial(material, 1)

                # set relative position
                model.reparentTo(node)
                model.setPos(*shape.pose.origin)
                model.setQuat(Quat(*shape.pose.quat))
                model.setScale(*shape.pose.scale)

    def _update_view(self, scene_view):
        """Apply scene state.

        Arguments:
            settings {SceneView} -- view settings, e.g. camera, light, viewport parameters
        """

        if scene_view.camera is not None:
            yfov, znear, zfar, aspect = decompose(scene_view.camera.projection_matrix)
            conv_mat = Mat4.convert_mat(p3d.CSZupRight, p3d.CSYupRight)
            self.cam.set_mat(conv_mat * Mat4(*scene_view.camera.pose_matrix.ravel(),))
            self.cam.node().get_lens().set_near_far(znear, zfar)
            self.cam.node().get_lens().set_fov(np.rad2deg(yfov*aspect), np.rad2deg(yfov))

    def _update_state(self, scene_state):
        """Apply scene state.

        Arguments:
            scene_state {SceneState} -- transformations of all objects in the scene
        """

        for uid, node in self.env_nodes.items():
            node.set_mat(Mat4(*scene_state.matrix(uid).ravel()))

