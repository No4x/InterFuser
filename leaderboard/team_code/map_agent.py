import numpy as np
from PIL import Image, ImageDraw

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

from team_code.base_agent import BaseAgent
from team_code.planner import RoutePlanner

import pygame
import torch
from utils_tf.map_utils import MapImage, PIXELS_PER_METER
from utils_tf import lts_rendering
import carla
import os
class MapAgent(BaseAgent):
    def sensors(self):
        result = super().sensors()
        # result.append(
        #     {
        #         "type": "sensor.camera.semantic_segmentation",
        #         "x": 0.0,
        #         "y": 0.0,
        #         "z": 100.0,
        #         "roll": 0.0,
        #         "pitch": -90.0,
        #         "yaw": 0.0,
        #         "width": 512,
        #         "height": 512,
        #         "fov": 5 * 10.0,
        #         "id": "map",
        #     }
        # )
        result+=[{
                "type": "sensor.camera.semantic_segmentation",
                "x": 0.0,
                "y": 0.0,
                "z": 100.0,
                "roll": 0.0,
                "pitch": -90.0,
                "yaw": 0.0,
                "width": 512,
                "height": 512,
                "fov": 5 * 10.0,
                "id": "map",
            },
            {
                'type': 'sensor.opendrive_map',
                'reading_frequency': 1e-6,
                'id': 'hd_map'
            }
        ]

        return result

    def set_global_plan(self, global_plan_gps, global_plan_world_coord):
        super().set_global_plan(global_plan_gps, global_plan_world_coord)

        self._plan_HACK = global_plan_world_coord
        self._plan_gps_HACK = global_plan_gps

    def _init(self, hd_map):
        super()._init()

        self._vehicle = CarlaDataProvider.get_hero_actor()
        self._world = self._vehicle.get_world()

        self._waypoint_planner = RoutePlanner(4.0, 50)
        self._waypoint_planner.set_route(self._plan_gps_HACK, True)
        print(len(self._waypoint_planner.route))

        self._traffic_lights = list()

        #tf create map for renderer

        self.world_map = carla.Map("RouteMap", hd_map[1]['opendrive'])

        map_image = MapImage(self._world, self.world_map, PIXELS_PER_METER)
        make_image = lambda x: np.swapaxes(pygame.surfarray.array3d(x), 0, 1).mean(axis=-1)
        road = make_image(map_image.map_surface)
        lane = make_image(map_image.lane_surface)

        self.global_map = np.zeros((1, 15,) + road.shape)
        self.global_map[:, 0, ...] = road / 255.
        self.global_map[:, 1, ...] = lane / 255.

        torch.cuda.empty_cache()
        cuda_device = os.environ.get('CUDA_DEVICE')
        self.global_map = torch.tensor(self.global_map, device=f'cuda_device:{cuda_device}', dtype=torch.float32)

        world_offset = torch.tensor(map_image._world_offset, device=f'cuda:{cuda_device}', dtype=torch.float32)

        self.map_dims = self.global_map.shape[2:4]
        self.renderer = lts_rendering.Renderer(world_offset, self.map_dims, data_generation=True)


    def tick(self, input_data):
        self._actors = self._world.get_actors()
        self._traffic_lights = get_nearby_lights(
            self._vehicle, self._actors.filter("*traffic_light*")
        )
        self._stop_signs = get_nearby_lights(
            self._vehicle, self._actors.filter("*stop*")
        )

        topdown = input_data["map"][1][:, :, 2]
        topdown = draw_traffic_lights(topdown, self._vehicle, self._traffic_lights)
        topdown = draw_stop_signs(topdown, self._vehicle, self._stop_signs)

        lidar = input_data['lidar']

        cars = self.get_bev_cars(lidar=lidar)
        result = super().tick(input_data)
        result["topdown"] = topdown
        result['topdown_tf'] = self.render_BEV()

        result['cars']=cars
        return result

    def render_BEV(self):
        semantic_grid = self.global_map

        vehicle_position = self._vehicle.get_location()
        ego_pos_list = [self._vehicle.get_transform().location.x, self._vehicle.get_transform().location.y]
        ego_yaw_list = [self._vehicle.get_transform().rotation.yaw / 180 * np.pi]

        # fetch local birdview per agent
        ego_pos = torch.tensor([self._vehicle.get_transform().location.x, self._vehicle.get_transform().location.y],
                               device='cuda', dtype=torch.float32)
        ego_yaw = torch.tensor([self._vehicle.get_transform().rotation.yaw / 180 * np.pi], device='cuda',
                               dtype=torch.float32)
        birdview = self.renderer.get_local_birdview(
            semantic_grid,
            ego_pos,
            ego_yaw
        )

        self._actors = self._world.get_actors()
        vehicles = self._actors.filter('*vehicle*')
        for vehicle in vehicles:
            if (vehicle.get_location().distance(self._vehicle.get_location()) < self.detection_radius):
                if (vehicle.id != self._vehicle.id):
                    pos = torch.tensor([vehicle.get_transform().location.x, vehicle.get_transform().location.y],
                                       device='cuda', dtype=torch.float32)
                    yaw = torch.tensor([vehicle.get_transform().rotation.yaw / 180 * np.pi], device='cuda',
                                       dtype=torch.float32)
                    veh_x_extent = int(max(vehicle.bounding_box.extent.x * 2, 1) * PIXELS_PER_METER)
                    veh_y_extent = int(max(vehicle.bounding_box.extent.y * 2, 1) * PIXELS_PER_METER)

                    self.vehicle_template = torch.ones(1, 1, veh_x_extent, veh_y_extent, device='cuda')
                    self.renderer.render_agent_bv(
                        birdview,
                        ego_pos,
                        ego_yaw,
                        self.vehicle_template,
                        pos,
                        yaw,
                        channel=5
                    )

        ego_pos_batched = []
        ego_yaw_batched = []
        pos_batched = []
        yaw_batched = []
        template_batched = []
        channel_batched = []

        # -----------------------------------------------------------
        # Pedestrian rendering
        # -----------------------------------------------------------
        walkers = self._actors.filter('*walker*')
        for walker in walkers:
            ego_pos_batched.append(ego_pos_list)
            ego_yaw_batched.append(ego_yaw_list)
            pos_batched.append([walker.get_transform().location.x, walker.get_transform().location.y])
            yaw_batched.append([walker.get_transform().rotation.yaw / 180 * np.pi])
            channel_batched.append(6)
            template_batched.append(np.ones([20, 7]))

        if len(ego_pos_batched) > 0:
            ego_pos_batched_torch = torch.tensor(ego_pos_batched, device='cuda', dtype=torch.float32).unsqueeze(1)
            ego_yaw_batched_torch = torch.tensor(ego_yaw_batched, device='cuda', dtype=torch.float32).unsqueeze(1)
            pos_batched_torch = torch.tensor(pos_batched, device='cuda', dtype=torch.float32).unsqueeze(1)
            yaw_batched_torch = torch.tensor(yaw_batched, device='cuda', dtype=torch.float32).unsqueeze(1)
            # template_batched_torch = torch.tensor(template_batched, device='cuda', dtype=torch.float32).unsqueeze(1)
            template_batched_np = np.array(template_batched)
            template_batched_torch = torch.tensor(template_batched_np, device='cuda', dtype=torch.float32).unsqueeze(1)
            channel_batched_torch = torch.tensor(channel_batched, device='cuda', dtype=torch.float32)

            self.renderer.render_agent_bv_batched(
                birdview,
                ego_pos_batched_torch,
                ego_yaw_batched_torch,
                template_batched_torch,
                pos_batched_torch,
                yaw_batched_torch,
                channel=channel_batched_torch,
            )

        ego_pos_batched = []
        ego_yaw_batched = []
        pos_batched = []
        yaw_batched = []
        template_batched = []
        channel_batched = []

        # -----------------------------------------------------------
        # Traffic light rendering
        # -----------------------------------------------------------
        traffic_lights = self._actors.filter('*traffic_light*')
        for traffic_light in traffic_lights:
            trigger_box_global_pos = traffic_light.get_transform().transform(traffic_light.trigger_volume.location)
            trigger_box_global_pos = carla.Location(x=trigger_box_global_pos.x, y=trigger_box_global_pos.y,
                                                    z=trigger_box_global_pos.z)
            if (trigger_box_global_pos.distance(vehicle_position) > self.light_radius):
                continue
            ego_pos_batched.append(ego_pos_list)
            ego_yaw_batched.append(ego_yaw_list)
            pos_batched.append([traffic_light.get_transform().location.x, traffic_light.get_transform().location.y])
            yaw_batched.append([traffic_light.get_transform().rotation.yaw / 180 * np.pi])
            template_batched.append(np.ones([4, 4]))
            if str(traffic_light.state) == 'Green':
                channel_batched.append(4)
            elif str(traffic_light.state) == 'Yellow':
                channel_batched.append(3)
            elif str(traffic_light.state) == 'Red':
                channel_batched.append(2)

        if len(ego_pos_batched) > 0:
            ego_pos_batched_torch = torch.tensor(ego_pos_batched, device='cuda', dtype=torch.float32).unsqueeze(1)
            ego_yaw_batched_torch = torch.tensor(ego_yaw_batched, device='cuda', dtype=torch.float32).unsqueeze(1)
            pos_batched_torch = torch.tensor(pos_batched, device='cuda', dtype=torch.float32).unsqueeze(1)
            yaw_batched_torch = torch.tensor(yaw_batched, device='cuda', dtype=torch.float32).unsqueeze(1)
            # template_batched_torch = torch.tensor(template_batched, device='cuda', dtype=torch.float32).unsqueeze(1)
            template_batched_np = np.array(template_batched)
            template_batched_torch = torch.tensor(template_batched_np, device='cuda', dtype=torch.float32).unsqueeze(1)
            channel_batched_torch = torch.tensor(channel_batched, device='cuda', dtype=torch.int)

            self.renderer.render_agent_bv_batched(
                birdview,
                ego_pos_batched_torch,
                ego_yaw_batched_torch,
                template_batched_torch,
                pos_batched_torch,
                yaw_batched_torch,
                channel=channel_batched_torch,
            )

        return birdview

    def get_bev_cars(self, lidar=None):
        results = []
        ego_rotation = self._vehicle.get_transform().rotation
        ego_matrix = np.array(self._vehicle.get_transform().get_matrix())

        ego_extent = self._vehicle.bounding_box.extent
        ego_dx = np.array([ego_extent.x, ego_extent.y, ego_extent.z]) * 2.
        ego_yaw = ego_rotation.yaw / 180 * np.pi

        # also add ego box for visulization
        relative_yaw = 0
        relative_pos = self.get_relative_transform(ego_matrix, ego_matrix)

        # add vehicle velocity and brake flag
        ego_transform = self._vehicle.get_transform()
        ego_control = self._vehicle.get_control()
        ego_velocity = self._vehicle.get_velocity()
        ego_speed = self._get_forward_speed(transform=ego_transform, velocity=ego_velocity)  # In m/s
        ego_brake = ego_control.brake

        # the position is in lidar coordinates
        result = {"class": "Car",
                  "extent": [ego_dx[2], ego_dx[0], ego_dx[1]],  # NOTE: height stored in first dimension
                  "position": [relative_pos[0], relative_pos[1], relative_pos[2]],
                  "yaw": relative_yaw,
                  "num_points": -1,
                  "distance": -1,
                  "speed": ego_speed,
                  "brake": ego_brake,
                  "id": int(self._vehicle.id),
                  'ego_matrix': self._vehicle.get_transform().get_matrix()
                  }
        results.append(result)

        self._actors = self._world.get_actors()
        vehicles = self._actors.filter('*vehicle*')
        for vehicle in vehicles:
            if (vehicle.get_location().distance(self._vehicle.get_location()) < 50):
                if (vehicle.id != self._vehicle.id):
                    vehicle_rotation = vehicle.get_transform().rotation
                    vehicle_matrix = np.array(vehicle.get_transform().get_matrix())
                    vehicle_id = vehicle.id

                    vehicle_extent = vehicle.bounding_box.extent
                    dx = np.array([vehicle_extent.x, vehicle_extent.y, vehicle_extent.z]) * 2.
                    yaw = vehicle_rotation.yaw / 180 * np.pi

                    relative_yaw = yaw - ego_yaw
                    relative_pos = self.get_relative_transform(ego_matrix, vehicle_matrix)

                    vehicle_transform = vehicle.get_transform()
                    vehicle_control = vehicle.get_control()
                    vehicle_velocity = vehicle.get_velocity()
                    vehicle_speed = self._get_forward_speed(transform=vehicle_transform,
                                                            velocity=vehicle_velocity)  # In m/s
                    vehicle_brake = vehicle_control.brake

                    # filter bbox that didn't contains points of contains less points
                    if not lidar is None:
                        num_in_bbox_points = self.get_points_in_bbox(ego_matrix, vehicle_matrix, dx, lidar)
                    else:
                        num_in_bbox_points = -1

                    distance = np.linalg.norm(relative_pos)

                    result = {
                        "class": "Car",
                        "extent": [dx[2], dx[0], dx[1]],  # NOTE: height stored in first dimension
                        "position": [relative_pos[0], relative_pos[1], relative_pos[2]],
                        "yaw": relative_yaw,
                        "num_points": int(num_in_bbox_points),
                        "distance": distance,
                        "speed": vehicle_speed,
                        "brake": vehicle_brake,
                        "id": int(vehicle_id),
                        "ego_matrix": vehicle.get_transform().get_matrix()
                    }
                    results.append(result)

        return results

    def get_relative_transform(self, ego_matrix, vehicle_matrix):
        """
        return the relative transform from ego_pose to vehicle pose
        """
        relative_pos = vehicle_matrix[:3, 3] - ego_matrix[:3, 3]
        rot = ego_matrix[:3, :3].T
        relative_pos = rot @ relative_pos

        # transform to right handed system
        relative_pos[1] = - relative_pos[1]

        # transform relative pos to virtual lidar system
        rot = np.eye(3)
        trans = - np.array([1.3, 0.0, 2.5])
        relative_pos = rot @ relative_pos + trans

        return relative_pos

    def get_points_in_bbox(self, ego_matrix, vehicle_matrix, dx, lidar):
        # inverse transform
        Tr_lidar_2_ego = self.get_lidar_to_vehicle_transform()

        # construct transform from lidar to vehicle
        Tr_lidar_2_vehicle = np.linalg.inv(vehicle_matrix) @ ego_matrix @ Tr_lidar_2_ego

        # transform lidar to vehicle coordinate
        lidar_vehicle = Tr_lidar_2_vehicle[:3, :3] @ lidar[1][:, :3].T + Tr_lidar_2_vehicle[:3, 3:]

        # check points in bbox
        x, y, z = dx / 2.
        # why should we use swap?
        x, y = y, x
        num_points = ((lidar_vehicle[0] < x) & (lidar_vehicle[0] > -x) &
                      (lidar_vehicle[1] < y) & (lidar_vehicle[1] > -y) &
                      (lidar_vehicle[2] < z) & (lidar_vehicle[2] > -z)).sum()
        return num_points

    def get_lidar_to_vehicle_transform(self):
        # yaw = -90
        rot = np.array([[0, 1, 0],
                        [-1, 0, 0],
                        [0, 0, 1]], dtype=np.float32)
        T = np.eye(4)

        T[0, 3] = 1.3
        T[1, 3] = 0.0
        T[2, 3] = 2.5
        T[:3, :3] = rot
        return T


def get_nearby_lights(vehicle, lights, pixels_per_meter=5.5, size=512, radius=5):
    result = list()

    transform = vehicle.get_transform()
    pos = transform.location
    theta = np.radians(90 + transform.rotation.yaw)
    R = np.array(
        [
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)],
        ]
    )

    for light in lights:
        delta = light.get_transform().location - pos

        target = R.T.dot([delta.x, delta.y])
        target *= pixels_per_meter
        target += size // 2

        if min(target) < 0 or max(target) >= size:
            continue

        trigger = light.trigger_volume
        light.get_transform().transform(trigger.location)
        dist = trigger.location.distance(vehicle.get_location())
        a = np.sqrt(
            trigger.extent.x**2 + trigger.extent.y**2 + trigger.extent.z**2
        )
        b = np.sqrt(
            vehicle.bounding_box.extent.x**2
            + vehicle.bounding_box.extent.y**2
            + vehicle.bounding_box.extent.z**2
        )

        if dist > a + b:
            continue

        result.append(light)

    return result


def draw_traffic_lights(
    image, vehicle, lights, pixels_per_meter=5.5, size=512, radius=5
):
    image = Image.fromarray(image)
    draw = ImageDraw.Draw(image)

    transform = vehicle.get_transform()
    pos = transform.location
    theta = np.radians(90 + transform.rotation.yaw)
    R = np.array(
        [
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)],
        ]
    )

    for light in lights:
        delta = light.get_transform().location - pos

        target = R.T.dot([delta.x, delta.y])
        target *= pixels_per_meter
        target += size // 2

        if min(target) < 0 or max(target) >= size:
            continue

        trigger = light.trigger_volume
        light.get_transform().transform(trigger.location)
        dist = trigger.location.distance(vehicle.get_location())
        a = np.sqrt(
            trigger.extent.x**2 + trigger.extent.y**2 + trigger.extent.z**2
        )
        b = np.sqrt(
            vehicle.bounding_box.extent.x**2
            + vehicle.bounding_box.extent.y**2
            + vehicle.bounding_box.extent.z**2
        )

        if dist > a + b:
            continue

        x, y = target
        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius), 23 + light.state.real
        )  # 13 changed to 23 for carla 0.9.10

    return np.array(image)


def draw_stop_signs(image, vehicle, lights, pixels_per_meter=5.5, size=512, radius=5):
    image = Image.fromarray(image)
    draw = ImageDraw.Draw(image)

    transform = vehicle.get_transform()
    pos = transform.location
    theta = np.radians(90 + transform.rotation.yaw)
    R = np.array(
        [
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)],
        ]
    )

    for light in lights:
        delta = light.get_transform().location - pos

        target = R.T.dot([delta.x, delta.y])
        target *= pixels_per_meter
        target += size // 2

        if min(target) < 0 or max(target) >= size:
            continue

        trigger = light.trigger_volume
        light.get_transform().transform(trigger.location)
        dist = trigger.location.distance(vehicle.get_location())
        a = np.sqrt(
            trigger.extent.x**2 + trigger.extent.y**2 + trigger.extent.z**2
        )
        b = np.sqrt(
            vehicle.bounding_box.extent.x**2
            + vehicle.bounding_box.extent.y**2
            + vehicle.bounding_box.extent.z**2
        )

        if dist > a + b:
            continue

        x, y = target
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), 26)

    return np.array(image)
