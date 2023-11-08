import os
import time
import datetime
import pathlib
import json
import random
import shapely
import math
from collections import deque
from itertools import chain

import numpy as np
import cv2
import carla
from carla_birdeye_view import BirdViewProducer, BirdViewCropType, PixelDimensions
from PIL import Image

from team_code.map_agent import MapAgent
from team_code.pid_controller import PIDController
from agents.navigation.local_planner import RoadOption
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

from collections import deque, defaultdict
import statistics
from utils_tf.nav_planner import PIDController, RoutePlanner, interpolate_trajectory


WEATHERS = {
    "ClearNoon": carla.WeatherParameters.ClearNoon,
    "ClearSunset": carla.WeatherParameters.ClearSunset,
    "CloudyNoon": carla.WeatherParameters.CloudyNoon,
    "CloudySunset": carla.WeatherParameters.CloudySunset,
    "WetNoon": carla.WeatherParameters.WetNoon,
    "WetSunset": carla.WeatherParameters.WetSunset,
    "MidRainyNoon": carla.WeatherParameters.MidRainyNoon,
    "MidRainSunset": carla.WeatherParameters.MidRainSunset,
    "WetCloudyNoon": carla.WeatherParameters.WetCloudyNoon,
    "WetCloudySunset": carla.WeatherParameters.WetCloudySunset,
    "HardRainNoon": carla.WeatherParameters.HardRainNoon,
    "HardRainSunset": carla.WeatherParameters.HardRainSunset,
    "SoftRainNoon": carla.WeatherParameters.SoftRainNoon,
    "SoftRainSunset": carla.WeatherParameters.SoftRainSunset,
    "ClearNight": carla.WeatherParameters(5.0,0.0,0.0,10.0,-1.0,-90.0,60.0,75.0,1.0,0.0),
    "CloudyNight": carla.WeatherParameters(60.0,0.0,0.0,10.0,-1.0,-90.0,60.0,0.75,0.1,0.0),
    "WetNight": carla.WeatherParameters(5.0,0.0,50.0,10.0,-1.0,-90.0,60.0,75.0,1.0,60.0),
    "WetCloudyNight": carla.WeatherParameters(60.0,0.0,50.0,10.0,-1.0,-90.0,60.0,0.75,0.1,60.0),
    "SoftRainNight": carla.WeatherParameters(60.0,30.0,50.0,30.0,-1.0,-90.0,60.0,0.75,0.1,60.0),
    "MidRainyNight": carla.WeatherParameters(80.0,60.0,60.0,60.0,-1.0,-90.0,60.0,0.75,0.1,80.0),
    "HardRainNight": carla.WeatherParameters(100.0,100.0,90.0,100.0,-1.0,-90.0,100.0,0.75,0.1,100.0),
}
WEATHERS_IDS = list(WEATHERS)


def get_entry_point():
    return "AutoPilot"


def _numpy(carla_vector, normalize=False):
    result = np.float32([carla_vector.x, carla_vector.y])

    if normalize:
        return result / (np.linalg.norm(result) + 1e-4)

    return result


def _location(x, y, z):
    return carla.Location(x=float(x), y=float(y), z=float(z))


def _orientation(yaw):
    return np.float32([np.cos(np.radians(yaw)), np.sin(np.radians(yaw))])


def get_collision(p1, v1, p2, v2):
    A = np.stack([v1, -v2], 1)
    b = p2 - p1

    if abs(np.linalg.det(A)) < 1e-3:
        return False, None

    x = np.linalg.solve(A, b)
    collides = all(x >= 0) and all(x <= 1)  # how many seconds until collision

    return collides, p1 + x[0] * v1


def check_episode_has_noise(lat_noise_percent, long_noise_percent):
    lat_noise = False
    long_noise = False
    if random.randint(0, 101) < lat_noise_percent:
        lat_noise = True

    if random.randint(0, 101) < long_noise_percent:
        long_noise = True

    return lat_noise, long_noise


class AutoPilot(MapAgent):

    # for stop signs
    PROXIMITY_THRESHOLD = 30.0  # meters
    SPEED_THRESHOLD = 0.1
    WAYPOINT_STEP = 1.0  # meters

    def setup(self, path_to_conf_file):
        super().setup(path_to_conf_file)

        self.render_bev = False
        self.ignore_stop_signs = False
        self.target_speed = 4.0
        self.frame_rate_tf = 20  # Brame rate used by kinematic bicycle model for forecasting
        self.frame_rate_sim = 20  # CARLA framerate, used for visualizations
        self.extrapolation_seconds = 4.0  # Amount of seconds we look into the future to predict collisions at junctions
        self.angle                = 0.0   # Angle to the next waypoint. Normalized in [-1, 1] corresponding to [-90, 90]
        self.stop_sign_hazard     = False
        self.traffic_light_hazard = False
        self.walker_hazard        = [False for i in range(int(self.extrapolation_seconds * self.frame_rate_tf))]
        self.vehicle_hazard       = [False for i in range(int(self.extrapolation_seconds * self.frame_rate_tf))]
        self.ignore_stop_sign     = False
        self.cleared_stop_signs   = []                       #tf

        self.center_bb_light_x = -2.0
        self.center_bb_light_y = 0.0
        self.center_bb_light_z = 0.0

        self.extent_bb_light_x = 4.5
        self.extent_bb_light_y = 1.5
        self.extent_bb_light_z = 2.0

        # Obstacle detection
        self.extrapolation_seconds_no_junction = 1.0  # Amount of seconds we look into the future to predict collisions (>= 1 frame)
        self.extrapolation_seconds = 4.0  # Amount of seconds we look into the future to predict collisions at junctions
        self.waypoint_seconds = 4.0  # Amount of seconds we look into the future to store waypoint labels
        self.detection_radius = 30.0  # Distance of obstacles (in meters) in which we will check for collisions
        self.light_radius = 15.0  # Distance of traffic lights considered relevant (in meters)

        # Speed buffer for detecting "stuck" vehicles
        self.vehicle_speed_buffer = defaultdict(lambda: {"velocity": [], "throttle": [], "brake": []})
        self.stuck_buffer_size = 30
        self.stuck_vel_threshold = 0.1
        self.stuck_throttle_threshold = 0.1
        self.stuck_brake_threshold = 0.1

        self.vehicle_model = EgoModel(dt=(1.0 / self.frame_rate_tf))
        self.ego_model     = EgoModel(dt=(1.0 / self.frame_rate_tf))
        self.ego_model_gps = EgoModel(dt=(1.0 / self.frame_rate_sim))

        self._waypoint_planner_extrapolation = RoutePlanner(3.5, 50)
        self._turn_controller_extrapolation = PIDController(K_P=1.25, K_I=0.75, K_D=0.3, n=40)
        self._speed_controller_extrapolation = PIDController(K_P=5.0, K_I=0.5, K_D=1.0, n=40)
        # Initialize controls
        self.steer_tf = 0.0
        self.throttle_tf = 0.0
        self.brake_tf = 0.0
        self.angle_search_range = 0
        self.clip_delta = 0.25  # Max angular error for turn controller
        self.clip_throttle = 0.75  # Max throttle (0-1)
        self.junction = False

    def _init(self,hd_map):
        super()._init(hd_map)

        self._turn_controller = PIDController(K_P=1.25, K_I=0.75, K_D=0.3, n=40)
        self._speed_controller = PIDController(K_P=5.0, K_I=0.5, K_D=1.0, n=40)

        # for stop signs
        self._target_stop_sign = None  # the stop sign affecting the ego vehicle
        self._stop_completed = False  # if the ego vehicle has completed the stop sign
        self._affected_by_stop = (
            False  # if the ego vehicle is influenced by a stop sign
        )

        lights_list = self._world.get_actors().filter("*traffic_light*")
        self._map = self._world.get_map()
        self._list_traffic_lights = []
        for light in lights_list:
            center, waypoints = self.get_traffic_light_waypoints(light)
            self._list_traffic_lights.append((light, center, waypoints))
        (
            self._list_traffic_waypoints,
            self._dict_traffic_lights,
        ) = self._gen_traffic_light_dict(self._list_traffic_lights)

        if self.weather_id is not None:
            weather = WEATHERS[WEATHERS_IDS[self.weather_id]]
            self._world.set_weather(weather)

        if self.destory_hazard_actors:
            self.hazard_actors_dict = {}

        # disturb waypoint
        if self.waypoint_disturb > 0.01:
            print("Setup waypoint disturb!")
            updated_route = self.disturb_waypoints(self._waypoint_planner.route)
            self._waypoint_planner.route = updated_route

        self.birdview_producer = BirdViewProducer(
            CarlaDataProvider.get_client(),  # carla.Client
            target_size=PixelDimensions(width=400, height=400),
            pixels_per_meter=4,
            crop_type=BirdViewCropType.FRONT_AND_REAR_AREA,
        )

        self.world_map = carla.Map("RouteMap", hd_map[1]['opendrive'])
        trajectory = [item[0].location for item in self._global_plan_world_coord]
        self.dense_route, _ = interpolate_trajectory(self.world_map, trajectory)
        self._waypoint_planner_extrapolation.set_route(self.dense_route, True)
        self._waypoint_planner_extrapolation.save()
    def disturb_waypoints(self, route):
        updated_route = deque()
        np.random.seed(self.waypoint_disturb_seed)
        for pos, command in route:
            if command == RoadOption.LANEFOLLOW or command == RoadOption.STRAIGHT:
                waypoint = self._map.get_waypoint(carla.Location(pos[1], -pos[0]))
                if np.random.random() < 0.5:
                    loc = self.rotate_point(
                        carla.Vector3D(
                            self.waypoint_disturb * np.random.random() * 0.8,
                            0.0,
                            waypoint.transform.location.z,
                        ),
                        waypoint.transform.rotation.yaw - 90,
                    )
                else:
                    loc = self.rotate_point(
                        carla.Vector3D(
                            self.waypoint_disturb * np.random.random() * 0.8,
                            0.0,
                            waypoint.transform.location.z,
                        ),
                        waypoint.transform.rotation.yaw + 90,
                    )
                loc = loc + waypoint.transform.location
                pos = np.array([-loc.y, loc.x])
            updated_route.append((pos, command))
        return updated_route

    def _get_angle_to(self, pos, theta, target):
        R = np.array(
            [
                [np.cos(theta), -np.sin(theta)],
                [np.sin(theta), np.cos(theta)],
            ]
        )

        aim = R.T.dot(target - pos)
        angle = -np.degrees(np.arctan2(-aim[1], aim[0]))
        angle = 0.0 if np.isnan(angle) else angle

        return angle

    def _get_control(self, target, far_target, near_command, far_command, tick_data):
        pos = self._get_position(tick_data)
        theta = tick_data["compass"]
        speed = tick_data["speed"]

        # Steering.
        angle_unnorm = self._get_angle_to(pos, theta, target)
        angle = angle_unnorm / 90
        self.angle=angle

        steer = self._turn_controller.step(angle)
        steer = np.clip(steer, -1.0, 1.0)
        steer = round(steer, 3)

        # Acceleration.
        angle_far_unnorm = self._get_angle_to(pos, theta, far_target)
        should_slow = abs(angle_far_unnorm) > 45.0 or abs(angle_unnorm) > 5.0
        self.should_slow = should_slow
        target_speed = 4.0 if should_slow else 6.5
        brake = self._should_brake(near_command)
        self.brake_tf=self._get_brake()
        self.should_brake = brake
        target_speed = target_speed if not brake else 0.0

        delta = np.clip(target_speed - speed, 0.0, 0.25)
        throttle = self._speed_controller.step(delta)
        throttle = np.clip(throttle, 0.0, 0.75)

        if brake:
            steer *= 0.5
            throttle = 0.0
        pos_tf = self._get_position(tick_data['gps'][1][:2])
        pos_tf = np.average(self.gps_buffer, axis=0)  # Denoised position
        self._waypoint_planner_extrapolation.load()
        self.waypoint_route_extrapolation = self._waypoint_planner_extrapolation.run_step(pos_tf)
        self._waypoint_planner_extrapolation.save()  #tf

        return steer, throttle, brake, target_speed

    def run_step(self, input_data, timestamp):
        if not self.initialized:
            if ('hd_map' in input_data.keys()):
                self._init(input_data['hd_map'])
            #self._init()


        data = self.tick(input_data)
        gps = self._get_position(data)
        loc = self._vehicle.get_location()
        self._loc = [loc.x, loc.y]
        near_node, near_command = self._waypoint_planner.run_step(gps)
        far_node, far_command = self._command_planner.run_step(gps)

        steer, throttle, brake, target_speed = self._get_control(
            near_node, far_node, near_command, far_command, data
        )
        self.is_junction = self._map.get_waypoint(
            self._vehicle.get_location()
        ).is_junction
        self.actors_data = self.collect_actor_data()

        light = self._find_closest_valid_traffic_light(
            self._vehicle.get_location(), min_dis=50
        )
        if light is not None:
            self.affected_light_id = light.id
        else:
            self.affected_light_id = -1

        control = carla.VehicleControl()
        control.steer = steer + 1e-2 * np.random.randn()
        control.throttle = throttle
        control.brake = float(brake)

        self.birdview = BirdViewProducer.as_rgb(
            self.birdview_producer.produce(agent_vehicle=self._vehicle)
        )

        self.stop_sign_hazard = True if len(self.is_stop_sign_present) > 0 else False
        self.traffic_light_hazard = True if len(self.is_red_light_present) > 0 else False
        self.junction = self.is_junction


        if self.step % self.save_skip_frames == 0 and self.save_path is not None:
            self.save(
                near_node,
                far_node,
                near_command,
                steer,
                throttle,
                brake,
                target_speed,
                data,
            )
        if self.destory_hazard_actors:
            self._detect_and_destory_hazard_actors()

        return control

    def collect_actor_data(self):
        data = {}
        vehicles = self._world.get_actors().filter("*vehicle*")
        for actor in vehicles:
            loc = actor.get_location()
            if loc.distance(self._vehicle.get_location()) > 50:
                continue
            _id = actor.id
            data[_id] = {}
            data[_id]["loc"] = [loc.x, loc.y, loc.z]
            ori = actor.get_transform().rotation.get_forward_vector()
            data[_id]["ori"] = [ori.x, ori.y, ori.z]
            box = actor.bounding_box.extent
            data[_id]["box"] = [box.x, box.y]
            vel = actor.get_velocity()
            data[_id]["vel"] = [vel.x, vel.y, vel.z]
            data[_id]["tpe"] = 0

        walkers = self._world.get_actors().filter("*walker*")
        for actor in walkers:
            loc = actor.get_location()
            if loc.distance(self._vehicle.get_location()) > 50:
                continue
            _id = actor.id
            data[_id] = {}
            data[_id]["loc"] = [loc.x, loc.y, loc.z]
            ori = actor.get_transform().rotation.get_forward_vector()
            data[_id]["ori"] = [ori.x, ori.y, ori.z]
            box = actor.bounding_box.extent
            data[_id]["box"] = [box.x, box.y]
            vel = actor.get_velocity()
            data[_id]["vel"] = [vel.x, vel.y, vel.z]
            data[_id]["tpe"] = 1

        lights = self._world.get_actors().filter("*traffic_light*")
        for actor in lights:
            loc = actor.get_location()
            if loc.distance(self._vehicle.get_location()) > 70:
                continue
            _id = actor.id
            data[_id] = {}
            data[_id]["loc"] = [loc.x, loc.y, loc.z]
            ori = actor.get_transform().rotation.get_forward_vector()
            data[_id]["ori"] = [ori.x, ori.y, ori.z]
            vel = actor.get_velocity()
            data[_id]["sta"] = int(actor.state)
            data[_id]["tpe"] = 2

            trigger = actor.trigger_volume
            box = trigger.extent
            loc = trigger.location
            ori = trigger.rotation.get_forward_vector()
            data[_id]["taigger_loc"] = [loc.x, loc.y, loc.z]
            data[_id]["trigger_ori"] = [ori.x, ori.y, ori.z]
            data[_id]["trigger_box"] = [box.x, box.y]
        return data

    def _should_brake(self, command):
        actors = self._world.get_actors()

        vehicle = self._is_vehicle_hazard(actors.filter("*vehicle*"), command)
        lane_vehicle = self._is_lane_vehicle_hazard(actors.filter("*vehicle*"), command)
        junction_vehicle = self._is_junction_vehicle_hazard(
            actors.filter("*vehicle*"), command
        )
        light = self._is_light_red(actors.filter("*traffic_light*"))
        walker = self._is_walker_hazard(actors.filter("*walker*"))
        bike = self._is_bike_hazard(actors.filter("*vehicle*"))
        stop_sign = self._is_stop_sign_hazard(actors.filter("*stop*"))

        # record the reason for braking
        self.is_vehicle_present = [x.id for x in vehicle]
        self.is_lane_vehicle_present = [x.id for x in lane_vehicle]
        self.is_junction_vehicle_present = [x.id for x in junction_vehicle]
        self.is_pedestrian_present = [x.id for x in walker]
        self.is_bike_present = [x.id for x in bike]
        self.is_red_light_present = [x.id for x in light]
        self.is_stop_sign_present = [x.id for x in stop_sign]




        return any(
            len(x) > 0
            for x in [
                vehicle,
                lane_vehicle,
                junction_vehicle,
                bike,
                light,
                walker,
                stop_sign,
            ]
        )

    def _point_inside_boundingbox(self, point, bb_center, bb_extent):
        A = carla.Vector2D(bb_center.x - bb_extent.x, bb_center.y - bb_extent.y)
        B = carla.Vector2D(bb_center.x + bb_extent.x, bb_center.y - bb_extent.y)
        D = carla.Vector2D(bb_center.x - bb_extent.x, bb_center.y + bb_extent.y)
        M = carla.Vector2D(point.x, point.y)

        AB = B - A
        AD = D - A
        AM = M - A
        am_ab = AM.x * AB.x + AM.y * AB.y
        ab_ab = AB.x * AB.x + AB.y * AB.y
        am_ad = AM.x * AD.x + AM.y * AD.y
        ad_ad = AD.x * AD.x + AD.y * AD.y

        return am_ab > 0 and am_ab < ab_ab and am_ad > 0 and am_ad < ad_ad

    def _get_forward_speed(self, transform=None, velocity=None):
        """Convert the vehicle transform directly to forward speed"""
        if not velocity:
            velocity = self._vehicle.get_velocity()
        if not transform:
            transform = self._vehicle.get_transform()

        vel_np = np.array([velocity.x, velocity.y, velocity.z])
        pitch = np.deg2rad(transform.rotation.pitch)
        yaw = np.deg2rad(transform.rotation.yaw)
        orientation = np.array(
            [np.cos(pitch) * np.cos(yaw), np.cos(pitch) * np.sin(yaw), np.sin(pitch)]
        )
        speed = np.dot(vel_np, orientation)
        return speed

    def _is_actor_affected_by_stop(self, actor, stop, multi_step=20):
        """
        Check if the given actor is affected by the stop
        """
        affected = False
        # first we run a fast coarse test
        current_location = actor.get_location()
        stop_location = stop.get_transform().location
        if stop_location.distance(current_location) > self.PROXIMITY_THRESHOLD:
            return affected

        stop_t = stop.get_transform()
        transformed_tv = stop_t.transform(stop.trigger_volume.location)

        # slower and accurate test based on waypoint's horizon and geometric test
        list_locations = [current_location]
        waypoint = self._map.get_waypoint(current_location)
        for _ in range(multi_step):
            if waypoint:
                waypoint = waypoint.next(self.WAYPOINT_STEP)[0]
                if not waypoint:
                    break
                list_locations.append(waypoint.transform.location)

        for actor_location in list_locations:
            if self._point_inside_boundingbox(
                actor_location, transformed_tv, stop.trigger_volume.extent
            ):
                affected = True

        return affected

    def _is_junction_vehicle_hazard(self, vehicle_list, command):
        res = []
        o1 = _orientation(self._vehicle.get_transform().rotation.yaw)
        x1 = self._vehicle.bounding_box.extent.x
        p1 = (
            self._vehicle.get_location()
            + x1 * self._vehicle.get_transform().get_forward_vector()
        )
        w1 = self._map.get_waypoint(p1)
        s1 = np.linalg.norm(_numpy(self._vehicle.get_velocity()))
        if command == RoadOption.RIGHT:
            shift_angle = 25
        elif command == RoadOption.LEFT:
            shift_angle = -25
        else:
            shift_angle = 0
        v1 = (4 * s1 + 5) * _orientation(
            self._vehicle.get_transform().rotation.yaw + shift_angle
        )

        for target_vehicle in vehicle_list:
            if target_vehicle.id == self._vehicle.id:
                continue

            o2 = _orientation(target_vehicle.get_transform().rotation.yaw)
            o2_left = _orientation(target_vehicle.get_transform().rotation.yaw - 15)
            o2_right = _orientation(target_vehicle.get_transform().rotation.yaw + 15)
            x2 = target_vehicle.bounding_box.extent.x

            p2 = target_vehicle.get_location()
            p2_hat = p2 - (x2 + 2) * target_vehicle.get_transform().get_forward_vector()
            w2 = self._map.get_waypoint(p2)
            s2 = np.linalg.norm(_numpy(target_vehicle.get_velocity()))

            v2 = (4 * s2 + 2 * x2 + 6) * o2
            v2_left = (4 * s2 + 2 * x2 + 6) * o2_left
            v2_right = (4 * s2 + 2 * x2 + 6) * o2_right

            angle_between_heading = np.degrees(np.arccos(np.clip(o1.dot(o2), -1, 1)))

            if self._vehicle.get_location().distance(p2) > 20:
                continue
            if w1.is_junction == False and w2.is_junction == False:
                continue
            if angle_between_heading < 15.0 or angle_between_heading > 165:
                continue
            collides, collision_point = get_collision(
                _numpy(p1), v1, _numpy(p2_hat), v2
            )
            if collides is None:
                collides, collision_point = get_collision(
                    _numpy(p1), v1, _numpy(p2_hat), v2_left
                )
            if collides is None:
                collides, collision_point = get_collision(
                    _numpy(p1), v1, _numpy(p2_hat), v2_right
                )

            light = self._find_closest_valid_traffic_light(
                target_vehicle.get_location(), min_dis=10
            )
            if (
                light is not None
                and light.state != carla.libcarla.TrafficLightState.Green
            ):
                continue
            if collides:
                res.append(target_vehicle)
        return res

    def _is_lane_vehicle_hazard(self, vehicle_list, command):
        res = []
        if (
            command != RoadOption.CHANGELANELEFT
            and command != RoadOption.CHANGELANERIGHT
        ):
            return []

        z = self._vehicle.get_location().z
        w1 = self._map.get_waypoint(self._vehicle.get_location())
        o1 = _orientation(self._vehicle.get_transform().rotation.yaw)
        p1 = self._vehicle.get_location()

        yaw_w1 = w1.transform.rotation.yaw
        lane_width = w1.lane_width
        location_w1 = w1.transform.location

        lft_shift = 0.5
        rgt_shift = 0.5
        if command == RoadOption.CHANGELANELEFT:
            rgt_shift += 1
        else:
            lft_shift += 1

        lft_lane_wp = self.rotate_point(
            carla.Vector3D(lft_shift * lane_width, 0.0, location_w1.z), yaw_w1 + 90
        )
        lft_lane_wp = location_w1 + carla.Location(lft_lane_wp)
        rgt_lane_wp = self.rotate_point(
            carla.Vector3D(rgt_shift * lane_width, 0.0, location_w1.z), yaw_w1 - 90
        )
        rgt_lane_wp = location_w1 + carla.Location(rgt_lane_wp)

        for target_vehicle in vehicle_list:
            if target_vehicle.id == self._vehicle.id:
                continue

            w2 = self._map.get_waypoint(target_vehicle.get_location())
            o2 = _orientation(target_vehicle.get_transform().rotation.yaw)
            p2 = target_vehicle.get_location()
            x2 = target_vehicle.bounding_box.extent.x
            p2_hat = p2 - target_vehicle.get_transform().get_forward_vector() * x2 * 2
            s2 = (
                target_vehicle.get_velocity()
                + target_vehicle.get_transform().get_forward_vector() * x2
            )
            s2_value = max(
                12,
                2
                + 2 * x2
                + 3.0 * np.linalg.norm(_numpy(target_vehicle.get_velocity())),
            )

            distance = p1.distance(p2)

            if distance > s2_value:
                continue
            if w1.road_id != w2.road_id or w1.lane_id * w2.lane_id < 0:
                continue
            if command == RoadOption.CHANGELANELEFT:
                if w1.lane_id > 0:
                    if w2.lane_id != w1.lane_id - 1:
                        continue
                if w1.lane_id < 0:
                    if w2.lane_id != w1.lane_id + 1:
                        continue
            if command == RoadOption.CHANGELANERIGHT:
                if w1.lane_id > 0:
                    if w2.lane_id != w1.lane_id + 1:
                        continue
                if w1.lane_id < 0:
                    if w2.lane_id != w1.lane_id - 1:
                        continue

            if self._are_vehicles_crossing_future(p2_hat, s2, lft_lane_wp, rgt_lane_wp):
                res.append(target_vehicle)
        return res

    def _are_vehicles_crossing_future(self, p1, s1, lft_lane, rgt_lane):
        p1_hat = carla.Location(x=p1.x + 3 * s1.x, y=p1.y + 3 * s1.y)
        line1 = shapely.geometry.LineString([(p1.x, p1.y), (p1_hat.x, p1_hat.y)])
        line2 = shapely.geometry.LineString(
            [(lft_lane.x, lft_lane.y), (rgt_lane.x, rgt_lane.y)]
        )
        inter = line1.intersection(line2)
        return not inter.is_empty

    def _is_stop_sign_hazard(self, stop_sign_list):
        res = []
        if self._affected_by_stop:
            if not self._stop_completed:
                current_speed = self._get_forward_speed()
                if current_speed < self.SPEED_THRESHOLD:
                    self._stop_completed = True
                    return res
                else:
                    return [self._target_stop_sign]
            else:
                # reset if the ego vehicle is outside the influence of the current stop sign
                if not self._is_actor_affected_by_stop(
                    self._vehicle, self._target_stop_sign
                ):
                    self._affected_by_stop = False
                    self._stop_completed = False
                    self._target_stop_sign = None
                return res

        ve_tra = self._vehicle.get_transform()
        ve_dir = ve_tra.get_forward_vector()

        wp = self._map.get_waypoint(ve_tra.location)
        wp_dir = wp.transform.get_forward_vector()

        dot_ve_wp = ve_dir.x * wp_dir.x + ve_dir.y * wp_dir.y + ve_dir.z * wp_dir.z

        if dot_ve_wp > 0:  # Ignore all when going in a wrong lane
            for stop_sign in stop_sign_list:
                if self._is_actor_affected_by_stop(self._vehicle, stop_sign):
                    # this stop sign is affecting the vehicle
                    self._affected_by_stop = True
                    self._target_stop_sign = stop_sign
                    res.append(self._target_stop_sign)

        return res

    def _is_light_red(self, lights_list):
        if (
            self._vehicle.get_traffic_light_state()
            != carla.libcarla.TrafficLightState.Green
        ):
            affecting = self._vehicle.get_traffic_light()

            for light in self._traffic_lights:
                if light.id == affecting.id:
                    return [light]

        light = self._find_closest_valid_traffic_light(
            self._vehicle.get_location(), min_dis=8
        )
        if light is not None and light.state != carla.libcarla.TrafficLightState.Green:
            return [light]
        return []

    def _find_closest_valid_traffic_light(self, loc, min_dis):
        wp = self._map.get_waypoint(loc)
        min_wp = None
        min_distance = min_dis
        for waypoint in self._list_traffic_waypoints:
            if waypoint.road_id != wp.road_id or waypoint.lane_id * wp.lane_id < 0:
                continue
            dis = loc.distance(waypoint.transform.location)
            if dis <= min_distance:
                min_distance = dis
                min_wp = waypoint
        if min_wp is None:
            return None
        else:
            return self._dict_traffic_lights[min_wp][0]

    def get_traffic_light_waypoints(self, traffic_light):
        base_transform = traffic_light.get_transform()
        base_rot = base_transform.rotation.yaw
        area_loc = base_transform.transform(traffic_light.trigger_volume.location)

        # Discretize the trigger box into points
        area_ext = traffic_light.trigger_volume.extent
        x_values = np.arange(
            -0.9 * area_ext.x, 0.9 * area_ext.x, 1.0
        )  # 0.9 to avoid crossing to adjacent lanes

        area = []
        for x in x_values:
            point = self.rotate_point(carla.Vector3D(x, 0, area_ext.z), base_rot)
            point_location = area_loc + carla.Location(x=point.x, y=point.y)
            area.append(point_location)

        # Get the waypoints of these points, removing duplicates
        ini_wps = []
        for pt in area:
            wpx = self._map.get_waypoint(pt)
            # As x_values are arranged in order, only the last one has to be checked
            if (
                not ini_wps
                or ini_wps[-1].road_id != wpx.road_id
                or ini_wps[-1].lane_id != wpx.lane_id
            ):
                ini_wps.append(wpx)

        # Advance them until the intersection
        wps = []
        for wpx in ini_wps:
            while not wpx.is_intersection:
                next_wp = wpx.next(0.5)[0]
                if next_wp and not next_wp.is_intersection:
                    wpx = next_wp
                else:
                    break
            wps.append(wpx)

        return area_loc, wps

    def _gen_traffic_light_dict(self, traffic_lights_list):
        traffic_light_dict = {}
        waypoints_list = []
        for light, center, waypoints in traffic_lights_list:
            for waypoint in waypoints:
                traffic_light_dict[waypoint] = (light, center)
                waypoints_list.append(waypoint)
        return waypoints_list, traffic_light_dict

    def _is_walker_hazard(self, walkers_list):
        res = []
        p1 = _numpy(self._vehicle.get_location())
        v1 = 10.0 * _orientation(self._vehicle.get_transform().rotation.yaw)

        for walker in walkers_list:
            v2_hat = _orientation(walker.get_transform().rotation.yaw)
            s2 = np.linalg.norm(_numpy(walker.get_velocity()))

            if s2 < 0.05:
                v2_hat *= s2

            p2 = -3.0 * v2_hat + _numpy(walker.get_location())
            v2 = 8.0 * v2_hat

            collides, collision_point = get_collision(p1, v1, p2, v2)

            if collides:
                res.append(walker)

        return res

    def _is_bike_hazard(self, bikes_list):
        res = []
        o1 = _orientation(self._vehicle.get_transform().rotation.yaw)
        v1_hat = o1
        p1 = _numpy(self._vehicle.get_location())
        v1 = 10.0 * o1

        for bike in bikes_list:
            o2 = _orientation(bike.get_transform().rotation.yaw)
            s2 = np.linalg.norm(_numpy(bike.get_velocity()))
            v2_hat = o2
            p2 = _numpy(bike.get_location())

            p2_p1 = p2 - p1
            distance = np.linalg.norm(p2_p1)
            p2_p1_hat = p2_p1 / (distance + 1e-4)

            angle_to_car = np.degrees(np.arccos(np.clip(v1_hat.dot(p2_p1_hat), -1, 1)))
            angle_between_heading = np.degrees(np.arccos(np.clip(o1.dot(o2), -1, 1)))

            # to consider -ve angles too
            angle_to_car = min(angle_to_car, 360.0 - angle_to_car)
            angle_between_heading = min(
                angle_between_heading, 360.0 - angle_between_heading
            )
            if distance > 20:
                continue
            if angle_to_car > 30:
                continue
            if angle_between_heading < 80 and angle_between_heading > 100:
                continue

            p2_hat = -2.0 * v2_hat + _numpy(bike.get_location())
            v2 = 7.0 * v2_hat

            collides, collision_point = get_collision(p1, v1, p2_hat, v2)

            if collides:
                res.append(bike)

        return res

    def _is_vehicle_hazard(self, vehicle_list, command):
        res = []
        z = self._vehicle.get_location().z

        o1 = _orientation(self._vehicle.get_transform().rotation.yaw)
        p1 = _numpy(self._vehicle.get_location())
        s1 = max(
            10, 3.0 * np.linalg.norm(_numpy(self._vehicle.get_velocity()))
        )  # increases the threshold distance
        s1a = np.linalg.norm(_numpy(self._vehicle.get_velocity()))
        w1 = self._map.get_waypoint(self._vehicle.get_location())
        v1_hat = o1
        v1 = s1 * v1_hat

        for target_vehicle in vehicle_list:
            if target_vehicle.id == self._vehicle.id:
                continue
            if not target_vehicle.is_alive:
                continue

            o2 = _orientation(target_vehicle.get_transform().rotation.yaw)
            p2 = _numpy(target_vehicle.get_location())
            s2 = max(5.0, 2.0 * np.linalg.norm(_numpy(target_vehicle.get_velocity())))
            s2a = np.linalg.norm(_numpy(target_vehicle.get_velocity()))
            w2 = self._map.get_waypoint(target_vehicle.get_location())
            v2_hat = o2
            v2 = s2 * v2_hat

            p2_p1 = p2 - p1
            distance = np.linalg.norm(p2_p1)
            p2_p1_hat = p2_p1 / (distance + 1e-4)

            angle_to_car = np.degrees(np.arccos(np.clip(v1_hat.dot(p2_p1_hat), -1, 1)))
            angle_between_heading = np.degrees(np.arccos(np.clip(o1.dot(o2), -1, 1)))

            # to consider -ve angles too
            angle_to_car = min(angle_to_car, 360.0 - angle_to_car)
            angle_between_heading = min(
                angle_between_heading, 360.0 - angle_between_heading
            )

            if (
                not w2.is_junction
                and angle_between_heading > 45.0
                and s2a < 0.5
                and distance > 4
            ):
                if w1.road_id != w2.road_id:
                    continue
            if (angle_between_heading < 15
                and w1.road_id == w2.road_id
                and w1.lane_id != w2.lane_id
                and command != RoadOption.CHANGELANELEFT
                and command != RoadOption.CHANGELANERIGHT
            ):
                continue

            if angle_between_heading > 60.0 and not (
                angle_to_car < 15 and distance < s1
            ):
                continue
            elif angle_to_car > 30.0:
                continue
            elif distance > s1:
                continue

            res.append(target_vehicle)

        return res

    def _detect_and_destory_hazard_actors(self):
        actors = self._world.get_actors()
        vehicles = actors.filter("*vehicle*")
        walkers = actors.filter("*walkers*")
        for actor in chain(vehicles, walkers):
            if actor.id == self._vehicle.id:
                continue
            s = np.linalg.norm(_numpy(actor.get_velocity()))
            if actor.id not in self.hazard_actors_dict:
                self.hazard_actors_dict[actor.id] = 0
            if s < 0.05:
                self.hazard_actors_dict[actor.id] += 1
            else:
                self.hazard_actors_dict[actor.id] = 0
        for actor_id in self.hazard_actors_dict:
            if self.hazard_actors_dict[actor_id] > 1500:
                actor = actors.find(actor_id)
                print(
                    "destory hazard actor: id: %d, type: %s" % (actor.id, actor.type_id)
                )
                actor.destroy()
                self.hazard_actors_dict[actor_id] = 0

    def rotate_point(self, point, angle):
        """
        rotate a given point by a given angle
        """
        x_ = (
            math.cos(math.radians(angle)) * point.x
            - math.sin(math.radians(angle)) * point.y
        )
        y_ = (
            math.sin(math.radians(angle)) * point.x
            + math.cos(math.radians(angle)) * point.y
        )
        return carla.Vector3D(x_, y_, point.z)

    def _get_steer_extrapolation(self, route, pos, theta, speed, restore=True):
        if self._waypoint_planner_extrapolation.is_last: # end of route
            angle = 0.0

        if len(route) == 1:
            target = route[0][0]
            angle_unnorm = self._get_angle_to(pos, theta, target)
            angle = angle_unnorm / 90
        elif self.angle_search_range <= 2:
            target = route[1][0]
            angle_unnorm = self._get_angle_to(pos, theta, target)
            angle = angle_unnorm / 90
        else:
            search_range = min([len(route), self.angle_search_range])
            for i in range(1, search_range):
                target = route[i][0]
                angle_unnorm = self._get_angle_to(pos, theta, target)
                angle_new = angle_unnorm / 90
                if i==1:
                    angle = angle_new
                if np.abs(angle_new) < np.abs(angle):
                    angle = angle_new

        self.angle_tf = angle

        if restore: self._turn_controller_extrapolation.load()
        steer = self._turn_controller_extrapolation.step(angle)
        if restore: self._turn_controller_extrapolation.save()

        steer = np.clip(steer, -1.0, 1.0)
        steer = round(steer, 3)

        return steer

    def _get_throttle_extrapolation(self, target_speed, speed, restore=True):
        if self._waypoint_planner_extrapolation.is_last: # end of route
            target_speed = 0.0

        delta = np.clip(target_speed - speed, 0.0, self.clip_delta)

        if restore: self._speed_controller_extrapolation.load()
        throttle = self._speed_controller_extrapolation.step(delta)
        if restore: self._speed_controller_extrapolation.save()

        throttle = np.clip(throttle, 0.0, self.clip_throttle)

        return throttle

    def get_nearby_object(self, vehicle_position, actor_list, radius):
        nearby_objects = []
        for actor in actor_list:
            trigger_box_global_pos = actor.get_transform().transform(actor.trigger_volume.location)
            trigger_box_global_pos = carla.Location(x=trigger_box_global_pos.x, y=trigger_box_global_pos.y, z=trigger_box_global_pos.z)
            if (trigger_box_global_pos.distance(vehicle_position) < radius):
                nearby_objects.append(actor)
        return nearby_objects

    def dot_product(self, vector1, vector2):
        return (vector1.x * vector2.x + vector1.y * vector2.y + vector1.z * vector2.z)

    def cross_product(self, vector1, vector2):
        return carla.Vector3D(x=vector1.y * vector2.z - vector1.z * vector2.y,
                              y=vector1.z * vector2.x - vector1.x * vector2.z,
                              z=vector1.x * vector2.y - vector1.y * vector2.x)

    def get_separating_plane(self, rPos, plane, obb1, obb2):
        ''' Checks if there is a seperating plane
        rPos Vec3
        plane Vec3
        obb1  Bounding Box
        obb2 Bounding Box
        '''
        return (abs(self.dot_product(rPos, plane)) > (
                    abs(self.dot_product((obb1.rotation.get_forward_vector() * obb1.extent.x), plane)) +
                    abs(self.dot_product((obb1.rotation.get_right_vector() * obb1.extent.y), plane)) +
                    abs(self.dot_product((obb1.rotation.get_up_vector() * obb1.extent.z), plane)) +
                    abs(self.dot_product((obb2.rotation.get_forward_vector() * obb2.extent.x), plane)) +
                    abs(self.dot_product((obb2.rotation.get_right_vector() * obb2.extent.y), plane)) +
                    abs(self.dot_product((obb2.rotation.get_up_vector() * obb2.extent.z), plane)))
                )

    def check_obb_intersection(self, obb1, obb2):
        RPos = obb2.location - obb1.location
        return not (self.get_separating_plane(RPos, obb1.rotation.get_forward_vector(), obb1, obb2) or
                    self.get_separating_plane(RPos, obb1.rotation.get_right_vector(), obb1, obb2) or
                    self.get_separating_plane(RPos, obb1.rotation.get_up_vector(), obb1, obb2) or
                    self.get_separating_plane(RPos, obb2.rotation.get_forward_vector(), obb1, obb2) or
                    self.get_separating_plane(RPos, obb2.rotation.get_right_vector(), obb1, obb2) or
                    self.get_separating_plane(RPos, obb2.rotation.get_up_vector(), obb1, obb2) or
                    self.get_separating_plane(RPos, self.cross_product(obb1.rotation.get_forward_vector(),
                                                                       obb2.rotation.get_forward_vector()), obb1,
                                              obb2) or
                    self.get_separating_plane(RPos, self.cross_product(obb1.rotation.get_forward_vector(),
                                                                       obb2.rotation.get_right_vector()), obb1, obb2) or
                    self.get_separating_plane(RPos, self.cross_product(obb1.rotation.get_forward_vector(),
                                                                       obb2.rotation.get_up_vector()), obb1, obb2) or
                    self.get_separating_plane(RPos, self.cross_product(obb1.rotation.get_right_vector(),
                                                                       obb2.rotation.get_forward_vector()), obb1,
                                              obb2) or
                    self.get_separating_plane(RPos, self.cross_product(obb1.rotation.get_right_vector(),
                                                                       obb2.rotation.get_right_vector()), obb1, obb2) or
                    self.get_separating_plane(RPos, self.cross_product(obb1.rotation.get_right_vector(),
                                                                       obb2.rotation.get_up_vector()), obb1, obb2) or
                    self.get_separating_plane(RPos, self.cross_product(obb1.rotation.get_up_vector(),
                                                                       obb2.rotation.get_forward_vector()), obb1,
                                              obb2) or
                    self.get_separating_plane(RPos, self.cross_product(obb1.rotation.get_up_vector(),
                                                                       obb2.rotation.get_right_vector()), obb1, obb2) or
                    self.get_separating_plane(RPos, self.cross_product(obb1.rotation.get_up_vector(),
                                                                       obb2.rotation.get_up_vector()), obb1, obb2))

    def _get_brake(self, vehicle_hazard=None, light_hazard=None, walker_hazard=None, stop_sign_hazard=None):
        actors = self._world.get_actors()
        speed = self._get_forward_speed()

        vehicle_location = self._vehicle.get_location()
        vehicle_transform = self._vehicle.get_transform()

        vehicles = actors.filter('*vehicle*')

        # -----------------------------------------------------------
        # Red light detection
        # -----------------------------------------------------------
        if light_hazard is None:
            light_hazard = False
            self._active_traffic_light = None
            _traffic_lights = self.get_nearby_object(vehicle_location, actors.filter('*traffic_light*'),
                                                     self.light_radius)

            center_light_detector_bb = vehicle_transform.transform(
                carla.Location(x=self.center_bb_light_x, y=self.center_bb_light_y, z=self.center_bb_light_z))
            extent_light_detector_bb = carla.Vector3D(x=self.extent_bb_light_x, y=self.extent_bb_light_y,
                                                      z=self.extent_bb_light_z)
            light_detector_bb = carla.BoundingBox(center_light_detector_bb, extent_light_detector_bb)
            light_detector_bb.rotation = vehicle_transform.rotation

            for light in _traffic_lights:
                size = 0.1  # size of the points and bounding boxes used for visualization
                # box in which we will look for traffic light triggers.
                center_bounding_box = light.get_transform().transform(light.trigger_volume.location)
                center_bounding_box = carla.Location(center_bounding_box.x, center_bounding_box.y,
                                                     center_bounding_box.z)
                length_bounding_box = carla.Vector3D(light.trigger_volume.extent.x, light.trigger_volume.extent.y,
                                                     light.trigger_volume.extent.z)
                transform = carla.Transform(
                    center_bounding_box)  # can only create a bounding box from a transform.location, not from a location
                bounding_box = carla.BoundingBox(transform.location, length_bounding_box)

                gloabl_rot = light.get_transform().rotation
                bounding_box.rotation = carla.Rotation(pitch=light.trigger_volume.rotation.pitch + gloabl_rot.pitch,
                                                       yaw=light.trigger_volume.rotation.yaw + gloabl_rot.yaw,
                                                       roll=light.trigger_volume.rotation.roll + gloabl_rot.roll)

                if (self.check_obb_intersection(light_detector_bb, bounding_box) == True):
                    if ((light.state == carla.libcarla.TrafficLightState.Red)
                            or (light.state == carla.libcarla.TrafficLightState.Yellow)):
                        self._active_traffic_light = light
                        light_hazard = True

        # -----------------------------------------------------------
        # Stop sign detection
        # -----------------------------------------------------------
        if stop_sign_hazard is None:
            stop_sign_hazard = False
            if not self.ignore_stop_signs:
                stop_signs = self.get_nearby_object(vehicle_location, actors.filter('*stop*'), self.light_radius)
                center_vehicle_stop_sign_detector_bb = vehicle_transform.transform(self._vehicle.bounding_box.location)
                extent_vehicle_stop_sign_detector_bb = self._vehicle.bounding_box.extent
                vehicle_stop_sign_detector_bb = carla.BoundingBox(center_vehicle_stop_sign_detector_bb,
                                                                  extent_vehicle_stop_sign_detector_bb)
                vehicle_stop_sign_detector_bb.rotation = vehicle_transform.rotation

                for stop_sign in stop_signs:
                    center_bb_stop_sign = stop_sign.get_transform().transform(stop_sign.trigger_volume.location)
                    length_bb_stop_sign = carla.Vector3D(stop_sign.trigger_volume.extent.x,
                                                         stop_sign.trigger_volume.extent.y,
                                                         stop_sign.trigger_volume.extent.z)
                    transform_stop_sign = carla.Transform(center_bb_stop_sign)
                    bounding_box_stop_sign = carla.BoundingBox(transform_stop_sign.location, length_bb_stop_sign)
                    rotation_stop_sign = stop_sign.get_transform().rotation
                    bounding_box_stop_sign.rotation = carla.Rotation(
                        pitch=stop_sign.trigger_volume.rotation.pitch + rotation_stop_sign.pitch,
                        yaw=stop_sign.trigger_volume.rotation.yaw + rotation_stop_sign.yaw,
                        roll=stop_sign.trigger_volume.rotation.roll + rotation_stop_sign.roll)

                    if (self.check_obb_intersection(vehicle_stop_sign_detector_bb, bounding_box_stop_sign) == True):
                        if (not (stop_sign.id in self.cleared_stop_signs)):
                            if ((speed * 3.6) > 0.0):  # Conversion from m/s to km/h
                                stop_sign_hazard = True
                            else:
                                self.cleared_stop_signs.append(stop_sign.id)

                # reset past cleared stop signs
                for cleared_stop_sign in self.cleared_stop_signs:
                    remove_stop_sign = True
                    for stop_sign in stop_signs:
                        if (stop_sign.id == cleared_stop_sign):
                            remove_stop_sign = False  # stop sign is still around us hence it might be active
                    if (remove_stop_sign == True):
                        self.cleared_stop_signs.remove(cleared_stop_sign)

        # -----------------------------------------------------------
        # Obstacle detection
        # -----------------------------------------------------------
        if vehicle_hazard is None or walker_hazard is None:
            vehicle_hazard = False
            self.vehicle_hazard = [False for i in range(int(self.extrapolation_seconds * self.frame_rate_tf))]
            extrapolation_seconds = self.extrapolation_seconds  # amount of seconds we look into the future to predict collisions
            detection_radius = self.detection_radius  # distance in which we check for collisions
            number_of_future_frames = int(extrapolation_seconds * self.frame_rate_tf)
            number_of_future_frames_no_junction = int(self.extrapolation_seconds_no_junction * self.frame_rate_tf)

            # -----------------------------------------------------------
            # Walker detection
            # -----------------------------------------------------------
            walkers = actors.filter('*walker*')
            walker_hazard = False
            self.walker_hazard = [False for i in range(int(self.extrapolation_seconds * self.frame_rate_tf))]
            nearby_walkers = []
            for walker in walkers:
                if (walker.get_location().distance(vehicle_location) < detection_radius):
                    walker_future_bbs = []
                    walker_transform = walker.get_transform()
                    walker_velocity = walker.get_velocity()
                    walker_speed = self._get_forward_speed(transform=walker_transform,
                                                           velocity=walker_velocity)  # In m/s
                    walker_location = walker_transform.location
                    walker_direction = walker.get_control().direction

                    for i in range(number_of_future_frames):
                        if self.render_bev == False and self.junction == False and i > number_of_future_frames_no_junction:
                            break

                        # NOTE for perf. optimization: Could also just add velocity.x instead might be a bit faster
                        new_x = walker_location.x + walker_direction.x * walker_speed * (1.0 / self.frame_rate_tf)
                        new_y = walker_location.y + walker_direction.y * walker_speed * (1.0 / self.frame_rate_tf)
                        new_z = walker_location.z + walker_direction.z * walker_speed * (1.0 / self.frame_rate_tf)
                        walker_location = carla.Location(new_x, new_y, new_z)

                        transform = carla.Transform(walker_location)
                        bounding_box = carla.BoundingBox(transform.location, walker.bounding_box.extent)
                        bounding_box.rotation = carla.Rotation(
                            pitch=walker.bounding_box.rotation.pitch + walker_transform.rotation.pitch,
                            yaw=walker.bounding_box.rotation.yaw + walker_transform.rotation.yaw,
                            roll=walker.bounding_box.rotation.roll + walker_transform.rotation.roll)

                        walker_future_bbs.append(bounding_box)
                    nearby_walkers.append(walker_future_bbs)

            # -----------------------------------------------------------
            # Vehicle detection
            # -----------------------------------------------------------
            nearby_vehicles = {}
            tmp_near_vehicle_id = []
            tmp_stucked_vehicle_id = []
            for vehicle in vehicles:
                if (vehicle.id == self._vehicle.id):
                    continue
                if (vehicle.get_location().distance(vehicle_location) < detection_radius):
                    tmp_near_vehicle_id.append(vehicle.id)
                    veh_future_bbs = []
                    traffic_transform = vehicle.get_transform()
                    traffic_control = vehicle.get_control()
                    traffic_velocity = vehicle.get_velocity()
                    traffic_speed = self._get_forward_speed(transform=traffic_transform,
                                                            velocity=traffic_velocity)  # In m/s

                    self.vehicle_speed_buffer[vehicle.id]["velocity"].append(traffic_speed)
                    self.vehicle_speed_buffer[vehicle.id]["throttle"].append(traffic_control.throttle)
                    self.vehicle_speed_buffer[vehicle.id]["brake"].append(traffic_control.brake)
                    if len(self.vehicle_speed_buffer[vehicle.id]["velocity"]) > self.stuck_buffer_size:
                        self.vehicle_speed_buffer[vehicle.id]["velocity"] = self.vehicle_speed_buffer[vehicle.id][
                                                                                "velocity"][-self.stuck_buffer_size:]
                        self.vehicle_speed_buffer[vehicle.id]["throttle"] = self.vehicle_speed_buffer[vehicle.id][
                                                                                "throttle"][-self.stuck_buffer_size:]
                        self.vehicle_speed_buffer[vehicle.id]["brake"] = self.vehicle_speed_buffer[vehicle.id]["brake"][
                                                                         -self.stuck_buffer_size:]

                    next_loc = np.array([traffic_transform.location.x, traffic_transform.location.y])
                    action = np.array(
                        np.stack([traffic_control.steer, traffic_control.throttle, traffic_control.brake], axis=-1))
                    next_yaw = np.array([traffic_transform.rotation.yaw / 180.0 * np.pi])
                    next_speed = np.array([traffic_speed])

                    for i in range(number_of_future_frames):
                        if self.render_bev == False and self.junction == False and i > number_of_future_frames_no_junction:
                            break

                        next_loc, next_yaw, next_speed = self.vehicle_model.forward(next_loc, next_yaw, next_speed,
                                                                                    action)

                        delta_yaws = next_yaw.item() * 180.0 / np.pi

                        transform = carla.Transform(
                            carla.Location(x=next_loc[0].item(), y=next_loc[1].item(), z=traffic_transform.location.z))
                        bounding_box = carla.BoundingBox(transform.location, vehicle.bounding_box.extent)
                        bounding_box.rotation = carla.Rotation(pitch=float(traffic_transform.rotation.pitch),
                                                               yaw=float(delta_yaws),
                                                               roll=float(traffic_transform.rotation.roll))

                    if (statistics.mean(self.vehicle_speed_buffer[vehicle.id]["velocity"]) < self.stuck_vel_threshold
                            and statistics.mean(
                                self.vehicle_speed_buffer[vehicle.id]["throttle"]) > self.stuck_throttle_threshold
                            and statistics.mean(
                                self.vehicle_speed_buffer[vehicle.id]["brake"]) < self.stuck_brake_threshold):
                        tmp_stucked_vehicle_id.append(vehicle.id)

                    nearby_vehicles[vehicle.id] = veh_future_bbs

            # delete old vehicles
            to_delete = set(self.vehicle_speed_buffer.keys()).difference(tmp_near_vehicle_id)
            for d in to_delete:
                del self.vehicle_speed_buffer[d]
            # -----------------------------------------------------------
            # Intersection checks with ego vehicle
            # -----------------------------------------------------------

            next_loc_no_brake = np.array([vehicle_transform.location.x, vehicle_transform.location.y])
            next_yaw_no_brake = np.array([vehicle_transform.rotation.yaw / 180.0 * np.pi])
            next_speed_no_brake = np.array([speed])

            # NOTE intentionally set ego vehicle to move at the target speed (we want to know if there is an intersection if we would not brake)
            throttle_extrapolation = self._get_throttle_extrapolation(self.target_speed, speed)
            action_no_brake = np.array(np.stack([self.steer_tf, throttle_extrapolation, 0.0], axis=-1))

            back_only_vehicle_id = []
            ego_future = []

            for i in range(number_of_future_frames):
                if self.render_bev == False and self.junction == False and i > number_of_future_frames_no_junction:
                    alpha = 255
                    color_value = 50
                    break
                else:
                    alpha = 50
                    color_value = 255

                # calculate ego vehicle bounding box for the next timestep. We don't consider timestep 0 because it is from the past and has already happened.
                next_loc_no_brake, next_yaw_no_brake, next_speed_no_brake = self.ego_model.forward(next_loc_no_brake,
                                                                                                   next_yaw_no_brake,
                                                                                                   next_speed_no_brake,
                                                                                                   action_no_brake)
                next_loc_no_brake_temp = np.array([-next_loc_no_brake[1], next_loc_no_brake[0]])
                next_yaw_no_brake_temp = next_yaw_no_brake.item() + np.pi / 2  # in global coordinates

                waypoint_route_extrapolation_temp = self._waypoint_planner_extrapolation.run_step(
                    next_loc_no_brake_temp)
                steer_extrapolation_temp = self._get_steer_extrapolation(waypoint_route_extrapolation_temp,
                                                                         next_loc_no_brake_temp, next_yaw_no_brake_temp,
                                                                         next_speed_no_brake, restore=False)
                throttle_extrapolation_temp = self._get_throttle_extrapolation(self.target_speed, next_speed_no_brake,
                                                                               restore=False)
                brake_extrapolation_temp = 1.0 if self._waypoint_planner_extrapolation.is_last else 0.0
                action_no_brake = np.array(
                    np.stack([steer_extrapolation_temp, float(throttle_extrapolation_temp), brake_extrapolation_temp],
                             axis=-1))

                delta_yaws_no_brake = next_yaw_no_brake.item() * 180.0 / np.pi
                cosine = np.cos(next_yaw_no_brake.item())
                sine = np.sin(next_yaw_no_brake.item())

                extent = self._vehicle.bounding_box.extent
                extent_org = self._vehicle.bounding_box.extent
                extent.x = extent.x / 2.

                # front half
                transform = carla.Transform(carla.Location(x=next_loc_no_brake[0].item() + extent.x * cosine,
                                                           y=next_loc_no_brake[1].item() + extent.y * sine,
                                                           z=vehicle_transform.location.z))
                bounding_box = carla.BoundingBox(transform.location, extent)
                bounding_box.rotation = carla.Rotation(pitch=float(vehicle_transform.rotation.pitch),
                                                       yaw=float(delta_yaws_no_brake),
                                                       roll=float(vehicle_transform.rotation.roll))

                # back half
                transform_back = carla.Transform(carla.Location(x=next_loc_no_brake[0].item() - extent.x * cosine,
                                                                y=next_loc_no_brake[1].item() - extent.y * sine,
                                                                z=vehicle_transform.location.z))
                bounding_box_back = carla.BoundingBox(transform_back.location, extent)
                bounding_box_back.rotation = carla.Rotation(pitch=float(vehicle_transform.rotation.pitch),
                                                            yaw=float(delta_yaws_no_brake),
                                                            roll=float(vehicle_transform.rotation.roll))


                i_stuck = i
                print(i_stuck)
                for id, traffic_participant in nearby_vehicles.items():
                    print(f"00000000000{traffic_participant},2,{nearby_vehicles}")
                    if self.render_bev == False and self.junction == False and i > number_of_future_frames_no_junction:
                        break
                    if id in tmp_stucked_vehicle_id:
                        i_stuck = 0
                    back_intersect = (
                                self.check_obb_intersection(bounding_box_back, traffic_participant[i_stuck]) == True)
                    front_intersect = (self.check_obb_intersection(bounding_box, traffic_participant[i_stuck]) == True)
                    if id in back_only_vehicle_id:
                        back_only_vehicle_id.remove(id)
                        if back_intersect:
                            back_only_vehicle_id.append(id)
                        continue
                    if back_intersect and not front_intersect:
                        back_only_vehicle_id.append(id)
                    if front_intersect:
                        color = carla.Color(color_value, 0, 0, alpha)
                        if self.junction == True or i <= number_of_future_frames_no_junction:
                            vehicle_hazard = True
                        self.vehicle_hazard[i] = True

                for walker in nearby_walkers:
                    if self.render_bev == False and self.junction == False and i > number_of_future_frames_no_junction:
                        break
                    if (self.check_obb_intersection(bounding_box, walker[i]) == True):
                        color = carla.Color(color_value, 0, 0, alpha)
                        if self.junction == True or i <= number_of_future_frames_no_junction:
                            walker_hazard = True
                        self.walker_hazard[i] = True

            bremsweg = ((speed * 3.6) / 10.0) ** 2 / 2.0  # Bremsweg formula for emergency break
            safety_x = np.clip(bremsweg + 1.0, a_min=2.0, a_max=4.0)  # plus one meter is the car.

            center_safety_box = vehicle_transform.transform(carla.Location(x=safety_x, y=0.0, z=0.0))
            bounding_box = carla.BoundingBox(center_safety_box, self._vehicle.bounding_box.extent)
            bounding_box.rotation = vehicle_transform.rotation

            for _, traffic_participant in nearby_vehicles.items():
                if (self.check_obb_intersection(bounding_box, traffic_participant[
                    0]) == True):  # check the first BB of the traffic participant. We don't extrapolate into the future here.
                    color = carla.Color(255, 0, 0, 255)
                    vehicle_hazard = True
                    self.vehicle_hazard[0] = True

            for walker in nearby_walkers:
                if (self.check_obb_intersection(bounding_box, walker[
                    0]) == True):  # check the first BB of the traffic participant. We don't extrapolate into the future here.
                    color = carla.Color(255, 0, 0, 255)
                    walker_hazard = True
                    self.walker_hazard[0] = True

            self.future_states = {'walker': nearby_walkers, 'vehicle': nearby_vehicles}

        else:
            self.vehicle_hazard = vehicle_hazard
            self.walker_hazard = walker_hazard

        self.stop_sign_hazard = stop_sign_hazard
        self.traffic_light_hazard = light_hazard

        return (vehicle_hazard or light_hazard or walker_hazard or stop_sign_hazard)

class EgoModel():
    def __init__(self, dt=1. / 4):
        self.dt = dt

        # Kinematic bicycle model. Numbers are the tuned parameters from World on Rails
        self.front_wb = -0.090769015
        self.rear_wb = 1.4178275

        self.steer_gain = 0.36848336
        self.brake_accel = -4.952399
        self.throt_accel = 0.5633837

    def forward(self, locs, yaws, spds, acts):
        # Kinematic bicycle model. Numbers are the tuned parameters from World on Rails
        steer = acts[..., 0:1].item()
        throt = acts[..., 1:2].item()
        brake = acts[..., 2:3].astype(np.uint8)

        if (brake):
            accel = self.brake_accel
        else:
            accel = self.throt_accel * throt

        wheel = self.steer_gain * steer

        beta = math.atan(self.rear_wb / (self.front_wb + self.rear_wb) * math.tan(wheel))
        yaws = yaws.item()
        spds = spds.item()
        next_locs_0 = locs[0].item() + spds * math.cos(yaws + beta) * self.dt
        next_locs_1 = locs[1].item() + spds * math.sin(yaws + beta) * self.dt
        next_yaws = yaws + spds / self.rear_wb * math.sin(beta) * self.dt
        next_spds = spds + accel * self.dt
        next_spds = next_spds * (next_spds > 0.0)  # Fast ReLU

        next_locs = np.array([next_locs_0, next_locs_1])
        next_yaws = np.array(next_yaws)
        next_spds = np.array(next_spds)

        return next_locs, next_yaws, next_spds