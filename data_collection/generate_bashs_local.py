import os
import random

routes = {}
routes[
    "training_routes/routes_town01_short.xml"
] = "scenarios/town01_all_scenarios.json"
routes["training_routes/routes_town01_tiny.xml"] = "scenarios/town01_all_scenarios.json"
routes[
    "training_routes/routes_town02_short.xml"
] = "scenarios/town02_all_scenarios.json"
routes["training_routes/routes_town02_tiny.xml"] = "scenarios/town02_all_scenarios.json"
routes[
    "training_routes/routes_town03_short.xml"
] = "scenarios/town03_all_scenarios.json"
routes["training_routes/routes_town03_tiny.xml"] = "scenarios/town03_all_scenarios.json"
routes[
    "training_routes/routes_town04_short.xml"
] = "scenarios/town04_all_scenarios.json"
routes["training_routes/routes_town04_tiny.xml"] = "scenarios/town04_all_scenarios.json"
routes[
    "training_routes/routes_town05_short.xml"
] = "scenarios/town05_all_scenarios.json"
routes["training_routes/routes_town05_tiny.xml"] = "scenarios/town05_all_scenarios.json"
routes["training_routes/routes_town05_long.xml"] = "scenarios/town05_all_scenarios.json"
routes[
    "training_routes/routes_town06_short.xml"
] = "scenarios/town06_all_scenarios.json"
routes["training_routes/routes_town06_tiny.xml"] = "scenarios/town06_all_scenarios.json"
routes[
    "training_routes/routes_town07_short.xml"
] = "scenarios/town07_all_scenarios.json"
routes["training_routes/routes_town07_tiny.xml"] = "scenarios/town07_all_scenarios.json"
routes[
    "training_routes/routes_town10_short.xml"
] = "scenarios/town10_all_scenarios.json"
routes["training_routes/routes_town10_tiny.xml"] = "scenarios/town10_all_scenarios.json"
routes[
    "additional_routes/routes_town01_long.xml"
] = "scenarios/town01_all_scenarios.json"
routes[
    "additional_routes/routes_town02_long.xml"
] = "scenarios/town02_all_scenarios.json"
routes[
    "additional_routes/routes_town03_long.xml"
] = "scenarios/town03_all_scenarios.json"
routes[
    "additional_routes/routes_town04_long.xml"
] = "scenarios/town04_all_scenarios.json"
routes[
    "additional_routes/routes_town06_long.xml"
] = "scenarios/town06_all_scenarios.json"

towns=['town01','town02','town03','town04','town05','town06','town07','town10']
ip_ports = []

towns=['town01','town02','town03','town04','town05','town06','town07','town10']

for port in range(2000, 2040, 10):
    ip_ports.append(("localhost", port, port + 2000))


carla_seed = 2000
traffic_seed = 2000

configs = []
for i in range(14):
    configs.append("weather-%d.yaml" % i)


def generate_script(
    ip, port, tm_port, route, scenario, carla_seed, traffic_seed, config_path,town
):
    lines = []
    lines.append("export HOST=%s\n" % ip)
    lines.append("export PORT=%d\n" % port)
    lines.append("export TM_PORT=%d\n" % tm_port)
    lines.append("export ROUTES=${LEADERBOARD_ROOT}/data/%s\n" % route)
    lines.append("export SCENARIOS=${LEADERBOARD_ROOT}/data/%s\n" % scenario)
    lines.append("export CARLA_SEED=%d\n" % port)
    lines.append("export TRAFFIC_SEED=%d\n" % port)
    lines.append("export TEAM_CONFIG=${YAML_ROOT}/%s\n" % config_path)
    lines.append("export SAVE_PATH=${DATA_ROOT}/%s/data\n" % town)
    lines.append(
        "export CHECKPOINT_ENDPOINT=${DATA_ROOT}/%s/results/%s_%s.json\n"
        % (town, route.split("/")[1].split(".")[0],config_path.split('.')[0])
    )
    lines.append("\n")
    base = open("base_script.sh").readlines()

    for line in lines:
        base.insert(13, line)

    return base

for i,town in enumerate(towns):
    if not os.path.exists("bashs_local"):
        os.mkdir("bashs_local")
    os.makedirs(f"bashs_local/{town}",exist_ok=True)

    i %= 4
    ip, port, tm_port = ip_ports[i]
    for route in routes:
        if town in route:
            for j in range(8):
                script = generate_script(
                    ip,
                    port,
                    tm_port,
                    route,
                    routes[route],
                    carla_seed,
                    traffic_seed,
                    configs[j],
                    town
                )

                fw = open(
                    f"bashs_local/{town}/{route.split('/')[1].split('.')[0]}_{configs[j].split('.')[0]}.sh", "w"
                )
                for line in script:
                    fw.write(line)
