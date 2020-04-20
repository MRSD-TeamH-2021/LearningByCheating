import argparse
import time

from pathlib import Path

from benchmark import make_suite, get_suites, ALL_SUITES
from benchmark.run_benchmark import run_benchmark

import bird_view.utils.bz_utils as bzu

import xml.etree.ElementTree as ET
import carla 

def _agent_factory_hack(model_path, config, autopilot):
    """
    These imports before carla.Client() cause seg faults...
    """
    from bird_view.models.roaming import RoamingAgentMine

    if autopilot:
        return RoamingAgentMine

    import torch

    from bird_view.models import baseline
    from bird_view.models import birdview
    from bird_view.models import image

    model_args = config['model_args']
    model_name = model_args['model']
    model_to_class = {
            'birdview_dian': (birdview.BirdViewPolicyModelSS, birdview.BirdViewAgent),
            'image_ss': (image.ImagePolicyModelSS, image.ImageAgent),
            }

    model_class, agent_class = model_to_class[model_name]

    model = model_class(**config['model_args'])
    model.load_state_dict(torch.load(str(model_path)))
    model.eval()

    agent_args = config.get('agent_args', dict())
    agent_args['model'] = model

    return lambda: agent_class(**agent_args)

class TargetConfiguration(object):

    """
    This class provides the basic  configuration for a target location
    """

    transform = None

    def __init__(self, node):
        pos_x = float(node.attrib.get('x', 0))
        pos_y = float(node.attrib.get('y', 0))
        pos_z = float(node.attrib.get('z', 0))
        yaw = float(node.attrib.get('yaw', 0))

        self.transform = carla.Transform(carla.Location(x=pos_x, y=pos_y, z=pos_z), carla.Rotation(yaw=yaw))


def get_target_from_xml(filepath, scenario_name):
    if not filepath:
        print(filepath)
        print("Run Scenario flag is on but XML File not specified.")
        return None
    
    tree = ET.parse(filepath)
    target = None
    start = None
    print(scenario_name)

    h_start = []
    h_target = []

    for scenario in tree.iter("scenario"):
        if(scenario.attrib.get('name') == scenario_name):
            for start in scenario.iter("hero_start"):
                h_start.append(TargetConfiguration(start).transform)
                print(h_start)
            for target in scenario.iter("hero_target"):
                h_target.append(TargetConfiguration(target).transform)
    
    # print(h_start)
    # print(h_target)
    return h_start, h_target

def set_sync_mode(client, sync):
    world = client.get_world()

    settings = world.get_settings()
    settings.synchronous_mode = sync
    settings.fixed_delta_seconds = 0.1

    world.apply_settings(settings)

def run(model_path, port, suite, big_cam, seed, autopilot, resume, args, max_run=10, show=False):
    
    # Get target
    if (args.run_scenario):
        h_start, h_target = get_target_from_xml(args.scenario_config, args.scenario)
        if args.player_name == "hero":
            start_pose = h_start[0]
            target_pose = h_target[0]
        elif args.player_name == "hero1":
            start_pose = h_start[1]
            target_pose = h_target[1]
        if args.player_name == "hero2":
            start_pose = h_start[2]
            target_pose = h_target[2]
            
        if not (target_pose or start_pose):
            return

    print(start_pose)
    print(target_pose)
    log_dir = model_path.parent
    config = bzu.load_json(str(log_dir / 'config.json'))

    total_time = 0.0

    for suite_name in get_suites(suite):
        tick = time.time()

        benchmark_dir = log_dir / 'benchmark' / model_path.stem / ('%s_seed%d' % (suite_name, seed))
        benchmark_dir.mkdir(parents=True, exist_ok=True)

        agent_maker = _agent_factory_hack(model_path, config, autopilot)
        
        client = carla.Client('localhost', port)
        
        set_sync_mode(client, True)

        try:

            envs = [make_suite(suite_name, port=port, big_cam=big_cam, run_scenario=args.run_scenario, player_name=args.player_name+str(i), client=client) for i in range(len(h_start))]
            start_poses = h_start
            target_poses = h_target

            # envs = env
            # start_poses = start_pose
            # target_poses = target_pose

            run_benchmark(agent_maker, envs, benchmark_dir, seed, autopilot, resume, args, start_poses, target_poses, max_run=max_run, show=show, client=client)
        
        except Exception as e:
            print(e)

        finally:
            set_sync_mode(client, False)
        

        elapsed = time.time() - tick
        total_time += elapsed

        print('%s: %.3f hours.' % (suite_name, elapsed / 3600))

    print('Total time: %.3f hours.' % (total_time / 3600))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--port', type=int, default=2000)
    parser.add_argument('--suite', choices=ALL_SUITES, default='town1')
    parser.add_argument('--big_cam', action='store_true')
    parser.add_argument('--seed', type=int, default=2019)
    parser.add_argument('--autopilot', action='store_true', default=False)
    parser.add_argument('--show', action='store_true', default=False)
    parser.add_argument('--run_scenario', action='store_true', default=False)
    parser.add_argument('--scenario_config', default=None)
    parser.add_argument('--scenario', default=None)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--max-run', type=int, default=3)
    parser.add_argument('--player-name', default="hero")

    args = parser.parse_args()

    run(Path(args.model_path), args.port, args.suite, args.big_cam, args.seed, args.autopilot, args.resume, args, max_run=args.max_run, show=args.show)
