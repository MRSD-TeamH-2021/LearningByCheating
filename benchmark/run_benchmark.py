from pathlib import Path

import pandas as pd
import numpy as np
import tqdm
import time

import carla

import bird_view.utils.bz_utils as bzu
import bird_view.utils.carla_utils as cu

from bird_view.models.common import crop_birdview

import cv2
import numpy as np 

from concurrent.futures import ThreadPoolExecutor, as_completed
import operator


def _stick_together(a, b, axis=1):

    if axis == 1:
        h = min(a.shape[0], b.shape[0])

        r1 = h / a.shape[0]
        r2 = h / b.shape[0]
    
        a = cv2.resize(a, (int(r1 * a.shape[1]), int(r1 * a.shape[0])))
        b = cv2.resize(b, (int(r2 * b.shape[1]), int(r2 * b.shape[0])))

        return np.concatenate([a, b], 1)
        
    else:
        h = min(a.shape[1], b.shape[1])
        
        r1 = h / a.shape[1]
        r2 = h / b.shape[1]
    
        a = cv2.resize(a, (int(r1 * a.shape[1]), int(r1 * a.shape[0])))
        b = cv2.resize(b, (int(r2 * b.shape[1]), int(r2 * b.shape[0])))

        return np.concatenate([a, b], 0)

def _paint(observations, control, diagnostic, debug, env, show=False):
    
    WHITE = (255, 255, 255)
    RED = (255, 0, 0)
    CROP_SIZE = 192
    X = 176
    Y = 192 // 2
    R = 2

    birdview = cu.visualize_birdview(observations['birdview'])
    birdview = crop_birdview(birdview)

    if 'big_cam' in observations:
        canvas = np.uint8(observations['big_cam']).copy()
        rgb = np.uint8(observations['rgb']).copy()
    else:
        canvas = np.uint8(observations['rgb']).copy()

    def _write(text, i, j, canvas=canvas, fontsize=0.4):
        rows = [x * (canvas.shape[0] // 10) for x in range(10+1)]
        cols = [x * (canvas.shape[1] // 9) for x in range(9+1)]
        cv2.putText(
                canvas, text, (cols[j], rows[i]),
                cv2.FONT_HERSHEY_SIMPLEX, fontsize, WHITE, 1)
                
    _command = {
            1: 'LEFT',
            2: 'RIGHT',
            3: 'STRAIGHT',
            4: 'FOLLOW',
            }.get(observations['command'], '???')
            
    if 'big_cam' in observations:
        fontsize = 0.8
    else:
        fontsize = 0.4

    _write('Command: ' + _command, 1, 0, fontsize=fontsize)
    _write('Velocity: %.1f' % np.linalg.norm(observations['velocity']), 2, 0, fontsize=fontsize)

    _write('Steer: %.2f' % control.steer, 4, 0, fontsize=fontsize)
    _write('Throttle: %.2f' % control.throttle, 5, 0, fontsize=fontsize)
    _write('Brake: %.1f' % control.brake, 6, 0, fontsize=fontsize)

    _write('Collided: %s' % diagnostic['collided'], 1, 6, fontsize=fontsize)
    _write('Invaded: %s' % diagnostic['invaded'], 2, 6, fontsize=fontsize)
    _write('Lights Ran: %d/%d' % (env.traffic_tracker.total_lights_ran, env.traffic_tracker.total_lights), 3, 6, fontsize=fontsize)
    _write('Goal: %.1f' % diagnostic['distance_to_goal'], 4, 6, fontsize=fontsize)

    _write('Time: %d' % env._tick, 5, 6, fontsize=fontsize)
    _write('FPS: %.2f' % (env._tick / (diagnostic['wall'])), 6, 6, fontsize=fontsize)

    for x, y in debug.get('locations', []):
        x = int(X - x / 2.0 * CROP_SIZE)
        y = int(Y + y / 2.0 * CROP_SIZE)

        S = R // 2
        birdview[x-S:x+S+1,y-S:y+S+1] = RED

    for x, y in debug.get('locations_world', []):
        x = int(X - x * 4)
        y = int(Y + y * 4)

        S = R // 2
        birdview[x-S:x+S+1,y-S:y+S+1] = RED
    
    for x, y in debug.get('locations_birdview', []):
        S = R // 2
        birdview[x-S:x+S+1,y-S:y+S+1] = RED       
 
    for x, y in debug.get('locations_pixel', []):
        S = R // 2
        if 'big_cam' in observations:
            rgb[y-S:y+S+1,x-S:x+S+1] = RED
        else:
            canvas[y-S:y+S+1,x-S:x+S+1] = RED
        
    for x, y in debug.get('curve', []):
        x = int(X - x * 4)
        y = int(Y + y * 4)

        try:
            birdview[x,y] = [155, 0, 155]
        except:
            pass

    if 'target' in debug:
        x, y = debug['target'][:2]
        x = int(X - x * 4)
        y = int(Y + y * 4)
        birdview[x-R:x+R+1,y-R:y+R+1] = [0, 155, 155]

    ox, oy = observations['orientation']
    rot = np.array([
        [ox, oy],
        [-oy, ox]])
    u = observations['node'] - observations['position'][:2]
    v = observations['next'] - observations['position'][:2]
    u = rot.dot(u)
    x, y = u
    x = int(X - x * 4)
    y = int(Y + y * 4)
    v = rot.dot(v)
    x, y = v
    x = int(X - x * 4)
    y = int(Y + y * 4)

    if 'big_cam' in observations:
        _write('Network input/output', 1, 0, canvas=rgb)
        _write('Projected output', 1, 0, canvas=birdview)
        full = _stick_together(rgb, birdview)
    else:
        full = _stick_together(canvas, birdview)

    if 'image' in debug:
        full = _stick_together(full, cu.visualize_predicted_birdview(debug['image'], 0.01))
        
    if 'big_cam' in observations:
        full = _stick_together(canvas, full, axis=0)
    
    return full 
    
# def set_sync_mode(client, sync):
#     world = client.get_world()

#     settings = world.get_settings()
#     settings.synchronous_mode = sync
#     settings.fixed_delta_seconds = 0.05

#     world.apply_settings(settings)

def run_single(env, weather, start, target, agent_maker, seed, autopilot, args, show=False):
    # HACK: deterministic vehicle spawns.
    env.seed = seed
    env.init(start=start, target=target, weather=cu.PRESET_WEATHERS[weather], args=args)

    if not autopilot:
        agent = agent_maker()
    else:
        agent = agent_maker(env._player, resolution=1, threshold=7.5)
        agent.set_route(env._start_pose.location, env._target_pose.location)

    diagnostics = list()
    result = {
            'weather': weather,
            'start': start, 'target': target,
            'success': None, 't': None,
            'total_lights_ran': None,
            'total_lights': None,
            'collided': None,
            }

    while env.tick():
        # set_sync_mode(env._client, True)
        # world = env._client.get_world()
        # settings = world.get_settings()
        # print(settings.synchronous_mode)
        observations = env.get_observations()
        control = agent.run_step(observations)
        diagnostic = env.apply_control(control)

        _paint(observations, control, diagnostic, agent.debug, env, show=show)

        diagnostic.pop('viz_img')
        diagnostics.append(diagnostic)

        if env.is_failure() or env.is_success():
            result['success'] = env.is_success()
            result['total_lights_ran'] = env.traffic_tracker.total_lights_ran
            result['total_lights'] = env.traffic_tracker.total_lights
            result['collided'] = env.collided
            result['t'] = env._tick
            break
    env.clean_up()

    return result, diagnostics

def run_an_agent(env, agent, run_over, show, id_):

    result = {}

    if run_over:
        return id_, None, None, None, run_over, None, None

    env.tick()

    observations = env.get_observations()
    control = agent.run_step(observations)

    diagnostic = env.apply_control(control)
    frame = _paint(observations, control, diagnostic, agent.debug, env, show=show)
    diagnostic.pop('viz_img')
    player_id = env._player.id
    if env.is_failure() or env.is_success():
        result['success'+str(id_)] = env.is_success()
        result['total_lights_ran'+str(id_)] = env.traffic_tracker.total_lights_ran
        result['total_lights'+str(id_)] = env.traffic_tracker.total_lights
        result['collided'+str(id_)] = env.collided
        result['t'+str(id_)] = env._tick
        
        env.clean_up()
        run_over = True

    return id_, control, player_id, diagnostic, run_over, frame, result

def run_multiple(envs, weather, starts, targets, agent_maker, seed, autopilot, args, show=False, client=None):
    # HACK: deterministic vehicle spawns.
    
    print('RUN Multiple')
    
    pool = ThreadPoolExecutor(4)

    try:
        world = client.get_world()
    except:
        raise "Client object not passed"

    for i, env in enumerate(envs):
        env.seed = seed
        env.init(start=starts[i], target=targets[i], weather=cu.PRESET_WEATHERS[weather], args=args)

    agents = [agent_maker() for i in range(len(envs))]

    diagnostics = list()
    result = {
            'weather': weather,
            'start': starts, 'target': targets,
            'success': None, 't': None,
            'total_lights_ran': None,
            'total_lights': None,
            'collided': None,
            }
    num_agents = len(envs)
    run_over_list = [False] * num_agents

    while not all(run_over_list):

        world.tick()

        batch = []
        frames = []
        futures = []

        for i in range(num_agents):
            env = envs[i]
            agent = agents[i]
            run_over = run_over_list[i]

            futures.append(pool.submit(run_an_agent, env, agent, run_over, show, i))

        for x in as_completed(futures):
            i, control, player_id, diag, run_over, frame, res = x.result()
            
            run_over_list[i] = run_over

            if not run_over:
                diagnostics.append(diag)
                batch.append(carla.command.ApplyVehicleControl(player_id, control))
                result.update(res)
                frames.append((i,frame))
            else:
                frames.append((i,None))


        _ = client.apply_batch_sync(batch, False)

        # Display
        count = 0
        frame_to_disp = None
        
        if frames:
            frames.sort(key= operator.itemgetter(0))

        for i, frame in enumerate(frames):
            if not run_over_list[i]:
                if count == 0:
                    frame_to_disp = frame[1]
                else:
                    frame_to_disp = _stick_together(frame_to_disp, frame[1], axis=0)
                count += 1
        if frame_to_disp is not None:
            if show:
                bzu.show_image('Agent View', frame_to_disp)
            bzu.add_to_video(frame_to_disp)

    return result, diagnostics

def run_benchmark(agent_maker, env, benchmark_dir, seed, autopilot, resume, args, start_pose, target_pose, max_run=5, show=False, client=None):
    """
    benchmark_dir must be an instance of pathlib.Path
    """
    summary_csv = benchmark_dir / 'summary.csv'
    diagnostics_dir = benchmark_dir / 'diagnostics'
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    summary = list()
    total = 1 #len(list(env.all_tasks))
    
    if args.run_scenario:
        total = 1

    if summary_csv.exists() and resume:
        summary = pd.read_csv(summary_csv)
    else:
        summary = pd.DataFrame()

    num_run = 0

    if isinstance(env, list):
        environment = env[0]
    else:
        environment = env

    for weather, (start, target), run_name in tqdm.tqdm(environment.all_tasks, total=total):
        if resume and len(summary) > 0 and ((summary['start'] == start) \
                       & (summary['target'] == target) \
                       & (summary['weather'] == weather)).any():
            print (weather, start, target)
            continue

        if args.run_scenario:
            target = target_pose
            start = start_pose
            if num_run >=1:
                return

        diagnostics_csv = str(diagnostics_dir / ('%s.csv' % run_name))

        bzu.init_video(save_dir=str(benchmark_dir / 'videos'), save_path=run_name)

        if isinstance(env, list):
            result, diagnostics = run_multiple(env, weather, start, target, agent_maker, seed, autopilot, args, show=show, client=client)
        else:
            result, diagnostics = run_single(env, weather, start, target, agent_maker, seed, autopilot, args, show=show)

        summary = summary.append(result, ignore_index=True)

        # Do this every timestep just in case.
        pd.DataFrame(summary).to_csv(summary_csv, index=False)
        pd.DataFrame(diagnostics).to_csv(diagnostics_csv, index=False)

        num_run += 1

        if num_run >= max_run:
            break
