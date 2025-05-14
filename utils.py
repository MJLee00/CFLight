import traci 
import numpy as np 
import torch
import traci.constants as tc


def updateTargetGraph(mainQN, targetQN, tau):
    for target_param, param in zip(targetQN.parameters(), mainQN.parameters()):
        target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)


def get_state():
    for veh_id in traci.vehicle.getIDList():
        traci.vehicle.subscribe(veh_id, (tc.VAR_POSITION, tc.VAR_SPEED))
    p = traci.vehicle.getAllContextSubscriptionResults()
    p_state = np.zeros((60, 60, 2))
    for x in p:
        if net_type == 0:
            ps = p[x][tc.VAR_POSITION]
            if ps[0] < 11700 or ps[0] > 12000 or ps[1] < 13200 or ps[1] > 13500:
                continue
            spd = p[x][tc.VAR_SPEED]
            p_state[int((ps[0] - 11700)/ 5), int((ps[1]-13200) / 5)] = [1, int(round(spd)/20)]
        elif net_type == 1:
            ps = p[x][tc.VAR_POSITION]
            spd = p[x][tc.VAR_SPEED]
            p_state[int((ps[0])/ 5), int((ps[1]) / 5)] = [1, int(round(spd)/20)]
    p_state = np.reshape(p_state, [-1, 3600, 2])
    return torch.FloatTensor(p_state)


def take_action(round_count, start_time, cf_generator, safety_check, collisions, tls, act, wait_time, wait_time_map, safe_weight, net_type):
    tls_id = tls[0]
    init_p = act * 2
    prev = -1
    changed = False
    collision_phase = -1
    p_state = np.zeros((60, 60, 2))
    step = 0
    phase_step_counter = 0
    conflict_flag = False
    green_duration = 10
    yellow_duration = 3
    traci.trafficlight.setRedYellowGreenState('0', traci.trafficlight.getAllProgramLogics('0')[0].phases[init_p].state)
    traci.trafficlight.setPhaseDuration(tls_id, green_duration)
    wait_t = 0
    collisions = 0
    cf_result = None
    while traci.simulation.getMinExpectedNumber() > 0:
        if step == green_duration * 2:
            traci.trafficlight.setRedYellowGreenState('0', traci.trafficlight.getAllProgramLogics('0')[0].phases[init_p + 1].state)
            traci.trafficlight.setPhaseDuration(tls_id, yellow_duration)
        if step == green_duration * 2 + yellow_duration * 2:
            break

        # LLM CF
        # if start_time > 1 and step == 0 and collisions > 0:
        #     llm_cf_action, llm_cf_next_state, llm_cf_metrics = cf_generator.analyze_actions(act)
        #     if llm_cf_action is not None:
        #         cf_result = llm_cf_action, llm_cf_next_state, llm_cf_metrics


        # Safety Check
        # if safety_check("", "PERMITTED_LEFT_TURN"):
        #     conflict_flag = True
        # if conflict_flag:
        #     if act == 0 or act == 1 or act == 2:
        #         traci.trafficlight.setRedYellowGreenState('0', traci.trafficlight.getAllProgramLogics('0')[0].phases[
        #         6].state)
        #         traci.trafficlight.setPhaseDuration(tls_id, green_duration - phase_step_counter * 2)
        #         init_p = 6
        #     elif act == 5 or act == 6 or act == 7:
        #         traci.trafficlight.setRedYellowGreenState('0', traci.trafficlight.getAllProgramLogics('0')[0].phases[
        #             16].state)
        #         traci.trafficlight.setPhaseDuration(tls_id, green_duration - phase_step_counter * 2)
        #         init_p = 16
        
        
        # LLM CF
        #llm_prior.add_vehicle_info(wait_t)
        #llm_prior.pre_action = act

        traci.simulationStep()
        phase_step_counter += 1
        collisions += traci.simulation.getCollidingVehiclesNumber()
        step += 1
        if step % 10 == 0:
            for veh_id in traci.vehicle.getIDList():
                if net_type == 0:
                    ps = traci.vehicle.getPosition(veh_id)
                    if ps[0] < 11700 or ps[0] > 12000 or ps[1] < 13200 or ps[1] > 13500:
                        continue
                    wait_time_map[veh_id] = traci.vehicle.getAccumulatedWaitingTime(veh_id)
                elif net_type == 1:
                    ps = traci.vehicle.getPosition(veh_id)
                    wait_time_map[veh_id] = traci.vehicle.getAccumulatedWaitingTime(veh_id)
        
        wait_temp = dict(wait_time_map)
        for veh_id in traci.vehicle.getIDList():
            ps = traci.vehicle.getPosition(veh_id)
            spd = traci.vehicle.getSpeed(veh_id)/20
            if net_type == 0:
                if ps[0] < 11700 or ps[0] > 12000 or ps[1] < 13200 or ps[1] > 13500:
                    continue
                p_state[int((ps[0] - 11700)/ 5), int((ps[1]-13200) / 5)] = [1, int(round(spd))]
            elif net_type == 1:
                p_state[int((ps[0])/ 5), int((ps[1]) / 5)] = [1, int(round(spd))]

        wait_t = sum(wait_temp[x] for x in wait_temp)

        d = False
        if traci.simulation.getMinExpectedNumber() == 0:
            d = True

        r = (wait_time - wait_t)/500 - collisions/safe_weight 
        p_state_tmp = torch.FloatTensor(np.reshape(p_state, [-1, 3600, 2]))
        
    return p_state_tmp, r, d, wait_t, collision_phase, collisions, (wait_time - wait_t)/500, cf_result



def take_action_safe_act(round_count, start_time, cf_generator, safety_check, collisions, tls, act, wait_time, wait_time_map, safe_weight, net_type):
    tls_id = tls[0]
    init_p = act * 2
    prev = -1
    changed = False
    collision_phase = -1
    p_state = np.zeros((60, 60, 2))
    step = 0
    phase_step_counter = 0
    conflict_flag = False
    green_duration = 10
    yellow_duration = 3
    traci.trafficlight.setRedYellowGreenState('0', traci.trafficlight.getAllProgramLogics('0')[0].phases[init_p].state)
    traci.trafficlight.setPhaseDuration(tls_id, green_duration)
    wait_t = 0
    collisions = 0
    cf_result = None
    while traci.simulation.getMinExpectedNumber() > 0:
        if step == green_duration * 2:
            traci.trafficlight.setRedYellowGreenState('0', traci.trafficlight.getAllProgramLogics('0')[0].phases[init_p + 1].state)
            traci.trafficlight.setPhaseDuration(tls_id, yellow_duration)
        if step == green_duration * 2 + yellow_duration * 2:
            break

        # LLM CF
        # if start_time > 1 and step == 0 and collisions > 0:
        #     llm_cf_action, llm_cf_next_state, llm_cf_metrics = cf_generator.analyze_actions(act)
        #     if llm_cf_action is not None:
        #         cf_result = llm_cf_action, llm_cf_next_state, llm_cf_metrics


        #Safety Check
        if safety_check("", "PERMITTED_LEFT_TURN"):
            conflict_flag = True
        if conflict_flag:
            if act == 0 or act == 1 or act == 2:
                traci.trafficlight.setRedYellowGreenState('0', traci.trafficlight.getAllProgramLogics('0')[0].phases[
                6].state)
                traci.trafficlight.setPhaseDuration(tls_id, green_duration - phase_step_counter * 2)
                init_p = 6
            elif act == 5 or act == 6 or act == 7:
                traci.trafficlight.setRedYellowGreenState('0', traci.trafficlight.getAllProgramLogics('0')[0].phases[
                    16].state)
                traci.trafficlight.setPhaseDuration(tls_id, green_duration - phase_step_counter * 2)
                init_p = 16
        
        
        # LLM CF
        #llm_prior.add_vehicle_info(wait_t)
        #llm_prior.pre_action = act

        traci.simulationStep()
        phase_step_counter += 1
        collisions += traci.simulation.getCollidingVehiclesNumber()
        step += 1
        if step % 10 == 0:
            for veh_id in traci.vehicle.getIDList():
                if net_type == 0:
                    ps = traci.vehicle.getPosition(veh_id)
                    if ps[0] < 11700 or ps[0] > 12000 or ps[1] < 13200 or ps[1] > 13500:
                        continue
                    wait_time_map[veh_id] = traci.vehicle.getAccumulatedWaitingTime(veh_id)
                elif net_type == 1:
                    ps = traci.vehicle.getPosition(veh_id)
                    wait_time_map[veh_id] = traci.vehicle.getAccumulatedWaitingTime(veh_id)
        
        wait_temp = dict(wait_time_map)
        for veh_id in traci.vehicle.getIDList():
            ps = traci.vehicle.getPosition(veh_id)
            spd = traci.vehicle.getSpeed(veh_id)/20
            if net_type == 0:
                if ps[0] < 11700 or ps[0] > 12000 or ps[1] < 13200 or ps[1] > 13500:
                    continue
                p_state[int((ps[0] - 11700)/ 5), int((ps[1]-13200) / 5)] = [1, int(round(spd))]
            elif net_type == 1:
                p_state[int((ps[0])/ 5), int((ps[1]) / 5)] = [1, int(round(spd))]

        wait_t = sum(wait_temp[x] for x in wait_temp)

        d = False
        if traci.simulation.getMinExpectedNumber() == 0:
            d = True

        r = (wait_time - wait_t)/500 #- collisions/safe_weight 
        p_state_tmp = torch.FloatTensor(np.reshape(p_state, [-1, 3600, 2]))
        
    return p_state_tmp, r, d, wait_t, collision_phase, collisions, (wait_time - wait_t)/500, cf_result



def take_action_safe_loss(round_count, start_time, cf_generator, safety_check, collisions, tls, act, wait_time, wait_time_map, safe_weight, net_type):
    tls_id = tls[0]
    init_p = act * 2
    prev = -1
    changed = False
    collision_phase = -1
    p_state = np.zeros((60, 60, 2))
    step = 0
    phase_step_counter = 0
    conflict_flag = False
    green_duration = 10
    yellow_duration = 3
    traci.trafficlight.setRedYellowGreenState('0', traci.trafficlight.getAllProgramLogics('0')[0].phases[init_p].state)
    traci.trafficlight.setPhaseDuration(tls_id, green_duration)
    wait_t = 0
    collisions = 0
    cf_result = None
    while traci.simulation.getMinExpectedNumber() > 0:
        if step == green_duration * 2:
            traci.trafficlight.setRedYellowGreenState('0', traci.trafficlight.getAllProgramLogics('0')[0].phases[init_p + 1].state)
            traci.trafficlight.setPhaseDuration(tls_id, yellow_duration)
        if step == green_duration * 2 + yellow_duration * 2:
            break

        # LLM CF
        # if start_time > 1 and step == 0 and collisions > 0:
        #     llm_cf_action, llm_cf_next_state, llm_cf_metrics = cf_generator.analyze_actions(act)
        #     if llm_cf_action is not None:
        #         cf_result = llm_cf_action, llm_cf_next_state, llm_cf_metrics

        # LLM CF
        #llm_prior.add_vehicle_info(wait_t)
        #llm_prior.pre_action = act

        traci.simulationStep()
        phase_step_counter += 1
        collisions += traci.simulation.getCollidingVehiclesNumber()
        step += 1
        if step % 10 == 0:
            for veh_id in traci.vehicle.getIDList():
                if net_type == 0:
                    ps = traci.vehicle.getPosition(veh_id)
                    if ps[0] < 11700 or ps[0] > 12000 or ps[1] < 13200 or ps[1] > 13500:
                        continue
                    wait_time_map[veh_id] = traci.vehicle.getAccumulatedWaitingTime(veh_id)
                elif net_type == 1:
                    ps = traci.vehicle.getPosition(veh_id)
                    wait_time_map[veh_id] = traci.vehicle.getAccumulatedWaitingTime(veh_id)
        
        wait_temp = dict(wait_time_map)
        for veh_id in traci.vehicle.getIDList():
            ps = traci.vehicle.getPosition(veh_id)
            spd = traci.vehicle.getSpeed(veh_id)/20
            if net_type == 0:
                if ps[0] < 11700 or ps[0] > 12000 or ps[1] < 13200 or ps[1] > 13500:
                    continue
                p_state[int((ps[0] - 11700)/ 5), int((ps[1]-13200) / 5)] = [1, int(round(spd))]
            elif net_type == 1:
                p_state[int((ps[0])/ 5), int((ps[1]) / 5)] = [1, int(round(spd))]

        wait_t = sum(wait_temp[x] for x in wait_temp)

        d = False
        if traci.simulation.getMinExpectedNumber() == 0:
            d = True

        r = (wait_time - wait_t)/500 #- collisions/safe_weight 
        p_state_tmp = torch.FloatTensor(np.reshape(p_state, [-1, 3600, 2]))
        
    return p_state_tmp, r, d, wait_t, collision_phase, collisions, (wait_time - wait_t)/500, cf_result




def take_action_q_loss(round_count, start_time, cf_generator, safety_check, collisions, tls, act, wait_time, wait_time_map, safe_weight, net_type):
    tls_id = tls[0]
    init_p = act * 2
    prev = -1
    changed = False
    collision_phase = -1
    p_state = np.zeros((60, 60, 2))
    step = 0
    phase_step_counter = 0
    conflict_flag = False
    green_duration = 10
    yellow_duration = 3
    traci.trafficlight.setRedYellowGreenState('0', traci.trafficlight.getAllProgramLogics('0')[0].phases[init_p].state)
    traci.trafficlight.setPhaseDuration(tls_id, green_duration)
    wait_t = 0
    collisions = 0
    cf_result = None
    while traci.simulation.getMinExpectedNumber() > 0:
        if step == green_duration * 2:
            traci.trafficlight.setRedYellowGreenState('0', traci.trafficlight.getAllProgramLogics('0')[0].phases[init_p + 1].state)
            traci.trafficlight.setPhaseDuration(tls_id, yellow_duration)
        if step == green_duration * 2 + yellow_duration * 2:
            break

        # LLM CF
        # if start_time > 1 and step == 0 and collisions > 0:
        #     llm_cf_action, llm_cf_next_state, llm_cf_metrics = cf_generator.analyze_actions(act)
        #     if llm_cf_action is not None:
        #         cf_result = llm_cf_action, llm_cf_next_state, llm_cf_metrics

        # LLM CF
        #llm_prior.add_vehicle_info(wait_t)
        #llm_prior.pre_action = act

        traci.simulationStep()
        phase_step_counter += 1
        collisions += traci.simulation.getCollidingVehiclesNumber()
        step += 1
        if step % 10 == 0:
            for veh_id in traci.vehicle.getIDList():
                if net_type == 0:
                    ps = traci.vehicle.getPosition(veh_id)
                    if ps[0] < 11700 or ps[0] > 12000 or ps[1] < 13200 or ps[1] > 13500:
                        continue
                    wait_time_map[veh_id] = traci.vehicle.getAccumulatedWaitingTime(veh_id)
                elif net_type == 1:
                    ps = traci.vehicle.getPosition(veh_id)
                    wait_time_map[veh_id] = traci.vehicle.getAccumulatedWaitingTime(veh_id)
        
        wait_temp = dict(wait_time_map)
        for veh_id in traci.vehicle.getIDList():
            ps = traci.vehicle.getPosition(veh_id)
            spd = traci.vehicle.getSpeed(veh_id)/20
            if net_type == 0:
                if ps[0] < 11700 or ps[0] > 12000 or ps[1] < 13200 or ps[1] > 13500:
                    continue
                p_state[int((ps[0] - 11700)/ 5), int((ps[1]-13200) / 5)] = [1, int(round(spd))]
            elif net_type == 1:
                p_state[int((ps[0])/ 5), int((ps[1]) / 5)] = [1, int(round(spd))]

        wait_t = sum(wait_temp[x] for x in wait_temp)

        d = False
        if traci.simulation.getMinExpectedNumber() == 0:
            d = True

        r = (wait_time - wait_t)/500 #- collisions/safe_weight 
        p_state_tmp = torch.FloatTensor(np.reshape(p_state, [-1, 3600, 2]))
        
    return p_state_tmp, r, d, wait_t, collision_phase, collisions, (wait_time - wait_t)/500, cf_result
