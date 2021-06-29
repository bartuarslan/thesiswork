import random
import simpy
import math
import matplotlib.pyplot as plt
import numpy as np
from math import sqrt
from statistics import mean
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import Adam, RMSprop

# import sys

# stdoutOrigin=sys.stdout
# sys.stdout = open("Output.txt", "w")

#run_time = 5184000
run_time = 864000
#transaction_interval = 7.6  # every 7.6 seconds create transaction
transaction_interval = 6.6
tiers = 5
bays = 25
lengthofbay = 0.5
heightoftier = 0.35
v_lift = 2
a_lift = 2
v_shuttle = 2
a_shuttle = 2
shuttle_no = 2
# shuttleNo = [1, 2, 3, 4, 5]
shuttleNo = []
for x in range(1, shuttle_no + 1):
    shuttleNo.append(x)
lift1No = [1, 2]

shuttle_locations = {
    1: {"tier": 5,
        "bay": 10},
    2: {"tier": 4,
        "bay": 10},
}

lift1_locations = {
    1: 1,
    2: 3
}
lift1_locations_ = {
    1: 1,
    2: 3
}

lift2_location = [3]
lift2_location_ = [3]

shuttle_util = [0] * shuttle_no
lift1_util = [0] * 2
lift2_util = [0]
shuttle_avail = [0] * shuttle_no
lift1_avail = [0] * 2
lift1_pick = [0] * 2
active_transactions = []
available_transactions = []
tier_buffer = []
tier_buffer_lift = []
flowtime = []
cycletime = []
tier_avail = [0] * tiers
lift1_buffer_control = [0] * tiers
proc_check = [0] * shuttle_no
trans_entered = 0
trans_left = 0
stable_time = 0
mean_cycletime = []
queue_state = [0] * (tiers * 2)
day_reset = 0

for shuttle in shuttleNo:
    tier_avail[shuttle_locations[shuttle]["tier"] - 1] = shuttle

def calctime(A, maxV, d):
    """
    A       constant acceleration, m/s/s
    maxV    maximum velocity, m/s
    return time in seconds required to travel
    d       distance, m
    """
    ta = float(maxV) / A  # time to accelerate to maxV
    da = A * ta * ta  # distance traveled during acceleration from 0 to maxV and back to 0
    if da > d:  # train never reaches full speed?
        return sqrt(4.0 * d / A)  # time needed to accelerate to half-way point then decelerate to destination
    else:
        return 2 * ta + (d - da) / maxV  # time to accelerate to maxV plus travel at maxV plus decelerate to destination


def source(env, interval):
    t_ID = 0
    timer_reset = 1
    global day_reset
    while True:
        t_ID += 1
        t_type = bool(random.getrandbits(1))
        t_tier = random.randint(1, tiers)
        t_bay = random.randint(1, bays)
        t_time = env.now
        t_actionindex = (bays * (t_tier - 1) + (t_bay - 1)) + t_type * bays * tiers
        side = random.randint(1, 2)
        t_info = [t_ID, t_type, t_tier, t_bay, t_time, t_actionindex]
        active_transactions.append(t_info)
        if t_type == 0:
            type1 = "Storage"
        else:
            type1 = "Retrieval"
        #print('%7.4f %s: Created as %s, Destination tier: %s, bay: %s' % (
        #    env.now, t_ID, type1, t_tier, t_bay))

        check_time = env.now

        if check_time - (day_reset*86400) >= 86400 and agent.epsilon > agent.epsilon_min:
            day_reset += 1
        #if agent.epsilon == agent.epsilon_min and timer_reset == 1:
            print("%7.4f: Avg cycle time %4.2f for day %1.0f" % (env.now, mean(cycletime), day_reset))
            print("%7.4f: Avg flow time %4.2f for day %1.0f" % (env.now, mean(flowtime), day_reset))
            print(agent.epsilon)
            shuttle_util[0] = 0
            shuttle_util[1] = 0
            lift1_util[0] = 0
            lift1_util[1] = 0
            lift2_util[0] = 0
            cycletime.clear()
            flowtime.clear()
            #mean_cycletime.clear()
            #active_transactions.clear()
            #agent.save_model()
            #interval = 7.6

        if agent.epsilon == agent.epsilon_min and timer_reset == 1:
            global stable_time
            stable_time = env.now
            print("%7.4f: Epsilon = min" % (stable_time))
            shuttle_util[0] = 0
            shuttle_util[1] = 0
            lift1_util[0] = 0
            lift1_util[1] = 0
            lift2_util[0] = 0
            timer_reset = 0
            cycletime.clear()
            flowtime.clear()
            active_transactions.clear()
            #agent.save_model()

        if proc_check[0] == 0:
            env.process(shuttle_action1(env, shuttle))

        if proc_check[1] == 0:
            env.process(shuttle_action2(env, shuttle))

        if t_ID % 500 == 0 and mean_cycletime:
            print("%7.4f: Avg cycle time %4.2f" % (env.now, mean_cycletime[-1]))
            print("%7.4f: Avg flow time %4.2f" % (env.now, mean(flowtime)))
            print(agent.epsilon)

        t = random.expovariate(1.0 / interval)
        yield env.timeout(t)


def shuttle_action1(env, shuttle, shuttleID=1):
    while True:
        proc_check[0] = proc_check[0] + 1
        if len(active_transactions) > 0 and shuttle_avail[shuttleID - 1] == 0:
            name = ""
            available_transactions.clear()
            queue_state = [0] * (tiers * 2)
            # -------- DQN implementation -------- #
            done = False
            if lift1_avail[0] == 0:
                avail_lift11 = 1
            else:
                avail_lift11 = 0
            if lift1_avail[1] == 0:
                avail_lift12 = 1
            else:
                avail_lift12 = 0
            for transaction in range(len(active_transactions)):
                for i in range(tiers):
                    if active_transactions[transaction][2] == i+1 and active_transactions[transaction][1] == 0:
                        queue_state[i*2] += 1
                        break
                    if active_transactions[transaction][2] == i+1 and active_transactions[transaction][1] == 1:
                        queue_state[i*2+1] += 1
                        break
            observation = np.array([shuttle_locations[shuttleID]["tier"], shuttle_locations[shuttleID]["bay"],
                                    lift1_locations_[1], avail_lift11, lift1_locations_[2], avail_lift12, lift2_location_[0]
                                    ])
            for transaction in range(len(active_transactions)):
                transaction_tier = active_transactions[transaction][2] - 1
                if tier_avail[transaction_tier] == 0 or tier_avail[transaction_tier] == shuttleID:
                    available_transactions.append(active_transactions[transaction])
            if available_transactions:
                action = agent.choose_action(observation)
                if action < (2 * tiers * bays):
                    liftno = 1
                    tmp_type = math.floor(action/(bays*tiers))
                    if ((action + 1) % bays) == 0:
                        tmp_bay = bays
                    else:
                        tmp_bay = (action + 1) % bays
                    tmp_tier = ((action - tmp_bay + 1) / bays) - (tmp_type * tiers) + 1
                else:
                    liftno = 2
                    tmp_action = action - (2 * tiers * bays)
                    tmp_type = math.floor(tmp_action / (bays * tiers))
                    if ((tmp_action + 1) % bays) == 0:
                        tmp_bay = bays
                    else:
                        tmp_bay = (tmp_action + 1) % bays
                    tmp_tier = ((tmp_action - tmp_bay + 1) / bays) - (tmp_type * tiers) + 1
                for ts in range(len(available_transactions)):
                    if available_transactions[ts][1] == tmp_type and available_transactions[ts][2] == tmp_tier and available_transactions[ts][3] == tmp_bay:
                        for transaction in range(len(active_transactions)):
                            if active_transactions[transaction][0] == available_transactions[ts][0]:
                                del active_transactions[transaction]
                                break
                        name = available_transactions[ts][0]
                        type = available_transactions[ts][1]
                        tier = available_transactions[ts][2]
                        bay = available_transactions[ts][3]
                        arrive = available_transactions[ts][4]
                        if shuttle_locations[shuttleID]["tier"] != tier:
                            lift2_location_[0] = tier
                        else:
                            lift2_location_[0] = lift2_location[0]
                        if liftno == 1:
                            if type == 0:
                                observation_ = np.array([tier, bay, tier, 1, lift1_locations_[2], 1, lift2_location_[0]
                                    ])
                            else:
                                observation_ = np.array([tier, 1, 1, 1, lift1_locations_[2], 1, lift2_location_[0]
                                    ])
                        else:
                            if type == 0:
                                observation_ = np.array([tier, bay, 1, lift1_locations_[1], tier, 1, lift2_location_[0]
                                    ])
                            else:
                                observation_ = np.array([tier, 1, 1, lift1_locations_[1], 1, 1, lift2_location_[0]
                                    ])
                        break
                # Process start
                global trans_entered
                global trans_left
                trans_entered += 1
                yield shuttle.get(lambda shuttleno: shuttleno == shuttleID)
                buffer_control = [tier, name]
                shuttle_avail[shuttleID - 1] = name
                tier_avail[tier - 1] = shuttleID
                pickup_time = env.now
                wait = pickup_time - arrive
                #print('%7.4f %s: Waited %6.3f, Chosen Shuttle: %s' % (env.now, name, wait, shuttleID))

                if tier != 1:
                    lift1_move = lift1_action(env, name, type, shuttleID, lift1, tier, bay, arrive, liftno, observation,
                                              action, observation_)
                    env.process(lift1_move)
                    lift1_use = 1
                elif tier == 1:
                    lift1_buffer_control[0] = name
                    lift1_use = 0
                if shuttle_locations[shuttleID]["tier"] != tier:
                    temp_tier = shuttle_locations[shuttleID]["tier"]
                    s_travel1 = abs(shuttle_locations[shuttleID]["bay"] - bays) * lengthofbay
                    t_st1 = calctime(a_shuttle, v_shuttle, s_travel1)
                    l2_travel1 = abs(lift2_location[0] - temp_tier) * heightoftier
                    t_l2t = calctime(a_lift, v_lift, l2_travel1)
                    l2_travel2 = abs(tier - temp_tier) * heightoftier
                    t_l2t2 = calctime(a_lift, v_lift, l2_travel2)
                    req_lift2 = lift2.request()
                    yield req_lift2
                    lift2_time1 = env.now
                    to1 = env.timeout(t_st1)
                    to2 = env.timeout(t_l2t)
                    # print('%7.4f %s: Shuttle:%s moving to Lift 2 buffer' % (env.now, name, shuttleID))
                    # print('%7.4f %s: Lift 2 moving to %s tier to pick up Shuttle %s' % (
                    # env.now, name, temp_tier, shuttleID))
                    yield to1 & to2
                    # print('%7.4f %s: Shuttle:%s moved to Lift 2 buffer' % (env.now, name, shuttleID))
                    to3 = env.timeout(t_l2t2)
                    tier_avail[temp_tier - 1] = 0
                    # print('%7.4f %s: Lift 2 moving to tier %s' % (env.now, name, tier))
                    yield to3
                    # print('%7.4f %s: Lift 2 moved to tier %s' % (env.now, name, tier))
                    shuttle_locations[shuttleID]["tier"] = tier
                    shuttle_locations[shuttleID]["bay"] = bays
                    lift2_util[0] = lift2_util[0] + (env.now - lift2_time1)
                    lift2.release(req_lift2)
                if type == 0:

                    shuttle_travel1 = abs(shuttle_locations[shuttleID]["bay"] - 0) * lengthofbay
                    time_shuttle_travel1 = calctime(a_shuttle, v_shuttle, shuttle_travel1)
                    shuttle_travel2 = bay * lengthofbay
                    time_shuttle_travel2 = calctime(a_shuttle, v_shuttle, shuttle_travel2)
                    t1 = env.timeout(time_shuttle_travel1)
                    # print('%7.4f %s: Shuttle:%s moving to buffer' % (env.now, name, shuttleID))
                    yield t1
                    # print('%7.4f %s: Shuttle:%s moved to buffer' % (env.now, name, shuttleID))
                    if lift1_use == 0:
                        t2 = env.timeout(time_shuttle_travel2)
                        # print('%7.4f %s: Shuttle:%s moving to bay %s' % (env.now, name, shuttleID, bay))
                        yield t2
                        # print('%7.4f %s: Shuttle:%s moved to bay %s' % (env.now, name, shuttleID, bay))

                        shuttle_avail[shuttleID - 1] = 0
                        shuttle.put(shuttleID)
                        shuttle_time = env.now - pickup_time
                        shuttle_util[shuttleID - 1] = shuttle_util[shuttleID - 1] + shuttle_time
                        shuttle_locations[shuttleID]["bay"] = bay
                        flow_time = env.now - pickup_time
                        cycle_time = env.now - arrive
                        flowtime.append(flow_time)
                        cycletime.append(cycle_time)
                        min_fl = 1/max(flowtime)
                        max_fl = 1/min(flowtime)
                        if max_fl == min_fl:
                            norm_reward = 1
                        else:
                            norm_reward = ((1/flow_time) - min_fl) / (max_fl - min_fl)
                        reward = norm_reward * 100
                        agent.remember(observation, action, reward, observation_, done)
                        agent.learn()
                        #print(
                        #    '%7.4f %s: Finished Shuttle:%s, Cycle time: %7.4f, Flow time: %7.4f' % (env.now, name, shuttleID, cycle_time, flow_time))

                        trans_left += 1
                        proc_check[0] = 0
                        env.process(shuttle_action1(env, shuttle, shuttleID))
                        break
                    elif lift1_use == 1:
                        found = 0
                        for i in range(len(tier_buffer_lift)):
                            tier_check = tier_buffer_lift[i][0]
                            name_check = tier_buffer_lift[i][1]
                            if tier_check == tier and name_check == name:
                                found = 1
                                del tier_buffer_lift[i]
                                t2 = env.timeout(time_shuttle_travel2)
                                # print('%7.4f %s: Shuttle:%s moving to bay %s' % (env.now, name, shuttleID, bay))
                                yield t2
                                # print('%7.4f %s: Shuttle:%s moved to bay %s' % (env.now, name, shuttleID, bay))

                                shuttle_avail[shuttleID - 1] = 0
                                shuttle.put(shuttleID)
                                shuttle_time = env.now - pickup_time
                                shuttle_util[shuttleID - 1] = shuttle_util[shuttleID - 1] + shuttle_time
                                shuttle_locations[shuttleID]["bay"] = bay
                                flow_time = env.now - pickup_time
                                cycle_time = env.now - arrive
                                flowtime.append(flow_time)
                                cycletime.append(cycle_time)
                                min_fl = 1 / max(flowtime)
                                max_fl = 1 / min(flowtime)
                                if max_fl == min_fl:
                                    norm_reward = 1
                                else:
                                    norm_reward = ((1 / flow_time) - min_fl) / (max_fl - min_fl)
                                reward = norm_reward * 100
                                agent.remember(observation, action, reward, observation_, done)
                                agent.learn()
                                #print('%7.4f %s: Finished Shuttle:%s, Cycle time: %7.4f, Flow time: %7.4f' % (env.now, name, shuttleID, cycle_time, flow_time))
                                trans_left += 1
                                break
                        if found == 1:
                            proc_check[0] = 0
                            env.process(shuttle_action1(env, shuttle, shuttleID))
                            break
                        if found == 0:
                            tier_buffer.append(buffer_control)
                            proc_check[0] = 0
                            break
                else:
                    shuttle_travel1 = abs(shuttle_locations[shuttleID]["bay"] - bay) * lengthofbay
                    time_shuttle_travel1 = calctime(a_shuttle, v_shuttle, shuttle_travel1)
                    shuttle_travel2 = bay * lengthofbay
                    time_shuttle_travel2 = calctime(a_shuttle, v_shuttle, shuttle_travel2)
                    ts1 = env.timeout(time_shuttle_travel1)

                    # print('%7.4f %s: Shuttle:%s moving to bay %s' % (env.now, name, shuttleID, bay))
                    yield ts1
                    # print('%7.4f %s: Shuttle:%s moved to bay %s' % (env.now, name, shuttleID, bay))

                    ts2 = env.timeout(time_shuttle_travel2)
                    # print('%7.4f %s: Shuttle:%s moving to buffer' % (env.now, name, shuttleID))
                    yield ts2
                    # print('%7.4f %s: Shuttle:%s moved to buffer' % (env.now, name, shuttleID))

                    shuttle_avail[shuttleID - 1] = 0
                    shuttle.put(shuttleID)
                    shuttle_time = env.now - pickup_time
                    shuttle_util[shuttleID - 1] = shuttle_util[shuttleID - 1] + shuttle_time
                    shuttle_locations[shuttleID]["bay"] = 0
                    if lift1_use == 0:
                        flow_time = env.now - pickup_time
                        cycle_time = env.now - arrive
                        flowtime.append(flow_time)
                        cycletime.append(cycle_time)
                        min_fl = 1 / max(flowtime)
                        max_fl = 1 / min(flowtime)
                        if max_fl == min_fl:
                            norm_reward = 1
                        else:
                            norm_reward = ((1 / flow_time) - min_fl) / (max_fl - min_fl)
                        reward = norm_reward * 100
                        agent.remember(observation, action, reward, observation_, done)
                        agent.learn()
                        #print(
                        #    '%7.4f %s: Finished Shuttle:%s, Cycle time: %7.4f, Flow time: %7.4f' % (env.now, name, shuttleID, cycle_time, flow_time))
                        trans_left += 1
                        proc_check[0] = 0
                    elif lift1_use == 1:
                        found = 0
                        for i in range(len(tier_buffer_lift)):
                            tier_check = tier_buffer_lift[i][0]
                            name_check = tier_buffer_lift[i][1]
                            if tier_check == tier and name_check == name:
                                found = 1
                                del tier_buffer_lift[i]
                                for no_lift1 in range(2):
                                    if lift1_avail[no_lift1] == name:
                                        lift1_travel1 = abs(1 - tier) * heightoftier
                                        time_lift1_travel1 = calctime(a_lift, v_lift, lift1_travel1)
                                        tl1 = env.timeout(time_lift1_travel1)

                                        #print('%7.4f %s: Lift1:%s moving to I/O' % (env.now, name, no_lift1 + 1))
                                        yield tl1
                                        #print('%7.4f %s: Lift1:%s moved to I/O' % (env.now, name, no_lift1 + 1))

                                        #print('%7.4f %s: Lift1:%s released' % (env.now, name, no_lift1 + 1))
                                        lift1.put(no_lift1 + 1)
                                        lift1_locations[no_lift1 + 1] = 1
                                        lift1_avail[no_lift1] = 0
                                        flow_time = env.now - pickup_time
                                        cycle_time = env.now - arrive
                                        flowtime.append(flow_time)
                                        cycletime.append(cycle_time)
                                        min_fl = 1 / max(flowtime)
                                        max_fl = 1 / min(flowtime)
                                        if max_fl == min_fl:
                                            norm_reward = 1
                                        else:
                                            norm_reward = ((1 / flow_time) - min_fl) / (max_fl - min_fl)
                                        reward = norm_reward * 100
                                        agent.remember(observation, action, reward, observation_, done)
                                        agent.learn()
                                        lift1_util[no_lift1] = lift1_util[no_lift1] + (env.now - lift1_pick[no_lift1])
                                        #print('%7.4f %s: Finished Lift1:%s, Cycle time: %7.4f, Flow time: %7.4f' % (
                                        #    env.now, name, no_lift1 + 1, cycle_time, flow_time))
                                        trans_left += 1
                                        proc_check[0] = 0
                                        break
                                break
                        if found == 0:
                            tier_buffer.append(buffer_control)
                            proc_check[0] = 0
            else:
                proc_check[0] = 0
                break
        else:
            proc_check[0] = 0
            break


def shuttle_action2(env, shuttle, shuttleID=2):
    while True:
        proc_check[1] = proc_check[1] + 1
        if len(active_transactions) > 0 and shuttle_avail[shuttleID - 1] == 0:
            name = ""
            available_transactions.clear()
            queue_state = [0] * (tiers * 2)
            # -------- DQN implementation -------- #
            done = False
            if lift1_avail[0] == 0:
                avail_lift11 = 1
            else:
                avail_lift11 = 0
            if lift1_avail[1] == 0:
                avail_lift12 = 1
            else:
                avail_lift12 = 0
            for transaction in range(len(active_transactions)):
                for i in range(tiers):
                    if active_transactions[transaction][2] == i+1 and active_transactions[transaction][1] == 0:
                        queue_state[i*2] += 1
                        break
                    if active_transactions[transaction][2] == i+1 and active_transactions[transaction][1] == 1:
                        queue_state[i*2+1] += 1
                        break
            observation = np.array([shuttle_locations[shuttleID]["tier"], shuttle_locations[shuttleID]["bay"],
                                    lift1_locations_[1], avail_lift11, lift1_locations_[2], avail_lift12, lift2_location_[0]
                                    ])
            for transaction in range(len(active_transactions)):
                transaction_tier = active_transactions[transaction][2] - 1
                if tier_avail[transaction_tier] == 0 or tier_avail[transaction_tier] == shuttleID:
                    available_transactions.append(active_transactions[transaction])
                    arrive = active_transactions[transaction][4]
            if available_transactions:
                action = agent.choose_action(observation)
                if action < (2 * tiers * bays):
                    liftno = 1
                    tmp_type = math.floor(action/(bays*tiers))
                    if ((action + 1) % bays) == 0:
                        tmp_bay = bays
                    else:
                        tmp_bay = (action + 1) % bays
                    tmp_tier = ((action - tmp_bay + 1) / bays) - (tmp_type * tiers) + 1
                else:
                    liftno = 2
                    tmp_action = action - (2 * tiers * bays)
                    tmp_type = math.floor(tmp_action / (bays * tiers))
                    if ((tmp_action + 1) % bays) == 0:
                        tmp_bay = bays
                    else:
                        tmp_bay = (tmp_action + 1) % bays
                    tmp_tier = ((tmp_action - tmp_bay + 1) / bays) - (tmp_type * tiers) + 1
                for ts in range(len(available_transactions)):
                    if available_transactions[ts][1] == tmp_type and available_transactions[ts][2] == tmp_tier and available_transactions[ts][3] == tmp_bay:
                        for transaction in range(len(active_transactions)):
                            if active_transactions[transaction][0] == available_transactions[ts][0]:
                                del active_transactions[transaction]
                                break
                        name = available_transactions[ts][0]
                        type = available_transactions[ts][1]
                        tier = available_transactions[ts][2]
                        bay = available_transactions[ts][3]
                        arrive = available_transactions[ts][4]
                        if shuttle_locations[shuttleID]["tier"] != tier:
                            lift2_location_[0] = tier
                        else:
                            lift2_location_[0] = lift2_location[0]
                        if liftno == 1:
                            if type == 0:
                                observation_ = np.array([tier, bay, tier, 1, lift1_locations_[2], 1, lift2_location_[0]
                                    ])
                            else:
                                observation_ = np.array([tier, 1, 1, 1, lift1_locations_[2], 1, lift2_location_[0]
                                    ])
                        else:
                            if type == 0:
                                observation_ = np.array([tier, bay, 1, lift1_locations_[1], tier, 1, lift2_location_[0]
                                    ])
                            else:
                                observation_ = np.array([tier, 1, 1, lift1_locations_[1], 1, 1, lift2_location_[0]
                                    ])
                # Process start
                global trans_entered
                global trans_left
                trans_entered += 1
                yield shuttle.get(lambda shuttleno: shuttleno == shuttleID)
                buffer_control = [tier, name]
                shuttle_avail[shuttleID - 1] = name
                tier_avail[tier - 1] = shuttleID
                pickup_time = env.now
                wait = pickup_time - arrive
                #print('%7.4f %s: Waited %6.3f, Chosen Shuttle: %s' % (env.now, name, wait, shuttleID))
                if tier != 1:
                    lift1_move = lift1_action(env, name, type, shuttleID, lift1, tier, bay, arrive, liftno, observation,
                                              action, observation_)
                    env.process(lift1_move)
                    lift1_use = 1
                elif tier == 1:
                    lift1_use = 0
                if shuttle_locations[shuttleID]["tier"] != tier:
                    temp_tier = shuttle_locations[shuttleID]["tier"]
                    s_travel1 = abs(shuttle_locations[shuttleID]["bay"] - bays) * lengthofbay
                    t_st1 = calctime(a_shuttle, v_shuttle, s_travel1)
                    l2_travel1 = abs(lift2_location[0] - temp_tier) * heightoftier
                    t_l2t = calctime(a_lift, v_lift, l2_travel1)
                    l2_travel2 = abs(tier - temp_tier) * heightoftier
                    t_l2t2 = calctime(a_lift, v_lift, l2_travel2)
                    req_lift2 = lift2.request()
                    yield req_lift2
                    lift2_time1 = env.now
                    to1 = env.timeout(t_st1)
                    to2 = env.timeout(t_l2t)
                    # print('%7.4f %s: Shuttle:%s moving to Lift 2 buffer' % (env.now, name, shuttleID))
                    # print('%7.4f %s: Lift 2 moving to %s tier to pick up Shuttle %s' % (
                    #    env.now, name, temp_tier, shuttleID))
                    yield to1 & to2
                    # print('%7.4f %s: Shuttle:%s moved to Lift 2 buffer' % (env.now, name, shuttleID))
                    to3 = env.timeout(t_l2t2)
                    tier_avail[temp_tier - 1] = 0
                    # print('%7.4f %s: Lift 2 moving to tier %s' % (env.now, name, tier))
                    yield to3
                    # print('%7.4f %s: Lift 2 moved to tier %s' % (env.now, name, tier))
                    shuttle_locations[shuttleID]["tier"] = tier
                    shuttle_locations[shuttleID]["bay"] = bays
                    lift2_util[0] = lift2_util[0] + (env.now - lift2_time1)
                    lift2.release(req_lift2)
                if type == 0:

                    shuttle_travel1 = abs(shuttle_locations[shuttleID]["bay"] - 0) * lengthofbay
                    time_shuttle_travel1 = calctime(a_shuttle, v_shuttle, shuttle_travel1)
                    shuttle_travel2 = bay * lengthofbay
                    time_shuttle_travel2 = calctime(a_shuttle, v_shuttle, shuttle_travel2)
                    t1 = env.timeout(time_shuttle_travel1)
                    # print('%7.4f %s: Shuttle:%s moving to buffer' % (env.now, name, shuttleID))
                    yield t1
                    # print('%7.4f %s: Shuttle:%s moved to buffer' % (env.now, name, shuttleID))
                    if lift1_use == 0:
                        t2 = env.timeout(time_shuttle_travel2)
                        # print('%7.4f %s: Shuttle:%s moving to bay %s' % (env.now, name, shuttleID, bay))
                        yield t2
                        # print('%7.4f %s: Shuttle:%s moved to bay %s' % (env.now, name, shuttleID, bay))

                        shuttle_avail[shuttleID - 1] = 0
                        shuttle.put(shuttleID)
                        shuttle_time = env.now - pickup_time
                        shuttle_util[shuttleID - 1] = shuttle_util[shuttleID - 1] + shuttle_time
                        shuttle_locations[shuttleID]["bay"] = bay
                        flow_time = env.now - pickup_time
                        cycle_time = env.now - arrive
                        flowtime.append(flow_time)
                        cycletime.append(cycle_time)
                        min_fl = 1 / max(flowtime)
                        max_fl = 1 / min(flowtime)
                        if max_fl == min_fl:
                            norm_reward = 1
                        else:
                            norm_reward = ((1 / flow_time) - min_fl) / (max_fl - min_fl)
                        reward = norm_reward * 100
                        agent.remember(observation, action, reward, observation_, done)
                        agent.learn()
                        #print('%7.4f %s: Finished Shuttle:%s, Cycle time: %7.4f, Flow time: %7.4f' % (
                        #    env.now, name, shuttleID, cycle_time, flow_time))
                        trans_left += 1
                        proc_check[1] = 0
                        env.process(shuttle_action2(env, shuttle, shuttleID))
                        break
                    elif lift1_use == 1:
                        found = 0
                        for i in range(len(tier_buffer_lift)):
                            tier_check = tier_buffer_lift[i][0]
                            name_check = tier_buffer_lift[i][1]
                            if tier_check == tier and name_check == name:
                                found = 1
                                del tier_buffer_lift[i]
                                t2 = env.timeout(time_shuttle_travel2)
                                # print('%7.4f %s: Shuttle:%s moving to bay %s' % (env.now, name, shuttleID, bay))
                                yield t2
                                # print('%7.4f %s: Shuttle:%s moved to bay %s' % (env.now, name, shuttleID, bay))

                                shuttle_avail[shuttleID - 1] = 0
                                shuttle.put(shuttleID)
                                shuttle_time = env.now - pickup_time
                                shuttle_util[shuttleID - 1] = shuttle_util[shuttleID - 1] + shuttle_time
                                shuttle_locations[shuttleID]["bay"] = bay
                                flow_time = env.now - pickup_time
                                cycle_time = env.now - arrive
                                flowtime.append(flow_time)
                                cycletime.append(cycle_time)
                                min_fl = 1 / max(flowtime)
                                max_fl = 1 / min(flowtime)
                                if max_fl == min_fl:
                                    norm_reward = 1
                                else:
                                    norm_reward = ((1 / flow_time) - min_fl) / (max_fl - min_fl)
                                reward = norm_reward * 100
                                agent.remember(observation, action, reward, observation_, done)
                                agent.learn()
                                #print('%7.4f %s: Finished Shuttle:%s, Cycle time: %7.4f, Flow time: %7.4f' % (
                                #    env.now, name, shuttleID, cycle_time, flow_time))
                                trans_left += 1
                                break
                        if found == 1:
                            proc_check[1] = 0
                            env.process(shuttle_action2(env, shuttle, shuttleID))
                            break
                        if found == 0:
                            tier_buffer.append(buffer_control)
                            proc_check[1] = 0
                            break
                else:
                    shuttle_travel1 = abs(shuttle_locations[shuttleID]["bay"] - bay) * lengthofbay
                    time_shuttle_travel1 = calctime(a_shuttle, v_shuttle, shuttle_travel1)
                    shuttle_travel2 = bay * lengthofbay
                    time_shuttle_travel2 = calctime(a_shuttle, v_shuttle, shuttle_travel2)
                    ts1 = env.timeout(time_shuttle_travel1)

                    # print('%7.4f %s: Shuttle:%s moving to bay %s' % (env.now, name, shuttleID, bay))
                    yield ts1
                    # print('%7.4f %s: Shuttle:%s moved to bay %s' % (env.now, name, shuttleID, bay))

                    ts2 = env.timeout(time_shuttle_travel2)
                    # print('%7.4f %s: Shuttle:%s moving to buffer' % (env.now, name, shuttleID))
                    yield ts2
                    # print('%7.4f %s: Shuttle:%s moved to buffer' % (env.now, name, shuttleID))

                    shuttle_avail[shuttleID - 1] = 0
                    shuttle.put(shuttleID)
                    shuttle_time = env.now - pickup_time
                    shuttle_util[shuttleID - 1] = shuttle_util[shuttleID - 1] + shuttle_time
                    shuttle_locations[shuttleID]["bay"] = 0
                    if lift1_use == 0:
                        flow_time = env.now - pickup_time
                        cycle_time = env.now - arrive
                        flowtime.append(flow_time)
                        cycletime.append(cycle_time)
                        min_fl = 1 / max(flowtime)
                        max_fl = 1 / min(flowtime)
                        if max_fl == min_fl:
                            norm_reward = 1
                        else:
                            norm_reward = ((1 / flow_time) - min_fl) / (max_fl - min_fl)
                        reward = norm_reward * 100
                        agent.remember(observation, action, reward, observation_, done)
                        agent.learn()
                        #print(
                        #    '%7.4f %s: Finished Shuttle:%s, Cycle time: %7.4f, Flow time: %7.4f' % (env.now, name, shuttleID, cycle_time, flow_time))
                        trans_left += 1
                        proc_check[1] = 0
                    elif lift1_use == 1:
                        found = 0
                        for i in range(len(tier_buffer_lift)):
                            tier_check = tier_buffer_lift[i][0]
                            name_check = tier_buffer_lift[i][1]
                            if tier_check == tier and name_check == name:
                                found = 1
                                del tier_buffer_lift[i]
                                for no_lift1 in range(2):
                                    if lift1_avail[no_lift1] == name:
                                        lift1_travel1 = abs(1 - tier) * heightoftier
                                        time_lift1_travel1 = calctime(a_lift, v_lift, lift1_travel1)
                                        tl1 = env.timeout(time_lift1_travel1)

                                        #print('%7.4f %s: Lift1:%s moving to I/O' % (env.now, name, no_lift1 + 1))
                                        yield tl1
                                        #print('%7.4f %s: Lift1:%s moved to I/O' % (env.now, name, no_lift1 + 1))

                                        #print('%7.4f %s: Lift1:%s released' % (env.now, name, no_lift1 + 1))
                                        lift1.put(no_lift1 + 1)
                                        lift1_locations[no_lift1 + 1] = 1
                                        lift1_avail[no_lift1] = 0
                                        flow_time = env.now - pickup_time
                                        cycle_time = env.now - arrive
                                        flowtime.append(flow_time)
                                        cycletime.append(cycle_time)
                                        min_fl = 1 / max(flowtime)
                                        max_fl = 1 / min(flowtime)
                                        if max_fl == min_fl:
                                            norm_reward = 1
                                        else:
                                            norm_reward = ((1 / flow_time) - min_fl) / (max_fl - min_fl)
                                        reward = norm_reward * 100
                                        agent.remember(observation, action, reward, observation_, done)
                                        agent.learn()
                                        lift1_util[no_lift1] = lift1_util[no_lift1] + (env.now - lift1_pick[no_lift1])
                                        #print('%7.4f %s: Finished Lift1:%s, Cycle time: %7.4f, Flow time: %7.4f' % (
                                        #    env.now, name, no_lift1 + 1, cycle_time, flow_time))
                                        trans_left += 1
                                        proc_check[1] = 0
                                        break
                                break
                        if found == 0:
                            tier_buffer.append(buffer_control)
                            proc_check[1] = 0
            else:
                proc_check[1] = 0
                break
        else:
            proc_check[1] = 0
            break


def lift1_action(env, name, type, shuttleID, lift1, tier, bay, arrive, liftno, observation, action, observation_):
    if type == 0:
        lift1_locations_[liftno] = tier
    else:
        lift1_locations_[liftno] = 1
    yield lift1.get(lambda lift_no: lift_no == liftno)
    global trans_left
    buffer_control = [tier, name]
    lift1_avail[liftno - 1] = name
    pickup_time = env.now
    lift1_pick[liftno - 1] = pickup_time
    if type == 0:

        lift1_travel1 = abs(lift1_locations[liftno] - 1) * heightoftier
        time_lift1_travel1 = calctime(a_lift, v_lift, lift1_travel1)
        lift1_travel2 = abs(1 - tier) * heightoftier
        time_lift1_travel2 = calctime(a_lift, v_lift, lift1_travel2)
        t1 = env.timeout(time_lift1_travel1)
        #print('%7.4f %s: Lift1:%s moving to I/O' % (env.now, name, liftno))
        yield t1
        #print('%7.4f %s: Lift1:%s moved to I/O' % (env.now, name, liftno))

        t2 = env.timeout(time_lift1_travel2)
        #print('%7.4f %s: Lift1:%s moving to tier %s' % (env.now, name, liftno, tier))
        yield t2
        #print('%7.4f %s: Lift1:%s moved to tier %s' % (env.now, name, liftno, tier))
        #print('%7.4f %s: Lift1:%s released' % (env.now, name, liftno))
        lift1_buffer_control[tier - 1] = name
        tier_buffer_lift.append(buffer_control)
        lift1.put(liftno)
        lift1_avail[liftno - 1] = 0
        lift1_locations[liftno] = tier
        lift1_time = env.now - pickup_time
        lift1_util[liftno - 1] = lift1_util[liftno - 1] + lift1_time
        for i in range(len(tier_buffer)):
            tier_check = tier_buffer[i][0]
            name_check = tier_buffer[i][1]
            if tier_check == tier and name_check == name:
                del tier_buffer[i]
                shuttle_travel2 = bay * lengthofbay
                time_shuttle_travel2 = calctime(a_shuttle, v_shuttle, shuttle_travel2)
                ts2 = env.timeout(time_shuttle_travel2)
                # print('%7.4f %s: Shuttle:%s moving to bay %s' % (env.now, name, shuttleID, bay))
                yield ts2
                # print('%7.4f %s: Shuttle:%s moved to bay %s' % (env.now, name, shuttleID, bay))

                shuttle_avail[shuttleID - 1] = 0
                shuttle.put(shuttleID)
                shuttle_time = env.now - pickup_time
                shuttle_util[shuttleID - 1] = shuttle_util[shuttleID - 1] + shuttle_time
                shuttle_locations[shuttleID]["bay"] = bay
                flow_time = env.now - pickup_time
                cycle_time = env.now - arrive
                flowtime.append(flow_time)
                cycletime.append(cycle_time)
                min_fl = 1 / max(flowtime)
                max_fl = 1 / min(flowtime)
                if max_fl == min_fl:
                    norm_reward = 1
                else:
                    norm_reward = ((1 / flow_time) - min_fl) / (max_fl - min_fl)
                reward = norm_reward * 100
                agent.remember(observation, action, reward, observation_, False)
                agent.learn()
                #print('%7.4f %s: Finished Shuttle:%s, Cycle time: %7.4f, Flow time: %7.4f' % (env.now, name, shuttleID, cycle_time, flow_time))
                trans_left += 1
                if shuttleID == 1:
                    if proc_check[0] == 0:
                        env.process(shuttle_action1(env, shuttle))
                elif shuttleID == 2:
                    if proc_check[1] == 0:
                        env.process(shuttle_action2(env, shuttle))
                break

    else:

        lift1_travel1 = abs(lift1_locations[liftno] - tier) * heightoftier
        time_lift1_travel1 = calctime(a_lift, v_lift, lift1_travel1)
        lift1_travel2 = abs(1 - tier) * heightoftier
        time_lift1_travel2 = calctime(a_lift, v_lift, lift1_travel2)
        t1 = env.timeout(time_lift1_travel1)
        # print('%7.4f %s: Lift1:%s moving to tier %s' % (env.now, name, liftno, tier))
        yield t1
        # print('%7.4f %s: Lift1:%s moved to tier %s' % (env.now, name, liftno, tier))

        lift1_buffer_control[tier - 1] = name
        tier_buffer_lift.append(buffer_control)

        for i in range(len(tier_buffer)):
            tier_check = tier_buffer[i][0]
            name_check = tier_buffer[i][1]
            if tier_check == tier and name_check == name:
                del tier_buffer[i]
                t2 = env.timeout(time_lift1_travel2)
                # print('%7.4f %s: Lift1:%s moving to I/O' % (env.now, name, liftno))
                yield t2
                # print('%7.4f %s: Lift1:%s moved to I/O' % (env.now, name, liftno))
                # print('%7.4f %s: Lift1:%s released' % (env.now, name, liftno))
                lift1.put(liftno)
                lift1_locations[liftno] = 1
                lift1_avail[liftno - 1] = 0
                flow_time = env.now - pickup_time
                cycle_time = env.now - arrive
                flowtime.append(flow_time)
                cycletime.append(cycle_time)
                min_fl = 1 / max(flowtime)
                max_fl = 1 / min(flowtime)
                if max_fl == min_fl:
                    norm_reward = 1
                else:
                    norm_reward = ((1 / flow_time) - min_fl) / (max_fl - min_fl)
                reward = norm_reward * 100
                agent.remember(observation, action, reward, observation_, False)
                agent.learn()
                lift1_util[liftno - 1] = lift1_util[liftno - 1] + (env.now - lift1_pick[liftno - 1])
                #print('%7.4f %s: Finished Lift1:%s, Cycle time: %7.4f, Flow time: %7.4f' % (env.now, name, liftno, cycle_time, flow_time))
                trans_left += 1
                break


class ReplayBuffer(object):
    def __init__(self, max_size, input_shape, n_actions, discrete=False):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.discrete = discrete
        self.state_memory = np.zeros((self.mem_size, input_shape))
        self.new_state_memory = np.zeros((self.mem_size, input_shape))
        dtype = np.int8 if self.discrete else np.float32
        self.action_memory = np.zeros((self.mem_size, n_actions), dtype=dtype)
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.float32)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward
        self.terminal_memory[index] = 1 - int(done)
        if self.discrete:
            actions = np.zeros(self.action_memory.shape[1])
            actions[action] = 1.0
            self.action_memory[index] = actions
        else:
            self.action_memory[index] = action
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size)

        states = self.state_memory[batch]
        states_ = self.new_state_memory[batch]
        rewards = self.reward_memory[batch]
        actions = self.action_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, actions, rewards, states_, terminal


def build_dqn(lr, n_actions, input_dims, fc1_dims, fc2_dims):
    model = Sequential([
        Dense(fc1_dims, input_shape=(input_dims,)),
        Activation('relu'),
        Dense(fc2_dims),
        Activation('relu'),
        Dense(n_actions)])
    model.compile(optimizer=Adam(lr=lr), loss='mse', metrics=['mse', 'mae', 'mape'])
    #model.compile(optimizer=RMSprop(lr=lr))

    return model


class Agent(object):
    def __init__(self, alpha, gamma, n_actions, epsilon, batch_size, input_dims,
                 epsilon_dec=0.9999, epsilon_end=0.01, mem_size=1000000, fname='dqn_5_25.h5'):
        self.action_space = [i for i in range(n_actions)]
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_dec = epsilon_dec
        self.epsilon_min = epsilon_end
        self.batch_size = batch_size
        self.model_file = fname
        self.memory = ReplayBuffer(mem_size, input_dims, n_actions, discrete=True)
        self.q_eval = build_dqn(alpha, n_actions, input_dims, 256, 256)

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def choose_action(self, state):
        state = state[np.newaxis, :]
        rand = np.random.random()
        temp_actions = []
        temp_predicts = [i for i in range(self.n_actions)]
        for i in range(len(available_transactions)):
            temp_actions.append(available_transactions[i][5])
            temp_actions.append(available_transactions[i][5] + (bays * tiers * 2))
        if rand < self.epsilon:
            action = np.random.choice(temp_actions)
        else:
            actions = self.q_eval.predict(state)
            for i in range(len(temp_predicts)):
                if temp_predicts[i] in temp_actions:
                    temp_predicts[i] = actions[0][i]
                else:
                    temp_predicts[i] = -100
            action = np.argmax(temp_predicts)
        return action

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return
        state, action, reward, new_state, done = self.memory.sample_buffer(self.batch_size)
        action_values = np.array(self.action_space, dtype=np.int16)
        action_indices = np.dot(action, action_values)
        q_eval = self.q_eval.predict(state)
        q_next = self.q_eval.predict(new_state)
        q_target = q_eval.copy()
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        q_target[batch_index, action_indices] = reward + self.gamma * np.max(q_next, axis=1) * done
        _ = self.q_eval.fit(state, q_target, verbose=0)
        self.epsilon = self.epsilon * self.epsilon_dec if self.epsilon > self.epsilon_min else self.epsilon_min
        mean_cycletime.append(mean(cycletime)) if len(cycletime) > 0 else mean_cycletime.append(cycletime)
        #print(self.epsilon)

    def save_model(self):
        self.q_eval.save(self.model_file)

    def load_model(self):
        self.q_eval = load_model(self.model_file)


env = simpy.Environment()
shuttle = simpy.FilterStore(env, capacity=shuttle_no)
shuttle.items = shuttleNo
lift1 = simpy.FilterStore(env, capacity=2)
lift1.items = lift1No
lift2 = simpy.Resource(env, capacity=1)
env.process(source(env, transaction_interval))
agent = Agent(gamma=0.2, epsilon=1.0, alpha=0.001, input_dims=7, n_actions=tiers * bays * 2 * 2,
              mem_size=1000000, batch_size=64, epsilon_end=0.01)
#agent.load_model()
env.run(until=run_time)
# sys.stdout.close()
# sys.stdout=stdoutOrigin

shuttle_utilization = [x / (run_time - stable_time) for x in shuttle_util]
lift1_utilization = [x / (run_time - stable_time) for x in lift1_util]
lift2_utilization = [x / (run_time - stable_time) for x in lift2_util]
if cycletime:
    print("Average cycle time is: %6.3f seconds." % mean(cycletime))
    print("Average flow time is: %6.3f seconds." % mean(flowtime))
for a in range(len(shuttle_utilization)):
    print("Shuttle %s utilization is %4.2f" % (a + 1, shuttle_utilization[a]))
for a in range(len(lift1_utilization)):
    print("Lift 1:%s utilization is %4.2f" % (a + 1, lift1_utilization[a]))
print("Lift 2 utilization is %4.2f" % (lift2_utilization[0]))
print("Number of transactions entered the system: ", trans_entered)
print("Number of transactions left the system: ", trans_left)
#agent.save_model()
c_mean = np.array(mean_cycletime)
time = np.arange(0, len(c_mean))
plt.plot(time, c_mean)
plt.show()