import random
import simpy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from math import sqrt
from statistics import mean
import sys

#stdoutOrigin = sys.stdout
#sys.stdout = open("Output.txt", "w")

run_time = 1728000
transaction_interval = 12  # every 7.6 seconds create transaction
tiers = 5
bays = 50
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
"""
shuttle_locations = {
    1: {"tier": 1,
        "bay": 10},
    2: {"tier": 2,
        "bay": 10},
    3: {"tier": 3,
        "bay": 10},
    4: {"tier": 4,
        "bay": 10},
    5: {"tier": 5,
        "bay": 10}
}"""

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

lift2_location = [3]

bay_util = {
    1: {"bay": 0,
        "side": 0},
    2: {"bay": 0,
        "side": 0},
    3: {"bay": 0,
        "side": 0},
    4: {"bay": 0,
        "side": 0},
    5: {"bay": 0,
        "side": 0}
}

shuttle_util = [0] * shuttle_no
lift1_util = [0] * 2
lift2_util = [0]
shuttle_avail = [0] * shuttle_no
lift1_avail = [0] * 2
lift1_pick = [0] * 2
active_transactions = []
tier_buffer = []
tier_buffer_lift = []
flowtime = []
cycletime = []
mean_cycletime = []
tier_avail = [0] * tiers
lift1_buffer_control = [0] * tiers
proc_check = [0] * shuttle_no
trans_entered = 0
trans_left = 0

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
    print("5tier")
    while True:
        t_ID += 1
        t_type = bool(random.getrandbits(1))
        t_tier = random.randint(1, tiers)
        t_bay = random.randint(1, bays)
        t_time = env.now
        side = random.randint(1, 2)
        t_info = [t_ID, t_type, t_tier, t_bay, t_time]
        active_transactions.append(t_info)
        if t_type == 0:
            type1 = "Storage"
        else:
            type1 = "Retrieval"
        #print('%7.4f %s: Created as %s, Destination tier: %s, bay: %s' % (
        #    env.now, t_ID, type1, t_tier, t_bay))

        check_time = env.now

        if check_time >= (run_time/2) and timer_reset == 1:
            shuttle_util[0] = 0
            shuttle_util[1] = 0
            lift1_util[0] = 0
            lift1_util[1] = 0
            lift2_util[0] = 0
            cycletime.clear()
            mean_cycletime.clear()
            timer_reset = 0

        if proc_check[0] == 0:
            env.process(shuttle_action1(env, shuttle))

        if proc_check[1] == 0:
            env.process(shuttle_action2(env, shuttle, 2))
        if t_ID % 1000 == 0 and mean_cycletime:
            print("%7.4f: Avg cycle time %4.2f" % (env.now, mean_cycletime[-1]))
        t = random.expovariate(1.0 / interval)
        yield env.timeout(t)

def shuttle_action1(env, shuttle, shuttleID=1):
    while True:
        proc_check[0] = proc_check[0] + 1
        if len(active_transactions) > 0 and shuttle_avail[shuttleID - 1] == 0:
            name = ""
            # -------- DQN implementation
            observation = 1
            for transaction in range(len(active_transactions)):
                transaction_tier = active_transactions[transaction][2] - 1
                if tier_avail[transaction_tier] == 0 or tier_avail[transaction_tier] == shuttleID:
                    name = active_transactions[transaction][0]
                    type = active_transactions[transaction][1]
                    tier = active_transactions[transaction][2]
                    bay = active_transactions[transaction][3]
                    arrive = active_transactions[transaction][4]
                    del active_transactions[transaction]
                    break
            # Process start
            if name != "":
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
                    lift1_move = lift1_action(env, name, type, shuttleID, lift1, tier, bay, arrive)
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
                    #print('%7.4f %s: Shuttle:%s moving to Lift 2 buffer' % (env.now, name, shuttleID))
                    #print('%7.4f %s: Lift 2 moving to %s tier to pick up Shuttle %s' % (
                    #env.now, name, temp_tier, shuttleID))
                    yield to1 & to2
                    #print('%7.4f %s: Shuttle:%s moved to Lift 2 buffer' % (env.now, name, shuttleID))
                    to3 = env.timeout(t_l2t2)
                    tier_avail[temp_tier - 1] = 0
                    #print('%7.4f %s: Lift 2 moving to tier %s' % (env.now, name, tier))
                    yield to3
                    #print('%7.4f %s: Lift 2 moved to tier %s' % (env.now, name, tier))
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
                    #print('%7.4f %s: Shuttle:%s moving to buffer' % (env.now, name, shuttleID))
                    yield t1
                    #print('%7.4f %s: Shuttle:%s moved to buffer' % (env.now, name, shuttleID))
                    if lift1_use == 0:
                        t2 = env.timeout(time_shuttle_travel2)
                        #print('%7.4f %s: Shuttle:%s moving to bay %s' % (env.now, name, shuttleID, bay))
                        yield t2
                        #print('%7.4f %s: Shuttle:%s moved to bay %s' % (env.now, name, shuttleID, bay))

                        shuttle_avail[shuttleID - 1] = 0
                        shuttle.put(shuttleID)
                        shuttle_time = env.now - pickup_time
                        shuttle_util[shuttleID - 1] = shuttle_util[shuttleID - 1] + shuttle_time
                        shuttle_locations[shuttleID]["bay"] = bay
                        flow_time = env.now - pickup_time
                        cycle_time = env.now - arrive
                        flowtime.append(flow_time)
                        cycletime.append(cycle_time)
                        mean_cycletime.append(mean(cycletime)) if len(cycletime) > 0 else mean_cycletime.append(cycletime)
                        #print('%7.4f %s: Finished Shuttle:%s, Cycle time: %7.4f' % (env.now, name, shuttleID, cycle_time))
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
                                #print('%7.4f %s: Shuttle:%s moving to bay %s' % (env.now, name, shuttleID, bay))
                                yield t2
                                #print('%7.4f %s: Shuttle:%s moved to bay %s' % (env.now, name, shuttleID, bay))

                                shuttle_avail[shuttleID - 1] = 0
                                shuttle.put(shuttleID)
                                shuttle_time = env.now - pickup_time
                                shuttle_util[shuttleID - 1] = shuttle_util[shuttleID - 1] + shuttle_time
                                shuttle_locations[shuttleID]["bay"] = bay
                                flow_time = env.now - pickup_time
                                cycle_time = env.now - arrive
                                flowtime.append(flow_time)
                                cycletime.append(cycle_time)
                                mean_cycletime.append(mean(cycletime)) if len(cycletime) > 0 else mean_cycletime.append(cycletime)
                                #print('%7.4f %s: Finished Shuttle:%s, Cycle time: %7.4f' % (env.now, name, shuttleID, cycle_time))
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

                    #print('%7.4f %s: Shuttle:%s moving to bay %s' % (env.now, name, shuttleID, bay))
                    yield ts1
                    #print('%7.4f %s: Shuttle:%s moved to bay %s' % (env.now, name, shuttleID, bay))

                    ts2 = env.timeout(time_shuttle_travel2)
                    #print('%7.4f %s: Shuttle:%s moving to buffer' % (env.now, name, shuttleID))
                    yield ts2
                    #print('%7.4f %s: Shuttle:%s moved to buffer' % (env.now, name, shuttleID))

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
                        mean_cycletime.append(mean(cycletime)) if len(cycletime) > 0 else mean_cycletime.append(cycletime)
                        #print('%7.4f %s: Finished Shuttle:%s, Cycle time: %7.4f' % (env.now, name, shuttleID, cycle_time))
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

                                        #print('%7.4f %s: Lift1:%s moving to I/O' % (env.now, name, no_lift1+1))
                                        yield tl1
                                        #print('%7.4f %s: Lift1:%s moved to I/O' % (env.now, name, no_lift1+1))

                                        #print('%7.4f %s: Lift1:%s released' % (env.now, name, no_lift1 + 1))
                                        lift1.put(no_lift1+1)
                                        lift1_locations[no_lift1+1] = 1
                                        lift1_avail[no_lift1] = 0
                                        flow_time = env.now - pickup_time
                                        cycle_time = env.now - arrive
                                        flowtime.append(flow_time)
                                        cycletime.append(cycle_time)
                                        mean_cycletime.append(mean(cycletime)) if len(cycletime) > 0 else mean_cycletime.append(cycletime)
                                        lift1_util[no_lift1] = lift1_util[no_lift1] + (env.now - lift1_pick[no_lift1])
                                        #print('%7.4f %s: Finished Lift1:%s, Cycle time: %7.4f' % (env.now, name, no_lift1+1, cycle_time))
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
            yield env.timeout(0.0001)
            for transaction in range(len(active_transactions)):
                transaction_tier = active_transactions[transaction][2] - 1
                if tier_avail[transaction_tier] == 0 or tier_avail[transaction_tier] == shuttleID:
                    name = active_transactions[transaction][0]
                    type = active_transactions[transaction][1]
                    tier = active_transactions[transaction][2]
                    bay = active_transactions[transaction][3]
                    arrive = active_transactions[transaction][4]
                    del active_transactions[transaction]
                    break
            # Process start
            if name != "":
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
                    lift1_move = lift1_action(env, name, type, shuttleID, lift1, tier, bay, arrive)
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
                    #print('%7.4f %s: Shuttle:%s moving to Lift 2 buffer' % (env.now, name, shuttleID))
                    #print('%7.4f %s: Lift 2 moving to %s tier to pick up Shuttle %s' % (
                    #    env.now, name, temp_tier, shuttleID))
                    yield to1 & to2
                    #print('%7.4f %s: Shuttle:%s moved to Lift 2 buffer' % (env.now, name, shuttleID))
                    to3 = env.timeout(t_l2t2)
                    tier_avail[temp_tier - 1] = 0
                    #print('%7.4f %s: Lift 2 moving to tier %s' % (env.now, name, tier))
                    yield to3
                    #print('%7.4f %s: Lift 2 moved to tier %s' % (env.now, name, tier))
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
                    #print('%7.4f %s: Shuttle:%s moving to buffer' % (env.now, name, shuttleID))
                    yield t1
                    #print('%7.4f %s: Shuttle:%s moved to buffer' % (env.now, name, shuttleID))
                    if lift1_use == 0:
                        t2 = env.timeout(time_shuttle_travel2)
                        #print('%7.4f %s: Shuttle:%s moving to bay %s' % (env.now, name, shuttleID, bay))
                        yield t2
                        #print('%7.4f %s: Shuttle:%s moved to bay %s' % (env.now, name, shuttleID, bay))

                        shuttle_avail[shuttleID - 1] = 0
                        shuttle.put(shuttleID)
                        shuttle_time = env.now - pickup_time
                        shuttle_util[shuttleID - 1] = shuttle_util[shuttleID - 1] + shuttle_time
                        shuttle_locations[shuttleID]["bay"] = bay
                        flow_time = env.now - pickup_time
                        cycle_time = env.now - arrive
                        flowtime.append(flow_time)
                        cycletime.append(cycle_time)
                        mean_cycletime.append(mean(cycletime)) if len(cycletime) > 0 else mean_cycletime.append(cycletime)
                        #print('%7.4f %s: Finished Shuttle:%s, Cycle time: %7.4f' % (env.now, name, shuttleID, cycle_time))
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
                                #print('%7.4f %s: Shuttle:%s moving to bay %s' % (env.now, name, shuttleID, bay))
                                yield t2
                                #print('%7.4f %s: Shuttle:%s moved to bay %s' % (env.now, name, shuttleID, bay))

                                shuttle_avail[shuttleID - 1] = 0
                                shuttle.put(shuttleID)
                                shuttle_time = env.now - pickup_time
                                shuttle_util[shuttleID - 1] = shuttle_util[shuttleID - 1] + shuttle_time
                                shuttle_locations[shuttleID]["bay"] = bay
                                flow_time = env.now - pickup_time
                                cycle_time = env.now - arrive
                                flowtime.append(flow_time)
                                cycletime.append(cycle_time)
                                mean_cycletime.append(mean(cycletime)) if len(cycletime) > 0 else mean_cycletime.append(cycletime)
                                #print('%7.4f %s: Finished Shuttle:%s, Cycle time: %7.4f' % (env.now, name, shuttleID, cycle_time))
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

                    #print('%7.4f %s: Shuttle:%s moving to bay %s' % (env.now, name, shuttleID, bay))
                    yield ts1
                    #print('%7.4f %s: Shuttle:%s moved to bay %s' % (env.now, name, shuttleID, bay))

                    ts2 = env.timeout(time_shuttle_travel2)
                    #print('%7.4f %s: Shuttle:%s moving to buffer' % (env.now, name, shuttleID))
                    yield ts2
                    #print('%7.4f %s: Shuttle:%s moved to buffer' % (env.now, name, shuttleID))

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
                        mean_cycletime.append(mean(cycletime)) if len(cycletime) > 0 else mean_cycletime.append(cycletime)
                        #print('%7.4f %s: Finished Shuttle:%s, Cycle time: %7.4f' % (env.now, name, shuttleID, cycle_time))
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
                                        mean_cycletime.append(mean(cycletime)) if len(cycletime) > 0 else mean_cycletime.append(cycletime)
                                        lift1_util[no_lift1] = lift1_util[no_lift1] + (env.now - lift1_pick[no_lift1])
                                        #print('%7.4f %s: Finished Lift1:%s, Cycle time: %7.4f' % (env.now, name, no_lift1 + 1, cycle_time))
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


def lift1_action(env, name, type, shuttleID, lift1, tier, bay, arrive):
    req2 = yield lift1.get()
    global trans_left
    buffer_control = [tier, name]
    lift1_avail[req2 - 1] = name
    pickup_time = env.now
    lift1_pick[req2 - 1] = pickup_time
    if type == 0:

        lift1_travel1 = abs(lift1_locations[req2] - 1) * heightoftier
        time_lift1_travel1 = calctime(a_lift, v_lift, lift1_travel1)
        lift1_travel2 = abs(1 - tier) * heightoftier
        time_lift1_travel2 = calctime(a_lift, v_lift, lift1_travel2)
        t1 = env.timeout(time_lift1_travel1)
        #print('%7.4f %s: Lift1:%s moving to I/O' % (env.now, name, req2))
        yield t1
        #print('%7.4f %s: Lift1:%s moved to I/O' % (env.now, name, req2))

        t2 = env.timeout(time_lift1_travel2)
        #print('%7.4f %s: Lift1:%s moving to tier %s' % (env.now, name, req2, tier))
        yield t2
        #print('%7.4f %s: Lift1:%s moved to tier %s' % (env.now, name, req2, tier))
        #print('%7.4f %s: Lift1:%s released' % (env.now, name, req2))
        lift1_buffer_control[tier - 1] = name
        tier_buffer_lift.append(buffer_control)
        lift1.put(req2)
        lift1_locations[req2] = tier
        lift1_time = env.now - pickup_time
        lift1_util[req2 - 1] = lift1_util[req2 - 1] + lift1_time
        for i in range(len(tier_buffer)):
            tier_check = tier_buffer[i][0]
            name_check = tier_buffer[i][1]
            if tier_check == tier and name_check == name:
                del tier_buffer[i]
                shuttle_travel2 = bay * lengthofbay
                time_shuttle_travel2 = calctime(a_shuttle, v_shuttle, shuttle_travel2)
                ts2 = env.timeout(time_shuttle_travel2)
                #print('%7.4f %s: Shuttle:%s moving to bay %s' % (env.now, name, shuttleID, bay))
                yield ts2
                #print('%7.4f %s: Shuttle:%s moved to bay %s' % (env.now, name, shuttleID, bay))

                shuttle_avail[shuttleID - 1] = 0
                shuttle.put(shuttleID)
                shuttle_time = env.now - pickup_time
                shuttle_util[shuttleID - 1] = shuttle_util[shuttleID - 1] + shuttle_time
                shuttle_locations[shuttleID]["bay"] = bay
                flow_time = env.now - pickup_time
                cycle_time = env.now - arrive
                flowtime.append(flow_time)
                cycletime.append(cycle_time)
                mean_cycletime.append(mean(cycletime)) if len(cycletime) > 0 else mean_cycletime.append(cycletime)
                #print('%7.4f %s: Finished Shuttle:%s, Cycle time: %7.4f' % (env.now, name, shuttleID, cycle_time))
                trans_left += 1
                if shuttleID == 1:
                    if proc_check[0] == 0:
                        env.process(shuttle_action1(env, shuttle))
                elif shuttleID == 2:
                    if proc_check[1] == 0:
                        env.process(shuttle_action2(env, shuttle))
                break

    else:

        lift1_travel1 = abs(lift1_locations[req2] - tier) * heightoftier
        time_lift1_travel1 = calctime(a_lift, v_lift, lift1_travel1)
        lift1_travel2 = abs(1 - tier) * heightoftier
        time_lift1_travel2 = calctime(a_lift, v_lift, lift1_travel2)
        t1 = env.timeout(time_lift1_travel1)
        #print('%7.4f %s: Lift1:%s moving to tier %s' % (env.now, name, req2, tier))
        yield t1
        #print('%7.4f %s: Lift1:%s moved to tier %s' % (env.now, name, req2, tier))

        lift1_buffer_control[tier - 1] = name
        tier_buffer_lift.append(buffer_control)

        for i in range(len(tier_buffer)):
            tier_check = tier_buffer[i][0]
            name_check = tier_buffer[i][1]
            if tier_check == tier and name_check == name:
                del tier_buffer[i]
                t2 = env.timeout(time_lift1_travel2)
                #print('%7.4f %s: Lift1:%s moving to I/O' % (env.now, name, req2))
                yield t2
                #print('%7.4f %s: Lift1:%s moved to I/O' % (env.now, name, req2))
                #print('%7.4f %s: Lift1:%s released' % (env.now, name, req2))
                lift1.put(req2)
                lift1_locations[req2] = 1
                flow_time = env.now - pickup_time
                cycle_time = env.now - arrive
                flowtime.append(flow_time)
                cycletime.append(cycle_time)
                mean_cycletime.append(mean(cycletime)) if len(cycletime) > 0 else mean_cycletime.append(cycletime)
                lift1_util[req2 - 1] = lift1_util[req2 - 1] + (env.now - lift1_pick[req2 - 1])
                #print('%7.4f %s: Finished Lift1:%s, Cycle time: %7.4f' % (env.now, name, req2, cycle_time))
                trans_left += 1
                break


env = simpy.Environment()
shuttle = simpy.FilterStore(env, capacity=shuttle_no)
shuttle.items = shuttleNo
lift1 = simpy.FilterStore(env, capacity=2)
lift1.items = lift1No
lift2 = simpy.Resource(env, capacity=1)
env.process(source(env, transaction_interval))
env.run(until=run_time)

#sys.stdout.close()
#sys.stdout = stdoutOrigin

shuttle_utilization = [x / (run_time / 2) for x in shuttle_util]
lift1_utilization = [x / (run_time / 2) for x in lift1_util]
lift2_utilization = [x / (run_time/2) for x in lift2_util]

print("Average cycle time is: %6.3f seconds." % mean(cycletime))
print("Average flow time is: %6.3f seconds." % mean(flowtime))
for a in range(len(shuttle_utilization)):
    print("Shuttle %s utilization is %4.2f" % (a + 1, shuttle_utilization[a]))
for a in range(len(lift1_utilization)):
    print("Lift 1:%s utilization is %4.2f" % (a + 1, lift1_utilization[a]))
print("Lift 2 utilization is %4.2f" % (lift2_utilization[0]))
print("Number of transactions entered the system: %s", trans_entered)
print("Number of transactions left the system: %s", trans_left)
print("Max cycle time: %4.2f" % max(cycletime))
c_mean = np.array(mean_cycletime)
time = np.arange(0, len(c_mean))
plt.plot(time, c_mean)
plt.show()