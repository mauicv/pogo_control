import torch


def posture_reward(state, condition):
    [
        front_right_top,
        front_right_bottom,
        front_left_top,
        front_left_bottom,
        back_right_top,
        back_right_bottom,
        back_left_top,
        back_left_bottom,
        front_right_top_vel,
        front_right_bottom_vel,
        front_left_top_vel,
        front_left_bottom_vel,
        back_right_top_vel,
        back_right_bottom_vel,
        back_left_top_vel,
        back_left_bottom_vel,
        ax,
        ay,
        az,
        gx,
        gy,
        gz,
        roll,
        pitch
    ] = state

    flbe = 1 - abs(front_left_bottom - 0.4)
    frbe = 1 - abs(front_right_bottom - 0.4)
    brbe = 1 - abs(back_right_bottom - 0.4)
    blbe = 1 - abs(back_left_bottom - 0.4)
    flte = 1 - abs(front_left_top - -0.3)
    frte = 1 - abs(front_right_top - -0.3)
    brte = 1 - abs(back_right_top - -0.3)
    blte = 1 - abs(back_left_top - -0.3)
    pe = 1 - abs(pitch)
    re = 1 - abs(roll)

    posture_reward = 0
    for item in [flbe, frbe, brbe, blbe, flte, frte, brte, blte, pe, re]:
        posture_reward += item

    posture_close = True
    for item in [flbe, frbe, brbe, blbe, flte, frte, brte, blte]:
        if item < 0.75:
            posture_close = False

    return posture_reward, posture_close


def overturn_penalty(state, condition):
    [overturned, *_] = condition
    if overturned:
        return -10
    return 0


def velocity_reward(state, condition):
    [
        overturned,
        last_mpus6050_sample_ts,
        last_servo_set_ts,
        x,
        y,
        distance,
        vx,
        vy,
        speed,
        yaw,
        last_detection_ts
    ] = condition
    if last_distance is None:
        last_distance = distance
    distance_delta_reward = distance - last_distance
    last_distance = distance
    return -distance_delta_reward


def default_standing_reward(states, conditions):
    rewards = []
    for state, condition in zip(states, conditions):
        posture_reward, _ = posture_reward(state, condition)
        overturn_penalty = overturn_penalty(state, condition)

        rewards.append(
            10 * posture_reward +
            overturn_penalty
        )
    rewards = torch.tensor(rewards)
    return (rewards)[:, None]


def default_velocity_reward(states, conditions):
    rewards = []
    last_distance = None

    for state, condition in zip(states, conditions):
        posture_reward, posture_close = posture_reward(state, condition)
        overturn_penalty = overturn_penalty(state, condition)
        if posture_close:
            velocity_reward = velocity_reward(state, condition)
        else:
            velocity_reward = 0


        rewards.append(posture_reward + velocity_reward + overturn_penalty)
    return torch.tensor(rewards)[:, None]
