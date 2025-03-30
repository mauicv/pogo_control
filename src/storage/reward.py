import torch


def default_standing_reward(states, conditions):
    rewards = []
    for state, condition in zip(states, conditions):
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

        [overturned, *_] = condition


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

        overturn_penalty = 0
        if overturned:
            overturn_penalty = -10

        rewards.append(
            10 * posture_reward +
            overturn_penalty
        )
    rewards = torch.tensor(rewards)
    return (rewards)[:, None]


def sanity_check_reward_function(states, conditions):
    rewards = []
    for state in states:
        [
            front_right_top,
            front_right_bottom,
            front_left_top,
            front_left_bottom,
            back_right_top,
            back_right_bottom,
            back_left_top,
            back_left_bottom,
            *_
        ] = state
        reward = - 100 * (
            (front_left_bottom - 0.4)**2 +
            (front_right_bottom - 0.4)**2 +
            (back_right_bottom - 0.4)**2 +
            (back_left_bottom - 0.4)**2 +
            (front_left_top - -0.3)**2 +
            (front_right_top - -0.3)**2 +
            (back_right_top - -0.3)**2 +
            (back_left_top - -0.3)**2
        )
        rewards.append(reward)
    return torch.tensor(rewards)[:, None]


def default_velocity_reward(states, conditions):
    rewards = []
    last_distance = None

    for state, condition in zip(states, conditions):
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

        overturn_penalty = 0
        if overturned:
            overturn_penalty = -10

        rewards.append(-distance_delta_reward + overturn_penalty)
    return torch.tensor(rewards)[:, None]
