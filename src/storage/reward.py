from scipy.signal import butter, filtfilt
import torch


def compute_posture_reward(state, condition):
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

    # posture-1
    front_left_bottom_target=0.3
    front_right_bottom_target=0.3
    back_right_bottom_target=0.3
    back_left_bottom_target=0.3
    front_left_top_target=-0.3
    front_right_top_target=-0.3
    back_right_top_target=-0.3
    back_left_top_target=-0.3


    flbe = 1 - 4 * abs(front_left_bottom - front_left_bottom_target)
    frbe = 1 - 4 * abs(front_right_bottom - front_right_bottom_target)
    brbe = 1 - 4 * abs(back_right_bottom - back_right_bottom_target)
    blbe = 1 - 4 * abs(back_left_bottom - back_left_bottom_target)
    flte = 1 - 4 * abs(front_left_top - front_left_top_target)
    frte = 1 - 4 * abs(front_right_top - front_right_top_target)
    brte = 1 - 4 * abs(back_right_top - back_right_top_target)
    blte = 1 - 4 * abs(back_left_top - back_left_top_target)
    pe = 1 - 4 * abs(pitch)
    re = 1 - 4 * abs(roll)

    posture_reward = 0
    for item in [flbe, frbe, brbe, blbe, flte, frte, brte, blte, pe, re]:
        posture_reward += item

    posture_close = True
    for item in [flbe, frbe, brbe, blbe, flte, frte, brte, blte]:
        if item < 0.0:
            posture_close = False

    return torch.sigmoid(posture_reward/5), posture_close


def compute_overturn_penalty(state, condition):
    [overturned, *_] = condition
    if overturned:
        return -2
    return 0


def compute_velocity_reward(state, distance, last_distance=None):
    if last_distance is None:
        last_distance = distance
    distance_delta_reward = distance - last_distance
    last_distance = distance
    reward = -distance_delta_reward
    reward = min(reward, 5)
    return torch.tanh(torch.tensor(reward)), last_distance


def default_standing_reward(states, conditions):
    rewards = []
    for state, condition in zip(states, conditions):
        posture_reward, _ = compute_posture_reward(state, condition)
        overturn_penalty = compute_overturn_penalty(state, condition)

        rewards.append(
            posture_reward +
            overturn_penalty
        )
    rewards = torch.tensor(rewards)
    return (rewards)[:, None]


def butter_lowpass_filter(data, cutoff, fs, order=4):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    filtered = filtfilt(b, a, data)
    return filtered


def default_velocity_reward(states, conditions):
    cutoff_freq = 5.0
    sampling_rate = 20
    order = 4

    distances = [condition[5] for condition in conditions]
    if len(distances) <= 15:
        filtered_distances = distances
    else:
        filtered_distances = butter_lowpass_filter(distances, cutoff_freq, sampling_rate, order)

    rewards = []
    last_distance = None
    for state, condition, distance in zip(states, conditions, filtered_distances):
        overturned = condition[0]
        posture_reward, _ = compute_posture_reward(state, condition)
        overturn_penalty = compute_overturn_penalty(state, condition)

        if len(distances) > 15:
            velocity_reward, last_distance = compute_velocity_reward(state, distance, last_distance)
        else:
            velocity_reward = 0

        if posture_reward < 0.6 and velocity_reward > 0:
                velocity_reward = 0

        if overturned:
            velocity_reward = 0
            posture_reward = 0

        posture_reward = min(posture_reward, 0.7)

        rewards.append(0.6 * posture_reward + velocity_reward + overturn_penalty)
    return torch.tensor(rewards)[:, None]