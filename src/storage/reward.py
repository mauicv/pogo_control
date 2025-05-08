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
        if item < 0.7:
            posture_close = False

    return posture_reward, posture_close


def compute_overturn_penalty(state, condition):
    [overturned, *_] = condition
    if overturned:
        return -50
    return 0


def compute_velocity_reward(state, distance, last_distance=None):
    if last_distance is None:
        last_distance = distance
    distance_delta_reward = distance - last_distance
    last_distance = distance
    reward = -distance_delta_reward
    reward = min(reward, 5)
    return reward, last_distance


def default_standing_reward(states, conditions):
    rewards = []
    for state, condition in zip(states, conditions):
        posture_reward, _ = compute_posture_reward(state, condition)
        overturn_penalty = compute_overturn_penalty(state, condition)

        rewards.append(
            10 * posture_reward +
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
    cutoff_freq = 2.0   # Hz
    sampling_rate = 20  # Hz
    order = 4

    distances = [condition[5] for condition in conditions]
    filtered_distances = butter_lowpass_filter(distances, cutoff_freq, sampling_rate, order)
    
    rewards = []
    last_distance = None
    for state, condition, distance in zip(states, conditions, filtered_distances):
        overturned = condition[0]
        posture_reward, posture_close = compute_posture_reward(state, condition)
        overturn_penalty = compute_overturn_penalty(state, condition)
        velocity_reward, last_distance = compute_velocity_reward(state, distance, last_distance)
        if not posture_close:
            velocity_reward = 0
        if overturned:
            velocity_reward = 0
            posture_reward = 0
        rewards.append(posture_reward + 10 * velocity_reward + overturn_penalty)
    return torch.tensor(rewards)[:, None]
