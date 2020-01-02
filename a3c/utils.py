import torch
import numpy as np


# def v_wrap(np_array, dtype=np.float32):
#     # if np_array.dtype != dtype:
#     #     np_array = np_array.astype(dtype)
#     np_array = np.array(np_array, dtype=np.float32)
#     # print(np_array)
#     # print(dtype(np_array))
#     s = torch.from_numpy(np_array)
#     return s


def push_and_pull(optimizer, local_net, global_net, done, next_state, buffer_state, buffer_action, buffer_reward,
                  gamma):
    if done:
        next_state_value = 0.  # terminal
    else:
        next_state = torch.tensor(next_state, dtype=torch.float32)
        next_state_value = local_net.forward(next_state)[-1].data.numpy()

    # 根据贝尔曼方程计算当前状态所能得到的回报总和
    buffer_v_target = []
    for reward in buffer_reward[::-1]:  # reverse buffer r
        next_state_value = reward + gamma * next_state_value
        buffer_v_target.append(next_state_value)
    buffer_v_target.reverse()
    # eps = np.finfo(np.float32).eps.item()
    # buffer_v_target = torch.tensor(buffer_v_target, dtype=torch.float32)
    # # 根据期望和方差做标准归一化
    # buffer_v_target = (buffer_v_target-buffer_v_target.mean())/(buffer_v_target.std()+eps)

    buffer_state = torch.stack(buffer_state)
    buffer_action = torch.stack(buffer_action)
    buffer_v_target = torch.tensor(buffer_v_target, dtype=torch.float32).reshape(-1, 1)
    loss = local_net.get_loss(buffer_state, buffer_action, buffer_v_target)

    # calculate local gradients and push local parameters to global
    optimizer.zero_grad()
    loss.backward()
    for lp, gp in zip(local_net.parameters(), global_net.parameters()):
        gp._grad = lp.grad
    optimizer.step()

    # pull global parameters
    local_net.load_state_dict(global_net.state_dict())


def record(global_ep, global_ep_r, ep_r, res_queue, name):
    with global_ep.get_lock():
        global_ep.value += 1
    with global_ep_r.get_lock():
        if global_ep_r.value == 0.:
            global_ep_r.value = ep_r
        else:
            global_ep_r.value = global_ep_r.value * 0.99 + ep_r * 0.01
    res_queue.put(global_ep_r.value)
    print(name, "Ep:", global_ep.value, "| Ep_r: %.0f" % global_ep_r.value, )
