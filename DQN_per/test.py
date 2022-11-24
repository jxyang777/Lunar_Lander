from segment_tree import MinSegmentTree, SumSegmentTree
from ReplayBuffer import PrioritizedReplayBuffer
import torch
from Agent import DQNAgent, Network
import gym

memory = PrioritizedReplayBuffer(8, 10, 4, 0.6)
memory.store([0, 0,0,0,0,0,0,0], 1, 100, [1,1,1,1,1,1,1,1], False)
memory.len()



env_id = "LunarLander-v2"
env = gym.make(env_id)
episodes = 10
memory_size = 128
batch_size = 64
target_update = 200
epsilon_decay = 0.995

state = env.reset()
print(state)
state2 = env.step(env.action_space.sample())

dqn = Network(8, 4)
states = torch.FloatTensor([[state],[state2]])
next_q = dqn(states).detach()
print(next_q)

env.close()











# init_value=0.0
# capacity = 16
# tree = [init_value for _ in range(2 * capacity)]
# tree = SumSegmentTree(16)
# print("SumSegmentTree")
# print("Tree Len:", len(tree))

# tree[1] = 10
# tree[0] = 5
# tree[10] = 20
# # print("Tree:", tree)
# print(tree[1])
# for i in range(2):
#     print(tree[i+2], end=' ')
# print()
# for i in range(4):
#     print(tree[i+4], end=' ')
# print()
# for i in range(8):
#     print(tree[i+8], end=' ')
# print()
# for i in range(16):
#     print(tree[i+16], end=' ')
# print()

# sum = tree.sum(0, 10)
# print("Sum:", sum)


# print("MinSegmentTree")
# minTree = MinSegmentTree(16)
# for i in range(16):
#     minTree[i] = i

# print(minTree[1])
# for i in range(2):
#     print(minTree[i+2], end=' ')
# print()
# for i in range(4):
#     print(minTree[i+4], end=' ')
# print()
# for i in range(8):
#     print(minTree[i+8], end=' ')
# print()
# for i in range(16):
#     print(minTree[i+16], end=' ')
# print()

# min = minTree.min()
# print("Min:", min)