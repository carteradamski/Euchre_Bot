import random
from collections import deque
import torch


class ReplayBuffer:
    """
    Standard fixed-capacity experience replay buffer for DQN.

    Each entry is a transition:
        state       – 60-dim float tensor  (state BEFORE the action)
        action      – int 0-23             (card index that was played)
        reward      – float                (shaped reward for this transition)
        next_state  – 60-dim float tensor  (state AFTER the action; zeros if done)
        done        – float 0.0 or 1.0     (1.0 on the last trick of a hand)

    Usage:
        buf = ReplayBuffer(capacity=100_000)
        buf.push(state, action, reward, next_state, done)
        states, actions, rewards, next_states, dones = buf.sample(64)
    """

    def __init__(self, capacity=100_000):
        self.buffer = deque(maxlen=capacity)

    # ------------------------------------------------------------------
    def push(self, state, action, reward, next_state, done):
        """
        state, next_state : 60-dim float32 tensors  (already on CPU)
        action            : int
        reward            : float
        done              : bool  (converted to float internally)
        """
        self.buffer.append((
            state.clone().detach(),
            int(action),
            float(reward),
            next_state.clone().detach(),
            float(done),
        ))

    # ------------------------------------------------------------------
    def sample(self, batch_size):
        """
        Returns a mini-batch of transitions as stacked tensors.

        Returns:
            states      (batch_size, 60)
            actions     (batch_size,)   long
            rewards     (batch_size,)   float32
            next_states (batch_size, 60)
            dones       (batch_size,)   float32
        """
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            torch.stack(list(states)),                            # (B, 60)
            torch.tensor(actions, dtype=torch.long),              # (B,)
            torch.tensor(rewards, dtype=torch.float32),           # (B,)
            torch.stack(list(next_states)),                       # (B, 60)
            torch.tensor(dones, dtype=torch.float32),             # (B,)
        )

    # ------------------------------------------------------------------
    def __len__(self):
        return len(self.buffer)
