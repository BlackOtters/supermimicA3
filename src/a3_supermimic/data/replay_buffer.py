"""Time-major ring replay. Contexts are deduplicated per episode, never per step."""
from dataclasses import dataclass
import torch

# Emergency numerical cutoff, not an action clamp. Ordinary commands are unchanged.
# A failed v8 episode fed 1e16 actions back through the actor's observation history.
MAX_ABS_ACTION = 100.0


def valid_action(action):
    return torch.isfinite(action).all(-1) & (action.abs() <= MAX_ABS_ACTION).all(-1)


@dataclass
class WindowBatch:
    states: torch.Tensor
    a_exec: torch.Tensor
    xi: torch.Tensor
    done: torch.Tensor
    frame: torch.Tensor


@dataclass
class StartBatch:
    state: torch.Tensor
    prev_action: torch.Tensor
    xi: torch.Tensor
    frame: torch.Tensor
    hist_states: torch.Tensor
    hist_actions: torch.Tensor


class ReplayBuffer:
    def __init__(self, capacity, num_envs, state_dim, xi_dim, device, action_dim=31):
        if capacity < 2:
            raise ValueError("Replay capacity must be >=2 vector steps")
        self.capacity, self.num_envs = capacity, num_envs
        self.device = torch.device(device)
        self.count = 0
        shape = (capacity, num_envs)
        self.states = torch.empty(*shape, state_dim, device=device)
        self.next_states = torch.empty_like(self.states)
        self.actions = torch.empty(*shape, action_dim, device=device)
        # Actor history is clean; state delay memory and dynamics inputs remain executed.
        self.previous_clean_actions = torch.empty(*shape, 29, device=device)
        self.frame = torch.empty(*shape, dtype=torch.long, device=device)
        self.episode = torch.full(shape, -1, dtype=torch.long, device=device)
        self.context_ids = torch.full(shape, -1, dtype=torch.long, device=device)
        self.done = torch.empty(*shape, dtype=torch.bool, device=device)
        self.valid = torch.zeros(*shape, dtype=torch.bool, device=device)
        self.start_valid = torch.zeros_like(self.valid)
        self.contexts = torch.empty(max(2*num_envs, 16), xi_dim, device=device)
        self.context_count = 0
        self.current_context = torch.full((num_envs,), -1, dtype=torch.long, device=device)

    @property
    def size(self):
        return min(self.count, self.capacity) * self.num_envs

    def register_context(self, env_ids, xi):
        n = len(env_ids)
        if not n:
            return
        end = self.context_count + n
        if end > len(self.contexts):
            expanded = torch.empty(max(end, 2*len(self.contexts)), self.contexts.shape[1], device=self.device)
            expanded[:self.context_count] = self.contexts[:self.context_count]
            self.contexts = expanded
        self.contexts[self.context_count:end] = xi
        self.current_context[env_ids] = torch.arange(self.context_count, end, device=self.device)
        self.context_count = end

    def _compact_contexts(self):
        used = torch.unique(torch.cat((self.context_ids.flatten(), self.current_context)))
        used = used[used >= 0]
        mapping = torch.full((self.context_count + 1,), -1, dtype=torch.long, device=self.device)
        mapping[used] = torch.arange(len(used), device=self.device)
        compact = self.contexts[used].clone()
        self.contexts[:len(used)] = compact
        self.context_ids = mapping[self.context_ids]  # -1 maps to reserved sentinel
        self.current_context = mapping[self.current_context]
        self.context_count = len(used)

    @torch.no_grad()
    def append(self, state, action, next_state, frame, episode, done, *, previous_clean_action, valid=None, start_valid=None):
        if bool((self.current_context < 0).any()):
            raise RuntimeError("Register each environment's context before recording")
        i = self.count % self.capacity
        self.states[i].copy_(state); self.next_states[i].copy_(next_state)
        self.previous_clean_actions[i].copy_(previous_clean_action)
        self.actions[i].copy_(action); self.frame[i].copy_(frame)
        self.episode[i].copy_(episode); self.context_ids[i].copy_(self.current_context)
        self.done[i].copy_(done.bool())
        finite_state = torch.isfinite(state).all(-1) & torch.isfinite(next_state).all(-1)
        usable = finite_state & valid_action(action) & valid_action(previous_clean_action)
        self.valid[i] = usable if valid is None else usable & valid
        self.start_valid[i] = usable if start_valid is None else usable & start_valid
        self.count += 1
        if self.count % self.capacity == 0:
            self._compact_contexts()

    def _candidates(self, n, span, *, windows, history=1):
        available = min(self.count, self.capacity)
        if available < span:
            raise RuntimeError(f"Replay needs {span} contiguous steps, has {available}")
        oldest = self.count - available
        if not windows and self.count > self.capacity:
            oldest += history - 1  # Exclude starts whose real history has been overwritten.
        picked_t, picked_e, remaining = [], [], n
        offsets = torch.arange(span, device=self.device)[:, None]
        for _ in range(64):
            draw = max(2*remaining, 64)
            t = torch.randint(oldest, self.count-span+1, (draw,), device=self.device)
            e = torch.randint(self.num_envs, (draw,), device=self.device)
            rows = (t[None] + offsets) % self.capacity
            eps = self.episode[rows, e]
            ok = (eps == eps[0]).all(0)
            if windows:
                ok &= self.valid[rows, e].all(0)
                ok &= ~self.done[rows[:-1], e].any(0)
                ctx = self.context_ids[rows, e]
                ok &= (ctx == ctx[0]).all(0)
            else:
                ok &= self.start_valid[rows[-1], e]
            selected = ok.nonzero().flatten()[:remaining]
            picked_t.append(t[selected]); picked_e.append(e[selected])
            remaining -= len(selected)
            if remaining == 0:
                return torch.cat(picked_t), torch.cat(picked_e)
        raise RuntimeError("Insufficient valid replay windows; collect more transitions or shorten the window")

    def sample_windows(self, n, k):
        t, e = self._candidates(n, k, windows=True)
        rows = (t[None] + torch.arange(k, device=self.device)[:, None]) % self.capacity
        states = torch.cat((self.states[rows[0], e][None], self.next_states[rows, e]), 0)
        return WindowBatch(states, self.actions[rows, e], self.contexts[self.context_ids[rows[0], e]],
                           self.done[rows, e].float(), self.frame[rows, e])

    def sample_starts(self, n, history):
        # Reconstruct reset-padded history rather than discarding short episodes.
        t, e = self._candidates(n, 1, windows=False, history=history)
        first = max(0, self.count-self.capacity)
        hist_t = (t[None] - torch.arange(history-1, -1, -1, device=self.device)[:, None]).clamp_min(first)
        episode = self.episode[t % self.capacity, e]
        same = self.episode[hist_t % self.capacity, e] == episode[None]
        earliest = torch.where(same, hist_t, t[None]).min(0).values
        hist_t = torch.maximum(hist_t, earliest[None])
        hist = self.states[hist_t % self.capacity, e]
        s = self.states[t % self.capacity, e]
        return StartBatch(s, self.previous_clean_actions[t % self.capacity, e],
                          self.contexts[self.context_ids[t % self.capacity, e]], self.frame[t % self.capacity, e],
                          hist, self.previous_clean_actions[hist_t % self.capacity, e])
