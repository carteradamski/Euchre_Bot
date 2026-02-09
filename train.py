"""
train.py  —  Self-play DQN training loop for euchre card play.

Usage:
    python train.py [--episodes 5000] [--eval-every 500] [--checkpoint-every 500]

What it does
------------
1.  Creates a shared Q-network and a frozen target copy.
2.  Runs full euchre hands with 4 AI players that all share the same network.
3.  After every trick, computes a shaped reward and pushes a transition into
    the replay buffer.  At the end of each hand, adds the terminal hand-reward
    to the last transition for each player.
4.  Once the buffer has enough samples, trains on random mini-batches using
    standard DQN (Bellman target + MSE loss).
5.  Decays epsilon (exploration) linearly over time.
6.  Every N episodes, runs an evaluation batch: trained AI vs heuristic-only AI.
7.  Saves model checkpoints periodically.

Architecture decisions
-----------------------
- All 4 players share ONE network.  This is deliberate: every position in euchre
  is strategically symmetric (team structure aside), so one policy covers all seats.
  The seat one-hot in the state vector lets the network differentiate.
- Trick rewards (+1/-1) provide short-horizon signal.  Hand rewards (±1 to ±4)
  are added to the LAST transition of each player's participation in that hand,
  providing the long-horizon outcome signal.
- The on_decision callback on AIPlayer lets us capture (state, action) with zero
  changes to the game engine or player interface.
"""

import argparse
import os
import sys
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Path setup so imports work whether you run from the project root or not
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from model          import EuchreQNetwork
from replay_buffer  import ReplayBuffer
from reward         import trick_reward, hand_reward
from AIplayer       import AIPlayer
from Pack           import Pack
from Card           import Suit
from state_encoder  import encode_state
from game_state     import GameState


# ===========================================================================
# Silence wrapper — suppresses all print() during training games
# ===========================================================================
import io, contextlib

@contextlib.contextmanager
def _silent():
    """Context manager that eats stdout."""
    f = io.StringIO()
    with contextlib.redirect_stdout(f):
        yield


# ===========================================================================
# Silent Game  — a stripped-down copy of euchre.Game that skips all prints
#                and returns richer info for the training loop.
# ===========================================================================

from euchre import Game   # inherit deal, decide_trump, give_points, etc.

class TrainingGame(Game):
    """
    Thin subclass that overrides play_game to play a single hand silently
    and return the GameState + per-trick winner info the training loop needs.

    We reuse deal() and decide_trump() from the parent unchanged.
    """

    def __init__(self, players, pack_filename="pack.txt", curriculum_mode=None):
        # Minimal init — we don't call super().__init__ because it reads
        # pack from file every time.  We manage the pack ourselves.
        self.players   = list(players)
        self.num_to_win = 99   # irrelevant for single-hand play
        self.shuffle   = True
        self.curriculum_mode = curriculum_mode  # None, 'card_play_only', 'no_alone', 'full'

        with open(pack_filename, 'r') as f:
            self.game_pack = Pack(f)

    def play_one_hand(self, dealer_index, force_trump=None):
        """
        Play exactly one hand silently.
        
        Args:
            dealer_index: which player is dealer
            force_trump: if provided, skip trump calling and use this suit
                        (for curriculum learning - card play only mode)

        Returns
        -------
        game_state   : GameState with full trick_history populated
        trick_winners: list of 5 bools, True if team 1 won that trick
        """
        # Clear hands
        for p in self.players:
            p.clear_hand()

        self.game_pack.shuffle()
        self.deal(dealer_index)
        upcard = self.game_pack.deal_one()

        # Suppress all prints from decide_trump / player methods
        with _silent():
            if force_trump is not None:
                # Curriculum mode: override trump calling
                # Monkey-patch players to always pass on trump (must return tuple format)
                original_make_trump = [p.make_trump for p in self.players]
                for p in self.players:
                    p.make_trump = lambda *args, **kwargs: (False, None)
                
                # Run normal game flow (which will record AI decisions properly)
                result_code, alone_info, gs = self.play_hand(upcard, dealer_index)
                
                # Restore original methods
                for i, p in enumerate(self.players):
                    p.make_trump = original_make_trump[i]
                
                # Override the trump that was set (everyone passed, so it's undefined)
                # and fix team1_calls to match forced trump
                gs.trump = force_trump
                gs.caller_index = dealer_index
                gs.team1_calls = (dealer_index == 0 or dealer_index == 2)
            else:
                result_code, alone_info, gs = self.play_hand(upcard, dealer_index)

        # Reconstruct per-trick winners from trick_history + gs
        # gs.team1_tricks and gs.team2_tricks give totals; we need per-trick.
        # Replay trick_history through Card_less to get per-trick winner.
        from Card import Card_less
        trick_winners = []
        for trick_cards in gs.trick_history:
            if len(trick_cards) == 0:
                trick_winners.append(False)
                continue
            # trick_cards is [(player_idx, Card), ...]
            # First entry is the leader
            led_card   = trick_cards[0][1]
            best_idx   = 0
            best_card  = trick_cards[0][1]
            for k in range(1, len(trick_cards)):
                if Card_less(best_card, trick_cards[k][1], led_card, gs.trump):
                    best_card = trick_cards[k][1]
                    best_idx  = k
            winner_player = trick_cards[best_idx][0]
            trick_winners.append(winner_player == 0 or winner_player == 2)

        return gs, trick_winners


# ===========================================================================
# Training loop
# ===========================================================================

def train(args):
    # ── hyper-parameters ──
    EPISODES         = args.episodes
    BATCH_SIZE       = 64
    BUFFER_CAP       = 100_000
    LR               = 1e-3
    GAMMA            = 0.99
    EPS_START        = 1.0
    EPS_END          = 0.04
    EPS_DECAY_STEPS  = EPISODES          # linear decay over all episodes
    TARGET_UPDATE    = 100               # episodes between target-net syncs
    EVAL_EVERY       = args.eval_every
    EVAL_HANDS       = 200               # hands per eval run
    CHECKPOINT_EVERY = args.checkpoint_every
    PACK_FILE        = os.path.join(SCRIPT_DIR, "pack.txt")

    # ── model + target ──
    model        = EuchreQNetwork()
    target_model = EuchreQNetwork()
    target_model.load_state_dict(model.state_dict())
    target_model.eval()

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    buffer    = ReplayBuffer(capacity=BUFFER_CAP)

    # ── resume from checkpoint ──
    start_episode = 0
    if args.resume:
        checkpoint_path = os.path.join(SCRIPT_DIR, args.resume)
        if os.path.exists(checkpoint_path):
            print(f"Loading checkpoint: {checkpoint_path}")
            model.load_state_dict(torch.load(checkpoint_path))
            target_model.load_state_dict(model.state_dict())
            # Extract episode number from filename (e.g., checkpoint_ep30000.pt -> 30000)
            import re
            match = re.search(r'ep(\d+)', args.resume)
            if match:
                start_episode = int(match.group(1))
                print(f"Resuming from episode {start_episode}")
        else:
            print(f"Checkpoint not found: {checkpoint_path}")
            print("Starting from scratch...")

    # ── epsilon schedule ──
    def get_epsilon(episode):
        frac = min(1.0, episode / max(1, EPS_DECAY_STEPS))
        return EPS_START + (EPS_END - EPS_START) * frac

    # ── metrics ──
    total_loss      = 0.0
    train_steps     = 0
    win_counts      = [0, 0]   # [team1_wins, team2_wins] across all episodes

    print("=" * 60)
    print("EUCHRE DQN TRAINING - CURRICULUM LEARNING")
    print(f"  Episodes : {EPISODES}")
    print(f"  Buffer   : {BUFFER_CAP}")
    print(f"  LR       : {LR}  |  gamma : {GAMMA}")
    print(f"  Epsilon  : {EPS_START} → {EPS_END} over {EPS_DECAY_STEPS} eps")
    print()
    print("  Curriculum Stages:")
    stage1_end = EPISODES // 4
    stage2_end = EPISODES // 2
    stage3_end = EPISODES * 3 // 4
    print(f"    Stage 1 (0-{stage1_end//1000}k):      Card play only (random trump)")
    print(f"    Stage 2 ({stage1_end//1000}k-{stage2_end//1000}k):    + Trump calling")
    print(f"    Stage 3 ({stage2_end//1000}k-{stage3_end//1000}k):    + Going alone")
    print(f"    Stage 4 ({stage3_end//1000}k-{EPISODES//1000}k):   Full game")
    if start_episode > 0:
        print(f"\n  Resuming from episode {start_episode}")
    print("=" * 60)

    for episode in range(start_episode, EPISODES):
        epsilon = get_epsilon(episode)
        dealer  = episode % 4
        
        # Determine curriculum stage (scale to EPISODES)
        if episode < EPISODES // 4:
            stage = 1  # Card play only
            force_trump = list(Suit)[episode % 4]  # Rotate through suits
        elif episode < EPISODES // 2:
            stage = 2  # Add trump calling
            force_trump = None
        elif episode < EPISODES * 3 // 4:
            stage = 3  # Add going alone (implement this in game logic)
            force_trump = None
        else:
            stage = 4  # Full game
            force_trump = None

        # ── create 4 AI players sharing the model ──
        players = [AIPlayer(f"AI_{i}", model=model, epsilon=epsilon) for i in range(4)]

        # ── per-player transition accumulators ──
        # Each player accumulates (state, action) pairs during the hand.
        # After each trick we know who won, so we can assign rewards.
        # Structure: pending[player_idx] = list of (state_vec, action_idx)
        pending = {i: [] for i in range(4)}

        def make_callback(player_idx):
            """Closure factory so each player's callback captures its own index."""
            def cb(state_vec, action_idx):
                pending[player_idx].append((state_vec.clone(), action_idx))
            return cb

        for i, p in enumerate(players):
            p.on_decision = make_callback(i)

        # ── play one hand ──
        game = TrainingGame(players, pack_filename=PACK_FILE)
        gs, trick_winners = game.play_one_hand(dealer, force_trump=force_trump)

        # ── assign rewards and push transitions ──
        # trick_winners[t] = True  →  team 1 won trick t
        # gs.trick_history[t]      = [(player_idx, Card), …]  who played in trick t
        #
        # For each player who played in trick t:
        #   reward = trick_reward(my_team_won)
        #   If t == 4 (last trick): reward += hand_reward(…) for that player
        #   next_state: the state that the SAME player will see at their next
        #               decision point.  If the hand is over, next_state = zeros
        #               and done = True.

        # First, collect all (player_idx, trick_idx) in order they acted
        # so we can link each action to its reward and the next state.
        # pending[i] has one entry per card that player i played (0–5 cards).
        # trick_history[t] tells us which players played in trick t, in order.

        # Build a timeline: for each player, which tricks did they play in?
        player_trick_order = {i: [] for i in range(4)}
        for t, trick_cards in enumerate(gs.trick_history):
            for (pidx, _) in trick_cards:
                player_trick_order[pidx].append(t)

        # Now push transitions
        num_tricks = len(gs.trick_history)
        for pidx in range(4):
            actions     = pending[pidx]      # list of (state_vec, action_idx)
            trick_idxs  = player_trick_order[pidx]  # which trick each action belongs to

            if len(actions) != len(trick_idxs):
                # Mismatch — skip this player (shouldn't happen)
                continue

            my_team_is_team1 = (pidx == 0 or pidx == 2)

            for step, (state_vec, action_idx) in enumerate(actions):
                t = trick_idxs[step]
                my_team_won = trick_winners[t] if my_team_is_team1 else (not trick_winners[t])

                # Trick reward
                r = trick_reward(my_team_won)

                # Terminal?
                is_last_action = (step == len(actions) - 1)
                done = is_last_action   # hand is over after this player's last card

                if done:
                    # Add hand reward
                    r += hand_reward(
                        team1_tricks     = gs.team1_tricks,
                        team1_calls      = gs.team1_calls,
                        going_alone      = gs.going_alone,
                        alone_team1      = gs.alone_team1,
                        my_team_is_team1 = my_team_is_team1,
                    )
                    next_state = torch.zeros(60)
                else:
                    # next_state = the state at the player's next action
                    next_state = actions[step + 1][0]

                buffer.push(state_vec, action_idx, r, next_state, done)

        # Track team wins
        if gs.team1_tricks >= 3:
            win_counts[0] += 1
        else:
            win_counts[1] += 1

        # ── train on a mini-batch ──
        if len(buffer) >= BATCH_SIZE:
            states, actions, rewards, next_states, dones = buffer.sample(BATCH_SIZE)

            # Current Q for the action taken
            q_all  = model(states)                                  # (B, 24)
            q_sa   = q_all.gather(1, actions.unsqueeze(1)).squeeze(1)  # (B,)

            # Target: r + γ * max Q_target(s') * (1 - done)
            with torch.no_grad():
                next_q     = target_model(next_states)              # (B, 24)
                max_next_q = next_q.max(dim=1)[0]                   # (B,)
                target     = rewards + GAMMA * max_next_q * (1.0 - dones)

            loss = F.mse_loss(q_sa, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss  += loss.item()
            train_steps += 1

        # ── sync target network ──
        if (episode + 1) % TARGET_UPDATE == 0:
            target_model.load_state_dict(model.state_dict())

        # ── logging ──
        if (episode + 1) % 100 == 0:
            avg_loss = total_loss / max(1, train_steps)
            print(f"  ep {episode+1:>6d} | stage={stage} | ε={epsilon:.3f} | "
                  f"loss={avg_loss:.6f} | steps={train_steps} | buf={len(buffer):>6d} | "
                  f"T1 wins={win_counts[0]}  T2 wins={win_counts[1]}")
            total_loss  = 0.0
            train_steps = 0

        # ── periodic evaluation ──
        if (episode + 1) % EVAL_EVERY == 0:
            print("\n  ── Evaluation ──")
            _evaluate(model, EVAL_HANDS, PACK_FILE)
            print()

        # ── checkpoint ──
        if (episode + 1) % CHECKPOINT_EVERY == 0:
            ckpt_path = os.path.join(SCRIPT_DIR, f"checkpoint_ep{episode+1}.pt")
            torch.save(model.state_dict(), ckpt_path)
            print(f"  [checkpoint saved: {ckpt_path}]")

    # Final checkpoint
    final_path = os.path.join(SCRIPT_DIR, "model_final.pt")
    torch.save(model.state_dict(), final_path)
    print(f"\nTraining complete.  Final model saved to {final_path}")


# ===========================================================================
# Evaluation  —  trained model (greedy) vs heuristic-only AI (no model)
# ===========================================================================

def _evaluate(model, num_hands, pack_file):
    """
    Play num_hands hands: players 0,2 use the trained model (greedy),
    players 1,3 use heuristic-only (model=None).
    Report win rate for the trained team.
    """
    model.eval()
    trained_wins = 0
    heuristic_wins = 0

    for h in range(num_hands):
        dealer = h % 4
        players = [
            AIPlayer("Trained_0", model=model, epsilon=0.0),
            AIPlayer("Heuristic_1", model=None),
            AIPlayer("Trained_2", model=model, epsilon=0.0),
            AIPlayer("Heuristic_3", model=None),
        ]
        game = TrainingGame(players, pack_filename=pack_file)
        gs, trick_winners = game.play_one_hand(dealer)

        if gs.team1_tricks >= 3:
            trained_wins   += 1
        else:
            heuristic_wins += 1

    model.train()
    total = trained_wins + heuristic_wins
    print(f"  Trained (T1) vs Heuristic (T2):  "
          f"{trained_wins}/{total} hands won  "
          f"({100*trained_wins/max(1,total):.1f}%)")


# ===========================================================================
# CLI
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description="Train euchre DQN via self-play")
    parser.add_argument("--episodes",         type=int, default=40000,
                        help="Total training episodes (hands)")
    parser.add_argument("--eval-every",       type=int, default=1000,
                        help="Run eval every N episodes")
    parser.add_argument("--checkpoint-every", type=int, default=5000,
                        help="Save checkpoint every N episodes")
    parser.add_argument("--resume",           type=str, default=None,
                        help="Resume from checkpoint file (e.g., checkpoint_ep30000.pt)")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
