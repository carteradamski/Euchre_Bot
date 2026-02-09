import sys
import os
from Player import Player_factory
from Card import Card, Suit, Card_less, SUIT_NAMES
from Pack import Pack
from game_state import GameState


class Game:
    def __init__(self, players_input, num, shuff, pack_filename):
        """Initialize a game of Euchre."""
        self.players = [None] * 4
        for i in range(4):
            self.players[i] = players_input[i]
        
        with open(pack_filename, 'r') as in_pack:
            self.game_pack = Pack(in_pack)
        
        self.num_to_win = num
        
        if shuff == "shuffle":
            self.shuffle = True
        else:
            self.shuffle = False
    
    def give_points(self, team1_points, team2_points, result, alone_info):
        """Award points based on the result of the hand."""
        going_alone = alone_info[0]
        alone_team1 = alone_info[1]
        
        # Determine base points
        if result == '0':  # Team 2 march (all 5 tricks)
            if going_alone and not alone_team1:
                team2_points += 4
                self.print_win(self.players[1], self.players[3])
                print("march while going alone!")
            else:
                team2_points += 2
                self.print_win(self.players[1], self.players[3])
                print("march!")
        elif result == '4':  # Team 2 euchre
            team2_points += 2
            self.print_win(self.players[1], self.players[3])
            print("euchred!")
        elif result == '1':  # Team 1 march (all 5 tricks)
            if going_alone and alone_team1:
                team1_points += 4
                self.print_win(self.players[0], self.players[2])
                print("march while going alone!")
            else:
                team1_points += 2
                self.print_win(self.players[0], self.players[2])
                print("march!")
        elif result == '2':  # Team 1 win (3-4 tricks)
            # If going alone and won 3-4 tricks, only get 1 point
            team1_points += 1
            self.print_win(self.players[0], self.players[2])
        elif result == '5':  # Team 2 win (3-4 tricks)
            # If going alone and won 3-4 tricks, only get 1 point
            team2_points += 1
            self.print_win(self.players[1], self.players[3])
        elif result == '3':  # Team 1 euchre
            team1_points += 2
            self.print_win(self.players[0], self.players[2])
            print("euchred!")
        else:
            assert False, "Invalid result"
        
        return team1_points, team2_points
    
    def play_game(self):
        """Play the full game until one team reaches the winning score."""
        team1_points = 0
        team2_points = 0
        round_num = 0
        
        print("=" * 60)
        print("EUCHRE GAME START")
        print(f"Playing to {self.num_to_win} points")
        print(f"Team 1: {self.players[0].get_name()} and {self.players[2].get_name()}")
        print(f"Team 2: {self.players[1].get_name()} and {self.players[3].get_name()}")
        print("=" * 60)
        print()
        
        while team1_points < self.num_to_win and team2_points < self.num_to_win:
            print("=" * 60)
            print(f"Hand {round_num}")
            print(f"{self.players[round_num % 4].get_name()} deals")
            print("=" * 60)
            
            # Clear all hands from previous round
            for player in self.players:
                player.clear_hand()
            
            if self.shuffle:
                self.game_pack.shuffle()
            
            self.deal(round_num % 4)
            upcard = self.game_pack.deal_one()
            print(f"{upcard} turned up")
            
            result_info = self.play_hand(upcard, round_num)
            result = result_info[0]
            alone_info = result_info[1]
            # result_info[2] is the GameState — available for training; ignored here
            
            team1_points, team2_points = self.give_points(team1_points, team2_points, result, alone_info)
            round_num += 1
            
            print(f"\n{self.players[0].get_name()} and {self.players[2].get_name()} have {team1_points} points")
            print(f"{self.players[1].get_name()} and {self.players[3].get_name()} have {team2_points} points")
            print()
        
        print("=" * 60)
        print("GAME OVER")
        print("=" * 60)
        if team1_points >= self.num_to_win:
            print(f"{self.players[0].get_name()} and {self.players[2].get_name()} win!")
        else:
            print(f"{self.players[1].get_name()} and {self.players[3].get_name()} win!")
        print("=" * 60)
        
        self.game_pack.reset()
    
    def deal(self, dealer_index):
        """Deal cards to all players in proper euchre fashion."""
        self.game_pack.reset()
        # Deal 3-2-3-2 pattern starting left of dealer, then 2-3-2-3
        # Left of dealer gets 3 cards
        self.players[(dealer_index + 1) % 4].add_card(self.game_pack.deal_one())
        self.players[(dealer_index + 1) % 4].add_card(self.game_pack.deal_one())
        self.players[(dealer_index + 1) % 4].add_card(self.game_pack.deal_one())
        # Across from dealer gets 2 cards
        self.players[(dealer_index + 2) % 4].add_card(self.game_pack.deal_one())
        self.players[(dealer_index + 2) % 4].add_card(self.game_pack.deal_one())
        # Right of dealer gets 3 cards
        self.players[(dealer_index + 3) % 4].add_card(self.game_pack.deal_one())
        self.players[(dealer_index + 3) % 4].add_card(self.game_pack.deal_one())
        self.players[(dealer_index + 3) % 4].add_card(self.game_pack.deal_one())
        # Dealer gets 2 cards
        self.players[dealer_index % 4].add_card(self.game_pack.deal_one())
        self.players[dealer_index % 4].add_card(self.game_pack.deal_one())
        # Left of dealer gets 2 more cards (total 5)
        self.players[(dealer_index + 1) % 4].add_card(self.game_pack.deal_one())
        self.players[(dealer_index + 1) % 4].add_card(self.game_pack.deal_one())
        # Across from dealer gets 3 more cards (total 5)
        self.players[(dealer_index + 2) % 4].add_card(self.game_pack.deal_one())
        self.players[(dealer_index + 2) % 4].add_card(self.game_pack.deal_one())
        self.players[(dealer_index + 2) % 4].add_card(self.game_pack.deal_one())
        # Right of dealer gets 2 more cards (total 5)
        self.players[(dealer_index + 3) % 4].add_card(self.game_pack.deal_one())
        self.players[(dealer_index + 3) % 4].add_card(self.game_pack.deal_one())
        # Dealer gets 3 more cards (total 5)
        self.players[dealer_index % 4].add_card(self.game_pack.deal_one())
        self.players[dealer_index % 4].add_card(self.game_pack.deal_one())
        self.players[dealer_index % 4].add_card(self.game_pack.deal_one())
    
    def play_hand(self, upcard, dealer_index):
        """
        Play one hand and return the result.
        Returns: tuple (result_code, alone_info, game_state)
            result_code  – single-char string used by give_points
            alone_info   – (going_alone, alone_team1)
            game_state   – the fully populated GameState at end of hand
                           (training loop uses this to compute terminal reward)
        """
        trump = Suit.SPADES
        trump_result = self.decide_trump(dealer_index, upcard)
        team1_calls = trump_result[0]
        trump = trump_result[1]
        caller_index = trump_result[2]

        # Ask if going alone
        going_alone = False
        alone_team1 = False
        sit_out_index = -1

        if caller_index >= 0:
            print(f"\n{self.players[caller_index].get_name()} called trump.")
            going_alone = self.players[caller_index].ask_go_alone()
            if going_alone:
                partner_index = (caller_index + 2) % 4
                sit_out_index = partner_index
                alone_team1 = (caller_index == 0 or caller_index == 2)
                print(f"{self.players[caller_index].get_name()} is going alone!")
                print(f"{self.players[partner_index].get_name()} sits out this hand.")
                print()

        # ── Build the GameState that will travel through the hand ──
        gs = GameState()
        gs.trump          = trump
        gs.dealer_index   = dealer_index % 4
        gs.upcard         = upcard
        gs.caller_index   = caller_index
        gs.team1_calls    = team1_calls
        gs.going_alone    = going_alone
        gs.alone_team1    = alone_team1

        starter = (dealer_index + 1) % 4
        team_one_wins = 0

        for i in range(5):
            gs.trick_index              = i
            gs.led_card                 = None
            gs.cards_played_this_trick  = []

            result = self.play_trick(starter, trump, sit_out_index, gs)
            starter       = result[0]
            team1_won_trick = result[1]

            if team1_won_trick:
                team_one_wins  += 1
                gs.team1_tricks += 1
            else:
                gs.team2_tricks += 1

            # Snapshot the completed trick into history
            gs.trick_history.append(list(gs.cards_played_this_trick))

        # Determine result code  (unchanged logic)
        if team_one_wins > 2 and not team1_calls:
            result_code = '3'          # Team 1 euchre
        elif team_one_wins < 3 and team1_calls:
            result_code = '4'          # Team 2 euchre
        elif team_one_wins == 0:
            result_code = '0'          # Team 2 march
        elif team_one_wins == 5:
            result_code = '1'          # Team 1 march
        elif team_one_wins > 2 and team1_calls:
            result_code = '2'          # Team 1 win
        else:
            result_code = '5'          # Team 2 win

        return (result_code, (going_alone, alone_team1), gs)
    
    def decide_trump(self, dealer_index, upcard):
        """
        Decide trump suit through bidding rounds.
        Returns: tuple (team1_calls, trump, caller_index)
        """
        trump = Suit.SPADES
        starter = (dealer_index + 1) % 4
        
        print("\n--- ROUND 1 BIDDING ---")
        for i in range(starter, starter + 4):
            player_index = i % 4
            is_dealer = (player_index == dealer_index % 4)
            round_num = 1
            
            result = self.players[player_index].make_trump(upcard, is_dealer, round_num, trump)
            
            if result[0]:  # Player ordered up
                trump = result[1]
                print(f"{self.players[player_index].get_name()} orders up {SUIT_NAMES[trump]}")
                
                print()
                
                # Dealer picks up upcard
                self.players[dealer_index % 4].add_and_discard(upcard)
                
                if player_index == 0 or player_index == 2:
                    return (True, trump, player_index)
                else:
                    return (False, trump, player_index)
            
            print(f"{self.players[player_index].get_name()} passes")
        
        print("\n--- ROUND 2 BIDDING ---")
        for i in range(starter, starter + 4):
            player_index = i % 4
            is_dealer = (player_index == dealer_index % 4)
            round_num = 2
            
            result = self.players[player_index].make_trump(upcard, is_dealer, round_num, trump)
            
            if result[0]:  # Player called trump
                trump = result[1]
                print(f"{self.players[player_index].get_name()} calls {SUIT_NAMES[trump]}")
                print()
                
                if player_index == 0 or player_index == 2:
                    return (True, trump, player_index)
                else:
                    return (False, trump, player_index)
            
            if not is_dealer:
                print(f"{self.players[player_index].get_name()} passes")
        
        # Should never get here as dealer must call in round 2
        return (False, trump, -1)
    
    def play_trick(self, starter_index, trump, sit_out_index=-1, game_state=None):
        """
        Play one trick.
        game_state: GameState (may be None for backward-compat; AI players need it).
        Returns: tuple (new_starter_index, team1_won)
        """
        trick_winner_index = 0
        cards = [None] * 4

        print(f"--- Trick (Trump: {SUIT_NAMES[trump]}) ---")

        # Lead card
        if starter_index == sit_out_index:
            starter_index = (starter_index + 1) % 4

        # ── set state for the leader ──
        if game_state is not None:
            game_state.my_index             = starter_index % 4
            game_state.led_card             = None   # I am leading
            game_state.cards_played_this_trick = []

        cards[0] = self.players[starter_index % 4].lead_card(trump, game_state)
        print(f"{cards[0]} led by {self.players[starter_index % 4].get_name()}")
        
        # Track cards played in order for display
        trick_display = [(self.players[starter_index % 4].get_name(), cards[0])]

        # Record what the leader played
        if game_state is not None:
            game_state.cards_played_this_trick.append((starter_index % 4, cards[0]))

        # Track the current highest card
        max_card = cards[0]
        trick_winner_index = 0

        # Other players follow
        for offset in range(1, 4):
            play_index = (starter_index + offset) % 4

            if play_index == sit_out_index:
                cards[offset] = None
                print(f"{self.players[play_index].get_name()} is sitting out")
                continue

            # Show cards played so far before this player's turn
            print(f"\n  Cards played so far:")
            for i, (player_name, card) in enumerate(trick_display, 1):
                trump_marker = ""
                if card.is_right_bower(trump):
                    trump_marker = " [TRUMP - RIGHT BOWER]"
                elif card.is_left_bower(trump):
                    trump_marker = " [TRUMP - LEFT BOWER]"
                elif card.is_trump(trump):
                    trump_marker = " [TRUMP]"
                print(f"    {i}. {player_name}: {card}{trump_marker}")
            print()

            # ── set state for this follower ──
            if game_state is not None:
                game_state.my_index = play_index
                game_state.led_card = cards[0]   # the card that was led

            cards[offset] = self.players[play_index].play_card(cards[0], trump, game_state)
            print(f"{cards[offset]} played by {self.players[play_index].get_name()}")
            
            # Add to trick display
            trick_display.append((self.players[play_index].get_name(), cards[offset]))
            
            # Update highest card if this card beats it
            if Card_less(max_card, cards[offset], cards[0], trump):
                max_card = cards[offset]
                trick_winner_index = offset

            # Record what this player played
            if game_state is not None:
                game_state.cards_played_this_trick.append((play_index, cards[offset]))

        # winner_absolute was already calculated during the trick
        winner_absolute = (trick_winner_index + starter_index) % 4
        print(f"\n🏆 {self.players[winner_absolute].get_name()} takes the trick")
        print()

        if winner_absolute == 0 or winner_absolute == 2:
            return (winner_absolute, True)
        else:
            return (winner_absolute, False)
    
    def print_win(self, teammate1, teammate2):
        """Print which team won the hand."""
        print(f"{teammate1.get_name()} and {teammate2.get_name()} win the hand")


def main():
    """Main function to run the Euchre game."""
    if len(sys.argv) < 12:
        print("Usage: python euchre.py <pack_filename> <shuffle|noshuffle> <points_to_win> " +
              "<player0_name> <player0_type> [r1_thresh r2_thresh] <player1_name> <player1_type> [r1_thresh r2_thresh] " +
              "<player2_name> <player2_type> [threshold] <player3_name> <player3_type> [threshold] " +
              "[--model <model_path>]")
        print("\nPlayer type: 'Human' or 'AI'")
        print("Optional threshold: 0.0-1.0 for AI trump decisions (same for both rounds)")
        print("Optional --model flag: Load a trained neural network for AI players")
        print("Example: python euchre.py pack.txt shuffle 10 Alice AI 0.3 Bob AI 0.5 Charlie AI 0.7 Dave Human")
        print("         (Alice: conservative, Bob: moderate, Charlie: aggressive)")
        print("Example with model: python euchre.py pack.txt shuffle 10 Alice AI Bob AI Charlie AI Dave Human --model model_final.pt")
        sys.exit(1)
    
    # Auto-load trained model or optionally train first
    model = None
    model_path = "model_final.pt"  # Default model location
    
    # Check for --model flag (override default)
    if "--model" in sys.argv:
        model_idx = sys.argv.index("--model")
        if model_idx + 1 < len(sys.argv):
            model_path = sys.argv[model_idx + 1]
    
    # Auto-train if model doesn't exist and --train flag is present
    if "--train" in sys.argv and not os.path.exists(model_path):
        print("=" * 60)
        print("No trained model found. Starting training session...")
        print("=" * 60)
        from train import train
        import argparse
        train_args = argparse.Namespace(
            episodes=50000,
            eval_every=250,
            checkpoint_every=500
        )
        train(train_args)
        print("\n" + "=" * 60)
        print("Training complete! Starting game...")
        print("=" * 60 + "\n")
    
    # Try to load the model
    if os.path.exists(model_path):
        try:
            import torch
            from model import EuchreQNetwork
            model = EuchreQNetwork()
            model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
            model.eval()
            print(f"✓ Loaded trained model from {model_path}")
            print("  AI players will use neural network for card play")
        except ImportError:
            print(f"⚠ Model file found but torch not installed")
            print("  AI players will use heuristics only")
            print("  (Install torch to use the trained model)")
            model = None
    else:
        print(f"ℹ No trained model found at '{model_path}'")
        print("  AI players will use heuristics only for card play")
        print("  (Run 'python3 train.py' to train a model, or use --train flag)")
    
    players = [None] * 4
    try:
        # Parse arguments - can have optional thresholds after AI player type
        arg_idx = 4  # Start after pack_filename, shuffle, and points_to_win
        
        for i in range(4):
            name = sys.argv[arg_idx]
            player_type = sys.argv[arg_idx + 1]
            
            # Check if next arg is a threshold (number) for AI players
            if player_type.lower() == "ai" and arg_idx + 2 < len(sys.argv) and "--" not in sys.argv[arg_idx + 2]:
                try:
                    thresh = float(sys.argv[arg_idx + 2])
                    # Use same threshold for both rounds
                    players[i] = Player_factory(name, player_type, model=model, epsilon=0.0, 
                                               round1_threshold=thresh, round2_threshold=thresh)
                    arg_idx += 3  # name, type, threshold
                except (ValueError, IndexError):
                    # Not valid threshold, use defaults
                    players[i] = Player_factory(name, player_type, model=model, epsilon=0.0)
                    arg_idx += 2  # name, type
            else:
                players[i] = Player_factory(name, player_type, model=model, epsilon=0.0)
                arg_idx += 2  # name, type
                
    except ValueError as e:
        print(f"Error creating players: {e}")
        print("Valid player types are: 'Human' and 'AI'")
        sys.exit(1)
    
    print(f"./euchre.exe {' '.join(sys.argv[1:])}")
    print()
    
    try:
        game = Game(players, int(sys.argv[3]), sys.argv[2], sys.argv[1])
        game.play_game()
    except FileNotFoundError:
        print(f"Error: Could not find pack file '{sys.argv[1]}'")
        sys.exit(1)
    except Exception as e:
        print(f"Error during game: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()