from Player import Player
from Card import Card, Suit, Rank, SUIT_NAMES, Suit_next, card_to_index, index_to_card


class AIPlayer(Player):
    """
    AI Player that uses an optional DQN model for card-play decisions.

    If model is None (or game_state is not passed), every decision falls back
    to simple heuristics so the player can still participate in a normal game
    while the model is being trained or tested.

    Constructor
    -----------
    name  : str
    model : EuchreQNetwork or None
    epsilon : float   exploration rate for epsilon-greedy play during training.
                      0.0 = always pick best Q; 1.0 = always random.
                      The training loop updates this externally via player.epsilon = …
    """

    def __init__(self, name, model=None, epsilon=0.03, round1_threshold=0.5, round2_threshold=0.2):
        super().__init__(name)
        self.model   = model
        self.epsilon = epsilon          # training loop sets this; default = greedy
        self.round1_threshold = round1_threshold  # threshold for ordering up in round 1
        self.round2_threshold = round2_threshold  # threshold for calling trump in round 2

        # ── training-loop hooks (set externally before each hand) ──
        # The training loop attaches these so it can capture transitions
        # without AIPlayer needing to know anything about replay buffers.
        self.on_decision = None         # callback(state_vec, action_index) or None

    # ============================================================
    # Internal helpers
    # ============================================================

    def _choose_card(self, game_state):
        """
        Core decision routine shared by lead_card and play_card.

        1. Attach current hand to game_state.
        2. Encode → 60-dim state vector.
        3. Build 24-dim legal mask.
        4. If model exists: run Q-network, mask, epsilon-greedy select.
           Else: pick a random legal card.
        5. Fire on_decision callback (for training loop).
        6. Remove the chosen card from hand and return it.
        """
        
        import torch
        from state_encoder import encode_state, build_legal_mask

        game_state.my_hand = list(self.hand)   # encoder reads this

        state_vec  = encode_state(game_state)       # (60,)
        legal_mask = build_legal_mask(game_state)   # (24,)

        if self.model is not None:
            # epsilon-greedy
            if torch.rand(1).item() < self.epsilon:
                # random legal action
                legal_indices = torch.where(legal_mask > 0)[0]
                action_index  = legal_indices[torch.randint(len(legal_indices), (1,)).item()].item()
            else:
                with torch.no_grad():
                    q_values = self.model(state_vec.unsqueeze(0))          # (1, 24)
                    masked_q = q_values + (legal_mask.unsqueeze(0) - 1) * 1e9
                    action_index = torch.argmax(masked_q, dim=1).item()    # int 0-23
        else:
            # No model — pick first legal card (deterministic fallback)
            legal_indices = torch.where(legal_mask > 0)[0]
            action_index  = legal_indices[0].item()

        # ── training hook ──
        if self.on_decision is not None:
            self.on_decision(state_vec, action_index)

        # Convert index back to Card and pull it from hand
        chosen_card = index_to_card(action_index)
        removed     = self.remove_card(chosen_card)
        if removed is None:
            # Should never happen if mask is correct; defensive fallback
            removed = self.hand.pop(0)
        return removed

    # ============================================================
    # Card-play decisions  (use the model)
    # ============================================================

    def lead_card(self, trump, game_state=None):
        """Lead a card — first card of a trick."""
        self.print_hand(trump)
        print(f"\n{self.name}, pick a card to lead [0-{len(self.hand)-1}]: ", end="")

        if game_state is not None and self.model is not None:
            card = self._choose_card(game_state)
        else:
            # Fallback: play first card in hand
            card = self.hand.pop(0)

        print(card)   # echo the choice so output looks like Human
        return card

    def play_card(self, led_card, trump, game_state=None):
        """Play a card following the led card."""
        self.print_hand(trump)
        print(f"\nLed card: {led_card}")

        must_follow = self.can_follow_suit(led_card, trump)
        led_suit    = led_card.get_suit(trump)
        if must_follow:
            print(f"You must follow suit ({SUIT_NAMES[led_suit]})")

        print(f"{self.name}, pick a card to play [0-{len(self.hand)-1}]: ", end="")

        if game_state is not None and self.model is not None:
            card = self._choose_card(game_state)
        else:
            # Fallback: first legal card
            card = None
            if must_follow:
                for c in self.hand:
                    if c.get_suit(trump) == led_suit:
                        self.hand.remove(c)
                        card = c
                        break
            if card is None:
                card = self.hand.pop(0)

        print(card)
        return card

    # ============================================================
    # Discard decision  (after dealer picks up upcard in round 1)
    # ============================================================

    def add_and_discard(self, upcard):
        """
        Dealer picks up the upcard and discards one card.

        Scoring heuristic (used whether or not a model exists):
            For each of the 6 possible 5-card hands that result from
            discarding one card, compute a hand-strength score.
            Discard the card whose removal leaves the highest score.

        Hand-strength score (per card kept):
            +base_rank   (Nine=0 … Ace=5)
            +10          if the card is trump
            +20          if the card is the right bower
            +15          if the card is the left bower
        """
        self.hand.append(upcard)   # now 6 cards

        print(f"\n{self.name}, you picked up the {upcard}")
        self.print_hand(upcard.suit)    # show full 6-card hand with trump markers

        trump = upcard.suit        # trump is the upcard's suit (round 1 order-up)

        best_discard_idx = 0
        best_score       = float('-inf')

        for i in range(len(self.hand)):
            # Score the hand that would remain if we discard card i
            score = 0
            for j, card in enumerate(self.hand):
                if j == i:
                    continue
                rank_val = _euchre_rank_value(card)
                score   += rank_val
                if card.is_right_bower(trump):
                    score += 20
                elif card.is_left_bower(trump):
                    score += 15
                elif card.is_trump(trump):
                    score += 10
            if score > best_score:
                best_score       = score
                best_discard_idx = i

        discarded = self.hand.pop(best_discard_idx)
        print(f"\nPick a card to discard [0-5]: {best_discard_idx}")
        print(f"Discarded: {discarded}")

    # ============================================================
    # Trump-calling decisions  (Monte Carlo simulation)
    # ============================================================

    def make_trump(self, upcard, is_dealer, round_num, trump):
        """
        Decide whether to order up / call trump using Monte Carlo simulation.

        Round 1: Run 100 simulations with upcard suit as trump.
                 Order up if expected value is positive.
        Round 2: Run 100 simulations for each remaining suit.
                 Call the suit with highest EV if positive, or if dealer
                 (forced to call) just pick the highest EV suit.
        """
        self.print_hand()
        print(f"\nUpcard: {upcard}")

        if round_num == 1:
            candidate = upcard.suit
            
            # During training (epsilon > 0), use simple epsilon-greedy for trump
            # to avoid Monte Carlo feedback loops that destabilize training
            if self.epsilon > 0:
                # Random decision based on threshold
                import random
                order_up = random.random() < self.round1_threshold
            else:
                # Inference mode: use Monte Carlo simulation
                ev = _simulate_trump_decision(self.hand, upcard, candidate, is_dealer, self.model, 1000)
                order_up = ev > self.round1_threshold

            print(f"Order up {SUIT_NAMES[upcard.suit]} as trump? (y/n): ", end="")
            if order_up:
                print("y")
                return (True, upcard.suit)
            else:
                print("n")
                return (False, trump)

        else:   # round 2
            if is_dealer:
                print("You are the dealer and must call trump!")

            print("Available suits to call:")
            available_suits = []
            for suit in Suit:
                if suit != upcard.suit:
                    available_suits.append(suit)
                    print(f"  {SUIT_NAMES[suit]}")

            # Simulate each available suit
            best_suit = available_suits[0]
            best_ev   = float('-inf')
            
            # During training (epsilon > 0), use random selection for trump
            if self.epsilon > 0:
                import random
                best_suit = random.choice(available_suits)
                best_ev = 0.2 if is_dealer else 0.1  # Dummy values
            else:
                # Inference mode: use Monte Carlo simulation
                for suit in available_suits:
                    ev = _simulate_trump_decision(self.hand, None, suit, is_dealer, self.model, 100)
                    if ev > best_ev:
                        best_ev = ev
                        best_suit = suit

            if is_dealer:
                print(f"Enter suit to call: ", end="")
            else:
                print(f"Enter suit to call (or 'pass'): ", end="")

            # Call if EV is strong enough, or if dealer (must call)
            # Round 2 should be even more conservative since no upcard pickup
            if best_ev > self.round2_threshold or is_dealer:
                print(SUIT_NAMES[best_suit])
                return (True, best_suit)
            else:
                print("pass")
                return (False, trump)

    # ============================================================
    # Go-alone decision  (heuristic)
    # ============================================================

    def ask_go_alone(self):
        return False

    # ============================================================
    # Display
    # ============================================================

    def print_hand(self, trump=None):
        """Print the player's current hand with indices."""
        print(f"\n{self.name}'s hand:")
        for i, card in enumerate(self.hand):
            trump_marker = ""
            if trump is not None and card.is_trump(trump):
                if card.is_right_bower(trump):
                    trump_marker = " [TRUMP - RIGHT BOWER]"
                elif card.is_left_bower(trump):
                    trump_marker = " [TRUMP - LEFT BOWER]"
                else:
                    trump_marker = " [TRUMP]"
            print(f"  [{i}] {card}{trump_marker}")


# ============================================================
# Module-level helpers  (not methods — used by the heuristics above)
# ============================================================

def _euchre_rank_value(card):
    """
    0-based value for the six euchre ranks.
    Nine=0, Ten=1, Jack=2, Queen=3, King=4, Ace=5
    """
    _order = {Rank.NINE: 0, Rank.TEN: 1, Rank.JACK: 2,
              Rank.QUEEN: 3, Rank.KING: 4, Rank.ACE: 5}
    return _order.get(card.rank, 0)


def _hand_trump_score(hand, trump_suit):
    """
    Score a 5-card hand assuming trump_suit is trump.
    Used by make_trump to decide whether / which suit to call.

    Per card:
        base rank value  (0–5)
        +20  right bower
        +15  left bower
        +10  other trump
    """
    score = 0
    for card in hand:
        score += _euchre_rank_value(card)
        if card.is_right_bower(trump_suit):
            score += 20
        elif card.is_left_bower(trump_suit):
            score += 15
        elif card.is_trump(trump_suit):
            score += 10
    return score


def _simulate_trump_decision(my_hand, upcard, trump_suit, is_dealer, model, num_sims=1000):
    """
    Monte Carlo simulation to estimate expected value of calling trump.
    
    Args:
        my_hand: list of Cards in my hand (5 cards)
        upcard: Card or None (for round 2)
        trump_suit: Suit to simulate
        is_dealer: bool, whether I'm the dealer
        model: EuchreQNetwork or None - if provided, uses ML model for decisions
        num_sims: number of simulations to run
    
    Returns:
        Expected value (positive = good for my team, negative = bad)
        Range is approximately -2 to +2 (points per hand)
    """
    import random
    import torch
    from Card import Card_less
    from state_encoder import encode_state, build_legal_mask
    from game_state import GameState
    
    # Build the deck of unknown cards
    all_cards = []
    for suit in Suit:
        for rank in [Rank.NINE, Rank.TEN, Rank.JACK, Rank.QUEEN, Rank.KING, Rank.ACE]:
            all_cards.append(Card(rank, suit))
    
    # Remove my cards from the deck
    remaining_cards = []
    for card in all_cards:
        is_in_my_hand = False
        for my_card in my_hand:
            if my_card.rank == card.rank and my_card.suit == card.suit:
                is_in_my_hand = True
                break
        if not is_in_my_hand:
            # Also remove upcard if it exists
            if upcard is not None and card.rank == upcard.rank and card.suit == upcard.suit:
                continue
            remaining_cards.append(card)
    
    # Run simulations
    total_score = 0.0
    
    for _ in range(num_sims):
        # Deal random cards to 3 opponents (4 cards each = 12 cards, leaving 7)
        # In round 1, dealer gets upcard so opponents get 4+4+4 (12 cards)
        # In round 2, all opponents get 5 cards each (15 cards)
        random.shuffle(remaining_cards)
        
        if is_dealer and upcard is not None:
            # Round 1: I'm dealer, I pick up upcard
            # My partner and 2 opponents each have 5 cards
            opp_hands = [
                remaining_cards[0:5],   # left opponent
                remaining_cards[5:10],  # partner
                remaining_cards[10:15], # right opponent
            ]
        else:
            # Round 2 or I'm not dealer: everyone has 5 cards
            opp_hands = [
                remaining_cards[0:5],
                remaining_cards[5:10],
                remaining_cards[10:15],
            ]
        
        # Simulate 5 tricks
        # Players are: me (0), left (1), partner (2), right (3)
        hands = [list(my_hand), opp_hands[0], opp_hands[1], opp_hands[2]]
        tricks_won_by_my_team = 0
        # Randomize starting leader to avoid bias
        leader = random.randint(0, 3)
        trick_history = []
        
        for trick_num in range(5):
            trick_cards = []
            
            # Each player plays a card
            for i in range(4):
                player_idx = (leader + i) % 4
                hand = hands[player_idx]
                
                if len(hand) == 0:
                    break
                
                # Build game state for model
                if model is not None:
                    gs = GameState()
                    gs.trump = trump_suit
                    gs.my_hand = list(hand)
                    gs.trick_history = list(trick_history)
                    gs.cards_played_this_trick = list(trick_cards)
                    gs.hand_leader = leader
                    gs.my_seat = player_idx
                    gs.my_index = player_idx  # REQUIRED for state encoding
                    gs.trick_index = trick_num
                    gs.team1_tricks = tricks_won_by_my_team if (player_idx == 0 or player_idx == 2) else (trick_num - tricks_won_by_my_team)
                    gs.team2_tricks = (trick_num - tricks_won_by_my_team) if (player_idx == 0 or player_idx == 2) else tricks_won_by_my_team
                    gs.led_card = trick_cards[0][1] if len(trick_cards) > 0 else None
                    
                    # Encode state and get legal mask
                    state_vec = encode_state(gs)
                    legal_mask = build_legal_mask(gs)
                    
                    # Use model to choose card (no epsilon, always greedy in simulation)
                    with torch.no_grad():
                        q_values = model(state_vec.unsqueeze(0))
                        masked_q = q_values + (legal_mask.unsqueeze(0) - 1) * 1e9
                        action_index = torch.argmax(masked_q, dim=1).item()
                    
                    played_card = index_to_card(action_index)
                    
                    # Ensure the card is actually in hand (safety check)
                    card_in_hand = None
                    for c in hand:
                        if c.rank == played_card.rank and c.suit == played_card.suit:
                            card_in_hand = c
                            break
                    
                    if card_in_hand is None:
                        # Fallback to heuristic if model made illegal move
                        if i == 0:
                            played_card = _sim_lead_card(hand, trump_suit)
                        else:
                            led_card = trick_cards[0][1]
                            played_card = _sim_follow_card(hand, led_card, trump_suit)
                    else:
                        played_card = card_in_hand
                else:
                    # No model - use heuristics
                    if i == 0:
                        played_card = _sim_lead_card(hand, trump_suit)
                    else:
                        led_card = trick_cards[0][1]
                        played_card = _sim_follow_card(hand, led_card, trump_suit)
                
                trick_cards.append((player_idx, played_card))
                hand.remove(played_card)
            
            # Determine trick winner
            if len(trick_cards) > 0:
                led_card = trick_cards[0][1]
                winner_idx = 0
                winner_card = trick_cards[0][1]
                
                for j in range(1, len(trick_cards)):
                    if Card_less(winner_card, trick_cards[j][1], led_card, trump_suit):
                        winner_card = trick_cards[j][1]
                        winner_idx = j
                
                winner_player = trick_cards[winner_idx][0]
                leader = winner_player
                
                # Add to trick history
                trick_history.append(list(trick_cards))
                
                # Check if my team won (players 0 and 2)
                if winner_player == 0 or winner_player == 2:
                    tricks_won_by_my_team += 1
        
        # Score the hand
        # When MY TEAM calls trump:
        #   - Win 3-4 tricks: +1 point
        #   - Win all 5 (march): +2 points
        #   - Get euchred (opponents win 3+): -2 points (very bad!)
        opponent_tricks = 5 - tricks_won_by_my_team
        
        if tricks_won_by_my_team >= 3:
            if tricks_won_by_my_team == 5:
                total_score += 2.0
            else:
                total_score += 1.0
        else:
            # Got euchred - we called trump and lost
            # This is always -2 points in euchre
            total_score -= 2.0
    
    return total_score / num_sims


def _sim_lead_card(hand, trump_suit):
    """Simple heuristic: lead highest trump, or highest card."""
    best_card = None
    best_score = -1
    
    for card in hand:
        score = _euchre_rank_value(card)
        if card.is_right_bower(trump_suit):
            score += 100
        elif card.is_left_bower(trump_suit):
            score += 90
        elif card.is_trump(trump_suit):
            score += 50
        
        if score > best_score:
            best_score = score
            best_card = card
    
    return best_card if best_card else hand[0]


def _sim_follow_card(hand, led_card, trump_suit):
    """Simple heuristic: follow suit if possible, otherwise dump lowest card."""
    led_suit = led_card.get_suit(trump_suit)
    
    # Try to follow suit
    follow_cards = [c for c in hand if c.get_suit(trump_suit) == led_suit]
    
    if follow_cards:
        # Play highest card that follows suit
        best = follow_cards[0]
        for card in follow_cards[1:]:
            from Card import Card_less
            if Card_less(best, card, led_card, trump_suit):
                best = card
        return best
    else:
        # Can't follow suit - dump lowest non-trump card, or lowest trump
        non_trump = [c for c in hand if not c.is_trump(trump_suit)]
        if non_trump:
            return min(non_trump, key=lambda c: _euchre_rank_value(c))
        else:
            return min(hand, key=lambda c: _euchre_rank_value(c))