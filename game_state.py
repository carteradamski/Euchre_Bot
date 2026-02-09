import copy


class GameState:
    """
    Everything an AI player is allowed to know at any decision point.
    The Game class in euchre.py fills this in before each player call.
    The AI reads it, attaches self.hand as my_hand, then hands it to the encoder.

    Fields set ONCE per hand (by play_hand after trump is decided):
        trump           - Suit enum
        dealer_index    - int 0-3
        upcard          - Card (the turned-up card, visible all hand)
        caller_index    - int 0-3, who called trump
        team1_calls     - bool, did team 1 (players 0,2) call trump
        going_alone     - bool
        alone_team1     - bool, is the lone player on team 1

    Fields updated EVERY trick (by play_trick before each player call):
        trick_index         - int 0-4, which trick we're on
        my_index            - int 0-3, THIS player's seat
        team1_tricks        - int, tricks won by team 1 so far
        team2_tricks        - int, tricks won by team 2 so far
        led_card            - Card or None.  None means I am the leader.
        cards_played_this_trick - list of (player_index, Card) in play order so far

    Field set by play_hand after each trick resolves:
        trick_history       - list of completed tricks, each a list of (player_index, Card)

    Field set by AIPlayer RIGHT BEFORE encoding (not by the game engine):
        my_hand             - list of Card, this player's current hand
    """

    def __init__(self):
        # --- per-hand constants ---
        self.trump = None
        self.dealer_index = None
        self.upcard = None
        self.caller_index = None
        self.team1_calls = False
        self.going_alone = False
        self.alone_team1 = False

        # --- per-trick, per-player-turn state ---
        self.trick_index = 0
        self.my_index = None
        self.team1_tricks = 0
        self.team2_tricks = 0
        self.led_card = None
        self.cards_played_this_trick = []

        # --- accumulates across the hand ---
        self.trick_history = []

        # --- set by AIPlayer, not by the engine ---
        self.my_hand = []

    def snapshot(self):
        """Deep copy — used by the training loop to freeze state before the action."""
        return copy.deepcopy(self)
