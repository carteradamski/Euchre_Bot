from abc import ABC, abstractmethod
from Card import Card, Suit, Rank, Card_less, Suit_next, string_to_suit, string_to_rank, SUIT_NAMES, RANK_NAMES

MAX_HAND_SIZE = 5

class Player(ABC):
    def __init__(self, name):
        """Initialize a player with a name and empty hand."""
        self.name = name
        self.hand = []
    
    def get_name(self):
        """Return the player's name."""
        return self.name
    
    def add_card(self, card):
        """Add a card to the player's hand."""
        if len(self.hand) < MAX_HAND_SIZE:
            self.hand.append(card)
    
    def clear_hand(self):
        """Remove all cards from hand."""
        self.hand = []
    
    @abstractmethod
    def make_trump(self, upcard, is_dealer, round_num, trump):
        """
        Decide whether to order up/make trump.
        Returns: tuple (bool, Suit) where bool indicates if ordering up,
                 and Suit is the trump suit
        """
        pass
    
    @abstractmethod
    def add_and_discard(self, upcard):
        """
        Called on dealer when trump is ordered up in round 1.
        Add upcard to hand and discard a card.
        """
        pass
    
    @abstractmethod
    def lead_card(self, trump, game_state=None):
        """
        Lead a card (first card of a trick).
        game_state: GameState or None.  AI players use it; Human ignores it.
        Returns: Card that was played
        """
        pass
    
    @abstractmethod
    def play_card(self, led_card, trump, game_state=None):
        """
        Play a card following the led card.
        game_state: GameState or None.  AI players use it; Human ignores it.
        Returns: Card that was played
        """
        pass
    
    @abstractmethod
    def ask_go_alone(self):
        """
        Ask if player wants to go alone.
        Returns: bool indicating if going alone
        """
        pass
    
    def can_follow_suit(self, led_card, trump):
        """Check if player has a card that follows suit."""
        led_suit = led_card.get_suit(trump)
        for card in self.hand:
            if card.get_suit(trump) == led_suit:
                return True
        return False
    
    def remove_card(self, card):
        """Remove a specific card from hand."""
        for i, c in enumerate(self.hand):
            if c.get_rank() == card.get_rank() and c.suit == card.suit:
                return self.hand.pop(i)
        return None


class Human(Player):
    def __init__(self, name):
        """Initialize a human player."""
        super().__init__(name)
    
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
    
    def make_trump(self, upcard, is_dealer, round_num, trump):
        """Human player decides whether to make trump."""
        self.print_hand()
        print(f"\nUpcard: {upcard}")
        
        if round_num == 1:
            # Round 1: Order up the upcard suit
            while True:
                print(f"Order up {SUIT_NAMES[upcard.suit]} as trump? (y/n): ", end="")
                response = input().strip().lower()
                if response in ['y', 'yes']:
                    return (True, upcard.suit)
                elif response in ['n', 'no']:
                    return (False, trump)
                else:
                    print("Invalid input. Please enter 'y' or 'n'.")
        else:
            # Round 2: Can call any suit except upcard suit
            if is_dealer:
                print("You are the dealer and must call trump!")
            
            print("Available suits to call:")
            available_suits = []
            for suit in Suit:
                if suit != upcard.suit:
                    available_suits.append(suit)
                    print(f"  {SUIT_NAMES[suit]}")
            
            while True:
                if is_dealer:
                    print("Enter suit to call: ", end="")
                else:
                    print("Enter suit to call (or 'pass'): ", end="")
                
                response = input().strip()
                
                if response.lower() == 'pass' and not is_dealer:
                    return (False, trump)
                
                suit = string_to_suit(response)
                if suit is not None and suit != upcard.suit:
                    return (True, suit)
                
                if is_dealer:
                    print(f"Invalid suit. Dealer must call trump! Choose from: {', '.join([SUIT_NAMES[s] for s in available_suits])}")
                else:
                    print(f"Invalid suit. Choose from: {', '.join([SUIT_NAMES[s] for s in available_suits])}, or type 'pass'")
    
    def ask_go_alone(self):
        """Ask human player if they want to go alone."""
        while True:
            print(f"\n{self.name}, do you want to go alone? (y/n): ", end="")
            response = input().strip().lower()
            if response in ['y', 'yes']:
                return True
            elif response in ['n', 'no']:
                return False
            else:
                print("Invalid input. Please enter 'y' or 'n'.")
    
    def add_and_discard(self, upcard):
        """Dealer adds upcard and discards a card."""
        # Add upcard directly to hand (bypassing MAX_HAND_SIZE check)
        self.hand.append(upcard)
        
        print(f"\n{self.name}, you picked up the {upcard}")
        self.print_hand()
        
        while True:
            print(f"\nPick a card to discard [0-{len(self.hand)-1}]: ", end="")
            try:
                choice = int(input().strip())
                if 0 <= choice < len(self.hand):
                    discarded = self.hand.pop(choice)
                    print(f"Discarded: {discarded}")
                    return
                else:
                    print(f"Invalid choice. Enter a number between 0 and {len(self.hand)-1}.")
            except (ValueError, IndexError):
                print("Invalid input. Enter a number.")
    
    def lead_card(self, trump, game_state=None):
        """Human player leads a card."""
        self.print_hand(trump)
        
        while True:
            print(f"\n{self.name}, pick a card to lead [0-{len(self.hand)-1}]: ", end="")
            try:
                choice = int(input().strip())
                if 0 <= choice < len(self.hand):
                    card = self.hand.pop(choice)
                    return card
                else:
                    print(f"Invalid choice. Enter a number between 0 and {len(self.hand)-1}.")
            except (ValueError, IndexError):
                print("Invalid input. Enter a number.")
    
    def play_card(self, led_card, trump, game_state=None):
        """Human player follows suit."""
        self.print_hand(trump)
        print(f"\nLed card: {led_card}")
        
        must_follow = self.can_follow_suit(led_card, trump)
        led_suit = led_card.get_suit(trump)
        
        if must_follow:
            print(f"You must follow suit ({SUIT_NAMES[led_suit]})")
        
        while True:
            print(f"{self.name}, pick a card to play [0-{len(self.hand)-1}]: ", end="")
            try:
                choice = int(input().strip())
                if 0 <= choice < len(self.hand):
                    card = self.hand[choice]
                    
                    # Validate that player follows suit if they can
                    if must_follow and card.get_suit(trump) != led_suit:
                        print(f"Illegal move! You must follow suit. Play a {SUIT_NAMES[led_suit]}.")
                        continue
                    
                    return self.hand.pop(choice)
                else:
                    print(f"Invalid choice. Enter a number between 0 and {len(self.hand)-1}.")
            except (ValueError, IndexError):
                print("Invalid input. Enter a number.")



def Player_factory(name, player_type, model=None, epsilon=0.0, round1_threshold=0.5, round2_threshold=0.2):
    """Factory function to create players.
    
    Args:
        name: Player name
        player_type: 'Human' or 'AI'
        model: Optional trained EuchreQNetwork model for AI players
        epsilon: Exploration rate (0.0 = always use model, 1.0 = random)
        round1_threshold: Threshold for ordering up trump in round 1 (AI only)
        round2_threshold: Threshold for calling trump in round 2 (AI only)
    """
    if player_type.lower() == "human":
        return Human(name)
    if(player_type.lower() == "ai"):
        from AIplayer import AIPlayer
        return AIPlayer(name, model=model, epsilon=epsilon, 
                       round1_threshold=round1_threshold, round2_threshold=round2_threshold)
    else:
        raise ValueError(f"Unknown player type: {player_type}. Only 'Human' and 'AI' are supported.")