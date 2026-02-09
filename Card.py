from enum import IntEnum

############### Rank implementation ###############

class Rank(IntEnum):
    TWO = 0
    THREE = 1
    FOUR = 2
    FIVE = 3
    SIX = 4
    SEVEN = 5
    EIGHT = 6
    NINE = 7
    TEN = 8
    JACK = 9
    QUEEN = 10
    KING = 11
    ACE = 12

RANK_NAMES = [
    "Two",
    "Three",
    "Four",
    "Five",
    "Six",
    "Seven",
    "Eight",
    "Nine",
    "Ten",
    "Jack",
    "Queen",
    "King",
    "Ace"
]

def string_to_rank(s):
    """Converts a string to a Rank enum value."""
    for r in Rank:
        if s == RANK_NAMES[r]:
            return r
    assert False, "Input string didn't match any rank"

############### Suit implementation ###############

class Suit(IntEnum):
    SPADES = 0
    HEARTS = 1
    CLUBS = 2
    DIAMONDS = 3

SUIT_NAMES = [
    "Spades",
    "Hearts",
    "Clubs",
    "Diamonds"
]

def string_to_suit(s):
    """Converts a string to a Suit enum value."""
    for suit in Suit:
        if s == SUIT_NAMES[suit]:
            return suit
    return None

def Suit_next(suit):
    """Returns the 'next' suit (the suit of the same color)."""
    if suit == Suit.SPADES:
        return Suit.CLUBS
    if suit == Suit.CLUBS:
        return Suit.SPADES
    if suit == Suit.DIAMONDS:
        return Suit.HEARTS
    if suit == Suit.HEARTS:
        return Suit.DIAMONDS
    return suit

############### Card class implementation ###############

class Card:
    def __init__(self, rank=None, suit=None):
        """Initialize a Card with given rank and suit, or default to Two of Spades."""
        if rank is None:
            self.rank = Rank.TWO
            self.suit = Suit.SPADES
        else:
            self.rank = rank
            self.suit = suit
    
    def get_rank(self):
        """Returns the rank of the card."""
        return self.rank
    
    def get_suit(self, trump=None):
        """Returns the suit of the card, accounting for trump if provided."""
        if trump is None:
            return self.suit
        
        if self.rank == Rank.JACK and self.suit == trump:
            return trump
        if self.rank == Rank.JACK:
            if self.suit == Suit_next(trump):
                return trump
            return self.suit
        return self.suit
    
    def is_face_or_ace(self):
        """Returns True if the card is a face card or Ace."""
        if self.rank in [Rank.JACK, Rank.QUEEN, Rank.KING, Rank.ACE]:
            return True
        return False
    
    def is_right_bower(self, trump):
        """Returns True if the card is the right bower (Jack of trump suit)."""
        if self.rank == Rank.JACK and self.suit == trump:
            return True
        return False
    
    def is_left_bower(self, trump):
        """Returns True if the card is the left bower (Jack of same color as trump)."""
        if self.rank == Rank.JACK:
            if trump == Suit.SPADES:
                if self.suit == Suit.CLUBS:
                    return True
            if trump == Suit.CLUBS:
                if self.suit == Suit.SPADES:
                    return True
            if trump == Suit.HEARTS:
                if self.suit == Suit.DIAMONDS:
                    return True
            if trump == Suit.DIAMONDS:
                if self.suit == Suit.HEARTS:
                    return True
        return False
    
    def is_trump(self, trump):
        """Returns True if the card is a trump card."""
        if self.get_suit(trump) == trump:
            return True
        if self.is_left_bower(trump):
            return True
        return False
    
    def __lt__(self, other):
        """Less than comparison based on rank only."""
        if self.get_rank() < other.get_rank():
            return True
        return False
    
    def __le__(self, other):
        """Less than or equal comparison based on rank only."""
        if self.get_rank() <= other.get_rank():
            return True
        return False
    
    def __gt__(self, other):
        """Greater than comparison based on rank only."""
        if self.get_rank() > other.get_rank():
            return True
        return False
    
    def __ge__(self, other):
        """Greater than or equal comparison based on rank only."""
        if self.get_rank() >= other.get_rank():
            return True
        return False
    
    def __eq__(self, other):
        """Equality comparison based on rank only."""
        if self.get_rank() == other.get_rank():
            return True
        return False
    
    def __ne__(self, other):
        """Inequality comparison based on rank only."""
        if self.get_rank() != other.get_rank():
            return True
        return False
    
    def __str__(self):
        """String representation of the card."""
        return f"{RANK_NAMES[self.rank]} of {SUIT_NAMES[self.suit]}"
    
    def __repr__(self):
        """String representation for debugging."""
        return self.__str__()

############### Card comparison functions ###############

def Card_less(a, b, trump_or_led, trump=None):
    """
    Compares two cards with trump consideration.
    Can be called with 3 arguments: Card_less(a, b, trump)
    Or with 4 arguments: Card_less(a, b, led_card, trump)
    """
    # Two-argument version: Card_less(a, b, trump)
    if trump is None:
        trump = trump_or_led
        
        if a.is_trump(trump) and not b.is_trump(trump):
            return False
        if not a.is_trump(trump) and b.is_trump(trump):
            return True
        if a.is_trump(trump) and b.is_trump(trump):
            if a.is_left_bower(trump) and b.is_right_bower(trump):
                return True
            if a.is_right_bower(trump) and b.is_left_bower(trump):
                return False
            if (a.is_right_bower(trump) or a.is_left_bower(trump)) and \
               not (b.is_right_bower(trump) or b.is_left_bower(trump)):
                return False
            if not (a.is_right_bower(trump) or a.is_left_bower(trump)) and \
               (b.is_right_bower(trump) or b.is_left_bower(trump)):
                return True
            if a.get_rank() < b.get_rank():
                return True
            if a.get_rank() > b.get_rank():
                return False
            if a.get_suit() < b.get_suit():
                return True
            return False
        if a.get_rank() < b.get_rank():
            return True
        if a.get_rank() > b.get_rank():
            return False
        if a.get_suit() < b.get_suit():
            return True
        return False
    
    # Four-argument version: Card_less(a, b, led_card, trump)
    else:
        led_card = trump_or_led
        
        if a.is_trump(trump) or b.is_trump(trump):
            return Card_less(a, b, trump)
        if a.get_suit(trump) == led_card.get_suit(trump) and \
           b.get_suit(trump) != led_card.get_suit(trump):
            return False
        if a.get_suit(trump) != led_card.get_suit(trump) and \
           b.get_suit(trump) == led_card.get_suit(trump):
            return True
        if a.get_suit(trump) == led_card.get_suit(trump) and \
           b.get_suit(trump) == led_card.get_suit(trump):
            return Card_less(a, b, trump)
        if a.get_suit(trump) != led_card.get_suit(trump) and \
           b.get_suit(trump) != led_card.get_suit(trump):
            return Card_less(a, b, trump)
        return False

############### 24-card index mapping (for ML encoding) ###############

# The six ranks present in a euchre deck, in the order used by the index.
# Index within a suit:  Nine=0, Ten=1, Jack=2, Queen=3, King=4, Ace=5
EUCHRE_RANKS = [Rank.NINE, Rank.TEN, Rank.JACK, Rank.QUEEN, Rank.KING, Rank.ACE]

def card_to_index(card):
    """
    Maps a Card to a unique int in [0, 23].

    Layout  (suit * 6 + rank_offset):
        Spades   0– 5    (Nine of Spades = 0, … , Ace of Spades = 5)
        Hearts   6–11
        Clubs   12–17
        Diamonds 18–23
    """
    rank_offset = EUCHRE_RANKS.index(card.rank)
    return int(card.suit) * 6 + rank_offset

def index_to_card(index):
    """Maps an int in [0, 23] back to a Card object."""
    suit  = Suit(index // 6)
    rank  = EUCHRE_RANKS[index % 6]
    return Card(rank, suit)
