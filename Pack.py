from Card import Card, Rank, Suit
import random

PACK_SIZE = 24


class Pack:
    def __init__(self, pack_input=None):
        """
        Initialize a Pack either with a standard euchre deck or from input.
        
        Args:
            pack_input: Optional file-like object to read cards from
        """
        self.next = 0
        self.cards = [None] * PACK_SIZE
        
        if pack_input is None:
            # Create standard euchre deck (Nine through Ace of each suit)
            for i in range(4):
                for j in range(7, 13):
                    card = Card(Rank(j), Suit(i))
                    self.cards[i * 6 + (j - 7)] = card
        else:
            # Read cards from input
            for i in range(PACK_SIZE):
                line = pack_input.readline().strip()
                if line:
                    parts = line.split()
                    if len(parts) >= 3:
                        rank_str = parts[0]
                        suit_str = parts[2]
                        
                        from Card import string_to_rank, string_to_suit
                        rank = string_to_rank(rank_str)
                        suit = string_to_suit(suit_str)
                        card = Card(rank, suit)
                        self.cards[i] = card
    
    def deal_one(self):
        """Deal one card from the pack."""
        if self.next < PACK_SIZE:
            deal = self.cards[self.next]
            self.next += 1
            return deal
        return self.cards[PACK_SIZE - 1]
    
    def reset(self):
        """Reset the pack to start dealing from the beginning."""
        self.next = 0
    
    def shuffle(self):
        """Shuffle the pack into a random order."""
        random.shuffle(self.cards)
        self.reset()
    
    def empty(self):
        """Check if all cards have been dealt."""
        if self.next == PACK_SIZE:
            return True
        return False