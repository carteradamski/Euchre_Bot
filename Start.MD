# Quick Start Guide - Euchre

## Running the Game

```bash
python3 euchre.py pack.txt shuffle 10 Alice AI Bob AI Charlie AI Dave AI
```

**Teams:**
- Team 1: Alice & Charlie (players 0 & 2)
- Team 2: Bob & Dave (players 1 & 3)

## How to Play

### 1. Calling Trump

Your hand is shown BEFORE you make any decision.

**Round 1 - Order Up:**
```
Alice's hand:
  [0] Nine of Spades
  [1] Ten of Hearts
  [2] Jack of Hearts [TRUMP - RIGHT BOWER]
  [3] Queen of Clubs
  [4] King of Diamonds

Upcard: Nine of Hearts
Order up Hearts as trump? (y/n): y
```

**Round 2 - Call a Suit:**
```
Available suits to call:
  Spades
  Clubs
  Diamonds
Enter suit to call (or 'pass'): Spades
```

### 2. Going Alone

After you call trump:
```
Alice, do you want to go alone? (y/n): y
```

**Going Alone Scoring:**
- All 5 tricks = **4 points**
- 3-4 tricks = **1 point**
- Euchred = Opponents get **2 points**

### 3. Playing Cards

You'll see your hand before EVERY play:
```
Bob's hand:
  [0] Jack of Hearts [TRUMP - RIGHT BOWER]
  [1] King of Hearts [TRUMP]
  [2] Ten of Clubs
  [3] Nine of Spades
  [4] Queen of Diamonds

Bob, pick a card to lead [0-4]: 0
```

**Following Suit:**
```
Charlie's hand:
  [0] Nine of Clubs
  [1] Ten of Clubs
  [2] Queen of Spades
  [3] King of Spades
  [4] Ace of Spades

Led card: Jack of Hearts
You must follow suit (Hearts)
Charlie, pick a card to play [0-4]: 2
Illegal move! You must follow suit. Play a Hearts.
```

The game **prevents illegal moves** - you MUST follow suit if you can!

## Quick Rules

1. **Trump beats everything** - Even a Nine of trump beats an Ace of non-trump
2. **Right Bower** - Jack of trump suit (highest card)
3. **Left Bower** - Jack of same color (second highest, counts as trump)
4. **Must follow suit** - Play the led suit if you have it
5. **Can't break rules** - Game validates all moves

## Trump Markers

- `[TRUMP]` - Regular trump card
- `[TRUMP - RIGHT BOWER]` - Jack of trump (highest)
- `[TRUMP - LEFT BOWER]` - Jack of same color (second highest)

## Scoring

- **March** (5 tricks): 2 points (4 if alone)
- **Win** (3-4 tricks): 1 point
- **Euchred** (<3 tricks): Opponents get 2 points

First to 10 points wins!

## Common Mistakes

❌ **Trying to not follow suit when you can**
```
You must follow suit (Hearts)
Pick a card to play [0-4]: 2
Illegal move! You must follow suit. Play a Hearts.
```

✅ **The game tells you when you must follow suit and prevents violations**

## See the Full README for Complete Rules

Happy playing! 🎴