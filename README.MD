# Euchre Game - Command Line Version

A complete, playable implementation of Euchre for the command line with full move validation.

## Features

- **Full Euchre Rules**: Complete implementation including trump calling, following suit, and proper scoring
- **Move Validation**: Prevents illegal moves (must follow suit when possible)
- **Going Alone**: Players can choose to go alone for bonus points
- **Clear Interface**: Always shows your hand before each action with trump markers
- **Illegal Move Prevention**: Game validates all moves and prevents rule violations

## How to Run

```bash
python euchre.py <pack_file> <shuffle|noshuffle> <points_to_win> \
    <player0_name> Human <player1_name> Human \
    <player2_name> Human <player3_name> Human
```

### Example Command

**4 Human Players:**
```bash
python euchre.py pack.txt shuffle 10 Alice Human Bob Human Charlie Human Dave Human
```

## Teams

- **Team 1**: Player 0 and Player 2 (1st and 3rd in command line)
- **Team 2**: Player 1 and Player 3 (2nd and 4th in command line)

Example:
- Alice (Player 0) and Charlie (Player 2) are Team 1
- Bob (Player 1) and Dave (Player 3) are Team 2

## Game Rules

### Objective
Be the first team to reach the target points (typically 10).

### Trump Calling

**Round 1:** Players can order up the turned-up card as trump
- If ordered up, dealer picks up the card and discards one
- The dealer's hand is shown with all 6 cards before discarding

**Round 2:** If no one orders up, players can call any suit EXCEPT the turned-up suit
- The dealer MUST call trump if everyone else passes

### Going Alone

After calling trump, you can choose to go alone:
- Your partner sits out the hand
- If you take all 5 tricks: **4 points**
- If you take 3-4 tricks: **1 point**
- If you get euchred (fewer than 3 tricks): Opponents get **2 points**

### Playing Tricks

1. **Lead Player** plays any card from their hand
2. **Following Players** must:
   - Follow suit if possible (play a card of the led suit)
   - If you can't follow suit, you may play any card
   - The game PREVENTS you from playing an illegal card

3. **Trump Cards** beat non-trump cards
4. **Highest trump** or **highest card of led suit** wins the trick

### Special Cards

- **Right Bower**: Jack of trump suit (highest card in the game)
  - Shown as [TRUMP - RIGHT BOWER]
- **Left Bower**: Jack of same color as trump (second highest card)
  - Shown as [TRUMP - LEFT BOWER]
  - Example: If Hearts is trump, Jack of Diamonds is the left bower
  - Example: If Spades is trump, Jack of Clubs is the left bower

### Scoring

- **March** (all 5 tricks): 
  - 2 points normally
  - 4 points if going alone
- **Win** (3-4 tricks): 1 point
- **Euchred**: If the team that called trump takes fewer than 3 tricks, opponents get 2 points

## Human Player Controls

Your hand is ALWAYS shown before you need to make a decision. You'll see:
- Card numbers [0-4] or [0-5] when discarding
- Trump markers: [TRUMP], [TRUMP - RIGHT BOWER], or [TRUMP - LEFT BOWER]
- The current game state

### Making Trump

**Round 1:**
- Type `y` or `yes` to order up the upcard suit
- Type `n` or `no` to pass

**Round 2:**
- Type the suit name: `Spades`, `Hearts`, `Clubs`, or `Diamonds`
- Type `pass` to pass (unless you're the dealer)
- Dealer MUST call trump - cannot pass

### Going Alone
- Type `y` or `yes` to go alone
- Type `n` or `no` to play with your partner

### Playing Cards
- Enter the number [0-4] of the card you want to play
- If you try to not follow suit when you can, the game will prevent it and ask again
- You cannot make illegal moves!

## Example Game Flow

```
Hand 0
Alice deals
Nine of Hearts turned up

--- ROUND 1 BIDDING ---

Bob's hand:
  [0] Jack of Hearts [TRUMP - RIGHT BOWER]
  [1] King of Hearts [TRUMP]
  [2] Ten of Clubs
  [3] Nine of Spades
  [4] Queen of Diamonds

Upcard: Nine of Hearts
Order up Hearts as trump? (y/n): y

Bob orders up Hearts

Bob, do you want to go alone? (y/n): n

Alice, you picked up the Nine of Hearts

Alice's hand:
  [0] Ace of Clubs
  [1] King of Clubs
  [2] Queen of Clubs
  [3] Jack of Clubs
  [4] Ten of Diamonds
  [5] Nine of Hearts [TRUMP]

Pick a card to discard [0-5]: 4
Discarded: Ten of Diamonds

--- Trick (Trump: Hearts) ---

Bob's hand:
  [0] Jack of Hearts [TRUMP - RIGHT BOWER]
  [1] King of Hearts [TRUMP]
  [2] Ten of Clubs
  [3] Nine of Spades
  [4] Queen of Diamonds

Bob, pick a card to lead [0-4]: 0
Jack of Hearts led by Bob

Charlie's hand:
  [0] Nine of Clubs
  [1] Ten of Clubs
  [2] Queen of Spades
  [3] King of Spades
  [4] Ace of Spades

Led card: Jack of Hearts
You must follow suit (Hearts)
Charlie, pick a card to play [0-4]: 1
Illegal move! You must follow suit. Play a Hearts.
Charlie, pick a card to play [0-4]: 0
Nine of Clubs played by Charlie
...
```

## Files Required

1. **Card.py** - Card, Rank, and Suit definitions
2. **Pack.py** - Deck management and shuffling
3. **Player.py** - Human player implementation
4. **euchre.py** - Main game logic
5. **pack.txt** - Card deck definition (24 cards, Nine through Ace)

## Move Validation

The game enforces:
- **Must follow suit** when you have cards of the led suit
- **Cannot play out of turn**
- **Cannot play cards you don't have**
- **Dealer must call trump in Round 2** if everyone passes Round 1
- All moves are validated before being accepted

## Trump System

The trump suit determines card suit for comparisons:
- Right Bower (Jack of trump) is considered trump suit
- Left Bower (Jack of same color) is ALSO considered trump suit
- Example: If Hearts is trump:
  - Jack of Hearts is the highest card (Right Bower)
  - Jack of Diamonds is second highest (Left Bower, counts as Hearts)
  - All other Hearts are trump
  - All other Diamonds are NOT trump (except Jack)
- Add new player types to `Player_factory()`
- AI can use the same game rules and validation

Enjoy your game of Euchre! 🎴
