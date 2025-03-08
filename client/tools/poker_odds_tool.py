from phevaluator import evaluate_cards
import random
from langchain_core.tools import tool


suits = ['d','s','c','h']
ranks = ['A','2','3','4','5','6','7','8','9','T','J','Q','K']
cards = []
for r in ranks:
    for s in suits:
        cards.append(r+s)

def simulate(hand, table, players):
    hands = []
    deck = random.sample(cards,len(cards)) #shuffle the deck
    hand = hand[:]
    table = table[:]

    full = table + hand
    deck = list(filter(lambda x: x not in full, deck))

    #deal cards to players
    for i in range(players):
        hn = []
        hn.append(deck[0])
        deck = deck[1:]
        hn.append(deck[0])
        deck = deck[1:]
        hands.append(hn)
    
    #flop, turn, river
    while len(table) < 5:
        card = deck.pop(0)
        table.append(card)
        full.append(card)
    my_hand_rank = evaluate_cards(full[0],full[1],full[2],full[3],full[4],full[5],full[6])

    result_state = 0  # 'WIN' - assume win until proven otherwise
    for check_hand in hands:
        all_cards = table + check_hand
        opponent = evaluate_cards(all_cards[0],all_cards[1],all_cards[2],all_cards[3],all_cards[4],all_cards[5],all_cards[6])
        # from the definition of the library we use for hand evaluation, larger evaluations correspond to less strong hands
        #so, the game is won by the player with the smallest hand evaluation
        if opponent < my_hand_rank:
            result_state = 1  # 'LOSE' - there is no comeback from losing, stop comparing
            break
        if opponent == my_hand_rank:
            result_state = 2  # 'SPLIT' - on split, we still might lose to another player
    return result_state

def monte_carlo(hand, table=None, players=2, samples=10000):
    if table is None:
        table = []
    dist = [0,0,0]

    for i in range(samples):
        outcome = simulate(hand, table, players)
        dist[outcome] += 1
    return list(map(lambda x: x/samples, dist))

@tool
def poker_odds(hand : list[str], flop : list[str]):
    """
    Calculate the odds of you winning, losing, or splitting the pot
    Convert these poker cards to evaluator format:
    - Replace card symbols with letters: ♣→c, ♦→d, ♥→h, ♠→s
    - Replace 10 with T
    - Replace face cards: Jack→J, Queen→Q, King→K, Ace→A
    - Format should be rank+suit with no spaces (e.g., "Qh", "Tc", "As")

    Examples:
    "Q♥" → "Qh"
    "10♣" → "Tc"
    "Ace of spades" → "As"
    "K♦, 7♥" → "Kd, 7h"
    Args:
        hand: List of 2 cards (e.g., ['Qs', 'Th'])
        flop: List of 0-5 community cards 
        players: Number of players (default: 2)
        samples: Number of Monte Carlo simulations (default: 10000)
        
    Returns:
        Dictionary with win, lose, and split probabilities
    """
    if flop is None:
        flop = []
    
    # Validate input
    if len(hand) != 2:
        raise ValueError("Hand must contain exactly 2 cards")
    if flop and len(flop) > 5:
        raise ValueError("Flop cannot contain more than 5 cards")
    
    result = monte_carlo(hand, flop, 2, 10000)
    print(result)
    return {
        "win": result[0],
        "lose": result[1],
        "split": result[2]
    }

if __name__ == "__main__":
    # Example 1: With flop
    my_hand = ['Qs', 'Th']
    flop = ['Ah']
    result = poker_odds(my_hand, flop)
    
    print("Example with flop:")
    print(f"Win probability: {result['win']:.2%}")
    print(f"Lose probability: {result['lose']:.2%}")
    print(f"Split probability: {result['split']:.2%}")
    
