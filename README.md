# Chess Game

A complete chess implementation built with Python and Pygame featuring both human vs human and human vs AI gameplay.

## Features

### Core Chess Features
- ✅ Complete chess rules implementation
- ✅ All piece movements (Pawn, Rook, Knight, Bishop, Queen, King)
- ✅ Special moves: Castling, En passant, Pawn promotion
- ✅ Check, checkmate, and stalemate detection
- ✅ Move validation and legal move checking

### Game Modes
- ✅ **Human vs Human**: Play against another person locally
- ✅ **Human vs AI**: Play against computer opponent with minimax algorithm

### Interface
- ✅ Modern graphical interface with pygame
- ✅ Visual piece representation using Unicode chess symbols
- ✅ Square highlighting for selected pieces and valid moves
- ✅ Check indication with red highlighting
- ✅ Game status display (current player, check status, game over)
- ✅ Menu system for game mode selection

### AI Features
- ✅ Minimax algorithm with alpha-beta pruning
- ✅ Configurable difficulty (search depth)
- ✅ Piece value evaluation
- ✅ Legal move generation and validation

## Installation

1. Install Python 3.7 or higher
2. Install pygame:
   ```bash
   pip install -r requirements.txt
   ```

## How to Run

```bash
python main.py
```

## How to Play

1. **Main Menu**: Choose between "Human vs Human" or "Human vs AI"
2. **Gameplay**:
   - Click on a piece to select it (highlighted in blue)
   - Valid moves are shown in green
   - Click on a valid move to make the move
   - Red highlighting indicates check
3. **Controls**:
   - "Menu" button: Return to main menu
   - "New Game" button: Start a new game

## Game Rules

### Standard Chess Rules
- White moves first
- Players alternate turns
- Capture opponent pieces by moving to their square
- Win by checkmate (opponent's king is in check with no legal moves)
- Draw by stalemate (no legal moves but not in check)

### Special Moves
- **Castling**: King and rook special move (kingside/queenside)
- **En Passant**: Special pawn capture
- **Pawn Promotion**: Pawns reaching the end automatically promote to Queen

### AI Difficulty
The AI uses a minimax algorithm with alpha-beta pruning. The default depth is 3, providing a challenging but not overwhelming opponent.

## Technical Details

### Architecture
- `ChessBoard`: Core game logic and rule implementation
- `Piece`: Individual chess piece representation
- `ChessAI`: Minimax algorithm implementation
- `ChessGame`: Main game loop and UI handling

### Key Algorithms
- **Move Generation**: Efficient legal move calculation
- **Check Detection**: Square attack analysis
- **Minimax with Alpha-Beta Pruning**: AI decision making
- **Board Evaluation**: Position scoring for AI

## File Structure

```
pygame/
├── main.py              # Entry point
├── chess_game.py        # Complete chess implementation
├── requirements.txt     # Dependencies
└── README.md           # This file
```

## Controls Summary

- **Mouse Click**: Select pieces and make moves
- **Menu Button**: Return to main menu
- **New Game Button**: Reset current game

Enjoy playing chess!