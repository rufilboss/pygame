#!/usr/bin/env python3
"""
Chess Game - A complete chess implementation with AI and multiplayer support

Features:
- Human vs Human gameplay
- Human vs AI gameplay with minimax algorithm
- Complete chess rules implementation
- Modern graphical interface
- Check, checkmate, and stalemate detection
- Castling and en passant support
- Pawn promotion
"""

from chess_game import ChessGame

if __name__ == "__main__":
    game = ChessGame()
    game.run()