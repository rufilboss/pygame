import pygame
import sys
import math
from enum import Enum
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass

# Initialize Pygame
pygame.init()

# Constants
BOARD_SIZE = 8
SQUARE_SIZE = 80
BOARD_WIDTH = BOARD_SIZE * SQUARE_SIZE
BOARD_HEIGHT = BOARD_SIZE * SQUARE_SIZE
SIDEBAR_WIDTH = 300
WINDOW_WIDTH = BOARD_WIDTH + SIDEBAR_WIDTH
WINDOW_HEIGHT = BOARD_HEIGHT + 100

# Colors
WHITE = (240, 217, 181)
BLACK = (181, 136, 99)
HIGHLIGHT = (255, 255, 0, 128)
VALID_MOVE = (0, 255, 0, 128)
CHECK = (255, 0, 0, 128)
SELECTED = (0, 0, 255, 128)
BG_COLOR = (50, 50, 50)
TEXT_COLOR = (255, 255, 255)
BUTTON_COLOR = (70, 70, 70)
BUTTON_HOVER = (90, 90, 90)

class PieceType(Enum):
    PAWN = "pawn"
    ROOK = "rook"
    KNIGHT = "knight"
    BISHOP = "bishop"
    QUEEN = "queen"
    KING = "king"

class Color(Enum):
    WHITE = "white"
    BLACK = "black"

class GameMode(Enum):
    MENU = "menu"
    HUMAN_VS_HUMAN = "human_vs_human"
    HUMAN_VS_AI = "human_vs_ai"

@dataclass
class Move:
    from_pos: Tuple[int, int]
    to_pos: Tuple[int, int]
    piece: 'Piece'
    captured_piece: Optional['Piece'] = None
    is_castling: bool = False
    is_en_passant: bool = False
    promotion_piece: Optional[PieceType] = None

class Piece:
    def __init__(self, piece_type: PieceType, color: Color, position: Tuple[int, int]):
        self.type = piece_type
        self.color = color
        self.position = position
        self.has_moved = False
        self.symbol = self._get_symbol()
    
    def _get_symbol(self) -> str:
        symbols = {
            (PieceType.KING, Color.WHITE): "♔",
            (PieceType.QUEEN, Color.WHITE): "♕",
            (PieceType.ROOK, Color.WHITE): "♖",
            (PieceType.BISHOP, Color.WHITE): "♗",
            (PieceType.KNIGHT, Color.WHITE): "♘",
            (PieceType.PAWN, Color.WHITE): "♙",
            (PieceType.KING, Color.BLACK): "♚",
            (PieceType.QUEEN, Color.BLACK): "♛",
            (PieceType.ROOK, Color.BLACK): "♜",
            (PieceType.BISHOP, Color.BLACK): "♝",
            (PieceType.KNIGHT, Color.BLACK): "♞",
            (PieceType.PAWN, Color.BLACK): "♟"
        }
        return symbols[(self.type, self.color)]

class ChessBoard:
    def __init__(self):
        self.board = [[None for _ in range(8)] for _ in range(8)]
        self.current_player = Color.WHITE
        self.move_history = []
        self.en_passant_target = None
        self.white_king_pos = (7, 4)
        self.black_king_pos = (0, 4)
        self.setup_board()
    
    def setup_board(self):
        # Place pawns
        for col in range(8):
            self.board[1][col] = Piece(PieceType.PAWN, Color.BLACK, (1, col))
            self.board[6][col] = Piece(PieceType.PAWN, Color.WHITE, (6, col))
        
        # Place other pieces
        piece_order = [PieceType.ROOK, PieceType.KNIGHT, PieceType.BISHOP, PieceType.QUEEN,
                      PieceType.KING, PieceType.BISHOP, PieceType.KNIGHT, PieceType.ROOK]
        
        for col, piece_type in enumerate(piece_order):
            self.board[0][col] = Piece(piece_type, Color.BLACK, (0, col))
            self.board[7][col] = Piece(piece_type, Color.WHITE, (7, col))
    
    def get_piece(self, row: int, col: int) -> Optional[Piece]:
        if 0 <= row < 8 and 0 <= col < 8:
            return self.board[row][col]
        return None
    
    def is_valid_position(self, row: int, col: int) -> bool:
        return 0 <= row < 8 and 0 <= col < 8
    
    def get_valid_moves(self, piece: Piece) -> List[Tuple[int, int]]:
        moves = []
        row, col = piece.position
        
        if piece.type == PieceType.PAWN:
            moves = self._get_pawn_moves(piece)
        elif piece.type == PieceType.ROOK:
            moves = self._get_rook_moves(piece)
        elif piece.type == PieceType.KNIGHT:
            moves = self._get_knight_moves(piece)
        elif piece.type == PieceType.BISHOP:
            moves = self._get_bishop_moves(piece)
        elif piece.type == PieceType.QUEEN:
            moves = self._get_queen_moves(piece)
        elif piece.type == PieceType.KING:
            moves = self._get_king_moves(piece)
        
        # Filter out moves that would put own king in check
        valid_moves = []
        for move_pos in moves:
            if self._is_legal_move(piece.position, move_pos):
                valid_moves.append(move_pos)
        
        return valid_moves
    
    def _get_pawn_moves(self, piece: Piece) -> List[Tuple[int, int]]:
        moves = []
        row, col = piece.position
        direction = -1 if piece.color == Color.WHITE else 1
        start_row = 6 if piece.color == Color.WHITE else 1
        
        # Forward moves
        new_row = row + direction
        if self.is_valid_position(new_row, col) and not self.get_piece(new_row, col):
            moves.append((new_row, col))
            
            # Double move from starting position
            if row == start_row:
                new_row = row + 2 * direction
                if self.is_valid_position(new_row, col) and not self.get_piece(new_row, col):
                    moves.append((new_row, col))
        
        # Captures
        for dc in [-1, 1]:
            new_row, new_col = row + direction, col + dc
            if self.is_valid_position(new_row, new_col):
                target = self.get_piece(new_row, new_col)
                if target and target.color != piece.color:
                    moves.append((new_row, new_col))
                # En passant
                elif self.en_passant_target == (new_row, new_col):
                    moves.append((new_row, new_col))
        
        return moves
    
    def _get_rook_moves(self, piece: Piece) -> List[Tuple[int, int]]:
        return self._get_line_moves(piece, [(0, 1), (0, -1), (1, 0), (-1, 0)])
    
    def _get_bishop_moves(self, piece: Piece) -> List[Tuple[int, int]]:
        return self._get_line_moves(piece, [(1, 1), (1, -1), (-1, 1), (-1, -1)])
    
    def _get_queen_moves(self, piece: Piece) -> List[Tuple[int, int]]:
        return self._get_line_moves(piece, [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)])
    
    def _get_line_moves(self, piece: Piece, directions: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        moves = []
        row, col = piece.position
        
        for dr, dc in directions:
            for i in range(1, 8):
                new_row, new_col = row + i * dr, col + i * dc
                if not self.is_valid_position(new_row, new_col):
                    break
                
                target = self.get_piece(new_row, new_col)
                if target:
                    if target.color != piece.color:
                        moves.append((new_row, new_col))
                    break
                else:
                    moves.append((new_row, new_col))
        
        return moves
    
    def _get_knight_moves(self, piece: Piece) -> List[Tuple[int, int]]:
        moves = []
        row, col = piece.position
        knight_moves = [(2, 1), (2, -1), (-2, 1), (-2, -1), (1, 2), (1, -2), (-1, 2), (-1, -2)]
        
        for dr, dc in knight_moves:
            new_row, new_col = row + dr, col + dc
            if self.is_valid_position(new_row, new_col):
                target = self.get_piece(new_row, new_col)
                if not target or target.color != piece.color:
                    moves.append((new_row, new_col))
        
        return moves
    
    def _get_king_moves(self, piece: Piece) -> List[Tuple[int, int]]:
        moves = []
        row, col = piece.position
        
        # Regular king moves
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                new_row, new_col = row + dr, col + dc
                if self.is_valid_position(new_row, new_col):
                    target = self.get_piece(new_row, new_col)
                    if not target or target.color != piece.color:
                        moves.append((new_row, new_col))
        
        # Castling
        if not piece.has_moved and not self.is_in_check(piece.color):
            # Kingside castling
            rook = self.get_piece(row, 7)
            if (rook and rook.type == PieceType.ROOK and not rook.has_moved and
                not self.get_piece(row, 5) and not self.get_piece(row, 6)):
                if not self._is_square_attacked((row, 5), piece.color) and not self._is_square_attacked((row, 6), piece.color):
                    moves.append((row, 6))
            
            # Queenside castling
            rook = self.get_piece(row, 0)
            if (rook and rook.type == PieceType.ROOK and not rook.has_moved and
                not self.get_piece(row, 1) and not self.get_piece(row, 2) and not self.get_piece(row, 3)):
                if not self._is_square_attacked((row, 3), piece.color) and not self._is_square_attacked((row, 2), piece.color):
                    moves.append((row, 2))
        
        return moves
    
    def _is_legal_move(self, from_pos: Tuple[int, int], to_pos: Tuple[int, int]) -> bool:
        # Simulate the move and check if it leaves the king in check
        piece = self.get_piece(*from_pos)
        if not piece:
            return False
        
        captured_piece = self.get_piece(*to_pos)
        
        # Make the move temporarily
        self.board[to_pos[0]][to_pos[1]] = piece
        self.board[from_pos[0]][from_pos[1]] = None
        piece.position = to_pos
        
        # Update king position if necessary
        if piece.type == PieceType.KING:
            if piece.color == Color.WHITE:
                old_king_pos = self.white_king_pos
                self.white_king_pos = to_pos
            else:
                old_king_pos = self.black_king_pos
                self.black_king_pos = to_pos
        
        # Check if the move leaves the king in check
        is_legal = not self.is_in_check(piece.color)
        
        # Undo the move
        self.board[from_pos[0]][from_pos[1]] = piece
        self.board[to_pos[0]][to_pos[1]] = captured_piece
        piece.position = from_pos
        
        # Restore king position if necessary
        if piece.type == PieceType.KING:
            if piece.color == Color.WHITE:
                self.white_king_pos = old_king_pos
            else:
                self.black_king_pos = old_king_pos
        
        return is_legal
    
    def is_in_check(self, color: Color) -> bool:
        king_pos = self.white_king_pos if color == Color.WHITE else self.black_king_pos
        return self._is_square_attacked(king_pos, color)
    
    def _is_square_attacked(self, pos: Tuple[int, int], defending_color: Color) -> bool:
        attacking_color = Color.BLACK if defending_color == Color.WHITE else Color.WHITE
        
        for row in range(8):
            for col in range(8):
                piece = self.get_piece(row, col)
                if piece and piece.color == attacking_color:
                    if pos in self._get_attacking_squares(piece):
                        return True
        return False
    
    def _get_attacking_squares(self, piece: Piece) -> List[Tuple[int, int]]:
        # Similar to get_valid_moves but without legal move filtering
        row, col = piece.position
        
        if piece.type == PieceType.PAWN:
            moves = []
            direction = -1 if piece.color == Color.WHITE else 1
            # Only capture moves for pawn attacks
            for dc in [-1, 1]:
                new_row, new_col = row + direction, col + dc
                if self.is_valid_position(new_row, new_col):
                    moves.append((new_row, new_col))
            return moves
        elif piece.type == PieceType.ROOK:
            return self._get_line_moves(piece, [(0, 1), (0, -1), (1, 0), (-1, 0)])
        elif piece.type == PieceType.KNIGHT:
            return self._get_knight_moves(piece)
        elif piece.type == PieceType.BISHOP:
            return self._get_line_moves(piece, [(1, 1), (1, -1), (-1, 1), (-1, -1)])
        elif piece.type == PieceType.QUEEN:
            return self._get_line_moves(piece, [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)])
        elif piece.type == PieceType.KING:
            moves = []
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    new_row, new_col = row + dr, col + dc
                    if self.is_valid_position(new_row, new_col):
                        moves.append((new_row, new_col))
            return moves
        
        return []
    
    def make_move(self, from_pos: Tuple[int, int], to_pos: Tuple[int, int]) -> bool:
        piece = self.get_piece(*from_pos)
        if not piece or piece.color != self.current_player:
            return False
        
        valid_moves = self.get_valid_moves(piece)
        if to_pos not in valid_moves:
            return False
        
        captured_piece = self.get_piece(*to_pos)
        is_castling = False
        is_en_passant = False
        
        # Handle special moves
        if piece.type == PieceType.KING and abs(to_pos[1] - from_pos[1]) == 2:
            # Castling
            is_castling = True
            rook_col = 7 if to_pos[1] > from_pos[1] else 0
            new_rook_col = 5 if to_pos[1] > from_pos[1] else 3
            rook = self.get_piece(from_pos[0], rook_col)
            self.board[from_pos[0]][new_rook_col] = rook
            self.board[from_pos[0]][rook_col] = None
            rook.position = (from_pos[0], new_rook_col)
            rook.has_moved = True
        
        elif piece.type == PieceType.PAWN and self.en_passant_target == to_pos:
            # En passant
            is_en_passant = True
            captured_pawn_row = from_pos[0]
            captured_piece = self.get_piece(captured_pawn_row, to_pos[1])
            self.board[captured_pawn_row][to_pos[1]] = None
        
        # Make the move
        self.board[to_pos[0]][to_pos[1]] = piece
        self.board[from_pos[0]][from_pos[1]] = None
        piece.position = to_pos
        piece.has_moved = True
        
        # Update king position
        if piece.type == PieceType.KING:
            if piece.color == Color.WHITE:
                self.white_king_pos = to_pos
            else:
                self.black_king_pos = to_pos
        
        # Handle pawn promotion
        if piece.type == PieceType.PAWN and (to_pos[0] == 0 or to_pos[0] == 7):
            piece.type = PieceType.QUEEN  # Auto-promote to queen
            piece.symbol = piece._get_symbol()
        
        # Set en passant target
        self.en_passant_target = None
        if piece.type == PieceType.PAWN and abs(to_pos[0] - from_pos[0]) == 2:
            self.en_passant_target = ((from_pos[0] + to_pos[0]) // 2, from_pos[1])
        
        # Record the move
        move = Move(from_pos, to_pos, piece, captured_piece, is_castling, is_en_passant)
        self.move_history.append(move)
        
        # Switch players
        self.current_player = Color.BLACK if self.current_player == Color.WHITE else Color.WHITE
        
        return True
    
    def is_checkmate(self, color: Color) -> bool:
        if not self.is_in_check(color):
            return False
        
        # Check if any piece can make a legal move
        for row in range(8):
            for col in range(8):
                piece = self.get_piece(row, col)
                if piece and piece.color == color:
                    if self.get_valid_moves(piece):
                        return False
        return True
    
    def is_stalemate(self, color: Color) -> bool:
        if self.is_in_check(color):
            return False
        
        # Check if any piece can make a legal move
        for row in range(8):
            for col in range(8):
                piece = self.get_piece(row, col)
                if piece and piece.color == color:
                    if self.get_valid_moves(piece):
                        return False
        return True

class ChessAI:
    def __init__(self, depth: int = 3):
        self.depth = depth
        self.piece_values = {
            PieceType.PAWN: 100,
            PieceType.KNIGHT: 320,
            PieceType.BISHOP: 330,
            PieceType.ROOK: 500,
            PieceType.QUEEN: 900,
            PieceType.KING: 20000
        }
    
    def get_best_move(self, board: ChessBoard) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
        _, best_move = self.minimax(board, self.depth, float('-inf'), float('inf'), True)
        return best_move
    
    def minimax(self, board: ChessBoard, depth: int, alpha: float, beta: float, maximizing: bool) -> Tuple[float, Optional[Tuple[Tuple[int, int], Tuple[int, int]]]]:
        if depth == 0 or self.is_game_over(board):
            return self.evaluate_board(board), None
        
        best_move = None
        
        if maximizing:
            max_eval = float('-inf')
            for move in self.get_all_moves(board, Color.BLACK):
                # Make move
                original_state = self.save_board_state(board)
                board.make_move(move[0], move[1])
                
                eval_score, _ = self.minimax(board, depth - 1, alpha, beta, False)
                
                # Undo move
                self.restore_board_state(board, original_state)
                
                if eval_score > max_eval:
                    max_eval = eval_score
                    best_move = move
                
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break
            
            return max_eval, best_move
        else:
            min_eval = float('inf')
            for move in self.get_all_moves(board, Color.WHITE):
                # Make move
                original_state = self.save_board_state(board)
                board.make_move(move[0], move[1])
                
                eval_score, _ = self.minimax(board, depth - 1, alpha, beta, True)
                
                # Undo move
                self.restore_board_state(board, original_state)
                
                if eval_score < min_eval:
                    min_eval = eval_score
                    best_move = move
                
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break
            
            return min_eval, best_move
    
    def get_all_moves(self, board: ChessBoard, color: Color) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
        moves = []
        for row in range(8):
            for col in range(8):
                piece = board.get_piece(row, col)
                if piece and piece.color == color:
                    valid_moves = board.get_valid_moves(piece)
                    for move in valid_moves:
                        moves.append(((row, col), move))
        return moves
    
    def evaluate_board(self, board: ChessBoard) -> float:
        score = 0
        
        for row in range(8):
            for col in range(8):
                piece = board.get_piece(row, col)
                if piece:
                    piece_value = self.piece_values[piece.type]
                    if piece.color == Color.BLACK:
                        score += piece_value
                    else:
                        score -= piece_value
        
        return score
    
    def is_game_over(self, board: ChessBoard) -> bool:
        return (board.is_checkmate(Color.WHITE) or board.is_checkmate(Color.BLACK) or
                board.is_stalemate(Color.WHITE) or board.is_stalemate(Color.BLACK))
    
    def save_board_state(self, board: ChessBoard) -> Dict:
        # Save current board state for undo
        state = {
            'board': [[piece for piece in row] for row in board.board],
            'current_player': board.current_player,
            'en_passant_target': board.en_passant_target,
            'white_king_pos': board.white_king_pos,
            'black_king_pos': board.black_king_pos,
            'move_history_len': len(board.move_history)
        }
        return state
    
    def restore_board_state(self, board: ChessBoard, state: Dict):
        # Restore board state
        board.board = state['board']
        board.current_player = state['current_player']
        board.en_passant_target = state['en_passant_target']
        board.white_king_pos = state['white_king_pos']
        board.black_king_pos = state['black_king_pos']
        board.move_history = board.move_history[:state['move_history_len']]

class ChessGame:
    def __init__(self):
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Chess Game")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)
        self.large_font = pygame.font.Font(None, 48)
        self.small_font = pygame.font.Font(None, 24)
        
        self.board = ChessBoard()
        self.ai = ChessAI()
        self.game_mode = GameMode.MENU
        self.selected_piece = None
        self.selected_pos = None
        self.valid_moves = []
        self.game_over = False
        self.winner = None
        
        # UI elements
        self.buttons = {}
        self.create_buttons()
    
    def create_buttons(self):
        button_width = 200
        button_height = 50
        start_y = 200
        
        self.buttons = {
            'human_vs_human': pygame.Rect(WINDOW_WIDTH // 2 - button_width // 2, start_y, button_width, button_height),
            'human_vs_ai': pygame.Rect(WINDOW_WIDTH // 2 - button_width // 2, start_y + 70, button_width, button_height),
            'back_to_menu': pygame.Rect(BOARD_WIDTH + 20, 20, 120, 40),
            'new_game': pygame.Rect(BOARD_WIDTH + 20, 70, 120, 40)
        }
    
    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            
            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left click
                    self.handle_click(event.pos)
        
        return True
    
    def handle_click(self, pos):
        if self.game_mode == GameMode.MENU:
            self.handle_menu_click(pos)
        else:
            self.handle_game_click(pos)
    
    def handle_menu_click(self, pos):
        if self.buttons['human_vs_human'].collidepoint(pos):
            self.game_mode = GameMode.HUMAN_VS_HUMAN
            self.reset_game()
        elif self.buttons['human_vs_ai'].collidepoint(pos):
            self.game_mode = GameMode.HUMAN_VS_AI
            self.reset_game()
    
    def handle_game_click(self, pos):
        # Check UI buttons
        if self.buttons['back_to_menu'].collidepoint(pos):
            self.game_mode = GameMode.MENU
            return
        elif self.buttons['new_game'].collidepoint(pos):
            self.reset_game()
            return
        
        # Check board clicks
        if pos[0] < BOARD_WIDTH and pos[1] < BOARD_HEIGHT and not self.game_over:
            col = pos[0] // SQUARE_SIZE
            row = pos[1] // SQUARE_SIZE
            
            if self.selected_piece:
                # Try to move the selected piece
                if (row, col) in self.valid_moves:
                    if self.board.make_move(self.selected_pos, (row, col)):
                        self.selected_piece = None
                        self.selected_pos = None
                        self.valid_moves = []
                        self.check_game_over()
                        
                        # AI move if in AI mode
                        if (self.game_mode == GameMode.HUMAN_VS_AI and 
                            self.board.current_player == Color.BLACK and not self.game_over):
                            self.make_ai_move()
                else:
                    # Select new piece or deselect
                    piece = self.board.get_piece(row, col)
                    if piece and piece.color == self.board.current_player:
                        self.select_piece(piece, (row, col))
                    else:
                        self.selected_piece = None
                        self.selected_pos = None
                        self.valid_moves = []
            else:
                # Select a piece
                piece = self.board.get_piece(row, col)
                if piece and piece.color == self.board.current_player:
                    # In AI mode, only allow human to move white pieces
                    if (self.game_mode == GameMode.HUMAN_VS_AI and 
                        self.board.current_player == Color.BLACK):
                        return
                    self.select_piece(piece, (row, col))
    
    def select_piece(self, piece, pos):
        self.selected_piece = piece
        self.selected_pos = pos
        self.valid_moves = self.board.get_valid_moves(piece)
    
    def make_ai_move(self):
        best_move = self.ai.get_best_move(self.board)
        if best_move:
            self.board.make_move(best_move[0], best_move[1])
            self.check_game_over()
    
    def check_game_over(self):
        if self.board.is_checkmate(Color.WHITE):
            self.game_over = True
            self.winner = "Black"
        elif self.board.is_checkmate(Color.BLACK):
            self.game_over = True
            self.winner = "White"
        elif self.board.is_stalemate(Color.WHITE) or self.board.is_stalemate(Color.BLACK):
            self.game_over = True
            self.winner = "Draw"
    
    def reset_game(self):
        self.board = ChessBoard()
        self.selected_piece = None
        self.selected_pos = None
        self.valid_moves = []
        self.game_over = False
        self.winner = None
    
    def draw_menu(self):
        self.screen.fill(BG_COLOR)
        
        # Title
        title = self.large_font.render("Chess Game", True, TEXT_COLOR)
        title_rect = title.get_rect(center=(WINDOW_WIDTH // 2, 100))
        self.screen.blit(title, title_rect)
        
        # Buttons
        mouse_pos = pygame.mouse.get_pos()
        
        for button_name, button_rect in self.buttons.items():
            if button_name in ['human_vs_human', 'human_vs_ai']:
                color = BUTTON_HOVER if button_rect.collidepoint(mouse_pos) else BUTTON_COLOR
                pygame.draw.rect(self.screen, color, button_rect)
                pygame.draw.rect(self.screen, TEXT_COLOR, button_rect, 2)
                
                text = "Human vs Human" if button_name == 'human_vs_human' else "Human vs AI"
                text_surface = self.font.render(text, True, TEXT_COLOR)
                text_rect = text_surface.get_rect(center=button_rect.center)
                self.screen.blit(text_surface, text_rect)
    
    def draw_board(self):
        for row in range(8):
            for col in range(8):
                color = WHITE if (row + col) % 2 == 0 else BLACK
                rect = pygame.Rect(col * SQUARE_SIZE, row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE)
                pygame.draw.rect(self.screen, color, rect)
                
                # Highlight selected square
                if self.selected_pos == (row, col):
                    highlight_surface = pygame.Surface((SQUARE_SIZE, SQUARE_SIZE), pygame.SRCALPHA)
                    highlight_surface.fill(SELECTED)
                    self.screen.blit(highlight_surface, rect)
                
                # Highlight valid moves
                if (row, col) in self.valid_moves:
                    highlight_surface = pygame.Surface((SQUARE_SIZE, SQUARE_SIZE), pygame.SRCALPHA)
                    highlight_surface.fill(VALID_MOVE)
                    self.screen.blit(highlight_surface, rect)
                
                # Highlight check
                piece = self.board.get_piece(row, col)
                if (piece and piece.type == PieceType.KING and 
                    self.board.is_in_check(piece.color)):
                    highlight_surface = pygame.Surface((SQUARE_SIZE, SQUARE_SIZE), pygame.SRCALPHA)
                    highlight_surface.fill(CHECK)
                    self.screen.blit(highlight_surface, rect)
    
    def draw_pieces(self):
        for row in range(8):
            for col in range(8):
                piece = self.board.get_piece(row, col)
                if piece:
                    text = self.large_font.render(piece.symbol, True, (0, 0, 0))
                    text_rect = text.get_rect(center=(col * SQUARE_SIZE + SQUARE_SIZE // 2,
                                                    row * SQUARE_SIZE + SQUARE_SIZE // 2))
                    self.screen.blit(text, text_rect)
    
    def draw_sidebar(self):
        # Background
        sidebar_rect = pygame.Rect(BOARD_WIDTH, 0, SIDEBAR_WIDTH, WINDOW_HEIGHT)
        pygame.draw.rect(self.screen, BG_COLOR, sidebar_rect)
        
        # Buttons
        mouse_pos = pygame.mouse.get_pos()
        
        for button_name, button_rect in self.buttons.items():
            if button_name in ['back_to_menu', 'new_game']:
                color = BUTTON_HOVER if button_rect.collidepoint(mouse_pos) else BUTTON_COLOR
                pygame.draw.rect(self.screen, color, button_rect)
                pygame.draw.rect(self.screen, TEXT_COLOR, button_rect, 2)
                
                text = "Menu" if button_name == 'back_to_menu' else "New Game"
                text_surface = self.small_font.render(text, True, TEXT_COLOR)
                text_rect = text_surface.get_rect(center=button_rect.center)
                self.screen.blit(text_surface, text_rect)
        
        # Game info
        y_offset = 130
        
        # Current player
        current_player_text = f"Current: {'White' if self.board.current_player == Color.WHITE else 'Black'}"
        text_surface = self.font.render(current_player_text, True, TEXT_COLOR)
        self.screen.blit(text_surface, (BOARD_WIDTH + 20, y_offset))
        y_offset += 40
        
        # Game mode
        mode_text = "Human vs Human" if self.game_mode == GameMode.HUMAN_VS_HUMAN else "Human vs AI"
        text_surface = self.small_font.render(mode_text, True, TEXT_COLOR)
        self.screen.blit(text_surface, (BOARD_WIDTH + 20, y_offset))
        y_offset += 30
        
        # Check status
        if self.board.is_in_check(Color.WHITE):
            check_text = "White in Check!"
            text_surface = self.font.render(check_text, True, (255, 0, 0))
            self.screen.blit(text_surface, (BOARD_WIDTH + 20, y_offset))
            y_offset += 40
        elif self.board.is_in_check(Color.BLACK):
            check_text = "Black in Check!"
            text_surface = self.font.render(check_text, True, (255, 0, 0))
            self.screen.blit(text_surface, (BOARD_WIDTH + 20, y_offset))
            y_offset += 40
        
        # Game over
        if self.game_over:
            if self.winner == "Draw":
                game_over_text = "Stalemate!"
            else:
                game_over_text = f"{self.winner} Wins!"
            text_surface = self.large_font.render(game_over_text, True, (255, 255, 0))
            self.screen.blit(text_surface, (BOARD_WIDTH + 20, y_offset))
    
    def draw_game(self):
        self.screen.fill(BG_COLOR)
        self.draw_board()
        self.draw_pieces()
        self.draw_sidebar()
    
    def run(self):
        running = True
        while running:
            running = self.handle_events()
            
            if self.game_mode == GameMode.MENU:
                self.draw_menu()
            else:
                self.draw_game()
            
            pygame.display.flip()
            self.clock.tick(60)
        
        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    game = ChessGame()
    game.run()