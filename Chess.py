import pygame
import sys
import math
import random
import copy

# --- Constants ---
WIDTH, HEIGHT = 640, 640
DIMENSION = 8
SQUARE_SIZE = HEIGHT // DIMENSION
MAX_FPS = 30
# <<< INCREASED DEPTH (Adjust based on performance) >>>
AI_SEARCH_DEPTH = 4 # Start with 4, maybe 5 if performance allows after optimizations
QUIESCENCE_DEPTH = 2 # How many extra steps to look for captures/checks

# --- Colors ---
# (Keep existing colors)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
LIGHT_SQUARE = (238, 238, 210)
DARK_SQUARE = (118, 150, 86)
HIGHLIGHT_COLOR = (255, 255, 0, 150) # Yellow highlight with transparency
MOVE_HIGHLIGHT_COLOR = (135, 152, 106, 150) # Faded dark green for possible moves
BLUE = (0, 100, 200) # Color for White pieces
TEXT_BLACK = (20, 20, 20) # Color for Black pieces
GAME_OVER_BG = (50, 50, 50, 200) # Semi-transparent dark background
WINNER_TEXT_COLOR = (255, 215, 0) # Gold-like color for winner text
DRAW_TEXT_COLOR = (200, 200, 200) # Light gray for draw text

# --- Piece Representation (Internal) ---
INITIAL_BOARD = [
    ["bR", "bN", "bB", "bQ", "bK", "bB", "bN", "bR"],
    ["bP", "bP", "bP", "bP", "bP", "bP", "bP", "bP"],
    [".", ".", ".", ".", ".", ".", ".", "."],
    [".", ".", ".", ".", ".", ".", ".", "."],
    [".", ".", ".", ".", ".", ".", ".", "."],
    [".", ".", ".", ".", ".", ".", ".", "."],
    ["wP", "wP", "wP", "wP", "wP", "wP", "wP", "wP"],
    ["wR", "wN", "wB", "wQ", "wK", "wB", "wN", "wR"],
]

# --- Piece Representation (Visual) ---
PIECE_SYMBOLS = {
    "wP": "P", "wR": "R", "wN": "N", "wB": "B", "wQ": "Q", "wK": "K",
    "bP": "p", "bR": "r", "bN": "n", "bB": "b", "bQ": "q", "bK": "k",
}
PIECE_COLORS = { "w": BLUE, "b": TEXT_BLACK }

# --- Piece Values (for AI Evaluation) ---
PIECE_VALUES = { "P": 100, "N": 320, "B": 330, "R": 500, "Q": 900, "K": 20000 } # Centipawns

# --- Positional Evaluation Bonus - Piece Square Tables (PSTs) ---
# Values represent bonuses/penalties for placing a piece on a specific square.
# These are typically defined from White's perspective and flipped for Black.
# Source: Simplified from common chess engine tables (e.g., Chess Programming Wiki)

# Mirror tables vertically for black
def mirror_pst(table):
    return table[::-1]

pst_pawn = [
    [ 0,  0,  0,  0,  0,  0,  0,  0],
    [50, 50, 50, 50, 50, 50, 50, 50], # Pawns gain value advancing
    [10, 10, 20, 30, 30, 20, 10, 10],
    [ 5,  5, 10, 25, 25, 10,  5,  5],
    [ 0,  0,  0, 20, 20,  0,  0,  0],
    [ 5, -5,-10,  0,  0,-10, -5,  5],
    [ 5, 10, 10,-20,-20, 10, 10,  5],
    [ 0,  0,  0,  0,  0,  0,  0,  0]
]
pst_knight = [
    [-50,-40,-30,-30,-30,-30,-40,-50],
    [-40,-20,  0,  0,  0,  0,-20,-40],
    [-30,  0, 10, 15, 15, 10,  0,-30],
    [-30,  5, 15, 20, 20, 15,  5,-30],
    [-30,  0, 15, 20, 20, 15,  0,-30],
    [-30,  5, 10, 15, 15, 10,  5,-30],
    [-40,-20,  0,  5,  5,  0,-20,-40],
    [-50,-40,-30,-30,-30,-30,-40,-50]
]
pst_bishop = [
    [-20,-10,-10,-10,-10,-10,-10,-20],
    [-10,  0,  0,  0,  0,  0,  0,-10],
    [-10,  0,  5, 10, 10,  5,  0,-10],
    [-10,  5,  5, 10, 10,  5,  5,-10],
    [-10,  0, 10, 10, 10, 10,  0,-10],
    [-10, 10, 10, 10, 10, 10, 10,-10],
    [-10,  5,  0,  0,  0,  0,  5,-10],
    [-20,-10,-10,-10,-10,-10,-10,-20]
]
pst_rook = [
    [ 0,  0,  0,  0,  0,  0,  0,  0],
    [ 5, 10, 10, 10, 10, 10, 10,  5], # Rooks like open files/ranks
    [-5,  0,  0,  0,  0,  0,  0, -5],
    [-5,  0,  0,  0,  0,  0,  0, -5],
    [-5,  0,  0,  0,  0,  0,  0, -5],
    [-5,  0,  0,  0,  0,  0,  0, -5],
    [-5,  0,  0,  0,  0,  0,  0, -5],
    [ 0,  0,  0,  5,  5,  0,  0,  0]
]
pst_queen = [
    [-20,-10,-10, -5, -5,-10,-10,-20],
    [-10,  0,  0,  0,  0,  0,  0,-10],
    [-10,  0,  5,  5,  5,  5,  0,-10],
    [ -5,  0,  5,  5,  5,  5,  0, -5],
    [  0,  0,  5,  5,  5,  5,  0, -5],
    [-10,  5,  5,  5,  5,  5,  0,-10],
    [-10,  0,  5,  0,  0,  0,  0,-10],
    [-20,-10,-10, -5, -5,-10,-10,-20]
]
# King safety is crucial - different tables for midgame vs endgame
pst_king_midgame = [
    [-30,-40,-40,-50,-50,-40,-40,-30], # Penalize king on back rank early
    [-30,-40,-40,-50,-50,-40,-40,-30],
    [-30,-40,-40,-50,-50,-40,-40,-30],
    [-30,-40,-40,-50,-50,-40,-40,-30],
    [-20,-30,-30,-40,-40,-30,-30,-20],
    [-10,-20,-20,-20,-20,-20,-20,-10],
    [ 20, 20,  0,  0,  0,  0, 20, 20], # Encourage castling short
    [ 20, 30, 10,  0,  0, 10, 30, 20]
]
pst_king_endgame = [
    [-50,-40,-30,-20,-20,-30,-40,-50], # Encourage king activity in endgame
    [-30,-20,-10,  0,  0,-10,-20,-30],
    [-30,-10, 20, 30, 30, 20,-10,-30],
    [-30,-10, 30, 40, 40, 30,-10,-30],
    [-30,-10, 30, 40, 40, 30,-10,-30],
    [-30,-10, 20, 30, 30, 20,-10,-30],
    [-30,-30,  0,  0,  0,  0,-30,-30],
    [-50,-30,-30,-30,-30,-30,-30,-50]
]

# Create black's tables by flipping white's
pst_pawn_b = mirror_pst(pst_pawn)
pst_knight_b = mirror_pst(pst_knight)
pst_bishop_b = mirror_pst(pst_bishop)
pst_rook_b = mirror_pst(pst_rook)
pst_queen_b = mirror_pst(pst_queen)
pst_king_midgame_b = mirror_pst(pst_king_midgame)
pst_king_endgame_b = mirror_pst(pst_king_endgame)

# Store all PSTs in a dictionary for easy access
PIECE_SQUARE_TABLES = {
    'wP': pst_pawn, 'wN': pst_knight, 'wB': pst_bishop, 'wR': pst_rook, 'wQ': pst_queen, 'wK': pst_king_midgame,
    'bP': pst_pawn_b, 'bN': pst_knight_b, 'bB': pst_bishop_b, 'bR': pst_rook_b, 'bQ': pst_queen_b, 'bK': pst_king_midgame_b,
    'wK_end': pst_king_endgame, 'bK_end': pst_king_endgame_b # Special tables for endgame king
}

CHECKMATE_SCORE = 100000 # A very large score for checkmate
STALEMATE_SCORE = 0      # Score for stalemate

# --- Game State Class ---
# (Keep existing GameState class, make_move, undo_move, _update_castling_rights)
class GameState:
    """Holds all current information about the game state."""
    def __init__(self):
        self.board = copy.deepcopy(INITIAL_BOARD)
        self.current_turn = 'w'
        self.game_over = False
        self.winner = None # 'w', 'b', or 'draw'
        self.selected_square = None
        self.selected_piece = None
        self.valid_moves_for_selected_piece = [] # Store moves for highlighting

        self.castling_rights = {'w_kingside': True, 'w_queenside': True,
                                'b_kingside': True, 'b_queenside': True}
        self.castling_rights_log = [copy.deepcopy(self.castling_rights)]

        self.en_passant_target = None # Stores as (row, col)
        self.en_passant_log = [self.en_passant_target]

        self.move_log = []

        # <<< Add piece counts for endgame detection >>>
        self.piece_counts = self._count_pieces()


    def make_move(self, move):
        """Applies a move to the board and updates game state."""
        start_pos, end_pos = move.start_sq, move.end_sq
        piece_moved = self.board[start_pos[0]][start_pos[1]]
        piece_captured = self.board[end_pos[0]][end_pos[1]] # Could be "."

        # Update board
        self.board[end_pos[0]][end_pos[1]] = piece_moved
        self.board[start_pos[0]][start_pos[1]] = "."
        self.move_log.append(move)

        # Update piece counts BEFORE handling special moves affecting count
        if piece_captured != ".":
            self.piece_counts[piece_captured] -= 1
        if move.is_pawn_promotion:
             self.piece_counts[piece_moved] -= 1 # Remove pawn
             promotion_piece = piece_moved[0] + 'Q' # Assuming Queen promotion for simplicity
             self.piece_counts[promotion_piece] = self.piece_counts.get(promotion_piece, 0) + 1 # Add Queen
             self.board[end_pos[0]][end_pos[1]] = promotion_piece
             move.promotion_piece_type = 'Q'
        elif move.is_en_passant:
            # Capture happens on target square, but pawn is removed elsewhere
            captured_pawn_piece = ('b' if piece_moved[0] == 'w' else 'w') + 'P'
            self.board[start_pos[0]][end_pos[1]] = "." # Remove captured pawn
            self.piece_counts[captured_pawn_piece] -= 1
            piece_captured = captured_pawn_piece # Ensure correct capture logging in Move obj

        # Update en passant target
        if piece_moved[1] == 'P' and abs(start_pos[0] - end_pos[0]) == 2:
            self.en_passant_target = ( (start_pos[0] + end_pos[0]) // 2, start_pos[1] )
        else:
            self.en_passant_target = None
        self.en_passant_log.append(self.en_passant_target)

        # Castling move
        if move.is_castle:
            if end_pos[1] - start_pos[1] == 2: # Kingside
                rook_start, rook_end = (start_pos[0], 7), (start_pos[0], 5)
            else: # Queenside
                rook_start, rook_end = (start_pos[0], 0), (start_pos[0], 3)
            rook = self.board[rook_start[0]][rook_start[1]]
            self.board[rook_start[0]][rook_start[1]] = "."
            self.board[rook_end[0]][rook_end[1]] = rook

        # Update castling rights
        self._update_castling_rights(move, piece_moved, piece_captured)
        self.castling_rights_log.append(copy.deepcopy(self.castling_rights))

        # Switch turn
        self.current_turn = 'b' if self.current_turn == 'w' else 'w'

        # Clear selection
        self.selected_square = None
        self.selected_piece = None
        self.valid_moves_for_selected_piece = []

    def undo_move(self):
        """Reverts the last move made."""
        if not self.move_log: return

        last_move = self.move_log.pop()
        start_pos, end_pos = last_move.start_sq, last_move.end_sq
        piece_moved_back = self.board[end_pos[0]][end_pos[1]] # Could be promoted piece
        piece_captured = last_move.piece_captured # Correct captured piece is stored

        # Revert board state
        if last_move.is_pawn_promotion:
            piece_moved_back = last_move.piece_moved # Get original pawn type ('wP' or 'bP')
            promoted_piece = self.board[end_pos[0]][end_pos[1]] # The Queen (usually)
            self.piece_counts[promoted_piece] -= 1 # Remove promoted piece count
            self.piece_counts[piece_moved_back] += 1 # Add pawn count back
        elif piece_captured != ".":
             self.piece_counts[piece_captured] += 1 # Add captured piece count back

        self.board[start_pos[0]][start_pos[1]] = piece_moved_back
        self.board[end_pos[0]][end_pos[1]] = piece_captured if not last_move.is_en_passant else "." # Put captured piece back (or '.')

        # Reverse en passant capture
        if last_move.is_en_passant:
            captured_pawn_pos = (start_pos[0], end_pos[1])
            self.board[captured_pawn_pos[0]][captured_pawn_pos[1]] = piece_captured # Put pawn back
            # No need to update piece_counts here, was handled above with piece_captured

        # Revert en passant target log
        self.en_passant_log.pop()
        self.en_passant_target = self.en_passant_log[-1] if self.en_passant_log else None

        # Reverse castling
        if last_move.is_castle:
            if end_pos[1] - start_pos[1] == 2: # Kingside
                rook_start, rook_end = (start_pos[0], 7), (start_pos[0], 5)
            else: # Queenside
                rook_start, rook_end = (start_pos[0], 0), (start_pos[0], 3)
            rook = self.board[rook_end[0]][rook_end[1]]
            self.board[rook_end[0]][rook_end[1]] = "."
            self.board[rook_start[0]][rook_start[1]] = rook

        # Revert castling rights log
        self.castling_rights_log.pop()
        self.castling_rights = copy.deepcopy(self.castling_rights_log[-1]) if self.castling_rights_log else {}

        # Switch turn back
        self.current_turn = 'b' if self.current_turn == 'w' else 'w'

        # Reset game over state potentially
        self.game_over = False
        self.winner = None
        # Reset selection
        self.selected_square = None
        self.selected_piece = None
        self.valid_moves_for_selected_piece = []

    def _update_castling_rights(self, move, piece_moved, piece_captured):
        """Updates castling rights based on the move made."""
        # (Keep existing logic)
        start_pos, end_pos = move.start_sq, move.end_sq
        color = piece_moved[0]

        if piece_moved[1] == 'K':
            self.castling_rights[color + '_kingside'] = False
            self.castling_rights[color + '_queenside'] = False
        elif piece_moved[1] == 'R':
            if color == 'w':
                if start_pos == (7, 0): self.castling_rights['w_queenside'] = False
                elif start_pos == (7, 7): self.castling_rights['w_kingside'] = False
            else:
                if start_pos == (0, 0): self.castling_rights['b_queenside'] = False
                elif start_pos == (0, 7): self.castling_rights['b_kingside'] = False

        opponent_color = 'b' if color == 'w' else 'w'
        if piece_captured != "." and piece_captured[1] == 'R':
           # Check if captured rook was on its starting square
           if opponent_color == 'w':
               if end_pos == (7, 0): self.castling_rights['w_queenside'] = False
               elif end_pos == (7, 7): self.castling_rights['w_kingside'] = False
           else:
               if end_pos == (0, 0): self.castling_rights['b_queenside'] = False
               elif end_pos == (0, 7): self.castling_rights['b_kingside'] = False

    def _count_pieces(self):
        """Counts all pieces on the board, useful for endgame detection."""
        counts = {}
        for r in range(DIMENSION):
            for c in range(DIMENSION):
                piece = self.board[r][c]
                if piece != ".":
                    counts[piece] = counts.get(piece, 0) + 1
        return counts

    # <<< NEW: Endgame Phase Detection >>>
    def is_endgame(self):
        """Determines if the game is likely in the endgame phase."""
        # Simple heuristic: fewer than X points of material (excluding kings)
        # Or if queens are off the board. Adjust threshold as needed.
        material_count = 0
        queens_on_board = self.piece_counts.get('wQ', 0) + self.piece_counts.get('bQ', 0) > 0

        for piece, count in self.piece_counts.items():
            if piece[1] != 'K': # Exclude kings
                 material_count += PIECE_VALUES.get(piece[1], 0) * count

        # Example thresholds (can be tuned)
        if not queens_on_board or material_count < 2 * (PIECE_VALUES['R'] + PIECE_VALUES['B'] + PIECE_VALUES['N']):
             return True
        return False


# --- Move Class ---
# (Keep existing Move class, slightly adjusted __init__ for clarity)
class Move:
    """Represents a single move in the game."""
    def __init__(self, start_sq, end_sq, board, is_en_passant=False, is_castle=False):
        self.start_sq = start_sq
        self.end_sq = end_sq
        self.piece_moved = board[start_sq[0]][start_sq[1]]
        self.piece_captured = board[end_sq[0]][end_sq[1]] # Store piece captured AT end square initially

        # Pawn promotion
        self.is_pawn_promotion = (self.piece_moved != "." and self.piece_moved[1] == 'P' and
                                 (end_sq[0] == 0 or end_sq[0] == DIMENSION - 1))
        self.promotion_piece_type = None # Set during make_move if promotion

        # En passant
        self.is_en_passant = is_en_passant
        if self.is_en_passant:
             # The actual pawn captured is adjacent horizontally to the landing square
             # Determine color of captured pawn based on mover's color
             captured_pawn_color = 'b' if self.piece_moved[0] == 'w' else 'w'
             self.piece_captured = captured_pawn_color + 'P' # Correctly identify the captured piece type

        # Castling
        self.is_castle = is_castle

        # <<< Add move score for ordering >>>
        self.move_score = 0 # Used for sorting moves in search

        self.move_id = (self.start_sq[0] * 100000 + self.start_sq[1] * 10000 +
                        self.end_sq[0] * 1000 + self.end_sq[1] * 100 +
                        (ord(self.promotion_piece_type) if self.promotion_piece_type else 0)
                       ) # More robust ID including promotion

    def __eq__(self, other):
        if isinstance(other, Move):
            # Only need start/end squares for basic equality check in player logic
            # Search might need more specific comparisons if using transposition tables
            return self.start_sq == other.start_sq and self.end_sq == other.end_sq \
                   and self.promotion_piece_type == other.promotion_piece_type # Need promotion match too
        return False

    def get_notation(self):
        # (Keep existing notation methods)
        return self.get_rank_file(self.start_sq) + self.get_rank_file(self.end_sq)

    def get_rank_file(self, pos):
        r, c = pos
        return chr(ord('a') + c) + str(DIMENSION - r)


# --- Helper Functions ---
# (Keep existing draw_board, draw_pieces, draw_highlights, get_square_from_pos, get_piece_color, find_king)
def draw_board(screen):
    """Draws the checkered board squares."""
    for r in range(DIMENSION):
        for c in range(DIMENSION):
            color = LIGHT_SQUARE if (r + c) % 2 == 0 else DARK_SQUARE
            pygame.draw.rect(screen, color, pygame.Rect(c * SQUARE_SIZE, r * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))

def draw_pieces(screen, board, font):
    """Draws the pieces on the board using text symbols."""
    for r in range(DIMENSION):
        for c in range(DIMENSION):
            piece = board[r][c]
            if piece != ".":
                symbol = PIECE_SYMBOLS.get(piece, '?') # Use get for safety
                color = PIECE_COLORS[piece[0]]
                text_surface = font.render(symbol, True, color)
                text_rect = text_surface.get_rect(center=(c * SQUARE_SIZE + SQUARE_SIZE // 2, r * SQUARE_SIZE + SQUARE_SIZE // 2))
                screen.blit(text_surface, text_rect)

def draw_highlights(screen, gs):
    """Draws highlights for selected square and possible moves."""
    if gs.selected_square:
        r, c = gs.selected_square
        highlight_surface = pygame.Surface((SQUARE_SIZE, SQUARE_SIZE), pygame.SRCALPHA)
        highlight_surface.fill(HIGHLIGHT_COLOR)
        screen.blit(highlight_surface, (c * SQUARE_SIZE, r * SQUARE_SIZE))

        move_highlight_surface = pygame.Surface((SQUARE_SIZE, SQUARE_SIZE), pygame.SRCALPHA)
        move_highlight_surface.fill(MOVE_HIGHLIGHT_COLOR)
        for move in gs.valid_moves_for_selected_piece:
            end_r, end_c = move.end_sq
            # Draw circles for potential moves
            center_x = end_c * SQUARE_SIZE + SQUARE_SIZE // 2
            center_y = end_r * SQUARE_SIZE + SQUARE_SIZE // 2
            radius = SQUARE_SIZE // 6
            if gs.board[end_r][end_c] != ".": # Capture indication
                 pygame.draw.circle(screen, MOVE_HIGHLIGHT_COLOR, (center_x, center_y), SQUARE_SIZE // 2, width=4)
            else: # Move indication
                 pygame.draw.circle(screen, MOVE_HIGHLIGHT_COLOR, (center_x, center_y), radius)


def get_square_from_pos(pos):
    """Converts pixel coordinates (x, y) to board coordinates (row, col)."""
    x, y = pos
    if 0 <= x < WIDTH and 0 <= y < HEIGHT:
        col = x // SQUARE_SIZE
        row = y // SQUARE_SIZE
        return row, col
    return None

def get_piece_color(piece):
    """Returns the color ('w' or 'b') of a piece string, or None."""
    if piece != ".":
        return piece[0]
    return None

def find_king(board, color):
    """Finds the position (row, col) of the king of the given color."""
    king_piece = color + 'K'
    for r in range(DIMENSION):
        for c in range(DIMENSION):
            if board[r][c] == king_piece:
                return (r, c)
    return None

# --- Check and Move Generation Logic ---
# (Keep existing is_square_attacked, is_in_check)
# (Keep existing get_pawn/knight/bishop/rook/queen/king/castle moves)
# (Modify get_all_legal_moves to use make_move/undo_move for perfect check validation)

def is_square_attacked(board, square_pos, attacker_color):
    """Checks if a given square is attacked by any piece of the attacker_color."""
    # (Keep existing logic, seems correct)
    r_target, c_target = square_pos
    opponent_color = 'w' if attacker_color == 'b' else 'b' # The color being attacked

    directions = ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1))
    knight_moves = ((-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1))
    pawn_attack_dirs = ((1, -1), (1, 1)) if attacker_color == 'w' else ((-1, -1), (-1, 1)) # White attacks rows--, Black attacks rows++

    # Sliding pieces (R, B, Q)
    for dr, dc in directions:
        for i in range(1, DIMENSION):
            r_check, c_check = r_target + dr * i, c_target + dc * i
            if 0 <= r_check < DIMENSION and 0 <= c_check < DIMENSION:
                piece = board[r_check][c_check]
                if piece != ".":
                    if piece[0] == attacker_color:
                        piece_type = piece[1]
                        is_straight = (dr == 0 or dc == 0)
                        is_diagonal = (dr != 0 and dc != 0)
                        if (piece_type == 'Q' or
                           (piece_type == 'R' and is_straight) or
                           (piece_type == 'B' and is_diagonal)):
                            return True
                    break # Hit a piece, stop searching this line
            else:
                break # Off board

    # Knights
    for dr, dc in knight_moves:
        r_check, c_check = r_target + dr, c_target + dc
        if 0 <= r_check < DIMENSION and 0 <= c_check < DIMENSION:
            if board[r_check][c_check] == attacker_color + 'N':
                return True

    # Pawns
    for dr, dc in pawn_attack_dirs:
        r_check, c_check = r_target + dr, c_target + dc
        if 0 <= r_check < DIMENSION and 0 <= c_check < DIMENSION:
            if board[r_check][c_check] == attacker_color + 'P':
                return True

    # Kings
    for dr in [-1, 0, 1]:
        for dc in [-1, 0, 1]:
            if dr == 0 and dc == 0: continue
            r_check, c_check = r_target + dr, c_target + dc
            if 0 <= r_check < DIMENSION and 0 <= c_check < DIMENSION:
                if board[r_check][c_check] == attacker_color + 'K':
                    return True
    return False

def is_in_check(board, color):
    """Checks if the king of the specified color is currently in check."""
    king_pos = find_king(board, color)
    if not king_pos: return False # Should not happen
    opponent_color = 'w' if color == 'b' else 'b'
    return is_square_attacked(board, king_pos, opponent_color)

def get_all_pseudo_legal_moves(gs):
    """Generates all moves for the current player IGNORING check safety."""
    # This generates moves faster, check safety is handled later.
    moves = []
    color = gs.current_turn
    for r_start in range(DIMENSION):
        for c_start in range(DIMENSION):
            piece = gs.board[r_start][c_start]
            if piece != "." and piece[0] == color:
                piece_type = piece[1]
                if piece_type == 'P': get_pawn_moves(gs, r_start, c_start, moves)
                elif piece_type == 'N': get_knight_moves(gs, r_start, c_start, moves)
                elif piece_type == 'B': get_bishop_moves(gs, r_start, c_start, moves)
                elif piece_type == 'R': get_rook_moves(gs, r_start, c_start, moves)
                elif piece_type == 'Q': get_queen_moves(gs, r_start, c_start, moves)
                elif piece_type == 'K': get_king_moves(gs, r_start, c_start, moves) # Generate king moves first
    # Add castling moves separately if conditions met (pre-check)
    king_pos = find_king(gs.board, color)
    if king_pos and not is_in_check(gs.board, color): # Can't castle out of check
        get_castle_moves(gs, king_pos, moves)
    return moves


def get_all_legal_moves(gs):
    """
    Generates ALL valid moves for the current player by checking king safety.
    Uses make_move/undo_move for 100% accuracy.
    """
    legal_moves = []
    current_turn_color = gs.current_turn
    pseudo_legal_moves = get_all_pseudo_legal_moves(gs)

    for move in pseudo_legal_moves:
        gs.make_move(move) # Simulate the move
        # Need to check the king of the player *who just moved*
        if not is_in_check(gs.board, current_turn_color):
            legal_moves.append(move)
        gs.undo_move() # Revert the board state

    return legal_moves

# --- Piece Specific Move Generation (Pseudo-Legal) ---
# (Keep existing functions: get_pawn_moves, get_knight_moves, _get_sliding_moves,
#  get_rook_moves, get_bishop_moves, get_queen_moves, get_king_moves, get_castle_moves)
# Make sure they add Move objects to the passed 'moves' list.

def get_pawn_moves(gs, r, c, moves):
    board = gs.board
    color = gs.current_turn
    opponent_color = 'w' if color == 'b' else 'b'
    direction = -1 if color == 'w' else 1
    start_row = 6 if color == 'w' else 1

    # 1. Forward Move
    one_step = r + direction
    if 0 <= one_step < DIMENSION and board[one_step][c] == ".":
        # Check for promotion
        if one_step == 0 or one_step == DIMENSION - 1:
            # In this simplified version, auto-promote to Queen for move generation
            # A real UI would ask, or AI might evaluate different promotions
            moves.append(Move((r, c), (one_step, c), board)) # Add promotion move later if needed
        else:
            moves.append(Move((r, c), (one_step, c), board))

        # 2. Double Forward Move
        if r == start_row:
            two_steps = r + 2 * direction
            if 0 <= two_steps < DIMENSION and board[two_steps][c] == ".":
                moves.append(Move((r, c), (two_steps, c), board))

    # 3. Captures
    for dc in [-1, 1]:
        capture_c = c + dc
        if 0 <= one_step < DIMENSION and 0 <= capture_c < DIMENSION:
            target_piece = board[one_step][capture_c]
            if target_piece != "." and target_piece[0] == opponent_color:
                 if one_step == 0 or one_step == DIMENSION - 1: # Capture with promotion
                    moves.append(Move((r, c), (one_step, capture_c), board)) # Add promotion move
                 else:
                    moves.append(Move((r, c), (one_step, capture_c), board))

    # 4. En Passant
    if gs.en_passant_target:
        target_r, target_c = gs.en_passant_target
        # Pawn must be on correct rank to perform en passant
        correct_rank = 3 if color == 'w' else 4
        if r == correct_rank and one_step == target_r and abs(c - target_c) == 1:
             moves.append(Move((r, c), gs.en_passant_target, board, is_en_passant=True))


def get_knight_moves(gs, r, c, moves):
    board = gs.board
    color = gs.current_turn
    knight_moves_deltas = ((-2, -1), (-2, 1), (-1, -2), (-1, 2),
                           (1, -2), (1, 2), (2, -1), (2, 1))
    for dr, dc in knight_moves_deltas:
        end_r, end_c = r + dr, c + dc
        if 0 <= end_r < DIMENSION and 0 <= end_c < DIMENSION:
            target_piece = board[end_r][end_c]
            if target_piece == "." or target_piece[0] != color:
                moves.append(Move((r, c), (end_r, end_c), board))

def _get_sliding_moves(gs, r, c, directions, moves):
    board = gs.board
    color = gs.current_turn
    opponent_color = 'w' if color == 'b' else 'b'
    for dr, dc in directions:
        for i in range(1, DIMENSION):
            end_r, end_c = r + dr * i, c + dc * i
            if 0 <= end_r < DIMENSION and 0 <= end_c < DIMENSION:
                target_piece = board[end_r][end_c]
                if target_piece == ".":
                    moves.append(Move((r, c), (end_r, end_c), board))
                elif target_piece[0] == opponent_color:
                    moves.append(Move((r, c), (end_r, end_c), board))
                    break # Can capture, but can't move further
                else: # Friendly piece
                    break # Blocked
            else: # Off board
                break

def get_rook_moves(gs, r, c, moves):
    directions = ((-1, 0), (1, 0), (0, -1), (0, 1))
    _get_sliding_moves(gs, r, c, directions, moves)

def get_bishop_moves(gs, r, c, moves):
    directions = ((-1, -1), (-1, 1), (1, -1), (1, 1))
    _get_sliding_moves(gs, r, c, directions, moves)

def get_queen_moves(gs, r, c, moves):
    directions = ((-1, 0), (1, 0), (0, -1), (0, 1),
                  (-1, -1), (-1, 1), (1, -1), (1, 1))
    _get_sliding_moves(gs, r, c, directions, moves)

def get_king_moves(gs, r, c, moves):
    board = gs.board
    color = gs.current_turn
    king_move_deltas = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))
    for dr, dc in king_move_deltas:
        end_r, end_c = r + dr, c + dc
        if 0 <= end_r < DIMENSION and 0 <= end_c < DIMENSION:
            target_piece = board[end_r][end_c]
            # Basic check - will be validated for safety later
            if target_piece == "." or target_piece[0] != color:
                 moves.append(Move((r, c), (end_r, end_c), board))
    # Castling generation moved to get_all_pseudo_legal_moves

def get_castle_moves(gs, king_pos, moves):
    """Generates castle moves if conditions are met (ignores check safety here)."""
    # (Keep existing logic, relies on is_square_attacked)
    board = gs.board
    color = gs.current_turn
    opponent_color = 'w' if color == 'b' else 'b'
    r = king_pos[0]

    # Kingside
    if gs.castling_rights[color + '_kingside']:
        if board[r][5] == "." and board[r][6] == ".":
            if not is_square_attacked(board, (r, 4), opponent_color) and \
               not is_square_attacked(board, (r, 5), opponent_color) and \
               not is_square_attacked(board, (r, 6), opponent_color):
                moves.append(Move(king_pos, (r, 6), board, is_castle=True))

    # Queenside
    if gs.castling_rights[color + '_queenside']:
        if board[r][1] == "." and board[r][2] == "." and board[r][3] == ".":
            if not is_square_attacked(board, (r, 4), opponent_color) and \
               not is_square_attacked(board, (r, 3), opponent_color) and \
               not is_square_attacked(board, (r, 2), opponent_color):
                moves.append(Move(king_pos, (r, 2), board, is_castle=True))


# --- AI Logic ---

# <<< IMPROVED Evaluation Function >>>
def evaluate_board(gs):
    """
    More sophisticated evaluation function using material and piece-square tables.
    Positive score is good for white, negative is good for black.
    """
    if gs.game_over:
        if gs.winner == 'w': return CHECKMATE_SCORE
        elif gs.winner == 'b': return -CHECKMATE_SCORE
        else: return STALEMATE_SCORE # Draw

    score = 0
    is_endgame_phase = gs.is_endgame() # Determine phase for king PST

    for r in range(DIMENSION):
        for c in range(DIMENSION):
            piece = gs.board[r][c]
            if piece != ".":
                piece_type = piece[1]
                piece_color = piece[0]
                material_value = PIECE_VALUES.get(piece_type, 0)
                positional_value = 0

                # Get appropriate PST based on piece and game phase (for king)
                pst_key = piece
                if piece_type == 'K':
                    pst_key += '_end' if is_endgame_phase else ''

                pst = PIECE_SQUARE_TABLES.get(pst_key)
                if pst:
                    positional_value = pst[r][c]

                if piece_color == 'w':
                    score += material_value + positional_value
                else: # Black piece
                    score -= (material_value + positional_value)

    # <<< Add Bonus for current check? (Optional, can sometimes cause issues) >>>
    # if is_in_check(gs.board, 'b'): # White is checking black
    #     score += 15 # Small bonus for check
    # elif is_in_check(gs.board, 'w'): # Black is checking white
    #     score -= 15

    return score

# <<< NEW: Move Scoring for Ordering >>>
def score_move(move):
    """Assigns a score to a move for sorting, prioritizing captures and promotions."""
    score = 0
    # Prioritize capturing higher value pieces with lower value pieces (MVV-LVA)
    if move.piece_captured != ".":
        attacker_value = PIECE_VALUES.get(move.piece_moved[1], 0)
        victim_value = PIECE_VALUES.get(move.piece_captured[1], 0)
        # Score is roughly: value of captured piece - value of attacking piece + base capture bonus
        score += 10 * victim_value - attacker_value + 1000 # Add base value for any capture

    # Promotions are valuable
    if move.is_pawn_promotion:
        # Assume Queen promotion has highest value
        score += PIECE_VALUES['Q'] # Add value of promoted piece

    # Checks can be good (could add a small bonus here, or rely on quiescence)
    # move.is_check = ... (Would need to implement check detection *for the move*)

    # Killer moves / History heuristic could be added here later

    return score


# <<< NEW: Quiescence Search >>>
def quiescence_search(gs, alpha, beta, depth):
    """
    Searches only 'noisy' moves (captures, promotions, maybe checks)
    to stabilize the evaluation at the end of the main search.
    """
    stand_pat_score = evaluate_board(gs) # Score of the current position without making a move

    if depth == 0:
         return stand_pat_score # Reached quiescence depth limit

    # Alpha-beta pruning based on the stand-pat score
    if gs.current_turn == 'w': # Maximizing player
        if stand_pat_score >= beta:
            return beta # Fail-high cutoff
        alpha = max(alpha, stand_pat_score)
    else: # Minimizing player
        if stand_pat_score <= alpha:
            return alpha # Fail-low cutoff
        beta = min(beta, stand_pat_score)

    # Generate only 'noisy' moves (captures, promotions for now)
    moves = get_all_legal_moves(gs) # Need legal moves to avoid illegal states
    noisy_moves = [m for m in moves if m.piece_captured != "." or m.is_pawn_promotion]
    # Add check-giving moves? Can be expensive to check here.

    # Order noisy moves for better pruning within quiescence
    noisy_moves.sort(key=score_move, reverse=True)

    for move in noisy_moves:
        gs.make_move(move)
        score = quiescence_search(gs, alpha, beta, depth - 1) # Recursive call, decrement q-depth
        gs.undo_move()

        if gs.current_turn == 'w': # Maximizing (was black's move)
            alpha = max(alpha, score)
            if alpha >= beta:
                return beta # Beta cutoff
        else: # Minimizing (was white's move)
            beta = min(beta, score)
            if beta <= alpha:
                return alpha # Alpha cutoff

    # Return the best score found (either stand-pat or from a noisy move)
    return alpha if gs.current_turn == 'w' else beta


# <<< MODIFIED Minimax with Move Ordering and Quiescence >>>
def minimax(gs, depth, alpha, beta, is_maximizing_player):
    """
    Minimax algorithm with alpha-beta pruning, move ordering, and quiescence search.
    """
    if depth == 0:
        # <<< Call Quiescence Search at leaf nodes >>>
        return quiescence_search(gs, alpha, beta, QUIESCENCE_DEPTH)

    legal_moves = get_all_legal_moves(gs)

    if not legal_moves:
        if is_in_check(gs.board, gs.current_turn):
             # Add depth penalty to checkmate score - prefer faster mates
            mate_score = CHECKMATE_SCORE if is_maximizing_player else -CHECKMATE_SCORE
            return mate_score - depth if is_maximizing_player else mate_score + depth
            # return -CHECKMATE_SCORE if is_maximizing_player else CHECKMATE_SCORE
        else:
            return STALEMATE_SCORE

    # <<< Move Ordering >>>
    # Score moves based on heuristics (captures, promotions)
    for move in legal_moves:
         move.move_score = score_move(move)
    # Sort moves: best score first (higher score is better)
    legal_moves.sort(key=lambda m: m.move_score, reverse=True)


    if is_maximizing_player: # White's turn (AI or simulated)
        max_eval = -math.inf
        for move in legal_moves:
            gs.make_move(move)
            eval_score = minimax(gs, depth - 1, alpha, beta, False) # Opponent's turn
            gs.undo_move()
            max_eval = max(max_eval, eval_score)
            alpha = max(alpha, eval_score)
            if beta <= alpha:
                # <<< Killer Move Heuristic could store 'move' here >>>
                break # Prune
        return max_eval
    else: # Black's turn (AI or simulated)
        min_eval = math.inf
        for move in legal_moves:
            gs.make_move(move)
            eval_score = minimax(gs, depth - 1, alpha, beta, True) # Opponent's turn
            gs.undo_move()
            min_eval = min(min_eval, eval_score)
            beta = min(beta, eval_score)
            if beta <= alpha:
                 # <<< Killer Move Heuristic could store 'move' here >>>
                break # Prune
        return min_eval

# <<< MODIFIED Find Best Move to use new minimax >>>
def find_best_move_ai(gs, depth):
    """
    Finds the best move for the AI using the enhanced Minimax search.
    Returns the best Move object found.
    """
    legal_moves = get_all_legal_moves(gs)
    if not legal_moves: return None

    # <<< Move Ordering at Root >>>
    for move in legal_moves:
         move.move_score = score_move(move)
    legal_moves.sort(key=lambda m: m.move_score, reverse=True)

    best_move = None
    best_score = -math.inf if gs.current_turn == 'w' else math.inf
    alpha = -math.inf
    beta = math.inf

    print(f"AI ({gs.current_turn}) evaluating {len(legal_moves)} moves at depth {depth}...")

    for i, move in enumerate(legal_moves):
        gs.make_move(move)
        # Call minimax for the opponent's perspective
        score = minimax(gs, depth - 1, alpha, beta, gs.current_turn == 'b') # True if next is white (max), False if next is black (min)
        gs.undo_move()

        # Debug print for move scores
        # print(f"  Move {i+1}/{len(legal_moves)}: {move.get_notation()} -> Score: {score}")

        if gs.current_turn == 'w': # White (Maximizer) is making the move
            if score > best_score:
                best_score = score
                best_move = move
            alpha = max(alpha, score)
            # Add randomness for equally scored top moves? Optional.
            # elif score == best_score and random.choice([True, False]):
            #      best_move = move
        else: # Black (Minimizer) is making the move
            if score < best_score:
                best_score = score
                best_move = move
            beta = min(beta, score)
            # Add randomness?
            # elif score == best_score and random.choice([True, False]):
            #      best_move = move

        # Early exit if checkmate found at root (optional but good)
        # if abs(best_score) >= CHECKMATE_SCORE - depth: # Found a forced mate
        #      print("Checkmate found at root!")
        #      break

    # Fallback if no best move assigned (shouldn't happen if legal_moves exist)
    if best_move is None and legal_moves:
        print("AI Warning: No best move found despite legal moves. Choosing randomly.")
        best_move = random.choice(legal_moves) # Should ideally not happen

    # Ensure best_move is a valid Move object from the original list
    # Find the original move object that corresponds to the best start/end squares
    # (This is belt-and-suspenders if best_move is assigned correctly above)
    final_best_move = None
    if best_move:
        for m in legal_moves:
             if m.start_sq == best_move.start_sq and m.end_sq == best_move.end_sq and m.promotion_piece_type == best_move.promotion_piece_type:
                 final_best_move = m
                 break
    if final_best_move is None and legal_moves:
         final_best_move = legal_moves[0] # Fallback if something went wrong

    print(f"AI ({gs.current_turn}) chose: {final_best_move.get_notation() if final_best_move else 'None'} with score: {best_score}")
    return final_best_move


# --- Game Over Screen ---
# (Keep existing draw_game_over)
def draw_game_over(screen, font, winner):
    """Draws the game over screen overlay."""
    overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    overlay.fill(GAME_OVER_BG)
    screen.blit(overlay, (0, 0))

    game_over_text = "Game Over"
    text_surface = font.render(game_over_text, True, WHITE)
    text_rect = text_surface.get_rect(center=(WIDTH // 2, HEIGHT // 2 - 60))
    screen.blit(text_surface, text_rect)

    if winner == 'draw':
        result_string = "Stalemate - It's a Draw!"
        result_color = DRAW_TEXT_COLOR
    else:
        winner_name = "Blue (Player)" if winner == 'w' else "Black (AI)"
        result_string = f"{winner_name} Wins!"
        result_color = WINNER_TEXT_COLOR

    result_surface = font.render(result_string, True, result_color)
    result_rect = result_surface.get_rect(center=(WIDTH // 2, HEIGHT // 2))
    screen.blit(result_surface, result_rect)

    restart_text = "Click anywhere to play again"
    restart_surface = font.render(restart_text, True, WHITE)
    restart_rect = restart_surface.get_rect(center=(WIDTH // 2, HEIGHT // 2 + 60))
    screen.blit(restart_surface, restart_rect)


# --- Main Game Function ---
# (Keep existing main loop structure, adapt AI turn call)
def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption(f"Pygame Chess - Player (Blue) vs AI (Depth {AI_SEARCH_DEPTH}+{QUIESCENCE_DEPTH})")
    clock = pygame.time.Clock()

    # Font loading (keep existing)
    try:
        piece_font = pygame.font.SysFont("Arial Unicode MS", SQUARE_SIZE * 2 // 3, bold=True)
        piece_font.render("♔", True, BLACK) # Test
        print("Using Arial Unicode MS font.")
    except Exception as e:
        print(f"Warning: Font error ({e}). Using default font.")
        piece_font = pygame.font.Font(None, SQUARE_SIZE * 3 // 4)
    try:
        game_over_font = pygame.font.SysFont("Arial", 50, bold=True)
    except Exception:
         game_over_font = pygame.font.Font(None, 60)

    gs = GameState()
    human_player_color = 'w' # Player is White
    player_turn = (gs.current_turn == human_player_color)
    ai_thinking = False
    running = True
    valid_player_moves = [] # Cache valid moves for the player

    # --- Pre-calculate valid moves for the first turn ---
    if player_turn:
        valid_player_moves = get_all_legal_moves(gs)


    while running:
        current_turn_color_for_input = gs.current_turn # Store turn at start of loop

        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False; break

            # --- Mouse Click Handling ---
            if event.type == pygame.MOUSEBUTTONDOWN:
                if gs.game_over:
                    print("Restarting game...")
                    gs = GameState()
                    player_turn = (gs.current_turn == human_player_color)
                    ai_thinking = False
                    valid_player_moves = get_all_legal_moves(gs) if player_turn else [] # Recalc initial moves
                    continue

                if player_turn and not ai_thinking and current_turn_color_for_input == human_player_color:
                    pos = pygame.mouse.get_pos()
                    square = get_square_from_pos(pos)

                    if square:
                        r, c = square
                        clicked_piece = gs.board[r][c]
                        clicked_piece_color = get_piece_color(clicked_piece)

                        if gs.selected_square: # A piece is already selected
                            start_r, start_c = gs.selected_square
                            target_r, target_c = square

                            # Find the intended move from the pre-calculated valid moves
                            intended_move = None
                            for valid_move in gs.valid_moves_for_selected_piece:
                                if valid_move.end_sq == square:
                                    intended_move = valid_move
                                    break

                            if intended_move:
                                print(f"Player moves: {intended_move.get_notation()}")
                                gs.make_move(intended_move)

                                # Check for game end AFTER player moves
                                valid_ai_moves = get_all_legal_moves(gs) # Check opponent moves
                                if not valid_ai_moves:
                                    if is_in_check(gs.board, gs.current_turn): # AI is checkmated
                                        gs.winner = human_player_color
                                        print("CHECKMATE! Player wins.")
                                    else: # Stalemate
                                        gs.winner = 'draw'
                                        print("STALEMATE! Draw.")
                                    gs.game_over = True

                                if not gs.game_over:
                                    player_turn = False
                                    ai_thinking = True
                                    print("AI's turn.")
                                # Clear selection regardless of move success here
                                gs.selected_square = None
                                gs.selected_piece = None
                                gs.valid_moves_for_selected_piece = []


                            # If click is on another friendly piece, change selection
                            elif clicked_piece != "." and clicked_piece_color == human_player_color:
                                gs.selected_square = square
                                gs.selected_piece = clicked_piece
                                # Filter pre-calculated moves for the new selection
                                gs.valid_moves_for_selected_piece = [m for m in valid_player_moves if m.start_sq == square]
                                print(f"Selected {gs.selected_piece} at {square}. Possible moves: {[m.get_notation() for m in gs.valid_moves_for_selected_piece]}")
                            # Otherwise (invalid move click or click empty square), deselect
                            else:
                                gs.selected_square = None
                                gs.selected_piece = None
                                gs.valid_moves_for_selected_piece = []

                        # --- Selecting a piece ---
                        elif clicked_piece != "." and clicked_piece_color == human_player_color:
                            gs.selected_square = square
                            gs.selected_piece = clicked_piece
                            # Filter pre-calculated moves for this selection
                            gs.valid_moves_for_selected_piece = [m for m in valid_player_moves if m.start_sq == square]
                            print(f"Selected {gs.selected_piece} at {square}. Possible moves: {[m.get_notation() for m in gs.valid_moves_for_selected_piece]}")
                        # --- Clicked empty/opponent piece without selection ---
                        else:
                            gs.selected_square = None
                            gs.selected_piece = None
                            gs.valid_moves_for_selected_piece = []

            # --- Keyboard Input Handling ---
            if event.type == pygame.KEYDOWN:
                 if event.key == pygame.K_u: # 'u' key for Undo
                      if not ai_thinking:
                           print("Attempting to undo...")
                           gs.undo_move() # Undo player or AI move
                           player_turn = (gs.current_turn == human_player_color)
                           # If AI was about to move, undo its move too
                           if not player_turn and len(gs.move_log) > 0:
                               print("Undoing AI move as well...")
                               gs.undo_move()
                               player_turn = (gs.current_turn == human_player_color)

                           # Recalculate valid moves for the current player
                           valid_player_moves = get_all_legal_moves(gs) if player_turn else []
                           gs.selected_square = None # Reset selection
                           gs.selected_piece = None
                           gs.valid_moves_for_selected_piece = []
                           gs.game_over = False # Game might not be over
                           gs.winner = None
                           print(f"Undo complete. {'Player' if player_turn else 'AI'}'s turn.")


        # --- AI Turn Logic ---
        if not player_turn and not gs.game_over and ai_thinking:
            # print("AI is thinking...") # Moved print to find_best_move_ai
            pygame.display.flip() # Show board before AI calculation

            ai_move = find_best_move_ai(gs, AI_SEARCH_DEPTH)

            if ai_move:
                gs.make_move(ai_move)

                # Check game end conditions AFTER AI moves
                valid_player_moves = get_all_legal_moves(gs) # Calculate player's next moves
                if not valid_player_moves:
                    if is_in_check(gs.board, gs.current_turn): # Player is checkmated
                        gs.winner = 'b' # AI wins
                        print("CHECKMATE! AI wins.")
                    else: # Stalemate
                        gs.winner = 'draw'
                        print("STALEMATE! Draw.")
                    gs.game_over = True

                if not gs.game_over:
                    player_turn = True # Switch back to player
                    print("Player's turn.")
            else:
                # This case should be caught by checkmate/stalemate check above
                print("AI Error: AI found no legal moves!")
                if is_in_check(gs.board, gs.current_turn): # AI is checkmated
                    gs.winner = human_player_color
                    print("CHECKMATE! Player wins (AI had no moves).")
                else: # Stalemate
                    gs.winner = 'draw'
                    print("STALEMATE! Draw (AI had no moves).")
                gs.game_over = True

            ai_thinking = False # AI finished

        # --- Drawing ---
        draw_board(screen)
        draw_highlights(screen, gs)
        draw_pieces(screen, gs.board, piece_font)

        if gs.game_over:
            draw_game_over(screen, game_over_font, gs.winner)

        pygame.display.flip()
        clock.tick(MAX_FPS)

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
