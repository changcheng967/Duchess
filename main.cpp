// Duchess Chess Engine - Phase 1: Foundation
// Single file implementation with bitboard representation and move generation

#include <iostream>
#include <vector>
#include <cstdint>
#include <random>
#include <string>
#include <array>
#include <algorithm>
#include <chrono>
#include <sstream>
#include <iomanip>
#include <fstream>

// ==================== CONSTANTS AND TYPES ====================

// Board squares
enum Square {
    A1, B1, C1, D1, E1, F1, G1, H1,
    A2, B2, C2, D2, E2, F2, G2, H2,
    A3, B3, C3, D3, E3, F3, G3, H3,
    A4, B4, C4, D4, E4, F4, G4, H4,
    A5, B5, C5, D5, E5, F5, G5, H5,
    A6, B6, C6, D6, E6, F6, G6, H6,
    A7, B7, C7, D7, E7, F7, G7, H7,
    A8, B8, C8, D8, E8, F8, G8, H8,
    NO_SQ
};

// Pieces
enum Piece {
    EMPTY = 0,
    W_PAWN, W_KNIGHT, W_BISHOP, W_ROOK, W_QUEEN, W_KING,
    B_PAWN, B_KNIGHT, B_BISHOP, B_ROOK, B_QUEEN, B_KING
};

// Colors
enum Color {
    WHITE = 0, BLACK = 1
};

// Castling rights
const int WHITE_KS = 1;
const int WHITE_QS = 2;
const int BLACK_KS = 4;
const int BLACK_QS = 8;

// Move flags
const int MOVE_CAPTURE = 1 << 4;
const int MOVE_PROMOTION = 1 << 5;
const int MOVE_ENPASSANT = 1 << 6;
const int MOVE_CASTLE = 1 << 7;

// Move encoding: 16 bits
// Bits 0-5: from square (0-63)
// Bits 6-11: to square (0-63)
// Bits 12-14: promotion piece (0-7)
// Bit 15: special flags
struct Move {
    uint16_t data;
    
    Move() : data(0) {}
    Move(int from, int to) : data(from | (to << 6)) {}
    Move(int from, int to, int flags) : data(from | (to << 6) | (flags << 12)) {}
    
    int from() const { return data & 0x3F; }
    int to() const { return (data >> 6) & 0x3F; }
    int flags() const { return (data >> 12) & 0xF; }
    int promotion() const { return (data >> 12) & 0x7; }
    
    bool is_capture() const { return (data >> 4) & 1; }
    bool is_promotion() const { return (data >> 5) & 1; }
    bool is_enpassant() const { return (data >> 6) & 1; }
    bool is_castle() const { return (data >> 7) & 1; }
    
    bool operator==(const Move& other) const { return data == other.data; }
    bool operator!=(const Move& other) const { return data != other.data; }
};

// ==================== BITBOARD UTILITIES ====================

using Bitboard = uint64_t;

// Bit manipulation
constexpr Bitboard SQ(int sq) { return 1ULL << sq; }
// Windows-compatible bit manipulation functions
#ifdef _MSC_VER
#include <intrin.h>
inline int popcount(Bitboard b) { return __popcnt64(b); }
inline int lsb(Bitboard b) {
    unsigned long index;
    if (_BitScanForward64(&index, b)) return index;
    return -1;
}
inline int msb(Bitboard b) {
    unsigned long index;
    if (_BitScanReverse64(&index, b)) return index;
    return -1;
}
#else
inline int popcount(Bitboard b) { return __builtin_popcountll(b); }
inline int lsb(Bitboard b) { return __builtin_ctzll(b); }
inline int msb(Bitboard b) { return 63 - __builtin_clzll(b); }
#endif
constexpr Bitboard clear_lsb(Bitboard b) { return b & (b - 1); }

// File and rank extraction
constexpr int file_of(int sq) { return sq & 7; }
constexpr int rank_of(int sq) { return sq >> 3; }
constexpr int make_sq(int file, int rank) { return (rank << 3) | file; }

// Direction vectors
constexpr int NORTH = 8, SOUTH = -8, EAST = 1, WEST = -1;
constexpr int NORTH_EAST = NORTH + EAST;
constexpr int NORTH_WEST = NORTH + WEST;
constexpr int SOUTH_EAST = SOUTH + EAST;
constexpr int SOUTH_WEST = SOUTH + WEST;

// ==================== ZOBRIST HASHING ====================

class Zobrist {
private:
    static constexpr int NUM_PIECES = 13;
    static constexpr int NUM_SQUARES = 64;
    static constexpr int NUM_CASTLING = 16;
    
    Bitboard piece_keys[NUM_PIECES][NUM_SQUARES];
    Bitboard side_key;
    Bitboard castling_keys[NUM_CASTLING];
    Bitboard enpassant_keys[8];
    
    std::mt19937_64 rng;
    
public:
    Zobrist() : rng(123456789) {
        // Initialize random keys
        for (int p = 1; p <= 12; p++) {
            for (int s = 0; s < NUM_SQUARES; s++) {
                piece_keys[p][s] = rng();
            }
        }
        side_key = rng();
        for (int i = 0; i < NUM_CASTLING; i++) {
            castling_keys[i] = rng();
        }
        for (int i = 0; i < 8; i++) {
            enpassant_keys[i] = rng();
        }
    }
    
    Bitboard hash_piece(int piece, int square) const {
        return piece_keys[piece][square];
    }
    
    Bitboard hash_side() const { return side_key; }
    Bitboard hash_castling(int rights) const { return castling_keys[rights]; }
    Bitboard hash_enpassant(int file) const { return enpassant_keys[file]; }
};

static Zobrist zobrist;

// ==================== ATTACK GENERATION ====================

class Attacks {
private:
    static std::array<std::array<Bitboard, 64>, 2> pawn_attacks_table;
    static std::array<Bitboard, 64> knight_attacks_table;
    static std::array<Bitboard, 64> king_attacks_table;
    static std::array<std::array<Bitboard, 64>, 8> line_attacks_table;
    
    static void init_pawn_attacks();
    static void init_knight_attacks();
    static void init_king_attacks();
    static void init_line_attacks();
    
public:
    static void init() {
        init_pawn_attacks();
        init_knight_attacks();
        init_king_attacks();
        init_line_attacks();
    }
    
    static Bitboard pawn_attacks(int color, int sq) {
        return pawn_attacks_table[color][sq];
    }
    
    static Bitboard knight_attacks(int sq) {
        return knight_attacks_table[sq];
    }
    
    static Bitboard king_attacks(int sq) {
        return king_attacks_table[sq];
    }
    
    static Bitboard bishop_attacks(int sq, Bitboard occupied);
    static Bitboard rook_attacks(int sq, Bitboard occupied);
    static Bitboard queen_attacks(int sq, Bitboard occupied);
};

std::array<std::array<Bitboard, 64>, 2> Attacks::pawn_attacks_table;
std::array<Bitboard, 64> Attacks::knight_attacks_table;
std::array<Bitboard, 64> Attacks::king_attacks_table;
std::array<std::array<Bitboard, 64>, 8> Attacks::line_attacks_table;

void Attacks::init_pawn_attacks() {
    for (int sq = 0; sq < 64; sq++) {
        int file = file_of(sq);
        int rank = rank_of(sq);
        
        // White pawn attacks
        Bitboard attacks = 0;
        if (rank < 7) {
            if (file > 0) attacks |= SQ(sq + NORTH_WEST);
            if (file < 7) attacks |= SQ(sq + NORTH_EAST);
        }
        pawn_attacks_table[WHITE][sq] = attacks;
        
        // Black pawn attacks
        attacks = 0;
        if (rank > 0) {
            if (file > 0) attacks |= SQ(sq + SOUTH_WEST);
            if (file < 7) attacks |= SQ(sq + SOUTH_EAST);
        }
        pawn_attacks_table[BLACK][sq] = attacks;
    }
}

void Attacks::init_knight_attacks() {
    const int offsets[] = {-17, -15, -10, -6, 6, 10, 15, 17};
    
    for (int sq = 0; sq < 64; sq++) {
        Bitboard attacks = 0;
        for (int offset : offsets) {
            int target = sq + offset;
            if (target >= 0 && target < 64) {
                int from_file = file_of(sq);
                int to_file = file_of(target);
                if (abs(from_file - to_file) <= 2) {
                    attacks |= SQ(target);
                }
            }
        }
        knight_attacks_table[sq] = attacks;
    }
}

void Attacks::init_king_attacks() {
    const int offsets[] = {-9, -8, -7, -1, 1, 7, 8, 9};
    
    for (int sq = 0; sq < 64; sq++) {
        Bitboard attacks = 0;
        for (int offset : offsets) {
            int target = sq + offset;
            if (target >= 0 && target < 64) {
                int from_file = file_of(sq);
                int to_file = file_of(target);
                if (abs(from_file - to_file) <= 1) {
                    attacks |= SQ(target);
                }
            }
        }
        king_attacks_table[sq] = attacks;
    }
}

void Attacks::init_line_attacks() {
    // Initialize for each direction
    const int directions[] = {NORTH, SOUTH, EAST, WEST, 
                             NORTH_EAST, NORTH_WEST, 
                             SOUTH_EAST, SOUTH_WEST};
    
    for (int dir_idx = 0; dir_idx < 8; dir_idx++) {
        int dir = directions[dir_idx];
        for (int sq = 0; sq < 64; sq++) {
            Bitboard attacks = 0;
            int target = sq + dir;
            
            while (target >= 0 && target < 64) {
                attacks |= SQ(target);
                
                // Check if we hit edge of board
                int from_file = file_of(sq);
                int to_file = file_of(target);
                if (abs(from_file - to_file) > 1) break;
                
                target += dir;
            }
            line_attacks_table[dir_idx][sq] = attacks;
        }
    }
}

Bitboard Attacks::bishop_attacks(int sq, Bitboard occupied) {
    Bitboard attacks = 0;
    
    // Northeast
    int target = sq + NORTH_EAST;
    while (target >= 0 && target < 64 && file_of(target) > file_of(sq)) {
        attacks |= SQ(target);
        if (occupied & SQ(target)) break;
        target += NORTH_EAST;
    }
    
    // Northwest
    target = sq + NORTH_WEST;
    while (target >= 0 && target < 64 && file_of(target) < file_of(sq)) {
        attacks |= SQ(target);
        if (occupied & SQ(target)) break;
        target += NORTH_WEST;
    }
    
    // Southeast
    target = sq + SOUTH_EAST;
    while (target >= 0 && target < 64 && file_of(target) > file_of(sq)) {
        attacks |= SQ(target);
        if (occupied & SQ(target)) break;
        target += SOUTH_EAST;
    }
    
    // Southwest
    target = sq + SOUTH_WEST;
    while (target >= 0 && target < 64 && file_of(target) < file_of(sq)) {
        attacks |= SQ(target);
        if (occupied & SQ(target)) break;
        target += SOUTH_WEST;
    }
    
    return attacks;
}

Bitboard Attacks::rook_attacks(int sq, Bitboard occupied) {
    Bitboard attacks = 0;
    
    // North
    int target = sq + NORTH;
    while (target >= 0 && target < 64) {
        attacks |= SQ(target);
        if (occupied & SQ(target)) break;
        target += NORTH;
    }
    
    // South
    target = sq + SOUTH;
    while (target >= 0 && target < 64) {
        attacks |= SQ(target);
        if (occupied & SQ(target)) break;
        target += SOUTH;
    }
    
    // East
    target = sq + EAST;
    while (target >= 0 && target < 64 && file_of(target) > file_of(sq)) {
        attacks |= SQ(target);
        if (occupied & SQ(target)) break;
        target += EAST;
    }
    
    // West
    target = sq + WEST;
    while (target >= 0 && target < 64 && file_of(target) < file_of(sq)) {
        attacks |= SQ(target);
        if (occupied & SQ(target)) break;
        target += WEST;
    }
    
    return attacks;
}

Bitboard Attacks::queen_attacks(int sq, Bitboard occupied) {
    return bishop_attacks(sq, occupied) | rook_attacks(sq, occupied);
}

// ==================== POSITION CLASS ====================

// Undo information for move undo
struct UndoInfo {
    int captured_piece;
    int castling_rights;
    int en_passant_square;
    int halfmove_clock;
    Bitboard hash;
};

class Position {
public:
    // Bitboards for each piece type
    Bitboard pieces[13]; // 0: empty (unused), 1-6: white, 7-12: black
    Bitboard occupied[2]; // White and black occupancy
    Bitboard all_occupied;
    
    // Game state
    int side_to_move;
    int castling_rights;
    int en_passant_square;
    int halfmove_clock;
    int fullmove_number;
    
    // Zobrist hash
    Bitboard hash;
    
    // Material count
    int material[2];
    
    // Move history for undo
    std::vector<UndoInfo> history;
    
private:
    
    // Update hash when piece moves
    void update_hash_remove(int piece, int square) {
        hash ^= zobrist.hash_piece(piece, square);
    }
    
    void update_hash_add(int piece, int square) {
        hash ^= zobrist.hash_piece(piece, square);
    }
    
    void update_hash_castling() {
        hash ^= zobrist.hash_castling(castling_rights);
    }
    
    void update_hash_enpassant() {
        if (en_passant_square != NO_SQ) {
            hash ^= zobrist.hash_enpassant(file_of(en_passant_square));
        }
    }
    
    void update_hash_side() {
        hash ^= zobrist.hash_side();
    }
    
    // Helper functions
    Bitboard get_attacks_to(int square, int attacker_color) const;
    bool is_square_attacked(int square, int attacker_color) const;
    void update_occupancy() {
        occupied[WHITE] = 0;
        occupied[BLACK] = 0;
        all_occupied = 0;
        
        for (int p = W_PAWN; p <= W_KING; p++) {
            occupied[WHITE] |= pieces[p];
            all_occupied |= pieces[p];
        }
        for (int p = B_PAWN; p <= B_KING; p++) {
            occupied[BLACK] |= pieces[p];
            all_occupied |= pieces[p];
        }
    }
    
public:
    Position();
    Position(const std::string& fen);
    
    // Accessors
    Bitboard get_pieces(int piece) const { return pieces[piece]; }
    Bitboard get_occupied(int color) const { return occupied[color]; }
    Bitboard get_all_occupied() const { return all_occupied; }
    int get_side_to_move() const { return side_to_move; }
    int get_castling_rights() const { return castling_rights; }
    int get_en_passant_square() const { return en_passant_square; }
    Bitboard get_hash() const { return hash; }
    
    // Helper method for move scoring
    int get_piece_at(int square) const {
        for (int p = 1; p <= 12; p++) {
            if (pieces[p] & SQ(square)) {
                return p;
            }
        }
        return EMPTY;
    }
    
    // Move generation
    std::vector<Move> generate_moves() const;
    std::vector<Move> generate_captures() const;
    std::vector<Move> generate_quiet_moves() const;
    std::vector<Move> generate_legal_moves() const;
    
    // Move execution
    bool make_move(const Move& move);
    void undo_move(const Move& move);
    
    // Game state
    bool is_check() const;
    bool is_checkmate() const;
    bool is_stalemate() const;
    bool is_repetition() const;
    bool is_insufficient_material() const;
    bool is_game_over() const;
    
    // FEN handling
    std::string to_fen() const;
    void from_fen(const std::string& fen);
    
    // Evaluation
    int evaluate() const;
    
    // Debug
    void print() const;
    
    // Helper method for move scoring
    int get_piece_at(int square) const {
        for (int p = 1; p <= 12; p++) {
            if (pieces[p] & SQ(square)) {
                return p;
            }
        }
        return EMPTY;
    }
};

// ==================== MOVE GENERATION ====================

std::vector<Move> Position::generate_moves() const {
    std::vector<Move> moves;
    
    // Generate all pseudo-legal moves
    moves.reserve(128); // Reserve space for efficiency
    
    int friendly = side_to_move;
    int enemy = 1 - side_to_move;
    
    // Pawns
    Bitboard pawns = pieces[friendly == WHITE ? W_PAWN : B_PAWN];
    while (pawns) {
        int from = lsb(pawns);
        pawns = clear_lsb(pawns);
        
        // Pawn moves
        int forward = friendly == WHITE ? NORTH : SOUTH;
        int rank = rank_of(from);
        
        // Single push
        int to = from + forward;
        if (!(all_occupied & SQ(to))) {
            if (rank_of(to) == (friendly == WHITE ? 7 : 0)) {
                // Promotion
                moves.emplace_back(from, to, MOVE_PROMOTION | W_QUEEN);
                moves.emplace_back(from, to, MOVE_PROMOTION | W_ROOK);
                moves.emplace_back(from, to, MOVE_PROMOTION | W_BISHOP);
                moves.emplace_back(from, to, MOVE_PROMOTION | W_KNIGHT);
            } else {
                moves.emplace_back(from, to);
                
                // Double push
                if ((friendly == WHITE && rank == 1) || (friendly == BLACK && rank == 6)) {
                    to = from + forward + forward;
                    if (!(all_occupied & SQ(to))) {
                        moves.emplace_back(from, to);
                    }
                }
            }
        }
        
        // Captures
        Bitboard attacks = Attacks::pawn_attacks(friendly, from);
        Bitboard targets = attacks & occupied[enemy];
        while (targets) {
            int to = lsb(targets);
            targets = clear_lsb(targets);
            
            if (rank_of(to) == (friendly == WHITE ? 7 : 0)) {
                // Promotion capture
                moves.emplace_back(from, to, MOVE_CAPTURE | MOVE_PROMOTION | W_QUEEN);
                moves.emplace_back(from, to, MOVE_CAPTURE | MOVE_PROMOTION | W_ROOK);
                moves.emplace_back(from, to, MOVE_CAPTURE | MOVE_PROMOTION | W_BISHOP);
                moves.emplace_back(from, to, MOVE_CAPTURE | MOVE_PROMOTION | W_KNIGHT);
            } else {
                moves.emplace_back(from, to, MOVE_CAPTURE);
            }
        }
        
        // En passant
        if (en_passant_square != NO_SQ) {
            if (attacks & SQ(en_passant_square)) {
                moves.emplace_back(from, en_passant_square, MOVE_CAPTURE | MOVE_ENPASSANT);
            }
        }
    }
    
    // Knights
    Bitboard knights = pieces[friendly == WHITE ? W_KNIGHT : B_KNIGHT];
    while (knights) {
        int from = lsb(knights);
        knights = clear_lsb(knights);
        
        Bitboard attacks = Attacks::knight_attacks(from) & ~occupied[friendly];
        while (attacks) {
            int to = lsb(attacks);
            attacks = clear_lsb(attacks);
            moves.emplace_back(from, to, occupied[enemy] & SQ(to) ? MOVE_CAPTURE : 0);
        }
    }
    
    // Bishops
    Bitboard bishops = pieces[friendly == WHITE ? W_BISHOP : B_BISHOP];
    while (bishops) {
        int from = lsb(bishops);
        bishops = clear_lsb(bishops);
        
        Bitboard attacks = Attacks::bishop_attacks(from, all_occupied) & ~occupied[friendly];
        while (attacks) {
            int to = lsb(attacks);
            attacks = clear_lsb(attacks);
            moves.emplace_back(from, to, occupied[enemy] & SQ(to) ? MOVE_CAPTURE : 0);
        }
    }
    
    // Rooks
    Bitboard rooks = pieces[friendly == WHITE ? W_ROOK : B_ROOK];
    while (rooks) {
        int from = lsb(rooks);
        rooks = clear_lsb(rooks);
        
        Bitboard attacks = Attacks::rook_attacks(from, all_occupied) & ~occupied[friendly];
        while (attacks) {
            int to = lsb(attacks);
            attacks = clear_lsb(attacks);
            moves.emplace_back(from, to, occupied[enemy] & SQ(to) ? MOVE_CAPTURE : 0);
        }
    }
    
    // Queens
    Bitboard queens = pieces[friendly == WHITE ? W_QUEEN : B_QUEEN];
    while (queens) {
        int from = lsb(queens);
        queens = clear_lsb(queens);
        
        Bitboard attacks = Attacks::queen_attacks(from, all_occupied) & ~occupied[friendly];
        while (attacks) {
            int to = lsb(attacks);
            attacks = clear_lsb(attacks);
            moves.emplace_back(from, to, occupied[enemy] & SQ(to) ? MOVE_CAPTURE : 0);
        }
    }
    
    // King
    int king_sq = lsb(pieces[friendly == WHITE ? W_KING : B_KING]);
    
    // King moves
    Bitboard attacks = Attacks::king_attacks(king_sq) & ~occupied[friendly];
    while (attacks) {
        int to = lsb(attacks);
        attacks = clear_lsb(attacks);
        moves.emplace_back(king_sq, to, occupied[enemy] & SQ(to) ? MOVE_CAPTURE : 0);
    }
    
    // Castling
    if (friendly == WHITE) {
        if (castling_rights & WHITE_KS) {
            if (!(all_occupied & (SQ(F1) | SQ(G1))) &&
                !is_square_attacked(E1, enemy) && !is_square_attacked(F1, enemy)) {
                moves.emplace_back(E1, G1, MOVE_CASTLE);
            }
        }
        if (castling_rights & WHITE_QS) {
            if (!(all_occupied & (SQ(B1) | SQ(C1) | SQ(D1))) &&
                !is_square_attacked(E1, enemy) && !is_square_attacked(D1, enemy)) {
                moves.emplace_back(E1, C1, MOVE_CASTLE);
            }
        }
    } else {
        if (castling_rights & BLACK_KS) {
            if (!(all_occupied & (SQ(F8) | SQ(G8))) &&
                !is_square_attacked(E8, enemy) && !is_square_attacked(F8, enemy)) {
                moves.emplace_back(E8, G8, MOVE_CASTLE);
            }
        }
        if (castling_rights & BLACK_QS) {
            if (!(all_occupied & (SQ(B8) | SQ(C8) | SQ(D8))) &&
                !is_square_attacked(E8, enemy) && !is_square_attacked(D8, enemy)) {
                moves.emplace_back(E8, C8, MOVE_CASTLE);
            }
        }
    }
    
    return moves;
}

std::vector<Move> Position::generate_captures() const {
    auto all_moves = generate_moves();
    std::vector<Move> captures;
    captures.reserve(all_moves.size());
    
    for (const auto& move : all_moves) {
        if (move.is_capture()) {
            captures.push_back(move);
        }
    }
    
    return captures;
}

std::vector<Move> Position::generate_quiet_moves() const {
    auto all_moves = generate_moves();
    std::vector<Move> quiet;
    quiet.reserve(all_moves.size());
    
    for (const auto& move : all_moves) {
        if (!move.is_capture() && !move.is_promotion() && !move.is_castle()) {
            quiet.push_back(move);
        }
    }
    
    return quiet;
}

// Generate legal moves (filter out moves that leave king in check)
std::vector<Move> Position::generate_legal_moves() const {
    auto pseudo_legal = generate_moves();
    std::vector<Move> legal;
    legal.reserve(pseudo_legal.size());
    
    int our_color = side_to_move;
    int king_piece = (our_color == WHITE) ? W_KING : B_KING;
    
    // Get king position BEFORE any moves
    Bitboard king_bb = pieces[king_piece];
    if (king_bb == 0) {
        // No king - invalid position
        return legal;
    }
    int king_sq = lsb(king_bb);
    
    for (const auto& move : pseudo_legal) {
        // Create a lightweight copy for validation
        Position temp = *this;
        
        // Make the move
        if (!temp.make_move(move)) continue;
        
        // Find king position after move (might have moved)
        Bitboard new_king_bb = temp.pieces[king_piece];
        if (new_king_bb == 0) continue; // King captured (invalid)
        
        int new_king_sq = lsb(new_king_bb);
        
        // Check if king is attacked AFTER the move
        // Use direct attack check WITHOUT calling generate_legal_moves again
        if (!temp.is_square_attacked(new_king_sq, 1 - our_color)) {
            legal.push_back(move);
        }
    }
    
    return legal;
}

// ==================== POSITION IMPLEMENTATION ====================

Position::Position() {
    from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
}

Position::Position(const std::string& fen) {
    from_fen(fen);
}

void Position::from_fen(const std::string& fen) {
    // Reset everything
    for (int i = 0; i <= 12; i++) pieces[i] = 0;
    occupied[WHITE] = occupied[BLACK] = all_occupied = 0;
    hash = 0;
    material[WHITE] = material[BLACK] = 0;
    history.clear();  // Clear history for new position
    
    std::istringstream ss(fen);
    std::string board, side, castling, enpassant;
    int halfmove, fullmove;
    
    ss >> board >> side >> castling >> enpassant >> halfmove >> fullmove;
    
    // Parse board
    int rank = 7; // Start at rank 8 (index 7)
    int file = 0;
    
    for (char c : board) {
        if (c == '/') {
            rank--;
            file = 0;
        } else if (isdigit(c)) {
            file += (c - '0'); // Skip files
        } else {
            int piece = EMPTY;
            
            switch (c) {
                case 'P': piece = W_PAWN; break;
                case 'N': piece = W_KNIGHT; break;
                case 'B': piece = W_BISHOP; break;
                case 'R': piece = W_ROOK; break;
                case 'Q': piece = W_QUEEN; break;
                case 'K': piece = W_KING; break;
                case 'p': piece = B_PAWN; break;
                case 'n': piece = B_KNIGHT; break;
                case 'b': piece = B_BISHOP; break;
                case 'r': piece = B_ROOK; break;
                case 'q': piece = B_QUEEN; break;
                case 'k': piece = B_KING; break;
            }
            
            if (piece != EMPTY) {
                int sq = make_sq(file, rank);
                pieces[piece] |= SQ(sq);
                hash ^= zobrist.hash_piece(piece, sq);
                file++;
            }
        }
    }
    
    // Update occupancy after placing all pieces
    update_occupancy();
    
    // Parse side to move
    side_to_move = (side == "w") ? WHITE : BLACK;
    
    // Parse castling rights
    castling_rights = 0;
    for (char c : castling) {
        switch (c) {
            case 'K': castling_rights |= WHITE_KS; break;
            case 'Q': castling_rights |= WHITE_QS; break;
            case 'k': castling_rights |= BLACK_KS; break;
            case 'q': castling_rights |= BLACK_QS; break;
        }
    }
    
    // Parse en passant
    if (enpassant != "-") {
        int file = enpassant[0] - 'a';
        int rank = enpassant[1] - '1';
        en_passant_square = make_sq(file, rank);
        hash ^= zobrist.hash_enpassant(file);
    } else {
        en_passant_square = NO_SQ;
    }
    
    halfmove_clock = halfmove;
    fullmove_number = fullmove;
    
    // Update hash for side and castling
    if (side_to_move == BLACK) hash ^= zobrist.hash_side();
    hash ^= zobrist.hash_castling(castling_rights);
}

std::string Position::to_fen() const {
    std::string fen;
    
    // Board
    for (int rank = 7; rank >= 0; rank--) {
        int empty = 0;
        for (int file = 0; file < 8; file++) {
            int sq = make_sq(file, rank);
            bool found = false;
            
            for (int p = W_PAWN; p <= B_KING; p++) {
                if (pieces[p] & SQ(sq)) {
                    if (empty > 0) {
                        fen += std::to_string(empty);
                        empty = 0;
                    }
                    char piece_char = ".PNBRQKpnbrqk"[p];
                    fen += piece_char;
                    found = true;
                    break;
                }
            }
            
            if (!found) empty++;
        }
        if (empty > 0) fen += std::to_string(empty);
        if (rank > 0) fen += "/";
    }
    
    // Side
    fen += (side_to_move == WHITE) ? " w " : " b ";
    
    // Castling
    std::string castling;
    if (castling_rights & WHITE_KS) castling += "K";
    if (castling_rights & WHITE_QS) castling += "Q";
    if (castling_rights & BLACK_KS) castling += "k";
    if (castling_rights & BLACK_QS) castling += "q";
    if (castling.empty()) castling = "-";
    fen += castling + " ";
    
    // En passant
    if (en_passant_square != NO_SQ) {
        char file = 'a' + file_of(en_passant_square);
        char rank = '1' + rank_of(en_passant_square);
        fen += std::string(1, file) + std::string(1, rank);
    } else {
        fen += "-";
    }
    
    // Halfmove and fullmove
    fen += " " + std::to_string(halfmove_clock) + " " + std::to_string(fullmove_number);
    
    return fen;
}


void Position::print() const {
    std::cout << "\nPosition:\n";
    for (int rank = 7; rank >= 0; rank--) {
        for (int file = 0; file < 8; file++) {
            int sq = make_sq(file, rank);
            bool found = false;
            
            for (int p = W_PAWN; p <= B_KING; p++) {
                if (pieces[p] & SQ(sq)) {
                    char piece_char = ".PNBRQKpnbrqk"[p];
                    std::cout << piece_char << " ";
                    found = true;
                    break;
                }
            }
            
            if (!found) std::cout << ". ";
        }
        std::cout << "\n";
    }
    
    std::cout << "\nSide: " << (side_to_move == WHITE ? "White" : "Black") << "\n";
    std::cout << "Castling: " << castling_rights << "\n";
    std::cout << "En passant: " << (en_passant_square == NO_SQ ? "-" : std::to_string(en_passant_square)) << "\n";
    std::cout << "Hash: 0x" << std::hex << hash << std::dec << "\n";
}

// ==================== NNUE EVALUATION ====================

// ==================== COMPLETE FIXED NNUE EVALUATION ====================
// Replace the entire NNUE namespace in your code with this implementation

// ==================== COMPLETE FIXED NNUE EVALUATION ====================
// Replace the entire NNUE namespace in your code with this implementation

namespace NNUE {
    // Stockfish NNUE architecture: HalfKAv2_hm (768->256x2->32->32->1)
    constexpr int FEATURE_TRANSFORMER_INPUT = 768;    // 12 piece types × 64 squares
    constexpr int FEATURE_TRANSFORMER_OUTPUT = 256;   // Feature transformer output per side
    constexpr int LAYER1_INPUT = 512;                 // 2 × 256 (both perspectives)
    constexpr int LAYER1_OUTPUT = 32;
    constexpr int LAYER2_INPUT = 32;
    constexpr int LAYER2_OUTPUT = 32;
    constexpr int OUTPUT_SIZE = 1;
    
    // Quantization scales (Stockfish uses fixed-point arithmetic)
    constexpr int FT_QUANT = 255;        // Feature transformer quantization
    constexpr int FT_SHIFT = 6;          // Feature transformer shift
    constexpr int WEIGHT_SCALE = 64;     // Weight scale
    constexpr int OUTPUT_SCALE = 16;     // Output scale
    
    // Network weights and biases
    alignas(64) static int16_t feature_weights[FEATURE_TRANSFORMER_INPUT][FEATURE_TRANSFORMER_OUTPUT];
    alignas(64) static int16_t feature_biases[FEATURE_TRANSFORMER_OUTPUT];
    alignas(64) static int8_t layer1_weights[LAYER1_INPUT][LAYER1_OUTPUT];
    alignas(64) static int32_t layer1_biases[LAYER1_OUTPUT];
    alignas(64) static int8_t layer2_weights[LAYER2_INPUT][LAYER2_OUTPUT];
    alignas(64) static int32_t layer2_biases[LAYER2_OUTPUT];
    alignas(64) static int8_t output_weights[LAYER2_OUTPUT];
    alignas(64) static int32_t output_bias;
    
    // NNUE file loaded flag
    static bool nnue_loaded = false;
    
    // Accumulator structure (stores feature transformer output)
    struct Accumulator {
        alignas(64) int16_t white[FEATURE_TRANSFORMER_OUTPUT];
        alignas(64) int16_t black[FEATURE_TRANSFORMER_OUTPUT];
        bool computed;
        
        Accumulator() : computed(false) {
            std::fill(std::begin(white), std::end(white), 0);
            std::fill(std::begin(black), std::end(black), 0);
        }
    };
    
    // LEB128 decompression helper
    class LEB128Reader {
    private:
        std::ifstream& file;
        size_t bytes_read;
        
    public:
        LEB128Reader(std::ifstream& f) : file(f), bytes_read(0) {}
        
        size_t get_bytes_read() const { return bytes_read; }
        
        // Read a signed LEB128-encoded integer
        int32_t read_int() {
            int32_t result = 0;
            int shift = 0;
            uint8_t byte;
            
            do {
                if (!file.read(reinterpret_cast<char*>(&byte), 1)) {
                    throw std::runtime_error("Unexpected EOF in LEB128 stream");
                }
                bytes_read++;
                result |= (byte & 0x7F) << shift;
                shift += 7;
            } while (byte & 0x80);
            
            // Sign extend if necessary
            if (shift < 32 && (byte & 0x40)) {
                result |= -(1 << shift);
            }
            
            return result;
        }
        
        // Read array of int16_t values
        void read_int16_array(int16_t* array, size_t count) {
            for (size_t i = 0; i < count; i++) {
                array[i] = static_cast<int16_t>(read_int());
            }
        }
        
        // Read array of int8_t values
        void read_int8_array(int8_t* array, size_t count) {
            for (size_t i = 0; i < count; i++) {
                array[i] = static_cast<int8_t>(read_int());
            }
        }
        
        // Read array of int32_t values
        void read_int32_array(int32_t* array, size_t count) {
            for (size_t i = 0; i < count; i++) {
                array[i] = read_int();
            }
        }
    };
    
    // Load NNUE weights from Stockfish format file (supports compressed and uncompressed)
    bool load_nnue(const std::string& filename) {
        std::ifstream file(filename, std::ios::binary);
        if (!file.is_open()) {
            return false;  // Silently fail, will try next path
        }
        
        // Get file size for validation
        file.seekg(0, std::ios::end);
        size_t file_size = file.tellg();
        file.seekg(0, std::ios::beg);
        
        std::cout << "\n=== Loading NNUE File ===" << std::endl;
        std::cout << "File: " << filename << std::endl;
        std::cout << "Size: " << file_size << " bytes (" << (file_size / 1024) << " KB)" << std::endl;
        
        // Read header (version 0x7AF32F20 = Stockfish NNUE)
        uint32_t version;
        file.read(reinterpret_cast<char*>(&version), sizeof(version));
        
        if (version != 0x7AF32F20 && version != 0x7AF32F16) {
            std::cerr << "ERROR: Invalid NNUE version: 0x" << std::hex << version << std::dec << std::endl;
            file.close();
            return false;
        }
        
        std::cout << "Version: 0x" << std::hex << version << std::dec << std::endl;
        
        // Read hash
        uint32_t hash_value;
        file.read(reinterpret_cast<char*>(&hash_value), sizeof(hash_value));
        std::cout << "Hash: 0x" << std::hex << hash_value << std::dec << std::endl;
        
        // Read architecture string with robust error handling
        std::string architecture;
        char c;
        int max_arch_length = 2048;
        int arch_bytes_read = 0;
        bool found_null = false;
        
        while (arch_bytes_read < max_arch_length) {
            if (!file.get(c)) {
                std::cerr << "ERROR: EOF while reading architecture string" << std::endl;
                file.close();
                return false;
            }
            
            arch_bytes_read++;
            
            if (c == '\0') {
                found_null = true;
                break;
            }
            
            // Add all characters (printable or not) for now
            architecture += c;
        }
        
        if (!found_null) {
            std::cerr << "ERROR: Architecture string too long" << std::endl;
            file.close();
            return false;
        }
        
        // Clean architecture string (remove non-printable chars)
        std::string clean_arch;
        for (char ch : architecture) {
            if (ch >= 32 && ch <= 126) {
                clean_arch += ch;
            }
        }
        
        std::cout << "Architecture: " << clean_arch << std::endl;
        std::cout << "Arch length: " << clean_arch.length() << " bytes" << std::endl;
        
        // Detect compression
        bool is_compressed = (clean_arch.find("COMPRESSED") != std::string::npos) ||
                            (file_size > 50000000);  // Files > 50MB are likely compressed
        
        std::cout << "Compression: " << (is_compressed ? "LEB128" : "Raw Binary") << std::endl;
        
        // Helper lambda to read section hash
        auto read_section = [&file]() {
            uint32_t hash;
            file.read(reinterpret_cast<char*>(&hash), sizeof(hash));
            return hash;
        };
        
        try {
            if (is_compressed) {
                // ===== COMPRESSED FORMAT (LEB128) =====
                std::cout << "\n--- Decompressing LEB128 Data ---" << std::endl;
                LEB128Reader reader(file);
                
                // Feature Transformer
                std::cout << "Reading Feature Transformer..." << std::endl;
                uint32_t ft_hash = read_section();
                std::cout << "  Section hash: 0x" << std::hex << ft_hash << std::dec << std::endl;
                
                reader.read_int16_array(feature_biases, FEATURE_TRANSFORMER_OUTPUT);
                std::cout << "  Biases: " << FEATURE_TRANSFORMER_OUTPUT << " values" << std::endl;
                
                // Weights are stored transposed: [output][input]
                for (int i = 0; i < FEATURE_TRANSFORMER_OUTPUT; i++) {
                    for (int j = 0; j < FEATURE_TRANSFORMER_INPUT; j++) {
                        feature_weights[j][i] = static_cast<int16_t>(reader.read_int());
                    }
                    if (i % 64 == 0) {
                        std::cout << "  Progress: " << i << "/" << FEATURE_TRANSFORMER_OUTPUT << "\r" << std::flush;
                    }
                }
                std::cout << "  Weights: " << (FEATURE_TRANSFORMER_INPUT * FEATURE_TRANSFORMER_OUTPUT)
                          << " values (" << reader.get_bytes_read() << " bytes read)" << std::endl;
                
                // Layer 1
                std::cout << "Reading Layer 1..." << std::endl;
                uint32_t l1_hash = read_section();
                std::cout << "  Section hash: 0x" << std::hex << l1_hash << std::dec << std::endl;
                
                reader.read_int32_array(layer1_biases, LAYER1_OUTPUT);
                
                for (int i = 0; i < LAYER1_OUTPUT; i++) {
                    for (int j = 0; j < LAYER1_INPUT; j++) {
                        layer1_weights[j][i] = static_cast<int8_t>(reader.read_int());
                    }
                }
                std::cout << "  Complete" << std::endl;
                
                // Layer 2
                std::cout << "Reading Layer 2..." << std::endl;
                uint32_t l2_hash = read_section();
                std::cout << "  Section hash: 0x" << std::hex << l2_hash << std::dec << std::endl;
                
                reader.read_int32_array(layer2_biases, LAYER2_OUTPUT);
                
                for (int i = 0; i < LAYER2_OUTPUT; i++) {
                    for (int j = 0; j < LAYER2_INPUT; j++) {
                        layer2_weights[j][i] = static_cast<int8_t>(reader.read_int());
                    }
                }
                std::cout << "  Complete" << std::endl;
                
                // Output Layer
                std::cout << "Reading Output Layer..." << std::endl;
                uint32_t out_hash = read_section();
                std::cout << "  Section hash: 0x" << std::hex << out_hash << std::dec << std::endl;
                
                output_bias = reader.read_int();
                reader.read_int8_array(output_weights, LAYER2_OUTPUT);
                std::cout << "  Complete" << std::endl;
                
                std::cout << "\nTotal bytes decompressed: " << reader.get_bytes_read() << std::endl;
                
            } else {
                // ===== UNCOMPRESSED FORMAT (RAW BINARY) =====
                std::cout << "\n--- Reading Raw Binary Data ---" << std::endl;
                
                // Feature Transformer
                std::cout << "Reading Feature Transformer..." << std::endl;
                uint32_t ft_hash = read_section();
                
                file.read(reinterpret_cast<char*>(feature_biases),
                          FEATURE_TRANSFORMER_OUTPUT * sizeof(int16_t));
                
                if (!file.good()) {
                    throw std::runtime_error("Failed to read feature biases");
                }
                
                file.read(reinterpret_cast<char*>(feature_weights),
                          FEATURE_TRANSFORMER_INPUT * FEATURE_TRANSFORMER_OUTPUT * sizeof(int16_t));
                
                if (!file.good()) {
                    throw std::runtime_error("Failed to read feature weights");
                }
                
                std::cout << "  Complete ("
                          << (FEATURE_TRANSFORMER_INPUT * FEATURE_TRANSFORMER_OUTPUT * sizeof(int16_t))
                          << " bytes)" << std::endl;
                
                // Layer 1
                std::cout << "Reading Layer 1..." << std::endl;
                uint32_t l1_hash = read_section();
                
                file.read(reinterpret_cast<char*>(layer1_biases),
                          LAYER1_OUTPUT * sizeof(int32_t));
                file.read(reinterpret_cast<char*>(layer1_weights),
                          LAYER1_INPUT * LAYER1_OUTPUT * sizeof(int8_t));
                
                if (!file.good()) {
                    throw std::runtime_error("Failed to read layer 1");
                }
                std::cout << "  Complete" << std::endl;
                
                // Layer 2
                std::cout << "Reading Layer 2..." << std::endl;
                uint32_t l2_hash = read_section();
                
                file.read(reinterpret_cast<char*>(layer2_biases),
                          LAYER2_OUTPUT * sizeof(int32_t));
                file.read(reinterpret_cast<char*>(layer2_weights),
                          LAYER2_INPUT * LAYER2_OUTPUT * sizeof(int8_t));
                
                if (!file.good()) {
                    throw std::runtime_error("Failed to read layer 2");
                }
                std::cout << "  Complete" << std::endl;
                
                // Output Layer
                std::cout << "Reading Output Layer..." << std::endl;
                uint32_t out_hash = read_section();
                
                file.read(reinterpret_cast<char*>(&output_bias), sizeof(int32_t));
                file.read(reinterpret_cast<char*>(output_weights),
                          LAYER2_OUTPUT * sizeof(int8_t));
                
                if (!file.good()) {
                    throw std::runtime_error("Failed to read output layer");
                }
                std::cout << "  Complete" << std::endl;
            }
            
        } catch (const std::exception& e) {
            std::cerr << "ERROR: " << e.what() << std::endl;
            file.close();
            return false;
        }
        
        file.close();
        
        // Validate weights
        std::cout << "\n--- Validation ---" << std::endl;
        std::cout << "Sample feature_biases[0]: " << feature_biases[0] << std::endl;
        std::cout << "Sample feature_biases[1]: " << feature_biases[1] << std::endl;
        std::cout << "Sample feature_biases[255]: " << feature_biases[255] << std::endl;
        std::cout << "Sample feature_weights[0][0]: " << feature_weights[0][0] << std::endl;
        std::cout << "Sample layer1_biases[0]: " << layer1_biases[0] << std::endl;
        std::cout << "Sample output_bias: " << output_bias << std::endl;
        
        // Check for all-zero weights (corruption indicator)
        int non_zero_count = 0;
        for (int i = 0; i < 100; i++) {
            if (feature_biases[i] != 0) non_zero_count++;
        }
        
        if (non_zero_count == 0) {
            std::cerr << "WARNING: All sampled biases are zero - file may be corrupted!" << std::endl;
            return false;
        }
        
        std::cout << "\n✓ NNUE loaded successfully!" << std::endl;
        std::cout << "========================\n" << std::endl;
        
        nnue_loaded = true;
        return true;
    }
    
    // Initialize NNUE (try to load file, fallback to random weights)
    void init() {
        // Try multiple possible paths
        std::vector<std::string> paths = {
            "nn-2962dca31855.nnue",                                    // Current directory (compressed)
            "C:\\Users\\chang\\Downloads\\nn-2962dca31855.nnue",      // Downloads (compressed)
            "raw_master.nnue",                                         // Current directory (uncompressed)
            "C:\\Users\\chang\\Downloads\\raw_master.nnue",           // Downloads (uncompressed)
            "./nn-2962dca31855.nnue",
            "../nn-2962dca31855.nnue",
            "C:\\Users\\chang\\Downloads\\Duchess\\nn-2962dca31855.nnue"
        };
        
        std::cout << "Searching for NNUE file..." << std::endl;
        
        for (const auto& path : paths) {
            std::cout << "Trying: " << path << "..." << std::endl;
            if (load_nnue(path)) {
                std::cout << "Successfully loaded from: " << path << std::endl;
                return;
            }
        }
        
        std::cout << "\n⚠ WARNING: No NNUE file found!" << std::endl;
        std::cout << "Using random weights (evaluation will be weak)" << std::endl;
        std::cout << "Download nn-2962dca31855.nnue from Stockfish and place it in:" << std::endl;
        std::cout << "  - Current directory, or" << std::endl;
        std::cout << "  - C:\\Users\\chang\\Downloads\\" << std::endl;
        
        // Fallback: Initialize with small random weights
        std::mt19937 rng(123456789);
        std::uniform_int_distribution<int> dist_ft(-127, 127);
        std::uniform_int_distribution<int> dist_layer(-127, 127);
        
        for (int i = 0; i < FEATURE_TRANSFORMER_OUTPUT; i++) {
            feature_biases[i] = 0;
        }
        
        for (int i = 0; i < FEATURE_TRANSFORMER_INPUT; i++) {
            for (int j = 0; j < FEATURE_TRANSFORMER_OUTPUT; j++) {
                feature_weights[i][j] = static_cast<int16_t>(dist_ft(rng));
            }
        }
        
        for (int i = 0; i < LAYER1_OUTPUT; i++) {
            layer1_biases[i] = 0;
        }
        
        for (int i = 0; i < LAYER1_INPUT; i++) {
            for (int j = 0; j < LAYER1_OUTPUT; j++) {
                layer1_weights[i][j] = static_cast<int8_t>(dist_layer(rng));
            }
        }
        
        for (int i = 0; i < LAYER2_OUTPUT; i++) {
            layer2_biases[i] = 0;
        }
        
        for (int i = 0; i < LAYER2_INPUT; i++) {
            for (int j = 0; j < LAYER2_OUTPUT; j++) {
                layer2_weights[i][j] = static_cast<int8_t>(dist_layer(rng));
            }
        }
        
        output_bias = 0;
        for (int i = 0; i < LAYER2_OUTPUT; i++) {
            output_weights[i] = static_cast<int8_t>(dist_layer(rng));
        }
        
        nnue_loaded = false;
    }
    
    // Convert piece and square to feature index (HalfKAv2 format)
    inline int get_feature_index(int piece, int square, int king_square, bool white_perspective) {
        int piece_type;
        if (piece >= W_PAWN && piece <= W_KING) {
            piece_type = piece - W_PAWN;
        } else if (piece >= B_PAWN && piece <= B_KING) {
            piece_type = (piece - B_PAWN) + 6;
        } else {
            return -1;
        }
        
        if (!white_perspective) {
            square ^= 56;
            king_square ^= 56;
        }
        
        return piece_type * 64 + square;
    }
    
    // Refresh accumulator from scratch
    void refresh_accumulator(const Position& pos, Accumulator& acc) {
        std::copy(std::begin(feature_biases), std::end(feature_biases), std::begin(acc.white));
        std::copy(std::begin(feature_biases), std::end(feature_biases), std::begin(acc.black));
        
        int white_king_sq = lsb(pos.get_pieces(W_KING));
        int black_king_sq = lsb(pos.get_pieces(B_KING));
        
        for (int piece = W_PAWN; piece <= B_KING; piece++) {
            Bitboard pieces = pos.get_pieces(piece);
            while (pieces) {
                int sq = lsb(pieces);
                pieces = clear_lsb(pieces);
                
                int idx_white = get_feature_index(piece, sq, white_king_sq, true);
                if (idx_white >= 0 && idx_white < FEATURE_TRANSFORMER_INPUT) {
                    for (int j = 0; j < FEATURE_TRANSFORMER_OUTPUT; j++) {
                        acc.white[j] += feature_weights[idx_white][j];
                    }
                }
                
                int idx_black = get_feature_index(piece, sq, black_king_sq, false);
                if (idx_black >= 0 && idx_black < FEATURE_TRANSFORMER_INPUT) {
                    for (int j = 0; j < FEATURE_TRANSFORMER_OUTPUT; j++) {
                        acc.black[j] += feature_weights[idx_black][j];
                    }
                }
            }
        }
        
        acc.computed = true;
    }
    
    // ClippedReLU activation
    inline int32_t clipped_relu(int32_t x) {
        return std::max(0, std::min(127, x));
    }
    
    // Evaluate position using NNUE
    int evaluate(const Position& pos) {
        Accumulator acc;
        refresh_accumulator(pos, acc);
        
        int16_t* our_acc = (pos.get_side_to_move() == WHITE) ? acc.white : acc.black;
        int16_t* their_acc = (pos.get_side_to_move() == WHITE) ? acc.black : acc.white;
        
        alignas(64) int32_t layer1_output[LAYER1_OUTPUT];
        
        for (int i = 0; i < LAYER1_OUTPUT; i++) {
            int32_t sum = layer1_biases[i];
            
            for (int j = 0; j < FEATURE_TRANSFORMER_OUTPUT; j++) {
                int32_t clipped = clipped_relu(our_acc[j]);
                sum += clipped * layer1_weights[j][i];
            }
            
            for (int j = 0; j < FEATURE_TRANSFORMER_OUTPUT; j++) {
                int32_t clipped = clipped_relu(their_acc[j]);
                sum += clipped * layer1_weights[j + FEATURE_TRANSFORMER_OUTPUT][i];
            }
            
            sum /= WEIGHT_SCALE;
            layer1_output[i] = clipped_relu(sum);
        }
        
        alignas(64) int32_t layer2_output[LAYER2_OUTPUT];
        
        for (int i = 0; i < LAYER2_OUTPUT; i++) {
            int32_t sum = layer2_biases[i];
            
            for (int j = 0; j < LAYER1_OUTPUT; j++) {
                sum += layer1_output[j] * layer2_weights[j][i];
            }
            
            sum /= WEIGHT_SCALE;
            layer2_output[i] = clipped_relu(sum);
        }
        
        int32_t output = output_bias;
        
        for (int i = 0; i < LAYER2_OUTPUT; i++) {
            output += layer2_output[i] * output_weights[i];
        }
        
        int eval = (output * 600) / (127 * OUTPUT_SCALE);
        
        return eval;
    }
    
    // Check if NNUE is properly loaded
    bool is_loaded() {
        return nnue_loaded;
    }
}

// ==================== MOVE EXECUTION ====================

bool Position::make_move(const Move& move) {
    int from = move.from();
    int to = move.to();
    int piece = -1;
    int captured = -1;
    
    // Find the moving piece
    for (int p = 1; p <= 12; p++) {  // Fixed: Check pieces 1-12
        if (pieces[p] & SQ(from)) {
            piece = p;
            break;
        }
    }
    
    if (piece == -1) return false; // No piece to move
    
    int color = (piece <= 6) ? WHITE : BLACK;
    int enemy = 1 - color;
    
    // Store undo information
    UndoInfo undo;
    undo.castling_rights = castling_rights;
    undo.en_passant_square = en_passant_square;
    undo.halfmove_clock = halfmove_clock;
    undo.hash = hash;
    
    // Handle captures
    if (move.is_capture()) {
        if (move.is_enpassant()) {
            // En passant capture
            int ep_target = to + (color == WHITE ? SOUTH : NORTH);
            for (int p = 1; p <= 12; p++) {
                if (pieces[p] & SQ(ep_target)) {
                    captured = p;
                    undo.captured_piece = captured;
                    break;
                }
            }
        } else {
            // Normal capture
            for (int p = 1; p <= 12; p++) {
                if (pieces[p] & SQ(to)) {
                    captured = p;
                    undo.captured_piece = captured;
                    break;
                }
            }
        }
    } else {
        undo.captured_piece = -1;
    }
    
    // Update hash
    update_hash_remove(piece, from);
    update_hash_side();
    
    // Handle captures
    if (move.is_capture()) {
        if (move.is_enpassant()) {
            // En passant capture
            int ep_target = to + (color == WHITE ? SOUTH : NORTH);
            for (int p = 1; p <= 12; p++) {
                if (pieces[p] & SQ(ep_target)) {
                    captured = p;
                    update_hash_remove(captured, ep_target);
                    pieces[captured] ^= SQ(ep_target);
                    occupied[enemy] ^= SQ(ep_target);
                    all_occupied ^= SQ(ep_target);
                    break;
                }
            }
        } else {
            // Normal capture
            for (int p = 1; p <= 12; p++) {
                if (pieces[p] & SQ(to)) {
                    captured = p;
                    update_hash_remove(captured, to);
                    pieces[captured] ^= SQ(to);
                    occupied[enemy] ^= SQ(to);
                    all_occupied ^= SQ(to);
                    break;
                }
            }
        }
    }
    
    // Move piece
    pieces[piece] ^= SQ(from) | SQ(to);
    occupied[color] ^= SQ(from) | SQ(to);
    all_occupied ^= SQ(from) | SQ(to);
    update_hash_add(piece, to);
    
    // Handle special moves
    if (move.is_castle()) {
        // King-side castling
        if (to == G1 || to == G8) {
            int rook_from = (color == WHITE) ? H1 : H8;
            int rook_to = (color == WHITE) ? F1 : F8;
            int rook = (color == WHITE) ? W_ROOK : B_ROOK;
            
            pieces[rook] ^= SQ(rook_from) | SQ(rook_to);
            occupied[color] ^= SQ(rook_from) | SQ(rook_to);
            all_occupied ^= SQ(rook_from) | SQ(rook_to);
            update_hash_remove(rook, rook_from);
            update_hash_add(rook, rook_to);
        }
        // Queen-side castling
        else if (to == C1 || to == C8) {
            int rook_from = (color == WHITE) ? A1 : A8;
            int rook_to = (color == WHITE) ? D1 : D8;
            int rook = (color == WHITE) ? W_ROOK : B_ROOK;
            
            pieces[rook] ^= SQ(rook_from) | SQ(rook_to);
            occupied[color] ^= SQ(rook_from) | SQ(rook_to);
            all_occupied ^= SQ(rook_from) | SQ(rook_to);
            update_hash_remove(rook, rook_from);
            update_hash_add(rook, rook_to);
        }
    }
    
    // Handle promotion
    if (move.is_promotion()) {
        int promotion_piece = move.promotion();
        int pawn = (color == WHITE) ? W_PAWN : B_PAWN;
        
        // Remove pawn
        pieces[pawn] ^= SQ(to);
        update_hash_remove(pawn, to);
        
        // Add promoted piece
        pieces[promotion_piece] ^= SQ(to);
        update_hash_add(promotion_piece, to);
    }
    
    // Update castling rights
    int old_castling = castling_rights;
    if (piece == W_KING || piece == B_KING) {
        castling_rights &= ~(color == WHITE ? (WHITE_KS | WHITE_QS) : (BLACK_KS | BLACK_QS));
    }
    if (piece == W_ROOK) {
        if (from == A1) castling_rights &= ~WHITE_QS;
        if (from == H1) castling_rights &= ~WHITE_KS;
    }
    if (piece == B_ROOK) {
        if (from == A8) castling_rights &= ~BLACK_QS;
        if (from == H8) castling_rights &= ~BLACK_KS;
    }
    if (old_castling != castling_rights) {
        update_hash_castling();
    }
    
    // Update en passant
    int old_ep = en_passant_square;
    en_passant_square = NO_SQ;
    if (piece == W_PAWN || piece == B_PAWN) {
        int rank = rank_of(from);
        int to_rank = rank_of(to);
        if (abs(to_rank - rank) == 2) {
            en_passant_square = from + (color == WHITE ? NORTH : SOUTH);
            update_hash_enpassant();
        }
    }
    if (old_ep != en_passant_square) {
        if (old_ep != NO_SQ) hash ^= zobrist.hash_enpassant(file_of(old_ep));
        if (en_passant_square != NO_SQ) hash ^= zobrist.hash_enpassant(file_of(en_passant_square));
    }
    
    // Update game state
    side_to_move = enemy;
    halfmove_clock = (piece == W_PAWN || piece == B_PAWN || captured != -1) ? 0 : halfmove_clock + 1;
    if (color == BLACK) fullmove_number++;
    
    // Store undo info
    history.push_back(undo);
    
    return true;
}

void Position::undo_move(const Move& move) {
    if (history.empty()) return;
    
    UndoInfo undo = history.back();
    history.pop_back();
    
    int from = move.from();
    int to = move.to();
    int piece = -1;
    
    // Find the moving piece
    for (int p = 1; p <= 12; p++) {
        if (pieces[p] & SQ(to)) {
            piece = p;
            break;
        }
    }
    
    if (piece == -1) return; // No piece to undo
    
    int color = (piece <= 6) ? WHITE : BLACK;
    int enemy = 1 - color;
    
    // Restore hash
    hash = undo.hash;
    
    // Restore game state
    castling_rights = undo.castling_rights;
    en_passant_square = undo.en_passant_square;
    halfmove_clock = undo.halfmove_clock;
    
    // Restore piece position
    pieces[piece] ^= SQ(from) | SQ(to);
    occupied[color] ^= SQ(from) | SQ(to);
    all_occupied ^= SQ(from) | SQ(to);
    
    // Restore captured piece if any
    if (undo.captured_piece != -1) {
        int capture_sq = to;
        
        // En passant: captured pawn is on different square
        if (move.is_enpassant()) {
            capture_sq = to + (color == WHITE ? SOUTH : NORTH);
        }
        
        pieces[undo.captured_piece] |= SQ(capture_sq);
        occupied[enemy] |= SQ(capture_sq);
        all_occupied |= SQ(capture_sq);
    }
    
    // Handle special moves
    if (move.is_castle()) {
        // King-side castling
        if (to == G1 || to == G8) {
            int rook_from = (color == WHITE) ? H1 : H8;
            int rook_to = (color == WHITE) ? F1 : F8;
            int rook = (color == WHITE) ? W_ROOK : B_ROOK;
            
            pieces[rook] ^= SQ(rook_from) | SQ(rook_to);
            occupied[color] ^= SQ(rook_from) | SQ(rook_to);
            all_occupied ^= SQ(rook_from) | SQ(rook_to);
        }
        // Queen-side castling
        else if (to == C1 || to == C8) {
            int rook_from = (color == WHITE) ? A1 : A8;
            int rook_to = (color == WHITE) ? D1 : D8;
            int rook = (color == WHITE) ? W_ROOK : B_ROOK;
            
            pieces[rook] ^= SQ(rook_from) | SQ(rook_to);
            occupied[color] ^= SQ(rook_from) | SQ(rook_to);
            all_occupied ^= SQ(rook_from) | SQ(rook_to);
        }
    }
    
    // Handle promotion
    if (move.is_promotion()) {
        int pawn = (color == WHITE) ? W_PAWN : B_PAWN;
        
        // Remove promoted piece from 'from' square
        pieces[move.promotion()] ^= SQ(from);
        
        // Restore pawn to 'from' square
        pieces[pawn] ^= SQ(from);
    }
    
    // Restore side to move
    side_to_move = color;
    
    // Update occupancy
    update_occupancy();
}

// ==================== GAME STATE CHECKS ====================

Bitboard Position::get_attacks_to(int square, int attacker_color) const {
    Bitboard attacks = 0;
    
    // Pawns
    int pawn_piece = attacker_color == WHITE ? W_PAWN : B_PAWN;
    Bitboard enemy_pawns = pieces[pawn_piece];
    Bitboard pawn_attack_sources = Attacks::pawn_attacks(1 - attacker_color, square);
    attacks |= enemy_pawns & pawn_attack_sources;
    
    // Knights
    attacks |= Attacks::knight_attacks(square) & pieces[attacker_color == WHITE ? W_KNIGHT : B_KNIGHT];
    
    // King
    attacks |= Attacks::king_attacks(square) & pieces[attacker_color == WHITE ? W_KING : B_KING];
    
    // Sliding pieces
    Bitboard bishops = pieces[attacker_color == WHITE ? W_BISHOP : B_BISHOP] | pieces[attacker_color == WHITE ? W_QUEEN : B_QUEEN];
    attacks |= Attacks::bishop_attacks(square, all_occupied) & bishops;
    
    Bitboard rooks = pieces[attacker_color == WHITE ? W_ROOK : B_ROOK] | pieces[attacker_color == WHITE ? W_QUEEN : B_QUEEN];
    attacks |= Attacks::rook_attacks(square, all_occupied) & rooks;
    
    return attacks;
}

bool Position::is_square_attacked(int square, int attacker_color) const {
    return get_attacks_to(square, attacker_color) != 0;
}

bool Position::is_check() const {
    int king_sq = lsb(pieces[side_to_move == WHITE ? W_KING : B_KING]);
    return is_square_attacked(king_sq, 1 - side_to_move);
}

bool Position::is_checkmate() const {
    if (!is_check()) return false;
    auto moves = generate_moves();
    for (const auto& move : moves) {
        // Would need to check if move gets out of check
        // Simplified for now
    }
    return false; // Simplified
}

bool Position::is_stalemate() const {
    if (is_check()) return false;
    auto moves = generate_moves();
    return moves.empty();
}

bool Position::is_repetition() const {
    // Simplified - would need to track history
    return false;
}

bool Position::is_insufficient_material() const {
    // Simplified - just check for bare kings
    return (occupied[WHITE] == pieces[W_KING]) &&
           (occupied[BLACK] == pieces[B_KING]);
}

bool Position::is_game_over() const {
    return is_checkmate() || is_stalemate() || is_insufficient_material() || halfmove_clock >= 100;
}

// ==================== PERFT TESTING ====================

class Perft {
private:
    static uint64_t perft_recursive(Position& pos, int depth) {
        if (depth == 0) return 1;
        
        auto moves = pos.generate_moves();
        uint64_t nodes = 0;
        
        for (const auto& move : moves) {
            Position temp = pos; // Copy position
            temp.make_move(move);
            nodes += perft_recursive(temp, depth - 1);
        }
        
        return nodes;
    }
    
public:
    static uint64_t perft(Position& pos, int depth) {
        return perft_recursive(pos, depth);
    }
    
    static void run_test_suite() {
        std::cout << "Running Perft test suite...\n";
        
        // Standard test positions
        std::vector<std::pair<std::string, std::vector<uint64_t>>> tests = {
            {"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", {20, 400, 8902, 197281, 4865609, 119060324}},
            {"r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", {48, 2039, 97862, 4085603, 193690690, 8031647685}},
            {"8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", {14, 191, 2812, 43238, 674624, 11030093}},
            {"r2q1rk1/pP1p2pp/Q4n2/bbp1p3/Np6/1B3NBn/pPPP1PPP/R4RK1 b - - 0 1", {6, 264, 9467, 422333, 15833292, 706045033}}
        };
        
        for (size_t i = 0; i < tests.size(); i++) {
            std::cout << "\nTest " << (i + 1) << ":\n";
            Position pos(tests[i].first);
            
            for (int depth = 1; depth <= 6; depth++) {
                auto start = std::chrono::high_resolution_clock::now();
                uint64_t nodes = perft(pos, depth);
                auto end = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
                
                uint64_t expected = tests[i].second[depth - 1];
                bool correct = (nodes == expected);
                
                std::cout << "Depth " << depth << ": " << nodes 
                         << " (expected: " << expected << ") "
                         << (correct ? "PASS" : "FAIL")
                         << " in " << duration.count() << "ms\n";
            }
        }
    }
};

// ==================== SEARCH OPTIMIZATIONS ====================

// Transposition Table
struct TTEntry {
    Bitboard hash;
    int depth;
    int score;
    int flag;  // 0=exact, 1=lower_bound, 2=upper_bound
    Move best_move;
};

static std::vector<TTEntry> transposition_table(1 << 20);  // 1M entries
static constexpr int EXACT = 0;
static constexpr int LOWER_BOUND = 1;
static constexpr int UPPER_BOUND = 2;

// Killer moves for move ordering
static Move killer_moves[2][100];  // 2 killers per depth

// History heuristic for move ordering
static int history_table[13][64][64];

// Global node counter
static uint64_t nodes_searched = 0;

// Move scoring for ordering
static int score_move(const Position& pos, const Move& move, int depth) {
    int score = 0;
    
    // MVV-LVA (Most Valuable Victim - Least Valuable Attacker)
    if (move.is_capture()) {
        int victim = pos.get_piece_at(move.to());
        int attacker = pos.get_piece_at(move.from());
        
        // Piece values: P=1, N=2, B=3, R=4, Q=5, K=6
        int victim_value = (victim == W_PAWN || victim == B_PAWN) ? 1 :
                          (victim == W_KNIGHT || victim == B_KNIGHT) ? 2 :
                          (victim == W_BISHOP || victim == B_BISHOP) ? 3 :
                          (victim == W_ROOK || victim == B_ROOK) ? 4 :
                          (victim == W_QUEEN || victim == B_QUEEN) ? 5 : 6;
        int attacker_value = (attacker == W_PAWN || attacker == B_PAWN) ? 1 :
                            (attacker == W_KNIGHT || attacker == B_KNIGHT) ? 2 :
                            (attacker == W_BISHOP || attacker == B_BISHOP) ? 3 :
                            (attacker == W_ROOK || attacker == B_ROOK) ? 4 :
                            (attacker == W_QUEEN || attacker == B_QUEEN) ? 5 : 6;
        
        score += 1000000 + (victim_value * 1000) - attacker_value;
    }
    
    // Promotion bonus
    if (move.is_promotion()) {
        score += 900000;
    }
    
    // Killer move bonus
    if (depth < 100) {
        if (killer_moves[0][depth] == move) score += 800000;
        else if (killer_moves[1][depth] == move) score += 700000;
    }
    
    // History heuristic bonus
    int piece = pos.get_piece_at(move.from());
    score += history_table[piece][move.from()][move.to()];
    
    // Center control bonus
    int to_file = file_of(move.to());
    int to_rank = rank_of(move.to());
    if ((to_file == 3 || to_file == 4) && (to_rank == 3 || to_rank == 4)) {
        score += 10000;
    }
    
    return score;
}

// Helper function to evaluate position
static int evaluate_position(const Position& pos) {
    return pos.evaluate();
}

// Quiescence search
static int quiescence(Position& pos, int alpha, int beta, int ply = 0) {
    // Limit quiescence depth
    if (ply >= 10) {
        return evaluate_position(pos);
    }
    
    nodes_searched++;
    
    // Stand-pat
    int stand_pat = evaluate_position(pos);
    if (stand_pat >= beta) return beta;
    if (alpha < stand_pat) alpha = stand_pat;
    
    // Delta pruning
    const int QUEEN_VALUE = 900;
    if (stand_pat + QUEEN_VALUE < alpha) {
        return alpha; // Even capturing a queen won't help
    }
    
    // Generate captures only
    auto captures = pos.generate_captures();
    
    // Score and sort captures
    std::vector<std::pair<Move, int>> scored_moves;
    scored_moves.reserve(captures.size());
    for (const auto& move : captures) {
        scored_moves.emplace_back(move, score_move(pos, move, 0));
    }
    std::sort(scored_moves.begin(), scored_moves.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });
    
    int our_color = pos.get_side_to_move();
    int king_piece = (our_color == WHITE) ? W_KING : B_KING;
    
    for (const auto& scored_move : scored_moves) {
        const Move& move = scored_move.first;
        
        Position temp = pos;
        if (!temp.make_move(move)) continue;
        
        // Legality check
        Bitboard king_bb = temp.pieces[king_piece];
        if (king_bb == 0) continue;
        
        int king_sq = lsb(king_bb);
        if (temp.is_square_attacked(king_sq, 1 - our_color)) {
            continue;
        }
        
        int score_after = -quiescence(temp, -beta, -alpha, ply + 1);
        
        if (score_after >= beta) return beta;
        if (score_after > alpha) alpha = score_after;
    }
    
    return alpha;
}

// Alpha-beta search with optimizations
static int alpha_beta(Position& pos, int depth, int alpha, int beta) {
    nodes_searched++;
    
    // Transposition table lookup
    Bitboard hash = pos.get_hash();
    int tt_index = hash % transposition_table.size();
    TTEntry& entry = transposition_table[tt_index];
    
    if (entry.hash == hash && entry.depth >= depth) {
        if (entry.flag == EXACT) return entry.score;
        if (entry.flag == LOWER_BOUND) alpha = std::max(alpha, entry.score);
        if (entry.flag == UPPER_BOUND) beta = std::min(beta, entry.score);
        if (alpha >= beta) return entry.score;
    }
    
    if (depth == 0) {
        return quiescence(pos, alpha, beta);
    }
    
    // Use pseudo-legal moves to avoid recursion
    auto moves = pos.generate_moves();
    if (moves.empty()) {
        if (pos.is_check()) {
            return -30000 + (100 - depth); // Prefer faster mates
        }
        return 0; // Stalemate
    }
    
    // Score and sort moves
    std::vector<std::pair<Move, int>> scored_moves;
    scored_moves.reserve(moves.size());
    for (const auto& move : moves) {
        scored_moves.emplace_back(move, score_move(pos, move, depth));
    }
    std::sort(scored_moves.begin(), scored_moves.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });
    
    int best_score = -30000;
    Move best_move;
    int legal_moves = 0;
    
    int our_color = pos.get_side_to_move();
    int king_piece = (our_color == WHITE) ? W_KING : B_KING;
    
    for (const auto& scored_move : scored_moves) {
        const Move& move = scored_move.first;
        
        // Make move on copy
        Position temp = pos;
        if (!temp.make_move(move)) continue;
        
        // Legality check: ensure our king is not in check
        Bitboard king_bb = temp.pieces[king_piece];
        if (king_bb == 0) continue; // King captured
        
        int king_sq = lsb(king_bb);
        if (temp.is_square_attacked(king_sq, 1 - our_color)) {
            continue; // Illegal move - king in check
        }
        
        legal_moves++;
        
        // Search
        int score_after;
        if (legal_moves == 1) {
            // Full search for first legal move
            score_after = -alpha_beta(temp, depth - 1, -beta, -alpha);
        } else {
            // Null window search
            score_after = -alpha_beta(temp, depth - 1, -alpha - 1, -alpha);
            if (score_after > alpha && score_after < beta) {
                // Re-search with full window
                score_after = -alpha_beta(temp, depth - 1, -beta, -alpha);
            }
        }
        
        if (score_after > best_score) {
            best_score = score_after;
            best_move = move;
            if (score_after > alpha) {
                alpha = score_after;
            }
        }
        
        if (alpha >= beta) {
            // Store killer move
            if (depth < 100 && !move.is_capture()) {
                killer_moves[1][depth] = killer_moves[0][depth];
                killer_moves[0][depth] = move;
            }
            
            // Update history heuristic
            int piece = pos.get_piece_at(move.from());
            if (piece >= 1 && piece <= 12) {
                history_table[piece][move.from()][move.to()] += depth * depth;
            }
            
            break; // Beta cutoff
        }
    }
    
    // No legal moves found
    if (legal_moves == 0) {
        if (pos.is_check()) {
            return -30000 + (100 - depth); // Checkmate
        }
        return 0; // Stalemate
    }
    
    // Store in transposition table
    entry.hash = hash;
    entry.depth = depth;
    entry.score = best_score;
    entry.best_move = best_move;
    
    if (best_score <= alpha) entry.flag = UPPER_BOUND;
    else if (best_score >= beta) entry.flag = LOWER_BOUND;
    else entry.flag = EXACT;
    
    return best_score;
}

// ==================== SEARCH ALGORITHM ====================

class Search {
private:
    static uint64_t nodes_searched;  // Node counter
    
    static std::pair<Move, int> find_best_move(Position& pos, int depth) {
        auto moves = pos.generate_legal_moves();
        if (moves.empty()) {
            return {Move(), -1000000};
        }
        
        Move best_move = moves[0];
        int best_score = -1000000;
        
        for (const auto& move : moves) {
            Position temp_pos = pos; // Create copy to avoid modifying original
            if (!temp_pos.make_move(move)) continue;
            
            nodes_searched = 0;  // Reset node counter
            int score = -alpha_beta(temp_pos, depth - 1, -1000000, 1000000);
            
            // Better move selection: prefer captures and center control
            if (score > best_score) {
                best_score = score;
                best_move = move;
            } else if (score == best_score) {
                // Tie-breaking: prefer captures, then center moves
                if (move.is_capture() && !best_move.is_capture()) {
                    best_score = score;
                    best_move = move;
                } else if (!move.is_capture() && !best_move.is_capture()) {
                    // Prefer center pawn moves
                    int to_file = file_of(best_move.to());
                    int best_to_file = file_of(best_move.to());
                    if ((to_file == 3 || to_file == 4) && (best_to_file != 3 && best_to_file != 4)) {
                        best_score = score;
                        best_move = move;
                    }
                }
            }
        }
        
        return {best_move, best_score};
    }

public:
    static void iterative_deepening(Position& pos, int max_depth, int time_limit_ms) {
        auto start_time = std::chrono::high_resolution_clock::now();
        
        Move best_move;
        int best_score = 0;
        
        for (int depth = 1; depth <= max_depth; depth++) {
            auto current_time = std::chrono::high_resolution_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                current_time - start_time).count();
            
            if (elapsed >= time_limit_ms) {
                break;
            }
            
            auto result = find_best_move(pos, depth);
            Move current_best = result.first;
            int current_score = result.second;
            
            // Calculate nodes per second
            uint64_t nps = (elapsed > 0) ? (nodes_searched * 1000 / elapsed) : 0;
            
            // Output search info
            std::cout << "info depth " << depth
                      << " score cp " << current_score
                      << " time " << elapsed
                      << " nodes " << nodes_searched
                      << " nps " << nps << "\n";
            
            best_move = current_best;
            best_score = current_score;
        }
        
        // Output best move
        std::cout << "bestmove ";
        if (best_move.data != 0) {
            char from_file = 'a' + file_of(best_move.from());
            char from_rank = '1' + rank_of(best_move.from());
            char to_file = 'a' + file_of(best_move.to());
            char to_rank = '1' + rank_of(best_move.to());
            std::cout << from_file << from_rank << to_file << to_rank;
            
            if (best_move.is_promotion()) {
                char promo = "  nbrq"[best_move.promotion() % 6];
                std::cout << promo;
            }
        } else {
            std::cout << "(none)";
        }
        std::cout << "\n";
    }
};

// Initialize static member
uint64_t Search::nodes_searched = 0;

// ==================== UCI PROTOCOL ====================

class UCI {
private:
    static Position current_position;  // Global position state
    
    static void uci_loop() {
        std::string command;
        
        while (std::getline(std::cin, command)) {
            std::istringstream ss(command);
            std::string token;
            ss >> token;
            
            if (token == "uci") {
                std::cout << "id name Duchess Chess Engine\n";
                std::cout << "id author changcheng967\n";
                std::cout << "option name Hash type spin default 16 min 1 max 1024\n";
                std::cout << "uciok\n";
            }
            else if (token == "isready") {
                std::cout << "readyok\n";
            }
            else if (token == "ucinewgame") {
                // Reset to starting position
                current_position.from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
            }
            else if (token == "position") {
                std::string fen;
                std::vector<std::string> moves;
                std::string part;
                
                ss >> part;
                if (part == "startpos") {
                    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
                } else if (part == "fen") {
                    // Read FEN
                    fen = "";
                    while (ss >> part && part != "moves") {
                        if (!fen.empty()) fen += " ";
                        fen += part;
                    }
                }
                
                // Read moves
                while (ss >> part) {
                    moves.push_back(part);
                }
                
                // Apply position and moves to global state
                current_position.from_fen(fen);
                
                // Apply moves
                for (const auto& move_str : moves) {
                    // Parse move string (e.g., "e2e4")
                    if (move_str.length() >= 4) {
                        int from_file = move_str[0] - 'a';
                        int from_rank = move_str[1] - '1';
                        int to_file = move_str[2] - 'a';
                        int to_rank = move_str[3] - '1';
                        
                        int from = from_file + from_rank * 8;
                        int to = to_file + to_rank * 8;
                        
                        Move move(from, to);
                        current_position.make_move(move);
                    }
                }
            }
            else if (token == "go") {
                // Parse go command
                int depth = 8; // Default depth
                int time = 900; // Default time in ms (1 second limit)
                
                while (ss >> token) {
                    if (token == "depth") ss >> depth;
                    else if (token == "wtime" || token == "btime") {
                        int time_ms; ss >> time_ms;
                        time = time_ms / 40; // Rough time allocation
                        if (time > 900) time = 900; // Cap at 900ms
                    }
                }
                
                // Start search with iterative deepening using current position
                Search::iterative_deepening(current_position, depth, time);
            }
            else if (token == "quit") {
                break;
            }
            else if (token == "perft") {
                int depth;
                ss >> depth;
                
                auto start = std::chrono::high_resolution_clock::now();
                uint64_t nodes = Perft::perft(current_position, depth);
                auto end = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
                
                std::cout << "Perft depth " << depth << ": " << nodes << " nodes in " << duration.count() << "ms\n";
            }
            else if (token == "test") {
                Perft::run_test_suite();
            }
        }
    }
    
public:
    static void start() {
        // Initialize starting position
        current_position.from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
        
        Attacks::init();
        NNUE::init(); // Initialize NNUE
        uci_loop();
    }
};

// Initialize static member
Position UCI::current_position;

// NNUE evaluation
int Position::evaluate() const {
    if (NNUE::is_loaded()) {
        return NNUE::evaluate(*this);
    } else {
        // Fallback to classical evaluation if NNUE not loaded
        const int PAWN_VALUE = 100;
        const int KNIGHT_VALUE = 320;
        const int BISHOP_VALUE = 330;
        const int ROOK_VALUE = 500;
        const int QUEEN_VALUE = 900;
        
        int score = 0;
        score += popcount(pieces[W_PAWN]) * PAWN_VALUE;
        score += popcount(pieces[W_KNIGHT]) * KNIGHT_VALUE;
        score += popcount(pieces[W_BISHOP]) * BISHOP_VALUE;
        score += popcount(pieces[W_ROOK]) * ROOK_VALUE;
        score += popcount(pieces[W_QUEEN]) * QUEEN_VALUE;
        
        score -= popcount(pieces[B_PAWN]) * PAWN_VALUE;
        score -= popcount(pieces[B_KNIGHT]) * KNIGHT_VALUE;
        score -= popcount(pieces[B_BISHOP]) * BISHOP_VALUE;
        score -= popcount(pieces[B_ROOK]) * ROOK_VALUE;
        score -= popcount(pieces[B_QUEEN]) * QUEEN_VALUE;
        
        return (side_to_move == WHITE) ? score : -score;
    }
}

// ==================== MAIN FUNCTION ====================

int main() {
    std::cout << "Duchess Chess Engine - Phase 1: Foundation\n";
    std::cout << "==========================================\n";
    
    // Initialize attack tables
    Attacks::init();
    
    // Test basic functionality
    Position pos;
    std::cout << "Starting position:\n";
    
    // Debug: Print piece bitboards
    for (int p = 1; p <= 12; p++) {
        std::cout << "Piece " << p << " (" << ".PNBRQKpnbrqk"[p] << "): ";
        Bitboard b = pos.get_pieces(p);
        for (int sq = 0; sq < 64; sq++) {
            if (b & SQ(sq)) {
                std::cout << sq << " ";
            }
        }
        std::cout << "\n";
    }
    
    pos.print();
    
    auto moves = pos.generate_moves();
    std::cout << "\nGenerated " << moves.size() << " pseudo-legal moves\n";
    
    // Debug castling
    std::cout << "Castling rights: " << pos.get_castling_rights() << "\n";
    std::cout << "White KS: " << (pos.get_castling_rights() & WHITE_KS) << "\n";
    std::cout << "White QS: " << (pos.get_castling_rights() & WHITE_QS) << "\n";
    
    // Test evaluation
    std::cout << "\nEvaluating position...\n";
    int eval = pos.evaluate();
    std::cout << "Classical evaluation: " << eval << " centipawns\n";
    
    // Test perft
    std::cout << "\nTesting perft (depth 3):\n";
    auto start = std::chrono::high_resolution_clock::now();
    uint64_t nodes = Perft::perft(pos, 3);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    std::cout << "Perft(3): " << nodes << " nodes in " << duration.count() << "ms\n";
    
    // Start UCI loop
    std::cout << "\nStarting UCI protocol...\n";
    UCI::start();
    
    return 0;
}