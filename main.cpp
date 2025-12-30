// Duchess Chess Engine - Phase 1: Foundation
// Single file implementation with bitboard representation and move generation
//
// Features implemented:
// - Bitboard representation for efficient piece storage and operations
// - Zobrist hashing for position identification and transposition tables
// - Complete move generation with pseudo-legal and legal move filtering
// - NNUE neural network evaluation (with fallback to classical evaluation)
// - Alpha-beta search with iterative deepening
// - Quiescence search with capture ordering and delta pruning
// - Move ordering with MVV-LVA, killer moves, and history heuristic
// - Transposition table with proper memory management
// - UCI protocol implementation with comprehensive command support
// - Perft testing for move generation validation
// - Comprehensive search statistics and debugging output

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
    if (b == 0) return -1;
    unsigned long index;
    if (_BitScanForward64(&index, b)) return static_cast<int>(index);
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
    // Public access for search
    bool is_square_attacked(int square, int attacker_color) const;
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
        // This is the correct way to check legality
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
        // Extract piece type (0-5) and color (0-1)
        int piece_type = (piece - 1) % 6;  // 0=pawn, 1=knight, 2=bishop, 3=rook, 4=queen, 5=king
        int piece_color = (piece <= W_KING) ? WHITE : BLACK;
        
        // Flip perspective for black
        if (!white_perspective) {
            square ^= 56;       // Flip piece square vertically
            king_square ^= 56;  // Flip king square vertically
            piece_color ^= 1;   // Flip piece color
        }
        
        // Stockfish HalfKAv2_hm formula:
        // index = piece_color * 6 * 64 + piece_type * 64 + square
        return piece_color * 384 + piece_type * 64 + square;
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
        
        // Scale output to centipawns (CORRECTED: 400 → 100)
        int eval = (output * 100) / (127 * OUTPUT_SCALE);
        
        // Apply game phase scaling
        int total_material = 0;
        for (int p = W_PAWN; p <= B_QUEEN; p++) {
            if (p != W_KING && p != B_KING) {
                total_material += popcount(pos.get_pieces(p));
            }
        }
        
        // Gradual endgame scaling (starts earlier, more gradual)
        if (total_material < 24) {
            eval = (eval * (total_material + 24)) / 48;
        }
        
        // Clamp to prevent extreme values
        eval = std::max(-3000, std::min(3000, eval));
        
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

// Transposition Table with better memory management
struct TTEntry {
    Bitboard hash = 0;
    int depth = 0;
    int score = 0;
    int flag = 0;  // 0=exact, 1=lower_bound, 2=upper_bound
    Move best_move;
};

// Memory-efficient transposition table
static std::vector<TTEntry> transposition_table;
static constexpr int EXACT = 0;
static constexpr int LOWER_BOUND = 1;
static constexpr int UPPER_BOUND = 2;

// Initialize transposition table with proper sizing
void init_transposition_table(int hash_mb = 16) {
    // Calculate size based on memory (rough estimate: 16 bytes per entry)
    size_t entries = (hash_mb * 1024 * 1024) / 16;
    // Round to power of 2 for better hashing
    size_t size = 1;
    while (size < entries) size <<= 1;
    transposition_table.resize(size);
    std::fill(transposition_table.begin(), transposition_table.end(), TTEntry{});
}

// Better hash index calculation
inline size_t tt_index(Bitboard hash) {
    return hash & (transposition_table.size() - 1);
}

// Killer moves for move ordering
static Move killer_moves[2][100];  // 2 killers per depth

// History heuristic for move ordering
static int history_table[13][64][64];

// Principal Variation table for move ordering
static Move pv_table[100][100];  // PV at each ply
static int pv_length[100];       // Length at each ply

// Countermove heuristic
static Move countermove_table[13][64];  // [piece][to_square]

// Continuation history
static int continuation_history[13][64][13][64];

// Capture history
static int capture_history[13][64][13];  // [attacker][to][victim]

// Global node counter
static uint64_t nodes_searched = 0;

// Global search control
static bool search_stopped = false;

// Move scoring for ordering with improved heuristics
static int score_move(const Position& pos, const Move& move, int depth, const Move& tt_move = Move(), int ply = 0, const Move& prev_move = Move()) {
    int score = 0;
    
    // PV move gets highest priority (after TT move)
    if (ply < pv_length[0] && move == pv_table[0][ply]) {
        return 20000000;  // Highest priority after TT
    }
    
    // TT move gets highest priority
    if (move == tt_move) {
        return 10000000;  // Highest score
    }
    
    // Countermove bonus
    if (ply > 0 && prev_move.data != 0) {
        int prev_piece = pos.get_piece_at(prev_move.to());
        if (move == countermove_table[prev_piece][prev_move.to()]) {
            score += 600000;
        }
    }
    
    // MVV-LVA (Most Valuable Victim - Least Valuable Attacker)
    if (move.is_capture()) {
        int victim = pos.get_piece_at(move.to());
        int attacker = pos.get_piece_at(move.from());
        
        // Piece values: P=100, N=320, B=330, R=500, Q=900, K=0 (can't capture king)
        int victim_value = 0;
        if (victim >= 1 && victim <= 12) {
            victim_value = (victim == W_PAWN || victim == B_PAWN) ? 100 :
                          (victim == W_KNIGHT || victim == B_KNIGHT) ? 320 :
                          (victim == W_BISHOP || victim == B_BISHOP) ? 330 :
                          (victim == W_ROOK || victim == B_ROOK) ? 500 :
                          (victim == W_QUEEN || victim == B_QUEEN) ? 900 : 0;
        }
        
        int attacker_value = 0;
        if (attacker >= 1 && attacker <= 12) {
            attacker_value = (attacker == W_PAWN || attacker == B_PAWN) ? 100 :
                            (attacker == W_KNIGHT || attacker == B_KNIGHT) ? 320 :
                            (attacker == W_BISHOP || attacker == B_BISHOP) ? 330 :
                            (attacker == W_ROOK || attacker == B_ROOK) ? 500 :
                            (attacker == W_QUEEN || attacker == B_QUEEN) ? 900 : 0;
        }
        
        // Improved MVV-LVA scoring
        score += 1000000 + (victim_value * 10) - attacker_value;
        
        // Capture history bonus
        if (attacker >= 1 && attacker <= 12 && victim >= 1 && victim <= 12) {
            score += capture_history[attacker][move.to()][victim];
        }
    }
    
    // Promotion bonus with piece preference
    if (move.is_promotion()) {
        int promo_piece = move.promotion();
        int promo_bonus = 0;
        switch (promo_piece) {
            case W_QUEEN: case B_QUEEN: promo_bonus = 900000; break;
            case W_ROOK: case B_ROOK: promo_bonus = 500000; break;
            case W_BISHOP: case B_BISHOP: promo_bonus = 330000; break;
            case W_KNIGHT: case B_KNIGHT: promo_bonus = 320000; break;
        }
        score += promo_bonus;
    }
    
    // Killer move bonus (improved ordering)
    if (depth < 100) {
        if (killer_moves[0][depth] == move) score += 800000;
        else if (killer_moves[1][depth] == move) score += 700000;
    }
    
    // History heuristic bonus (scaled by depth)
    int piece = pos.get_piece_at(move.from());
    if (piece >= 1 && piece <= 12) {
        score += history_table[piece][move.from()][move.to()] * (depth + 1);
        
        // Continuation history bonus
        if (ply > 0 && prev_move.data != 0) {
            int prev_piece = pos.get_piece_at(prev_move.to());
            score += continuation_history[prev_piece][prev_move.to()][piece][move.to()];
        }
    }
    
    // Center control bonus (improved)
    int to_file = file_of(move.to());
    int to_rank = rank_of(move.to());
    if ((to_file >= 2 && to_file <= 5) && (to_rank >= 2 && to_rank <= 5)) {
        score += 15000; // Increased bonus for central squares
    }
    
    // Development bonus for minor pieces (early game)
    if (depth >= 3) {
        int piece_type = piece;
        if ((piece_type == W_KNIGHT || piece_type == B_KNIGHT ||
             piece_type == W_BISHOP || piece_type == B_BISHOP) &&
            (to_rank >= 1 && to_rank <= 6)) {
            score += 5000;
        }
    }
    
    // Pawn push bonus (avoid repetitive scoring)
    if (piece == W_PAWN || piece == B_PAWN) {
        if (!move.is_capture() && !move.is_promotion()) {
            score += 1000; // Small bonus for quiet pawn moves
        }
    }
    
    return score;
}

// Helper function to evaluate position
static int evaluate_position(const Position& pos) {
    return pos.evaluate();
}

// Quiescence search with improved capture ordering and delta pruning
static int quiescence(Position& pos, int alpha, int beta, int ply = 0) {
    // Limit quiescence depth to prevent stack overflow
    if (ply >= 8) {
        return evaluate_position(pos);
    }
    
    nodes_searched++;
    
    // Stand-pat (null move)
    int stand_pat = evaluate_position(pos);
    if (stand_pat >= beta) return beta;
    if (alpha < stand_pat) alpha = stand_pat;
    
    // Delta pruning: if even capturing the most valuable piece won't help
    const int MAX_MATERIAL_GAIN = 1000; // Queen value + margin
    if (stand_pat + MAX_MATERIAL_GAIN < alpha) {
        return alpha;
    }
    
    // Generate captures and checks
    auto captures = pos.generate_captures();
    
    // Score and sort captures using improved move ordering
    std::vector<std::pair<Move, int>> scored_moves;
    scored_moves.reserve(captures.size());
    
    for (const auto& move : captures) {
        // Skip bad captures (losing trades)
        if (move.is_capture()) {
            int victim = pos.get_piece_at(move.to());
            int attacker = pos.get_piece_at(move.from());
            
            // Simple SEE (Static Exchange Evaluation) approximation
            int victim_value = (victim == W_PAWN || victim == B_PAWN) ? 100 :
                              (victim == W_KNIGHT || victim == B_KNIGHT) ? 320 :
                              (victim == W_BISHOP || victim == B_BISHOP) ? 330 :
                              (victim == W_ROOK || victim == B_ROOK) ? 500 :
                              (victim == W_QUEEN || victim == B_QUEEN) ? 900 : 0;
            int attacker_value = (attacker == W_PAWN || attacker == B_PAWN) ? 100 :
                                (attacker == W_KNIGHT || attacker == B_KNIGHT) ? 320 :
                                (attacker == W_BISHOP || attacker == B_BISHOP) ? 330 :
                                (attacker == W_ROOK || attacker == B_ROOK) ? 500 :
                                (attacker == W_QUEEN || attacker == B_QUEEN) ? 900 : 0;
            
            // Skip obviously losing captures (simplified)
            if (victim_value < attacker_value && ply > 0) {
                continue;
            }
        }
        
        scored_moves.emplace_back(move, score_move(pos, move, 0, Move(), ply));
    }
    
    // Sort captures by score (best first)
    std::sort(scored_moves.begin(), scored_moves.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });
    
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
        
        // Late move pruning in quiescence (skip low-score moves if we already have good result)
        if (ply >= 4 && scored_move.second < 100000) {
            continue;
        }
        
        int score_after = -quiescence(temp, -beta, -alpha, ply + 1);
        
        if (score_after >= beta) return beta;
        if (score_after > alpha) alpha = score_after;
    }
    
    return alpha;
}

// Alpha-beta search with optimizations
static int alpha_beta(Position& pos, int depth, int alpha, int beta, int ply = 0) {
    nodes_searched++;
    
    // Reverse Futility Pruning
    if (depth <= 6 && !pos.is_check() && alpha > -20000 && beta < 20000) {
        int eval = evaluate_position(pos);
        int rfp_margin = 80 * depth;
        
        if (eval - rfp_margin >= beta) {
            return eval;  // Position too good, opponent won't allow
        }
    }
    
    // Razoring
    if (depth <= 3 && !pos.is_check() && alpha > -20000 && beta < 20000) {
        int eval = evaluate_position(pos);
        int razor_margin = 300 + 100 * depth;
        
        if (eval + razor_margin < alpha) {
            // Position very bad, go straight to quiescence
            int q_score = quiescence(pos, alpha, beta, ply);
            if (q_score <= alpha) {
                return q_score;
            }
        }
    }
    
    // Transposition table lookup
    Bitboard hash = pos.get_hash();
    size_t tt_idx = tt_index(hash);
    TTEntry& entry = transposition_table[tt_idx];
    
    // Get TT move for better ordering
    Move tt_move = (entry.hash == hash && entry.depth > 0) ? entry.best_move : Move();
    
    if (entry.hash == hash && entry.depth >= depth) {
        if (entry.flag == EXACT) return entry.score;
        if (entry.flag == LOWER_BOUND) alpha = std::max(alpha, entry.score);
        if (entry.flag == UPPER_BOUND) beta = std::min(beta, entry.score);
        if (alpha >= beta) return entry.score;
    }
    
    // Internal Iterative Deepening (IID)
    if (depth >= 4 && !tt_move.data && !pos.is_check()) {
        int iid_depth = depth - 2;
        alpha_beta(pos, iid_depth, alpha, beta, ply);
        
        // Reload TT entry
        TTEntry& new_entry = transposition_table[tt_idx];
        if (new_entry.hash == hash) {
            tt_move = new_entry.best_move;
        }
    }
    
    // Null Move Pruning
    if (depth >= 3 && !pos.is_check()) {
        // Check if we have non-pawn material
        int our_color = pos.get_side_to_move();
        Bitboard non_pawns = (our_color == WHITE) ?
            (pos.get_pieces(W_KNIGHT) | pos.get_pieces(W_BISHOP) |
             pos.get_pieces(W_ROOK) | pos.get_pieces(W_QUEEN)) :
            (pos.get_pieces(B_KNIGHT) | pos.get_pieces(B_BISHOP) |
             pos.get_pieces(B_ROOK) | pos.get_pieces(B_QUEEN));
        
        if (non_pawns != 0) {
            // Make null move (pass turn to opponent)
            Position null_pos = pos;
            null_pos.side_to_move ^= 1;
            null_pos.en_passant_square = NO_SQ;
            null_pos.hash ^= zobrist.hash_side();
            if (pos.get_en_passant_square() != NO_SQ) {
                null_pos.hash ^= zobrist.hash_enpassant(file_of(pos.get_en_passant_square()));
            }
            
            // Reduction factor (R)
            int R = (depth > 6) ? 3 : 2;
            
            // Search with reduced depth
            int null_score = -alpha_beta(null_pos, depth - 1 - R, -beta, -beta + 1, ply + 1);
            
            // If null move fails high, we can prune
            if (null_score >= beta) {
                // Don't return mate scores from null move
                if (null_score > 20000) null_score = beta;
                return null_score;
            }
        }
    }
    
    if (depth == 0) {
        return quiescence(pos, alpha, beta, ply);
    }
    
    // Use pseudo-legal moves to avoid recursion
    auto moves = pos.generate_moves();
    if (moves.empty()) {
        if (pos.is_check()) {
            return -30000 + (100 - depth); // Prefer faster mates
        }
        return 0; // Stalemate
    }
    
    // Futility Pruning setup (calculate once)
    bool in_pv = (alpha != beta - 1);
    bool can_prune = false;
    
    if (depth <= 3 && !pos.is_check() && !in_pv) {
        int eval = evaluate_position(pos);
        int futility_margin = 100 + 150 * depth;
        can_prune = (eval + futility_margin <= alpha);
    }
    
    // Score and sort moves (ONLY ONCE!)
    std::vector<std::pair<Move, int>> scored_moves;
    scored_moves.reserve(moves.size());
    
    for (const auto& move : moves) {
        // Apply futility pruning here
        if (can_prune && !move.is_capture() && !move.is_promotion()) {
            continue;  // Skip futile quiet moves
        }
        
        scored_moves.emplace_back(move, score_move(pos, move, depth, tt_move, ply));
    }
    
    std::sort(scored_moves.begin(), scored_moves.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });
    
    int best_score = -30000;
    Move best_move;
    int legal_moves = 0;
    
    int our_color = pos.get_side_to_move();
    int king_piece = (our_color == WHITE) ? W_KING : B_KING;
    
    int moves_searched = 0;
    
    // Moves already scored and sorted above
    
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
        
        moves_searched++;
        legal_moves++;
        int score_after;
        
        // ===== LATE MOVE REDUCTIONS =====
        if (moves_searched > 3 && depth >= 3 &&
            !move.is_capture() && !move.is_promotion() &&
            !pos.is_check() && !temp.is_check()) {
            
            // Calculate reduction
            int reduction = 1;
            if (moves_searched > 6) reduction = 2;
            if (moves_searched > 12) reduction = 3;
            if (depth > 6) reduction++;  // Reduce more at high depths
            
            // Ensure we don't reduce below depth 1
            reduction = std::min(reduction, depth - 2);
            
            // Search with reduced depth
            score_after = -alpha_beta(temp, depth - 1 - reduction, -alpha - 1, -alpha, ply + 1);
            
            // If it fails high, re-search at full depth
            if (score_after > alpha) {
                score_after = -alpha_beta(temp, depth - 1, -beta, -alpha, ply + 1);
            }
        } else if (moves_searched == 1) {
            // Full window search for first move
            score_after = -alpha_beta(temp, depth - 1, -beta, -alpha, ply + 1);
        } else {
            // Null window search (PVS)
            score_after = -alpha_beta(temp, depth - 1, -alpha - 1, -alpha, ply + 1);
            
            // Re-search if it beats alpha
            if (score_after > alpha && score_after < beta) {
                score_after = -alpha_beta(temp, depth - 1, -beta, -alpha, ply + 1);
            }
        }
        
        if (score_after > best_score) {
            best_score = score_after;
            best_move = move;
            
            // Store PV
            if (ply < 99) {
                pv_table[ply][0] = move;
                for (int i = 0; i < pv_length[ply + 1]; i++) {
                    pv_table[ply][i + 1] = pv_table[ply + 1][i];
                }
                pv_length[ply] = pv_length[ply + 1] + 1;
            }
            
            if (score_after > alpha) {
                alpha = score_after;
            }
        }
        
        if (alpha >= beta) {
            // Store killer move
            if (depth < 100 && !move.is_capture()) {
                killer_moves[1][depth] = killer_moves[0][depth];
                killer_moves[0][depth] = move;
                
                // Update countermove
                if (ply > 0) {
                    int prev_piece = pos.get_piece_at(pv_table[0][ply - 1].to());
                    countermove_table[prev_piece][pv_table[0][ply - 1].to()] = move;
                }
            }
            
            // Update history heuristic
            int piece = pos.get_piece_at(move.from());
            if (piece >= 1 && piece <= 12) {
                history_table[piece][move.from()][move.to()] += depth * depth;
                
                // Update continuation history
                if (ply > 0) {
                    int prev_piece = pos.get_piece_at(pv_table[0][ply - 1].to());
                    continuation_history[prev_piece][pv_table[0][ply - 1].to()][piece][move.to()] += depth * depth;
                }
                
                // Update capture history
                if (move.is_capture()) {
                    int victim = pos.get_piece_at(move.to());
                    if (victim >= 1 && victim <= 12) {
                        capture_history[piece][move.to()][victim] += depth * depth;
                    }
                }
                
                // Age history table to prevent overflow
                if (history_table[piece][move.from()][move.to()] > 10000) {
                    for (int p = 1; p <= 12; p++) {
                        for (int f = 0; f < 64; f++) {
                            for (int t = 0; t < 64; t++) {
                                history_table[p][f][t] /= 2;
                            }
                        }
                    }
                }
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
    
    // Store in transposition table (only if better)
    if (entry.hash != hash || entry.depth <= depth) {
        entry.hash = hash;
        entry.depth = depth;
        entry.score = best_score;
        entry.best_move = best_move;
        
        if (best_score <= alpha) entry.flag = UPPER_BOUND;
        else if (best_score >= beta) entry.flag = LOWER_BOUND;
        else entry.flag = EXACT;
    }
    
    return best_score;
}

// ==================== SEARCH ALGORITHM ====================

class Search {
private:
    // Use global nodes_searched, no class static needed
    
    static std::pair<Move, int> find_best_move(Position& pos, int depth, int prev_score = 0) {
        auto moves = pos.generate_legal_moves();
        if (moves.empty()) {
            return {Move(), -1000000};
        }
        
        Move best_move = moves[0];
        int best_score = -1000000;
        
        // Aspiration windows for depth > 3
        if (depth > 3 && prev_score != 0) {
            int window = 50;
            int alpha = prev_score - window;
            int beta = prev_score + window;
            
            while (true) {
                best_score = -1000000;
                
                for (const auto& move : moves) {
                    Position temp_pos = pos;
                    if (!temp_pos.make_move(move)) continue;
                    
                    int score = -alpha_beta(temp_pos, depth - 1, -beta, -alpha, 1);
                    
                    if (score > best_score) {
                        best_score = score;
                        best_move = move;
                    }
                }
                
                // Check if we need to re-search
                if (best_score <= alpha) {
                    alpha -= window;
                    window *= 2;
                } else if (best_score >= beta) {
                    beta += window;
                    window *= 2;
                } else {
                    break;  // Success!
                }
                
                // Fallback to full window if aspiration fails
                if (window > 500) {
                    alpha = -1000000;
                    beta = 1000000;
                }
            }
        } else {
            // Full window for shallow depths
            for (const auto& move : moves) {
                Position temp_pos = pos;
                if (!temp_pos.make_move(move)) continue;
                
                int score = -alpha_beta(temp_pos, depth - 1, -1000000, 1000000, 1);
                
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
        }
        
        return {best_move, best_score};
    }

public:
    // Initialize search components
    static void init() {
        init_transposition_table(16); // 16MB default
        // Clear killer moves and history
        std::fill_n(killer_moves[0], 100, Move());
        std::fill_n(killer_moves[1], 100, Move());
        std::fill_n(&history_table[0][0][0], 13 * 64 * 64, 0);
    }
    
    static void iterative_deepening(Position& pos, int max_depth, int time_limit_ms) {
        auto start_time = std::chrono::high_resolution_clock::now();
        
        Move best_move;
        int best_score = 0;
        int prev_score = 0;
        ::nodes_searched = 0;
        
        uint64_t total_time = 0;
        int search_depth = 0;
        
        for (int depth = 1; depth <= max_depth; depth++) {
            // Check if GUI requested stop
            if (search_stopped) {
                search_stopped = false;  // Reset flag
                break;
            }
            
            auto current_time = std::chrono::high_resolution_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                current_time - start_time).count();
            
            // FIX: Only check time limit if it's not infinite (-1)
            if (time_limit_ms > 0 && elapsed >= time_limit_ms) {
                break;
            }
            
            uint64_t depth_nodes_start = nodes_searched;
            
            auto result = find_best_move(pos, depth, prev_score);
            Move current_best = result.first;
            int current_score = result.second;
            
            prev_score = current_score;
            
            uint64_t depth_nodes = ::nodes_searched - depth_nodes_start;
            total_time = elapsed;
            search_depth = depth;
            
            uint64_t nps = (elapsed > 0) ? (::nodes_searched * 1000 / elapsed) : 0;
            
            std::cout << "info depth " << depth
                      << " score cp " << current_score
                      << " time " << elapsed
                      << " nodes " << ::nodes_searched
                      << " nps " << nps
                      << " pv ";
            
            if (current_best.data != 0) {
                char from_file = 'a' + file_of(current_best.from());
                char from_rank = '1' + rank_of(current_best.from());
                char to_file = 'a' + file_of(current_best.to());
                char to_rank = '1' + rank_of(current_best.to());
                std::cout << from_file << from_rank << to_file << to_rank;
                
                if (current_best.is_promotion()) {
                    char promo = "  nbrq"[current_best.promotion() % 6];
                    std::cout << promo;
                }
            }
            std::cout << "\n";
            
            best_move = current_best;
            best_score = current_score;
            
            // For infinite search, continue until max_depth
            // GUI will send "stop" command to interrupt
        }
        
        std::cout << "info string Search completed: depth=" << search_depth
                  << " nodes=" << ::nodes_searched
                  << " time=" << total_time << "ms"
                  << " nps=" << (total_time > 0 ? (nodes_searched * 1000 / total_time) : 0) << "\n";
        
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

// Node counter is global, no class static initialization needed

// ==================== UCI PROTOCOL ====================

class UCI {
private:
    static Position current_position;  // Global position state
    static int hash_size_mb;           // Hash table size in MB
    
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
                std::cout << "option name Ponder type check default false\n";
                std::cout << "option name MultiPV type spin default 1 min 1 max 10\n";
                std::cout << "uciok\n";
            }
            else if (token == "isready") {
                std::cout << "readyok\n";
            }
            else if (token == "setoption") {
                std::string name, value;
                ss >> token; // "name"
                if (token == "name") {
                    ss >> name;
                    if (name == "Hash") {
                        ss >> token; // "value"
                        if (token == "value") {
                            ss >> hash_size_mb;
                            // Adjust transposition table size based on hash setting
                            size_t new_size = 1ULL << (10 + hash_size_mb / 4); // Rough approximation
                            transposition_table.resize(new_size);
                            std::cout << "info string Hash table resized to " << hash_size_mb << "MB\n";
                        }
                    }
                }
            }
            else if (token == "ucinewgame") {
                // Reset to starting position
                current_position.from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
                // Clear transposition table
                std::fill(transposition_table.begin(), transposition_table.end(), TTEntry{});
                // Clear killer moves and history
                std::fill_n(killer_moves[0], 100, Move());
                std::fill_n(killer_moves[1], 100, Move());
                std::fill_n(&history_table[0][0][0], 13 * 64 * 64, 0);
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
                    // Parse move string (e.g., "e2e4" or "e7e8q")
                    if (move_str.length() >= 4) {
                        int from_file = move_str[0] - 'a';
                        int from_rank = move_str[1] - '1';
                        int to_file = move_str[2] - 'a';
                        int to_rank = move_str[3] - '1';
                        
                        int from = from_file + from_rank * 8;
                        int to = to_file + to_rank * 8;
                        
                        Move move(from, to);
                        
                        // Handle promotion
                        if (move_str.length() >= 5) {
                            char promo = move_str[4];
                            int promo_piece = W_QUEEN;  // Default
                            switch (promo) {
                                case 'q': promo_piece = (current_position.get_side_to_move() == WHITE) ? W_QUEEN : B_QUEEN; break;
                                case 'r': promo_piece = (current_position.get_side_to_move() == WHITE) ? W_ROOK : B_ROOK; break;
                                case 'b': promo_piece = (current_position.get_side_to_move() == WHITE) ? W_BISHOP : B_BISHOP; break;
                                case 'n': promo_piece = (current_position.get_side_to_move() == WHITE) ? W_KNIGHT : B_KNIGHT; break;
                            }
                            move = Move(from, to, MOVE_PROMOTION | promo_piece);
                        }
                        
                        current_position.make_move(move);
                    }
                }
            }
            else if (token == "go") {
                search_stopped = false;  // Reset flag for new search
                
                // Parse go command
                int depth = 8; // Default depth
                int time = 900; // Default time in ms (1 second limit)
                int wtime = -1, btime = -1, winc = 0, binc = 0;
                bool ponder = false;
                int multipv = 1;
                
                while (ss >> token) {
                    if (token == "depth") ss >> depth;
                    else if (token == "wtime") ss >> wtime;
                    else if (token == "btime") ss >> btime;
                    else if (token == "winc") ss >> winc;
                    else if (token == "binc") ss >> binc;
                    else if (token == "ponder") ponder = true;
                    else if (token == "multipv") ss >> multipv;
                    else if (token == "infinite") time = -1; // Infinite search
                }
                
                // Time management
                if (wtime > 0 || btime > 0) {
                    int time_ms = (current_position.get_side_to_move() == WHITE) ? wtime : btime;
                    int inc_ms = (current_position.get_side_to_move() == WHITE) ? winc : binc;
                    
                    // Allocate time based on remaining time and increment
                    // Simple time allocation: use 10% of remaining time + 50% of increment
                    time = (time_ms / 10) + (inc_ms / 2);
                    
                    // Cap time to prevent excessive thinking
                    if (time > 30000) time = 30000; // 30 seconds max
                    if (time < 100) time = 100;     // 100ms minimum
                }
                
                std::cout << "info string Time management: depth=" << depth
                          << " time=" << time << "ms ponder=" << (ponder ? "true" : "false") << "\n";
                
                // Start search with iterative deepening using current position
                Search::iterative_deepening(current_position, depth, time);
            }
            else if (token == "quit") {
                break;
            }
            else if (token == "stop") {
                search_stopped = true;  // Set flag
                std::cout << "info string Search stopped\n";
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
            else if (token == "debug") {
                // Debug command to show current position
                current_position.print();
                std::cout << "info string Position hash: 0x" << std::hex << current_position.get_hash() << std::dec << "\n";
                std::cout << "info string Castling rights: " << current_position.get_castling_rights() << "\n";
                std::cout << "info string En passant: " << current_position.get_en_passant_square() << "\n";
            }
            else if (token == "eval") {
                // Evaluate current position
                int score = current_position.evaluate();
                std::cout << "info string Evaluation: " << score << " centipawns\n";
            }
            else if (token == "moves") {
                // List all legal moves
                auto moves = current_position.generate_legal_moves();
                std::cout << "info string Legal moves (" << moves.size() << "): ";
                for (const auto& move : moves) {
                    char from_file = 'a' + file_of(move.from());
                    char from_rank = '1' + rank_of(move.from());
                    char to_file = 'a' + file_of(move.to());
                    char to_rank = '1' + rank_of(move.to());
                    std::cout << from_file << from_rank << to_file << to_rank << " ";
                }
                std::cout << "\n";
            }
            else if (token == "help") {
                std::cout << "Available commands:\n";
                std::cout << "  uci              - Initialize UCI protocol\n";
                std::cout << "  isready          - Check if engine is ready\n";
                std::cout << "  ucinewgame       - Reset to starting position\n";
                std::cout << "  position fen ... - Set position from FEN\n";
                std::cout << "  position startpos [moves ...] - Set starting position with moves\n";
                std::cout << "  go [depth N] [wtime N] [btime N] - Start search\n";
                std::cout << "  perft N          - Run perft test to depth N\n";
                std::cout << "  test             - Run perft test suite\n";
                std::cout << "  debug            - Show debug information\n";
                std::cout << "  eval             - Evaluate current position\n";
                std::cout << "  moves            - List all legal moves\n";
                std::cout << "  quit             - Exit engine\n";
            }
            else if (!token.empty() && token[0] != '#') {
                // Unknown command
                std::cout << "info string Unknown command: " << token << "\n";
                std::cout << "info string Type 'help' for available commands\n";
            }
            else if (token == "test") {
                Perft::run_test_suite();
            }
            else if (token == "debug") {
                // Debug command to show current position
                current_position.print();
                std::cout << "info string Position hash: 0x" << std::hex << current_position.get_hash() << std::dec << "\n";
                std::cout << "info string Castling rights: " << current_position.get_castling_rights() << "\n";
                std::cout << "info string En passant: " << current_position.get_en_passant_square() << "\n";
            }
            else if (token == "eval") {
                // Evaluate current position
                int score = current_position.evaluate();
                std::cout << "info string Evaluation: " << score << " centipawns\n";
            }
            else if (token == "moves") {
                // List all legal moves
                auto moves = current_position.generate_legal_moves();
                std::cout << "info string Legal moves (" << moves.size() << "): ";
                for (const auto& move : moves) {
                    char from_file = 'a' + file_of(move.from());
                    char from_rank = '1' + rank_of(move.from());
                    char to_file = 'a' + file_of(move.to());
                    char to_rank = '1' + rank_of(move.to());
                    std::cout << from_file << from_rank << to_file << to_rank << " ";
                }
                std::cout << "\n";
            }
            else if (token == "help") {
                std::cout << "Available commands:\n";
                std::cout << "  uci              - Initialize UCI protocol\n";
                std::cout << "  isready          - Check if engine is ready\n";
                std::cout << "  ucinewgame       - Reset to starting position\n";
                std::cout << "  position fen ... - Set position from FEN\n";
                std::cout << "  position startpos [moves ...] - Set starting position with moves\n";
                std::cout << "  go [depth N] [wtime N] [btime N] - Start search\n";
                std::cout << "  perft N          - Run perft test to depth N\n";
                std::cout << "  test             - Run perft test suite\n";
                std::cout << "  debug            - Show debug information\n";
                std::cout << "  eval             - Evaluate current position\n";
                std::cout << "  moves            - List all legal moves\n";
                std::cout << "  quit             - Exit engine\n";
            }
            else if (!token.empty() && token[0] != '#') {
                // Unknown command
                std::cout << "info string Unknown command: " << token << "\n";
                std::cout << "info string Type 'help' for available commands\n";
            }
        }
    }
    
public:
    static void start() {
        hash_size_mb = 16; // Default hash size
        
        // Initialize starting position
        current_position.from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
        
        Attacks::init();
        NNUE::init(); // Initialize NNUE
        
        std::cout << "info string Duchess Chess Engine initialized\n";
        std::cout << "info string Type 'help' for available commands\n";
        
        uci_loop();
    }
};

// Initialize static members
Position UCI::current_position;
int UCI::hash_size_mb = 16;

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
    
    // Initialize all components
    Attacks::init();
    Search::init(); // Initialize search components
    NNUE::init();   // Initialize NNUE evaluation
    
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
    std::cout << "Evaluation: " << eval << " centipawns\n";
    
    // Test NNUE feature indexing (debug)
    if (NNUE::is_loaded()) {
        std::cout << "NNUE loaded successfully - testing feature indexing...\n";
        
        // Test a few pieces
        int white_pawn_sq = 8;  // a2
        int white_king_sq = 4;  // e1
        int black_king_sq = 60; // e8
        
        int idx1 = NNUE::get_feature_index(W_PAWN, white_pawn_sq, white_king_sq, true);
        int idx2 = NNUE::get_feature_index(B_PAWN, white_pawn_sq, black_king_sq, false);
        
        std::cout << "White pawn at a2 (white perspective): " << idx1 << "\n";
        std::cout << "Black pawn at a2 (black perspective): " << idx2 << "\n";
        std::cout << "Feature indices should be different for different perspectives\n";
    } else {
        std::cout << "NNUE not loaded - using classical evaluation\n";
    }
    
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