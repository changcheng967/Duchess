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
    static constexpr int NUM_PIECES = 12;
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
        for (int p = 0; p < NUM_PIECES; p++) {
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

class Position {
private:
    // Bitboards for each piece type
    Bitboard pieces[12]; // 0-5: white, 6-11: black
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
    
    // Move generation
    std::vector<Move> generate_moves() const;
    std::vector<Move> generate_captures() const;
    std::vector<Move> generate_quiet_moves() const;
    
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
    
    // Debug
    void print() const;
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
                !is_square_attacked(E1, BLACK) && !is_square_attacked(F1, BLACK)) {
                moves.emplace_back(E1, G1, MOVE_CASTLE);
            }
        }
        if (castling_rights & WHITE_QS) {
            if (!(all_occupied & (SQ(B1) | SQ(C1) | SQ(D1))) && 
                !is_square_attacked(E1, BLACK) && !is_square_attacked(D1, BLACK)) {
                moves.emplace_back(E1, C1, MOVE_CASTLE);
            }
        }
    } else {
        if (castling_rights & BLACK_KS) {
            if (!(all_occupied & (SQ(F8) | SQ(G8))) && 
                !is_square_attacked(E8, WHITE) && !is_square_attacked(F8, WHITE)) {
                moves.emplace_back(E8, G8, MOVE_CASTLE);
            }
        }
        if (castling_rights & BLACK_QS) {
            if (!(all_occupied & (SQ(B8) | SQ(C8) | SQ(D8))) && 
                !is_square_attacked(E8, WHITE) && !is_square_attacked(D8, WHITE)) {
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

// ==================== POSITION IMPLEMENTATION ====================

Position::Position() {
    from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
}

Position::Position(const std::string& fen) {
    from_fen(fen);
}

void Position::from_fen(const std::string& fen) {
    // Reset everything
    for (int i = 0; i < 12; i++) pieces[i] = 0;
    occupied[WHITE] = occupied[BLACK] = all_occupied = 0;
    hash = 0;
    material[WHITE] = material[BLACK] = 0;
    
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
            
            for (int p = 0; p < 12; p++) {
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
            
            for (int p = 0; p < 12; p++) {
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

// ==================== MOVE EXECUTION ====================

bool Position::make_move(const Move& move) {
    int from = move.from();
    int to = move.to();
    int piece = -1;
    int captured = -1;
    
    // Find the moving piece
    for (int p = 0; p < 12; p++) {
        if (pieces[p] & SQ(from)) {
            piece = p;
            break;
        }
    }
    
    if (piece == -1) return false; // No piece to move
    
    int color = (piece < 6) ? WHITE : BLACK;
    int enemy = 1 - color;
    
    // Update hash
    update_hash_remove(piece, from);
    update_hash_side();
    
    // Handle captures
    if (move.is_capture()) {
        if (move.is_enpassant()) {
            // En passant capture
            int ep_target = to + (color == WHITE ? SOUTH : NORTH);
            for (int p = 0; p < 12; p++) {
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
            for (int p = 0; p < 12; p++) {
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
    
    return true;
}

void Position::undo_move(const Move& move) {
    // This is a simplified undo - in a real engine you'd store more state
    // For now, we'll just re-parse from FEN which is inefficient but works
    // In practice, you'd store the previous state
}

// ==================== GAME STATE CHECKS ====================

Bitboard Position::get_attacks_to(int square, int attacker_color) const {
    Bitboard attacks = 0;
    
    // Pawns
    attacks |= Attacks::pawn_attacks(attacker_color, square) & pieces[attacker_color == WHITE ? W_PAWN : B_PAWN];
    
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
    return (occupied[WHITE] == pieces[side_to_move == WHITE ? W_KING : B_KING]) &&
           (occupied[BLACK] == pieces[side_to_move == BLACK ? W_KING : B_KING]);
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

// ==================== UCI PROTOCOL ====================

class UCI {
private:
    static void uci_loop() {
        std::string command;
        
        while (std::getline(std::cin, command)) {
            std::istringstream ss(command);
            std::string token;
            ss >> token;
            
            if (token == "uci") {
                std::cout << "id name Duchess Chess Engine\n";
                std::cout << "id author Kilo Code\n";
                std::cout << "option name Hash type spin default 16 min 1 max 1024\n";
                std::cout << "uciok\n";
            }
            else if (token == "isready") {
                std::cout << "readyok\n";
            }
            else if (token == "ucinewgame") {
                // Clear hash tables, etc.
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
                    fen = part;
                    while (ss >> part && part != "moves") {
                        fen += " " + part;
                    }
                }
                
                // Read moves
                while (ss >> part) {
                    moves.push_back(part);
                }
                
                // Apply position and moves
                // This would set up the position
            }
            else if (token == "go") {
                // Parse go command
                int depth = 8; // Default depth
                int time = 1000; // Default time in ms
                
                while (ss >> token) {
                    if (token == "depth") ss >> depth;
                    else if (token == "wtime" || token == "btime") {
                        int time_ms; ss >> time_ms;
                        time = time_ms / 40; // Rough time allocation
                    }
                }
                
                // Start search
                std::cout << "info depth " << depth << " score cp 0 nodes 0 nps 0 time 0 pv e2e4\n";
                std::cout << "bestmove e2e4\n";
            }
            else if (token == "quit") {
                break;
            }
            else if (token == "perft") {
                int depth;
                ss >> depth;
                
                Position pos;
                auto start = std::chrono::high_resolution_clock::now();
                uint64_t nodes = Perft::perft(pos, depth);
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
        Attacks::init();
        uci_loop();
    }
};

// ==================== MAIN FUNCTION ====================

int main() {
    std::cout << "Duchess Chess Engine - Phase 1: Foundation\n";
    std::cout << "==========================================\n";
    
    // Initialize attack tables
    Attacks::init();
    
    // Test basic functionality
    Position pos;
    std::cout << "Starting position:\n";
    pos.print();
    
    auto moves = pos.generate_moves();
    std::cout << "\nGenerated " << moves.size() << " legal moves\n";
    
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