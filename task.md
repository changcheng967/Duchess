# Duchess Chess Engine - Development Plan

## Project Overview
**Goal**: Build a competitive chess engine in a single C++ file that uses Stockfish's NNUE for evaluation while implementing all other components from scratch.

**Target Specifications**:
- Maximum ELO with no training
- No opening/closing books
- 1-second move time limit
- Single C++ file implementation
- Uses Stockfish NNUE evaluation
- Windows + Visual Studio 18 2026

---

## Architecture Overview

```
Duchess Engine Components:
├── Board Representation (Bitboards)
├── Move Generation
├── Search Algorithm (Alpha-Beta + Extensions)
├── Time Management
├── UCI Protocol Handler
└── NNUE Evaluation (from Stockfish)
```

---

## Phase 1: Foundation (Week 1-2)

### 1.1 Board Representation
- **Bitboard structure**: 12 bitboards (6 piece types × 2 colors)
- **Zobrist hashing**: For transposition table
- **Move structure**: Compact 16-bit or 32-bit move encoding
  - From square (6 bits)
  - To square (6 bits)
  - Promotion piece (3 bits)
  - Flags (castling, en passant, capture)

### 1.2 Move Generation
- **Pseudo-legal move generation** using magic bitboards
- **Legal move validation** (king safety check)
- **Move ordering** (critical for search efficiency):
  1. Hash move from transposition table
  2. Winning captures (MVV-LVA)
  3. Killer moves (2 per ply)
  4. History heuristic
  5. Quiet moves

### 1.3 Core Data Structures
```cpp
struct Position {
    uint64_t pieces[12];  // Bitboards for each piece type
    uint64_t occupied[2]; // White/Black occupancy
    uint64_t all_occupied;
    int side_to_move;
    int castling_rights;
    int en_passant_square;
    int halfmove_clock;
    uint64_t hash;
};
```

---

## Phase 2: NNUE Integration (Week 2-3)

### 2.1 Extract Stockfish NNUE
- Download Stockfish source code
- Extract NNUE evaluation files:
  - `nnue/nnue_architecture.h`
  - `nnue/nnue_evaluator.h`
  - `nnue/layers/*.h`
  - Pre-trained network file (`.nnue` file)

### 2.2 Minimal NNUE Wrapper
- **Incremental update**: Update accumulator after each move
- **Efficient refresh**: Only when necessary (e.g., after null move)
- **Network structure**: Typically 768→512×2→1 (HalfKAv2)
- **Input features**: King position + piece positions

### 2.3 Integration Strategy
```cpp
// Embed NNUE code directly in single file
namespace NNUE {
    // Paste Stockfish NNUE implementation
    // Simplify to remove dependencies
}

int evaluate(Position& pos) {
    return NNUE::evaluate(pos);
}
```

---

## Phase 3: Search Algorithm (Week 3-5)

### 3.1 Core Search: Negamax with Alpha-Beta
```cpp
int negamax(Position& pos, int depth, int alpha, int beta) {
    if (depth == 0) return quiescence(pos, alpha, beta);
    
    // Transposition table probe
    // Move generation and ordering
    // Loop through moves
    // Recursion with alpha-beta pruning
    
    return best_score;
}
```

### 3.2 Quiescence Search
- Search only captures and checks
- Prevent horizon effect
- Stand-pat pruning
- Delta pruning for efficiency

### 3.3 Search Enhancements (Priority Order)
1. **Transposition Table** (16MB-1GB)
   - Store: hash, depth, score, best move, node type
   - Replacement scheme: Always replace or depth-preferred

2. **Iterative Deepening**
   - Start from depth 1, increment until time runs out
   - Provides move ordering for next iteration

3. **Aspiration Windows**
   - Narrow alpha-beta window around previous score
   - Re-search if fails

4. **Null Move Pruning** (R=2 or 3)
   - Skip turn and search reduced depth
   - If still fails high, prune branch

5. **Late Move Reductions (LMR)**
   - Reduce depth for moves late in move list
   - Re-search at full depth if score improves

6. **Futility Pruning**
   - Skip quiet moves when far from alpha in frontier nodes

7. **Check Extensions**
   - Extend search by 1 ply when in check

8. **Singular Extensions**
   - Extend when one move is significantly better

---

## Phase 4: Time Management (Week 5)

### 4.1 Time Allocation Strategy
```cpp
// For 1-second constraint
int allocated_time = 900; // 900ms, leave 100ms buffer

// Dynamic adjustment:
// - Extend if best move changes
// - Extend if score drops significantly
// - Stop early if mate found or overwhelming advantage
```

### 4.2 Search Control
- Check time every 1024 nodes
- Soft limit: 900ms (can finish current depth)
- Hard limit: 1000ms (stop immediately)

---

## Phase 5: UCI Protocol (Week 6)

### 5.1 Required UCI Commands
```cpp
void uci_loop() {
    // "uci" -> respond with engine info
    // "isready" -> respond "readyok"
    // "ucinewgame" -> clear hash tables
    // "position [fen | startpos] moves ..."
    // "go" -> start search
    // "quit" -> exit
}
```

### 5.2 Output Format
```
info depth 10 score cp 25 nodes 123456 nps 500000 time 247 pv e2e4 e7e5
bestmove e2e4
```

---

## Phase 6: Optimization (Week 7-8)

### 6.1 Performance Targets
- **Nodes per second**: 500K-2M NPS (with NNUE)
- **Effective branching factor**: ~2.0-2.5
- **Transposition table hit rate**: >90%

### 6.2 Optimization Techniques
1. **Magic Bitboards**: Pre-computed attack tables
2. **Copy-Make vs Make-Unmake**: Profile both approaches
3. **Prefetching**: Hint transposition table access
4. **SIMD**: Use AVX2/AVX512 for NNUE (if available)
5. **Compiler flags**: `/O2 /Oi /Ot /GL /arch:AVX2`

### 6.3 Profiling
- Use Visual Studio Profiler
- Identify hotspots (likely: move generation, NNUE eval)
- Optimize critical paths

---

## Phase 7: Testing & Tuning (Week 8-10)

### 7.1 Correctness Testing
- **Perft testing**: Verify move generation
  - Position: `rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1`
  - Depth 6: 119,060,324 nodes
- **Tactical test suites**: WAC, Win at Chess
- **Mate problems**: Ensure mate finding

### 7.2 Strength Testing
- **Self-play**: Play against previous versions
- **Engine matches**: Test against known engines
  - Stockfish (should lose, but measure gap)
  - Older engines (Vice, Sunfish)
- **Expected ELO**: 2000-2400 with good NNUE and search

### 7.3 Parameter Tuning
- **Search parameters**: LMR reduction amounts, null move R
- **Evaluation weights**: (minimal, since using NNUE)
- **Time management**: Allocation ratios
- **Use SPSA or manual tuning**

---

## Implementation Roadmap

### Single File Structure
```cpp
// duchess.cpp - Single file chess engine

// 1. Includes and constants
#include <iostream>
#include <vector>
#include <cstdint>
// ... other standard library includes

// 2. Magic bitboard tables (can be generated at compile time)
namespace MagicBitboards { /* ... */ }

// 3. NNUE implementation (from Stockfish)
namespace NNUE { /* ... */ }

// 4. Position and move structures
struct Move { /* ... */ };
struct Position { /* ... */ };

// 5. Move generation
namespace MoveGen { /* ... */ }

// 6. Evaluation
int evaluate(Position& pos) { return NNUE::evaluate(pos); }

// 7. Search
namespace Search { /* ... */ }

// 8. Transposition table
namespace TT { /* ... */ }

// 9. UCI protocol
namespace UCI { /* ... */ }

// 10. Main
int main() { UCI::loop(); return 0; }
```

---

## Expected Performance

### With Proper Implementation:
- **ELO**: 2200-2400 (CCRL 40/4 scale)
- **Tactical strength**: Solves most WAC positions
- **Speed**: 500K-1M NPS with NNUE
- **Search depth**: 8-12 ply in 1 second (middlegame)

### Comparison:
- Stockfish 16: ~3500 ELO
- Duchess (target): ~2300 ELO
- Gap due to: Simpler search, no tuning, single-threaded

---

## Critical Success Factors

1. **Correct move generation**: Use perft to verify
2. **Efficient NNUE integration**: Incremental updates are essential
3. **Strong move ordering**: 90% of time saved here
4. **Transposition table**: Massive speedup
5. **LMR and null move**: Doubles effective search depth
6. **Time management**: Don't forfeit on time

---

## Resources Needed

### Code References:
- **Stockfish**: NNUE implementation
- **Chess Programming Wiki**: Algorithms and techniques
- **Bluefever Software**: Tutorial series
- **Magic bitboards**: Pre-computed tables or generators

### Testing Tools:
- **Cute Chess**: Engine tournament manager
- **Arena Chess GUI**: UCI interface
- **Perft positions**: Verification suite

### Network File:
- Download `nn-*.nnue` from Stockfish releases
- Embed in executable or load at runtime

---

## Potential Challenges

1. **NNUE complexity**: Large codebase to integrate
   - *Solution*: Start with simplified version, gradually add features

2. **Single file constraint**: Code organization
   - *Solution*: Use namespaces extensively, clear comments

3. **1-second time limit**: Must be efficient
   - *Solution*: Focus on NPS optimization, smart time management

4. **No opening book**: Weak in opening
   - *Solution*: Rely on NNUE evaluation, accept some opening weaknesses

5. **Windows/MSVC specific**: Portability concerns
   - *Solution*: Use standard C++17, avoid platform-specific code

---

## Success Metrics

- ✅ Compiles in single file
- ✅ Passes perft tests
- ✅ UCI compliant
- ✅ Moves within 1 second
- ✅ ELO > 2000
- ✅ Beats random mover 100% of time
- ✅ Solves >80% of WAC positions

---

## Next Steps

1. Set up Visual Studio project
2. Implement bitboard representation
3. Write move generator + perft test
4. Add basic search (no NNUE yet, use simple eval)
5. Integrate NNUE
6. Add search enhancements one by one
7. Test and tune
8. Celebrate! 🎉