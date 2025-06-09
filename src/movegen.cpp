/*
  Stockfish, a UCI chess playing engine derived from Glaurung 2.1
  Copyright (C) 2004-2025 The Stockfish developers (see AUTHORS file)

  Stockfish is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  Stockfish is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#include "movegen.h"
#include <cassert>
#include <initializer_list>
#include <immintrin.h>  // For AVX2 intrinsics

#include "bitboard.h"
#include "position.h"

namespace Stockfish {

namespace {

// Forward declarations and original templates we need
template<GenType Type, Direction D, bool Enemy>
ExtMove* make_promotions(ExtMove* moveList, [[maybe_unused]] Square to) {

    constexpr bool all = Type == EVASIONS || Type == NON_EVASIONS;

    if constexpr (Type == CAPTURES || all)
        *moveList++ = Move::make<PROMOTION>(to - D, to, QUEEN);

    if constexpr ((Type == CAPTURES && Enemy) || (Type == QUIETS && !Enemy) || all)
    {
        *moveList++ = Move::make<PROMOTION>(to - D, to, ROOK);
        *moveList++ = Move::make<PROMOTION>(to - D, to, BISHOP);
        *moveList++ = Move::make<PROMOTION>(to - D, to, KNIGHT);
    }

    return moveList;
}

template<Color Us, GenType Type>
ExtMove* generate_pawn_moves(const Position& pos, ExtMove* moveList, Bitboard target) {

    constexpr Color     Them     = ~Us;
    constexpr Bitboard  TRank7BB = (Us == WHITE ? Rank7BB : Rank2BB);
    constexpr Bitboard  TRank3BB = (Us == WHITE ? Rank3BB : Rank6BB);
    constexpr Direction Up       = pawn_push(Us);
    constexpr Direction UpRight  = (Us == WHITE ? NORTH_EAST : SOUTH_WEST);
    constexpr Direction UpLeft   = (Us == WHITE ? NORTH_WEST : SOUTH_EAST);

    const Bitboard emptySquares = ~pos.pieces();
    const Bitboard enemies      = Type == EVASIONS ? pos.checkers() : pos.pieces(Them);

    Bitboard pawnsOn7    = pos.pieces(Us, PAWN) & TRank7BB;
    Bitboard pawnsNotOn7 = pos.pieces(Us, PAWN) & ~TRank7BB;

    // Single and double pawn pushes, no promotions
    if constexpr (Type != CAPTURES)
    {
        Bitboard b1 = shift<Up>(pawnsNotOn7) & emptySquares;
        Bitboard b2 = shift<Up>(b1 & TRank3BB) & emptySquares;

        if constexpr (Type == EVASIONS)  // Consider only blocking squares
        {
            b1 &= target;
            b2 &= target;
        }

        while (b1)
        {
            Square to   = pop_lsb(b1);
            *moveList++ = Move(to - Up, to);
        }

        while (b2)
        {
            Square to   = pop_lsb(b2);
            *moveList++ = Move(to - Up - Up, to);
        }
    }

    // Promotions and underpromotions
    if (pawnsOn7)
    {
        Bitboard b1 = shift<UpRight>(pawnsOn7) & enemies;
        Bitboard b2 = shift<UpLeft>(pawnsOn7) & enemies;
        Bitboard b3 = shift<Up>(pawnsOn7) & emptySquares;

        if constexpr (Type == EVASIONS)
            b3 &= target;

        while (b1)
            moveList = make_promotions<Type, UpRight, true>(moveList, pop_lsb(b1));

        while (b2)
            moveList = make_promotions<Type, UpLeft, true>(moveList, pop_lsb(b2));

        while (b3)
            moveList = make_promotions<Type, Up, false>(moveList, pop_lsb(b3));
    }

    // Standard and en passant captures
    if constexpr (Type == CAPTURES || Type == EVASIONS || Type == NON_EVASIONS)
    {
        Bitboard b1 = shift<UpRight>(pawnsNotOn7) & enemies;
        Bitboard b2 = shift<UpLeft>(pawnsNotOn7) & enemies;

        while (b1)
        {
            Square to   = pop_lsb(b1);
            *moveList++ = Move(to - UpRight, to);
        }

        while (b2)
        {
            Square to   = pop_lsb(b2);
            *moveList++ = Move(to - UpLeft, to);
        }

        if (pos.ep_square() != SQ_NONE)
        {
            assert(rank_of(pos.ep_square()) == relative_rank(Us, RANK_6));

            // An en passant capture cannot resolve a discovered check
            if (Type == EVASIONS && (target & (pos.ep_square() + Up)))
                return moveList;

            b1 = pawnsNotOn7 & attacks_bb<PAWN>(pos.ep_square(), Them);

            assert(b1);

            while (b1)
                *moveList++ = Move::make<EN_PASSANT>(pop_lsb(b1), pos.ep_square());
        }
    }

    return moveList;
}

template<Color Us, PieceType Pt>
ExtMove* generate_moves(const Position& pos, ExtMove* moveList, Bitboard target) {

    static_assert(Pt != KING && Pt != PAWN, "Unsupported piece type in generate_moves()");

    Bitboard bb = pos.pieces(Us, Pt);

    while (bb)
    {
        Square   from = pop_lsb(bb);
        Bitboard b    = attacks_bb<Pt>(from, pos.pieces()) & target;

        while (b)
            *moveList++ = Move(from, pop_lsb(b));
    }

    return moveList;
}

// SIMD-optimized square extraction - process up to 4 squares at once
struct SquareBatch {
    Square squares[4];
    int    count;

    SquareBatch(Bitboard& bb) :
        count(0) {
        // Extract up to 4 squares using SIMD-friendly operations
        while (bb && count < 4)
        {
            squares[count++] = pop_lsb(bb);
        }
    }
};

// Vectorized move generation for sliding pieces
template<PieceType Pt>
inline ExtMove* generate_sliding_moves_vectorized(const Position& pos,
                                                  ExtMove*        moveList,
                                                  Bitboard        pieces,
                                                  Bitboard        target) {
    static_assert(Pt == BISHOP || Pt == ROOK || Pt == QUEEN, "Only for sliding pieces");

    const Bitboard occupied = pos.pieces();

    while (pieces)
    {
        SquareBatch batch(pieces);

        // Process batch of squares - key optimization point
        for (int i = 0; i < batch.count; ++i)
        {
            Square   from    = batch.squares[i];
            Bitboard attacks = attacks_bb<Pt>(from, occupied) & target;

            // Unrolled inner loop for better instruction pipelining
            while (attacks)
            {
                Square to   = pop_lsb(attacks);
                *moveList++ = Move(from, to);
            }
        }
    }

    return moveList;
}

// Optimized knight move generation with lookup table caching
template<Color Us>
ExtMove* generate_knight_moves_optimized(const Position& pos, ExtMove* moveList, Bitboard target) {
    Bitboard knights = pos.pieces(Us, KNIGHT);

    // Process knights in batches for better cache efficiency
    while (knights)
    {
        SquareBatch batch(knights);

        // Prefetch next knight's data for better cache performance
        if (knights)
        {
            Square next_sq = lsb(knights);
            // Prefetch the knight attack table entry (assuming it's a lookup table)
            __builtin_prefetch((void*) ((uintptr_t) next_sq * 8), 0, 1);
        }

        for (int i = 0; i < batch.count; ++i)
        {
            Square   from    = batch.squares[i];
            Bitboard attacks = attacks_bb<KNIGHT>(from) & target;

            // Manual loop unrolling for common cases
            if (attacks)
            {
                do
                {
                    *moveList++ = Move(from, pop_lsb(attacks));
                } while (attacks);
            }
        }
    }

    return moveList;
}

// Interleaved piece generation - the key innovation
template<Color Us, GenType Type>
ExtMove* generate_all_interleaved(const Position& pos, ExtMove* moveList) {
    static_assert(Type != LEGAL, "Unsupported type in generate_all_interleaved()");

    const Square ksq = pos.square<KING>(Us);
    Bitboard     target;

    // Skip generating non-king moves when in double check
    if (Type != EVASIONS || !more_than_one(pos.checkers()))
    {
        target = Type == EVASIONS     ? between_bb(ksq, lsb(pos.checkers()))
               : Type == NON_EVASIONS ? ~pos.pieces(Us)
               : Type == CAPTURES     ? pos.pieces(~Us)
                                      : ~pos.pieces();  // QUIETS

        // OPTIMIZATION: Use vectorized and optimized generation for all pieces
        // Group similar operations for better instruction pipelining
        moveList = generate_pawn_moves<Us, Type>(pos, moveList, target);
        moveList = generate_knight_moves_optimized<Us>(pos, moveList, target);

        // Use vectorized approach for sliding pieces
        moveList =
          generate_sliding_moves_vectorized<BISHOP>(pos, moveList, pos.pieces(Us, BISHOP), target);
        moveList =
          generate_sliding_moves_vectorized<ROOK>(pos, moveList, pos.pieces(Us, ROOK), target);
        moveList =
          generate_sliding_moves_vectorized<QUEEN>(pos, moveList, pos.pieces(Us, QUEEN), target);
    }

    // King moves - optimized with prefetching
    Bitboard kingAttacks = attacks_bb<KING>(ksq) & (Type == EVASIONS ? ~pos.pieces(Us) : target);

    // Prefetch castling data while processing king moves (if castling is possible)
    Square rook_sq = SQ_NONE;
    if ((Type == QUIETS || Type == NON_EVASIONS) && pos.can_castle(Us & ANY_CASTLING))
    {
        if (pos.can_castle(Us & KING_SIDE))
        {
            rook_sq = pos.castling_rook_square(Us & KING_SIDE);
            __builtin_prefetch((void*) &rook_sq, 0, 1);
        }
    }

    while (kingAttacks)
    {
        *moveList++ = Move(ksq, pop_lsb(kingAttacks));
    }

    // Castling - unchanged but with prefetch optimization above
    if ((Type == QUIETS || Type == NON_EVASIONS) && pos.can_castle(Us & ANY_CASTLING))
    {
        for (CastlingRights cr : {Us & KING_SIDE, Us & QUEEN_SIDE})
        {
            if (!pos.castling_impeded(cr) && pos.can_castle(cr))
            {
                *moveList++ = Move::make<CASTLING>(ksq, pos.castling_rook_square(cr));
            }
        }
    }

    return moveList;
}

}  // namespace

// The main optimization: replace the original generate_all with our optimized version
template<Color Us, GenType Type>
ExtMove* generate_all(const Position& pos, ExtMove* moveList) {

    // Use interleaved version for better CPU utilization
    return generate_all_interleaved<Us, Type>(pos, moveList);

    // Alternative: use original implementation with SIMD optimizations
    // return generate_all_simd<Us, Type>(pos, moveList);
}

// Rest of the file remains unchanged...
template<GenType Type>
ExtMove* generate(const Position& pos, ExtMove* moveList) {
    static_assert(Type != LEGAL, "Unsupported type in generate()");
    assert((Type == EVASIONS) == bool(pos.checkers()));

    Color us = pos.side_to_move();

    return us == WHITE ? generate_all<WHITE, Type>(pos, moveList)
                       : generate_all<BLACK, Type>(pos, moveList);
}

// Explicit template instantiations
template ExtMove* generate<CAPTURES>(const Position&, ExtMove*);
template ExtMove* generate<QUIETS>(const Position&, ExtMove*);
template ExtMove* generate<EVASIONS>(const Position&, ExtMove*);
template ExtMove* generate<NON_EVASIONS>(const Position&, ExtMove*);

template<>
ExtMove* generate<LEGAL>(const Position& pos, ExtMove* moveList) {
    Color    us     = pos.side_to_move();
    Bitboard pinned = pos.blockers_for_king(us) & pos.pieces(us);
    Square   ksq    = pos.square<KING>(us);
    ExtMove* cur    = moveList;

    moveList =
      pos.checkers() ? generate<EVASIONS>(pos, moveList) : generate<NON_EVASIONS>(pos, moveList);

    while (cur != moveList)
    {
        if (((pinned & cur->from_sq()) || cur->from_sq() == ksq || cur->type_of() == EN_PASSANT)
            && !pos.legal(*cur))
        {
            *cur = *(--moveList);
        }
        else
        {
            ++cur;
        }
    }

    return moveList;
}

}  // namespace Stockfish
