#ifndef CMLIR_FRONTEND_LOOP_HINTS_H
#define CMLIR_FRONTEND_LOOP_HINTS_H

#include "llvm/ADT/DenseMap.h"
#include <cstdint>
#include <optional>

namespace cmlir::frontend {

struct LoopHints {
  std::optional<uint64_t> unrollCount;     // loop_unroll(N)
  std::optional<bool> unrollFull;          // loop_unroll(full)
  std::optional<bool> unrollDisable;       // loop_unroll(disable)
  std::optional<bool> vectorize;           // loop_vectorize(enable|disable)
  std::optional<uint64_t> vectorizeWidth;  // loop_vectorize_width(N)
  std::optional<bool> interleave;          // loop_interleave(enable|disable)
  std::optional<uint64_t> interleaveCount; // loop_interleave_count(N)
};

using LoopHintMap = llvm::DenseMap<uint32_t, LoopHints>;

} // namespace cmlir::frontend

#endif // CMLIR_FRONTEND_LOOP_HINTS_H
