#ifndef CMLIRC_PRAGMA_HANDLER_H
#define CMLIRC_PRAGMA_HANDLER_H

#include "clang/Lex/Pragma.h"
#include "clang/Lex/Preprocessor.h"
#include "llvm/ADT/DenseMap.h"
#include <cstdint>
#include <optional>

namespace cmlirc {

struct LoopHints {
  std::optional<uint64_t> unrollCount;     // loop_unroll(N)
  std::optional<bool> unrollFull;          // loop_unroll(full)
  std::optional<bool> unrollDisable;       // loop_unroll(disable)
  std::optional<bool> vectorize;           // loop_vectorize(enable|disable)
  std::optional<uint64_t> vectorizeWidth;  // loop_vectorize_width(N)
  std::optional<bool> interleave;          // loop_interleave(enable|disable)
  std::optional<uint64_t> interleaveCount; // loop_interleave_count(N)
};

using LoopHintMap = llvm::DenseMap<unsigned, LoopHints>;

class CMLIRPragmaHandler : public clang::PragmaHandler {
public:
  explicit CMLIRPragmaHandler(LoopHintMap &hints)
      : clang::PragmaHandler("cmlir"), hints_(hints) {}

  void HandlePragma(clang::Preprocessor &PP, clang::PragmaIntroducer Introducer,
                    clang::Token &firstTok) override;

private:
  LoopHintMap &hints_;
};

} // namespace cmlirc
#endif // CMLIRC_PRAGMA_HANDLER_H
