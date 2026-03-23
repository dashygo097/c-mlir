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

using LoopHintMap = llvm::DenseMap<uint32_t, LoopHints>;

class CMLIRPragmaHandler : public clang::PragmaHandler {
public:
  explicit CMLIRPragmaHandler(LoopHintMap &loopHintMap)
      : clang::PragmaHandler("cmlir"), loopHintMap(loopHintMap) {}

  void HandlePragma(clang::Preprocessor &pp, clang::PragmaIntroducer introducer,
                    clang::Token &firstTok) override;

private:
  // components
  LoopHintMap &loopHintMap;
};

} // namespace cmlirc
#endif // CMLIRC_PRAGMA_HANDLER_H
