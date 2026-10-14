#ifndef CMLIR_FRONTEND_PRAGMA_HANDLER_H
#define CMLIR_FRONTEND_PRAGMA_HANDLER_H

#include "cmlir/Frontend/LoopHints.h"
#include "clang/Lex/Pragma.h"
#include "clang/Lex/Preprocessor.h"
#include "llvm/ADT/DenseMap.h"
#include <cstdint>
#include <optional>

namespace cmlir::frontend {

class CMLIRPragmaHandler : public clang::PragmaHandler {
public:
  explicit CMLIRPragmaHandler(LoopHintMap &loopHintMap)
      : clang::PragmaHandler("cmlir"), loopHintMap(loopHintMap) {}

  auto HandlePragma(clang::Preprocessor &pp, clang::PragmaIntroducer introducer,
                    clang::Token &firstTok) -> void override;

private:
  // components
  LoopHintMap &loopHintMap;
};

} // namespace cmlir::frontend

#endif // CMLIR_FRONTEND_PRAGMA_HANDLER_H
