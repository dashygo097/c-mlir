#include "./ASTConsumer.h"
#include "clang/Frontend/FrontendActions.h"

namespace cmlirc {
using namespace clang;

class CMLIRCFrontendAction : public ASTFrontendAction {
public:
  std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &CI,
                                                 StringRef file) override {
    return std::make_unique<CMLIRCASTConsumer>(&CI.getASTContext());
  }
};

} // namespace cmlirc
