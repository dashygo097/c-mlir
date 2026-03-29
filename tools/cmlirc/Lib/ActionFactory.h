#ifndef CMLIRC_ACTION_FACTORY_H
#define CMLIRC_ACTION_FACTORY_H

#include "./FrontendAction.h"

namespace cmlirc {

class CMLIRActionFactory : public clang::tooling::FrontendActionFactory {
public:
  explicit CMLIRActionFactory(llvm::raw_ostream *os) : outStream(os) {}

  auto create() -> std::unique_ptr<clang::FrontendAction> override {
    return std::make_unique<CMLIRFrontendAction>(outStream);
  }

private:
  llvm::raw_ostream *outStream;
};

} // namespace cmlirc

#endif // CMLIRC_ACTION_FACTORY_H
