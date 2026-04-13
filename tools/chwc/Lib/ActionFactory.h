#ifndef CHWC_ACTION_FACTORY_H
#define CHWC_ACTION_FACTORY_H

#include "./FrontendAction.h"

namespace chwc {

class CHWActionFactory : public clang::tooling::FrontendActionFactory {
public:
  explicit CHWActionFactory(llvm::raw_ostream *os) : outStream(os) {}

  auto create() -> std::unique_ptr<clang::FrontendAction> override {
    return std::make_unique<CHWFrontendAction>(outStream);
  }

private:
  llvm::raw_ostream *outStream;
};

} // namespace chwc

#endif // CHWC_ACTION_FACTORY_H
