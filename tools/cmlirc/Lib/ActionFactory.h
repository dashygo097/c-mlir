#ifndef CMLIRC_ACTION_FACTORY_H
#define CMLIRC_ACTION_FACTORY_H

#include "./FrontendAction.h"

namespace cmlirc {

class CMLIRActionFactory : public clang::tooling::FrontendActionFactory {
public:
  explicit CMLIRActionFactory(llvm::raw_ostream *out) : output_stream_(out) {}

  std::unique_ptr<clang::FrontendAction> create() override {
    return std::make_unique<CMLIRFrontendAction>(output_stream_);
  }

private:
  llvm::raw_ostream *output_stream_;
};

} // namespace cmlirc

#endif // CMLIRC_ACTION_FACTORY_H
