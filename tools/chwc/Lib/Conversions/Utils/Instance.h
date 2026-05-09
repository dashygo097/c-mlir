#ifndef CHWC_UTILS_INSTANCE_H
#define CHWC_UTILS_INSTANCE_H

#include "../../Converter.h"
#include "./Expr.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/Support/WithColor.h"
#include <optional>

namespace chwc::utils {

struct InstancePortAccess {
  const clang::FieldDecl *instanceFieldDecl{nullptr};
  HWInstanceInfo *instanceInfo{nullptr};
  const clang::FieldDecl *portDecl{nullptr};
};

inline auto sameField(const clang::FieldDecl *a, const clang::FieldDecl *b)
    -> bool {
  if (!a || !b) {
    return false;
  }

  return a->getCanonicalDecl() == b->getCanonicalDecl();
}

inline auto getFieldFromBaseExpr(clang::Expr *expr)
    -> const clang::FieldDecl * {
  expr = ignoreCasts(expr);

  if (auto *memberExpr = mlir::dyn_cast_or_null<clang::MemberExpr>(expr)) {
    return mlir::dyn_cast<clang::FieldDecl>(memberExpr->getMemberDecl());
  }

  if (auto *declRef = mlir::dyn_cast_or_null<clang::DeclRefExpr>(expr)) {
    return mlir::dyn_cast<clang::FieldDecl>(declRef->getDecl());
  }

  return nullptr;
}

inline auto parseInstancePortAccess(HWModuleContext &moduleContext,
                                    clang::Expr *expr)
    -> std::optional<InstancePortAccess> {
  expr = ignoreCasts(expr);

  auto *portMember = mlir::dyn_cast_or_null<clang::MemberExpr>(expr);
  if (!portMember) {
    return std::nullopt;
  }

  auto *portDecl =
      mlir::dyn_cast<clang::FieldDecl>(portMember->getMemberDecl());
  if (!portDecl) {
    return std::nullopt;
  }

  clang::Expr *ioExpr = ignoreCasts(portMember->getBase());
  auto *ioMember = mlir::dyn_cast_or_null<clang::MemberExpr>(ioExpr);
  if (!ioMember) {
    return std::nullopt;
  }

  if (ioMember->getMemberNameInfo().getAsString() != "io") {
    return std::nullopt;
  }

  const clang::FieldDecl *instanceField =
      getFieldFromBaseExpr(ioMember->getBase());
  if (!instanceField) {
    return std::nullopt;
  }

  auto instanceIt = moduleContext.instances.find(instanceField);
  if (instanceIt == moduleContext.instances.end()) {
    return std::nullopt;
  }

  InstancePortAccess access;
  access.instanceFieldDecl = instanceField;
  access.instanceInfo = &instanceIt->second;
  access.portDecl = portDecl;
  return access;
}

inline auto getPortIndex(llvm::ArrayRef<const clang::FieldDecl *> ports,
                         const clang::FieldDecl *portDecl)
    -> std::optional<unsigned> {
  for (unsigned i = 0; i < ports.size(); ++i) {
    if (sameField(ports[i], portDecl)) {
      return i;
    }
  }

  return std::nullopt;
}

inline auto isInstanceInputPort(const HWInstanceInfo &instanceInfo,
                                const clang::FieldDecl *portDecl) -> bool {
  return getPortIndex(instanceInfo.inputPorts, portDecl).has_value();
}

inline auto isInstanceOutputPort(const HWInstanceInfo &instanceInfo,
                                 const clang::FieldDecl *portDecl) -> bool {
  return getPortIndex(instanceInfo.outputPorts, portDecl).has_value();
}

inline auto getInstanceInputType(const HWInstanceInfo &instanceInfo,
                                 const clang::FieldDecl *portDecl)
    -> mlir::Type {
  std::optional<unsigned> index =
      getPortIndex(instanceInfo.inputPorts, portDecl);
  if (!index) {
    return nullptr;
  }

  return instanceInfo.inputTypes[*index];
}

inline auto getInstanceOutputType(const HWInstanceInfo &instanceInfo,
                                  const clang::FieldDecl *portDecl)
    -> mlir::Type {
  std::optional<unsigned> index =
      getPortIndex(instanceInfo.outputPorts, portDecl);
  if (!index) {
    return nullptr;
  }

  return instanceInfo.outputTypes[*index];
}

inline void writeInstanceInput(const InstancePortAccess &access,
                               mlir::Value value) {
  if (!access.instanceInfo || !access.portDecl || !value) {
    return;
  }

  if (!isInstanceInputPort(*access.instanceInfo, access.portDecl)) {
    llvm::WithColor::error() << "chwc: cannot assign to submodule output port: "
                             << access.portDecl->getNameAsString() << "\n";
    return;
  }

  access.instanceInfo->inputValues[access.portDecl] = value;
}

inline auto emitInstanceIfNeeded(HWModuleContext &moduleContext,
                                 HWInstanceInfo &instanceInfo,
                                 mlir::OpBuilder &builder, mlir::Location loc)
    -> bool {
  if (instanceInfo.instanceOp) {
    return true;
  }

  llvm::SmallVector<mlir::Value, 8> operands;
  llvm::SmallVector<mlir::Attribute, 8> argNames;
  llvm::SmallVector<mlir::Attribute, 8> resultNames;
  llvm::SmallVector<mlir::Type, 8> resultTypes;

  operands.push_back(moduleContext.clock);
  operands.push_back(moduleContext.reset);

  argNames.push_back(builder.getStringAttr("clk"));
  argNames.push_back(builder.getStringAttr("rst"));

  for (const clang::FieldDecl *inputPort : instanceInfo.inputPorts) {
    mlir::Value value = instanceInfo.inputValues.lookup(inputPort);
    if (!value) {
      llvm::WithColor::error()
          << "chwc: submodule input is not assigned before output read: "
          << instanceInfo.name << ".io." << inputPort->getNameAsString()
          << "\n";
      return false;
    }

    operands.push_back(value);
    argNames.push_back(builder.getStringAttr(inputPort->getNameAsString()));
  }

  for (unsigned i = 0; i < instanceInfo.outputPorts.size(); ++i) {
    const clang::FieldDecl *outputPort = instanceInfo.outputPorts[i];
    resultNames.push_back(builder.getStringAttr(outputPort->getNameAsString()));
    resultTypes.push_back(instanceInfo.outputTypes[i]);
  }

  mlir::OperationState state(loc, "hw.instance");
  state.addOperands(operands);
  state.addTypes(resultTypes);

  state.addAttribute("instanceName", builder.getStringAttr(instanceInfo.name));
  state.addAttribute("moduleName",
                     mlir::FlatSymbolRefAttr::get(builder.getContext(),
                                                  instanceInfo.moduleName));
  state.addAttribute("argNames",
                     mlir::ArrayAttr::get(builder.getContext(), argNames));
  state.addAttribute("resultNames",
                     mlir::ArrayAttr::get(builder.getContext(), resultNames));
  state.addAttribute("parameters",
                     mlir::ArrayAttr::get(builder.getContext(), {}));

  instanceInfo.instanceOp = builder.create(state);

  for (unsigned i = 0; i < instanceInfo.outputPorts.size(); ++i) {
    instanceInfo.outputValues[instanceInfo.outputPorts[i]] =
        instanceInfo.instanceOp->getResult(i);
  }

  return true;
}

inline auto readInstanceOutput(HWModuleContext &moduleContext,
                               const InstancePortAccess &access,
                               mlir::OpBuilder &builder, mlir::Location loc)
    -> mlir::Value {
  if (!access.instanceInfo || !access.portDecl) {
    return nullptr;
  }

  if (!isInstanceOutputPort(*access.instanceInfo, access.portDecl)) {
    llvm::WithColor::error()
        << "chwc: cannot read submodule input port as value: "
        << access.portDecl->getNameAsString() << "\n";
    return nullptr;
  }

  if (!emitInstanceIfNeeded(moduleContext, *access.instanceInfo, builder,
                            loc)) {
    return nullptr;
  }

  mlir::Value value = access.instanceInfo->outputValues.lookup(access.portDecl);
  if (!value) {
    llvm::WithColor::error() << "chwc: failed to read submodule output: "
                             << access.instanceInfo->name << ".io."
                             << access.portDecl->getNameAsString() << "\n";
    return nullptr;
  }

  return value;
}

} // namespace chwc::utils

#endif // CHWC_UTILS_INSTANCE_H
