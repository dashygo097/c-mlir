#ifndef CMLIRC_LOOPUTILS_H
#define CMLIRC_LOOPUTILS_H

#include "../../../Converter.h"
#include "mlir/IR/Builders.h"
#include "clang/AST/Expr.h"
#include "clang/AST/Stmt.h"

namespace cmlirc::detail {

void ensureYield(mlir::OpBuilder &builder, mlir::Location loc,
                 mlir::Block *block);

bool classifyCondOp(clang::BinaryOperatorKind op, bool &isIncrementing);

mlir::Value extractStep(clang::Expr *incExpr, const clang::VarDecl *var,
                        bool isIncrementing, mlir::OpBuilder &builder,
                        mlir::Location loc,
                        std::function<mlir::Value(clang::Expr *)> genExpr);

void adjustBounds(mlir::OpBuilder &b, mlir::Location loc,
                  clang::BinaryOperatorKind condOp, bool isIncrementing,
                  mlir::Value initVal, mlir::Value condVal, mlir::Value &lb,
                  mlir::Value &ub);

std::optional<CMLIRConverter::SimpleLoopInfo>
analyseForLoop(clang::ForStmt *forStmt, mlir::OpBuilder &builder,
               mlir::Location loc,
               std::function<mlir::Value(clang::Expr *)> genExpr);

} // namespace cmlirc::detail

#endif // CMLIRC_LOOPUTILS_H
