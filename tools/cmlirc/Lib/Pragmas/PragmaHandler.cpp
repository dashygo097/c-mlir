#include "PragmaHandler.h"
#include "clang/Basic/TokenKinds.h"
#include "clang/Lex/Preprocessor.h"
#include "llvm/Support/WithColor.h"
#include "llvm/Support/raw_ostream.h"

namespace cmlirc {

void skipToEOD(clang::Preprocessor &PP) {
  clang::Token tok;
  do {
    PP.Lex(tok);
  } while (tok.isNot(clang::tok::eod) && tok.isNot(clang::tok::eof));
}

bool isEndOfDirective(const clang::Token &tok) {
  return tok.is(clang::tok::eod) || tok.is(clang::tok::eof);
}

void CMLIRPragmaHandler::HandlePragma(clang::Preprocessor &PP,
                                      clang::PragmaIntroducer Introducer,
                                      clang::Token &firstTok) {
  clang::Token directiveTok;
  PP.Lex(directiveTok);

  if (directiveTok.isNot(clang::tok::identifier)) {
    skipToEOD(PP);
    return;
  }

  llvm::StringRef directive = directiveTok.getIdentifierInfo()->getName();
  uint32_t line =
      PP.getSourceManager().getSpellingLineNumber(Introducer.Loc) + 1;
  LoopHints &hints = hints_[line];

  clang::Token tok;

  auto parseUInt = [&](uint32_t &out) -> bool {
    PP.Lex(tok);
    if (tok.isNot(clang::tok::l_paren))
      return false;
    PP.Lex(tok);
    if (tok.isNot(clang::tok::numeric_constant))
      return false;
    if (llvm::StringRef(PP.getSpelling(tok)).getAsInteger(10, out))
      return false;
    PP.Lex(tok);
    return tok.is(clang::tok::r_paren);
  };

  auto parseBool = [&](bool &out) -> bool {
    PP.Lex(tok);
    if (tok.isNot(clang::tok::l_paren))
      return false;
    PP.Lex(tok);
    if (tok.isNot(clang::tok::identifier))
      return false;
    llvm::StringRef val = tok.getIdentifierInfo()->getName();
    if (val == "enable")
      out = true;
    else if (val == "disable")
      out = false;
    else
      return false;
    PP.Lex(tok);
    return tok.is(clang::tok::r_paren);
  };

  auto parseUnroll = [&]() -> bool {
    PP.Lex(tok);
    if (tok.isNot(clang::tok::l_paren))
      return false;

    PP.Lex(tok); // value token

    if (tok.is(clang::tok::identifier)) {
      llvm::StringRef val = tok.getIdentifierInfo()->getName();
      if (val == "full") {
        hints.unrollFull = true;
      } else if (val == "disable") {
        hints.unrollDisable = true;
      } else {
        return false;
      }
    } else if (tok.is(clang::tok::numeric_constant)) {
      uint32_t n = 0;
      if (llvm::StringRef(PP.getSpelling(tok)).getAsInteger(10, n))
        return false;
      if (n < 2)
        return false;
      hints.unrollCount = n;
    } else {
      return false;
    }

    PP.Lex(tok);
    return tok.is(clang::tok::r_paren);
  };

  bool ok = false;

  if (directive == "loop") {
    while (true) {
      PP.Lex(tok);

      if (isEndOfDirective(tok)) {
        ok = true;
        break;
      }

      if (tok.isNot(clang::tok::identifier))
        break;

      llvm::StringRef hint = tok.getIdentifierInfo()->getName();

      if (hint == "unroll") {
        if (!parseUnroll())
          break;
      } else if (hint == "vectorize") {
        bool en = false;
        if (!parseBool(en))
          break;
        hints.vectorize = en;
      } else if (hint == "vectorize_width") {
        uint32_t n = 0;
        if (!parseUInt(n))
          break;
        hints.vectorizeWidth = n;
      } else if (hint == "interleave") {
        bool en = false;
        if (!parseBool(en))
          break;
        hints.interleave = en;
      } else if (hint == "interleave_count") {
        uint32_t n = 0;
        if (!parseUInt(n))
          break;
        hints.interleaveCount = n;
      } else {
        break;
      }

      PP.Lex(tok);
      if (isEndOfDirective(tok)) {
        ok = true;
        break;
      }
      if (!tok.is(clang::tok::comma))
        break;
    }
  }

  if (!ok)
    llvm::WithColor::warning()
        << "cmlirc: malformed #pragma cmlir " << directive << " at line "
        << PP.getSourceManager().getSpellingLineNumber(Introducer.Loc) << "\n";

  if (!isEndOfDirective(tok))
    skipToEOD(PP);
}

} // namespace cmlirc
