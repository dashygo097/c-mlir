#include "PragmaHandler.h"
#include "clang/Basic/TokenKinds.h"
#include "clang/Lex/Preprocessor.h"
#include "llvm/Support/WithColor.h"
#include "llvm/Support/raw_ostream.h"

namespace cmlirc {

void skipToEOD(clang::Preprocessor &pp) {
  clang::Token tok;
  do {
    pp.Lex(tok);
  } while (tok.isNot(clang::tok::eod) && tok.isNot(clang::tok::eof));
}

bool isEndOfDirective(const clang::Token &tok) {
  return tok.is(clang::tok::eod) || tok.is(clang::tok::eof);
}

void CMLIRPragmaHandler::HandlePragma(clang::Preprocessor &pp,
                                      clang::PragmaIntroducer introducer,
                                      clang::Token &firstTok) {
  clang::Token directiveTok;
  pp.Lex(directiveTok);

  if (directiveTok.isNot(clang::tok::identifier)) {
    skipToEOD(pp);
    return;
  }

  llvm::StringRef directive = directiveTok.getIdentifierInfo()->getName();
  uint32_t line =
      pp.getSourceManager().getSpellingLineNumber(introducer.Loc) + 1;
  LoopHints &hints = loopHintMap[line];

  clang::Token tok;

  auto parseUInt = [&](uint32_t &out) -> bool {
    pp.Lex(tok);
    if (tok.isNot(clang::tok::l_paren))
      return false;
    pp.Lex(tok);
    if (tok.isNot(clang::tok::numeric_constant))
      return false;
    if (llvm::StringRef(pp.getSpelling(tok)).getAsInteger(10, out))
      return false;
    pp.Lex(tok);
    return tok.is(clang::tok::r_paren);
  };

  auto parseBool = [&](bool &out) -> bool {
    pp.Lex(tok);
    if (tok.isNot(clang::tok::l_paren))
      return false;
    pp.Lex(tok);
    if (tok.isNot(clang::tok::identifier))
      return false;
    llvm::StringRef val = tok.getIdentifierInfo()->getName();
    if (val == "enable")
      out = true;
    else if (val == "disable")
      out = false;
    else
      return false;
    pp.Lex(tok);
    return tok.is(clang::tok::r_paren);
  };

  auto parseUnroll = [&]() -> bool {
    pp.Lex(tok);
    if (tok.isNot(clang::tok::l_paren))
      return false;

    pp.Lex(tok); // value token

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
      if (llvm::StringRef(pp.getSpelling(tok)).getAsInteger(10, n))
        return false;
      if (n < 2)
        return false;
      hints.unrollCount = n;
    } else {
      return false;
    }

    pp.Lex(tok);
    return tok.is(clang::tok::r_paren);
  };

  bool ok = false;

  if (directive == "loop") {
    ok = true;
    pp.Lex(tok);

    while (!isEndOfDirective(tok)) {
      if (tok.is(clang::tok::comma)) {
        pp.Lex(tok);
        continue;
      }

      if (tok.isNot(clang::tok::identifier)) {
        ok = false;
        break;
      }

      llvm::StringRef hint = tok.getIdentifierInfo()->getName();

      if (hint == "unroll") {
        if (!parseUnroll()) {
          ok = false;
          break;
        }
      } else if (hint == "vectorize") {
        bool en = false;
        if (!parseBool(en)) {
          ok = false;
          break;
        }
        hints.vectorize = en;
      } else if (hint == "vectorize_width") {
        uint32_t n = 0;
        if (!parseUInt(n)) {
          ok = false;
          break;
        }
        hints.vectorizeWidth = n;
      } else if (hint == "interleave") {
        bool en = false;
        if (!parseBool(en)) {
          ok = false;
          break;
        }
        hints.interleave = en;
      } else if (hint == "interleave_count") {
        uint32_t n = 0;
        if (!parseUInt(n)) {
          ok = false;
          break;
        }
        hints.interleaveCount = n;
      } else {
        ok = false;
        break;
      }

      pp.Lex(tok);
    }
  }

  if (!ok)
    llvm::WithColor::warning()
        << "cmlirc: malformed #pragma cmlir " << directive << " at line "
        << pp.getSourceManager().getSpellingLineNumber(introducer.Loc) << "\n";

  if (!isEndOfDirective(tok))
    skipToEOD(pp);
}

} // namespace cmlirc
