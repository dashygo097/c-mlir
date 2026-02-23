#include "PragmaHandler.h"
#include "clang/Basic/TokenKinds.h"
#include "clang/Lex/Preprocessor.h"
#include "llvm/Support/raw_ostream.h"

namespace cmlirc {

static void skipToEOD(clang::Preprocessor &PP) {
  clang::Token tok;
  do {
    PP.Lex(tok);
  } while (tok.isNot(clang::tok::eod) && tok.isNot(clang::tok::eof));
}

void CMLIRPragmaHandler::HandlePragma(clang::Preprocessor &PP,
                                      clang::PragmaIntroducer Introducer,
                                      clang::Token &firstTok) {
  // In this Clang version, firstTok is the matched handler token ("cmlir").
  // Lex one more token to get the actual directive ("loop_unroll" etc).
  clang::Token directiveTok;
  PP.Lex(directiveTok);
  if (directiveTok.isNot(clang::tok::identifier)) {
    skipToEOD(PP);
    return;
  }

  llvm::StringRef directive = directiveTok.getIdentifierInfo()->getName();
  unsigned line =
      PP.getSourceManager().getSpellingLineNumber(Introducer.Loc) + 1;
  LoopHints &hints = hints_[line];

  // All sub-parsers lex their own tokens from scratch after directive is known.
  clang::Token tok;

  // ── parse ( N ) ───────────────────────────────────────────────────────────
  auto parseUInt = [&](uint64_t &out) -> bool {
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

  // ── parse ( enable | disable ) ────────────────────────────────────────────
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

  bool ok = false;

  if (directive == "loop_unroll") {
    // ( N | full | disable )
    PP.Lex(tok);
    if (tok.isNot(clang::tok::l_paren))
      goto done;
    PP.Lex(tok);

    if (tok.is(clang::tok::numeric_constant)) {
      uint64_t n = 0;
      if (llvm::StringRef(PP.getSpelling(tok)).getAsInteger(10, n))
        goto done;
      PP.Lex(tok);
      if (tok.isNot(clang::tok::r_paren))
        goto done;
      hints.unrollCount = n;
      ok = true;
    } else if (tok.is(clang::tok::identifier)) {
      llvm::StringRef val = tok.getIdentifierInfo()->getName();
      PP.Lex(tok);
      if (tok.isNot(clang::tok::r_paren))
        goto done;
      if (val == "full") {
        hints.unrollFull = true;
        ok = true;
      } else if (val == "disable") {
        hints.unrollDisable = true;
        ok = true;
      }
    }

  } else if (directive == "loop_vectorize") {
    bool en = false;
    if (parseBool(en)) {
      hints.vectorize = en;
      ok = true;
    }

  } else if (directive == "loop_vectorize_width") {
    uint64_t n = 0;
    if (parseUInt(n)) {
      hints.vectorizeWidth = n;
      ok = true;
    }

  } else if (directive == "loop_interleave") {
    bool en = false;
    if (parseBool(en)) {
      hints.interleave = en;
      ok = true;
    }

  } else if (directive == "loop_interleave_count") {
    uint64_t n = 0;
    if (parseUInt(n)) {
      hints.interleaveCount = n;
      ok = true;
    }
  }

done:
  if (!ok)
    llvm::errs() << "warning: malformed #pragma cmlir " << directive
                 << " at line "
                 << PP.getSourceManager().getSpellingLineNumber(Introducer.Loc)
                 << "\n";
  skipToEOD(PP);
}

} // namespace cmlirc
