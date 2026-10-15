#include "cmlir/Support/Printing.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/raw_ostream.h"

namespace cmlir {

namespace {

using Color = llvm::raw_ostream::Colors;

struct Token {
  size_t length;
  Color color = Color::SAVEDCOLOR;
  bool bold = false;
};

auto quotedLength(llvm::StringRef text) -> size_t {
  size_t end = 1;
  while (end < text.size()) {
    if (text[end] == '\\' && end + 1 < text.size()) {
      end += 2;
    } else if (text[end++] == '"') {
      break;
    }
  }
  return end;
}

auto identifierChar(char ch) -> bool {
  return llvm::isAlnum(ch) || ch == '_' || ch == '.' || ch == '$' ||
         ch == '-' || ch == '#';
}

auto nextToken(llvm::StringRef text) -> Token {
  if (text.starts_with("//")) {
    auto end = text.find('\n');
    return {end == llvm::StringRef::npos ? text.size() : end, Color::WHITE};
  }
  if (text.front() == '"') {
    return {quotedLength(text), Color::BLUE};
  }
  if (llvm::StringRef("%@^!#").contains(text.front())) {
    size_t length = 1;
    if (text.size() > 1 && text[1] == '"') {
      length += quotedLength(text.drop_front());
    } else {
      while (length < text.size() && identifierChar(text[length])) {
        ++length;
      }
    }
    switch (text.front()) {
    case '%':
      return {length, Color::WHITE, true};
    case '@':
      return {length, Color::YELLOW, true};
    case '^':
      return {length, Color::RED};
    default:
      return {length, Color::CYAN};
    }
  }
  if (llvm::isDigit(text.front()) ||
      ((text.front() == '-' || text.front() == '+') && text.size() > 1 &&
       llvm::isDigit(text[1]))) {
    size_t length = 1;
    while (length < text.size()) {
      char ch = text[length];
      if (llvm::isAlnum(ch) || ch == '.' || ch == '_') {
        ++length;
      } else if ((ch == '+' || ch == '-') &&
                 llvm::StringRef("eEpP").contains(text[length - 1])) {
        ++length;
      } else {
        break;
      }
    }
    return {length, Color::CYAN};
  }
  if (llvm::isAlpha(text.front()) || text.front() == '_') {
    size_t length = 1;
    while (length < text.size() && identifierChar(text[length])) {
      ++length;
    }
    auto word = text.take_front(length);
    if (word.contains('.')) {
      return {length, Color::MAGENTA, true};
    }
    if (llvm::StringSwitch<bool>(word)
            .Cases({"func", "return", "module", "do", "default"}, true)
            .Cases({"case", "to", "step", "iter_args", "else"}, true)
            .Cases({"if", "for", "while", "yield", "call"}, true)
            .Cases({"private", "public", "attributes"}, true)
            .Default(false)) {
      return {length, Color::RED, true};
    }
    if (llvm::StringSwitch<bool>(word)
            .Cases({"index", "memref", "tensor", "vector", "ptr"}, true)
            .Cases({"none", "vararg", "complex", "tuple", "bf16"}, true)
            .Default(false) ||
        (word.size() > 1 && (word.front() == 'i' || word.front() == 'f') &&
         llvm::all_of(word.drop_front(), llvm::isDigit))) {
      return {length, Color::CYAN};
    }
    return {length};
  }
  return {1};
}

} // namespace

auto printIR(mlir::Operation *operation, llvm::raw_ostream &out,
             llvm::ColorMode mode) -> void {
  bool useColor = mode == llvm::ColorMode::Enable ||
                  (mode == llvm::ColorMode::Auto &&
                   llvm::WithColor::defaultAutoDetectFunction()(out));
  if (!useColor) {
    operation->print(out);
    return;
  }
  std::string buffer;
  llvm::raw_string_ostream stream(buffer);
  operation->print(stream);
  llvm::StringRef text(buffer);
  bool wasEnabled = out.colors_enabled();
  out.enable_colors(true);
  auto restore =
      llvm::scope_exit([&]() -> void { out.enable_colors(wasEnabled); });
  while (!text.empty()) {
    auto token = nextToken(text);
    auto value = text.take_front(token.length);
    if (token.color == Color::SAVEDCOLOR) {
      out << value;
    } else {
      llvm::WithColor(out, token.color, token.bold, false,
                      llvm::ColorMode::Enable)
          << value;
    }
    text = text.drop_front(token.length);
  }
}

} // namespace cmlir
