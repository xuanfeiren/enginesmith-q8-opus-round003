// Minimal JSON reader (objects, arrays, strings, numbers, booleans, null) for requests and safetensors headers.
#pragma once
#include <cstdint>
#include <cstdlib>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace json {

struct Value {
  enum Kind { Null, Bool, Num, Str, Arr, Obj } kind = Null;
  bool b = false;
  double num = 0;
  int64_t i = 0;
  bool is_int = false;
  std::string s;
  std::vector<Value> arr;
  std::vector<std::pair<std::string, Value>> obj;

  const Value* get(const std::string& k) const {
    for (auto& kv : obj) if (kv.first == k) return &kv.second;
    return nullptr;
  }
  const Value& at(const std::string& k) const {
    auto* v = get(k);
    if (!v) throw std::runtime_error("missing key: " + k);
    return *v;
  }
  int64_t as_int() const {
    if (kind != Num) throw std::runtime_error("expected a number");
    return is_int ? i : (int64_t)num;
  }
};

class Parser {
 public:
  Parser(const char* p, const char* e) : p_(p), e_(e) {}
  Value parse() {
    Value v = value();
    ws();
    if (p_ != e_) throw std::runtime_error("trailing characters in JSON");
    return v;
  }

 private:
  const char* p_;
  const char* e_;
  void ws() { while (p_ < e_ && (*p_ == ' ' || *p_ == '\n' || *p_ == '\r' || *p_ == '\t')) p_++; }
  [[noreturn]] void fail(const char* m) { throw std::runtime_error(std::string("JSON: ") + m); }
  bool lit(const char* s) {
    const char* q = p_;
    for (; *s; s++, q++) if (q >= e_ || *q != *s) return false;
    p_ = q;
    return true;
  }
  Value value() {
    ws();
    if (p_ >= e_) fail("unexpected end");
    Value v;
    char c = *p_;
    if (c == '{') {
      v.kind = Value::Obj;
      p_++;
      ws();
      if (p_ < e_ && *p_ == '}') { p_++; return v; }
      for (;;) {
        ws();
        if (p_ >= e_ || *p_ != '"') fail("expected a key");
        std::string k = str();
        ws();
        if (p_ >= e_ || *p_ != ':') fail("expected ':'");
        p_++;
        v.obj.emplace_back(std::move(k), value());
        ws();
        if (p_ < e_ && *p_ == ',') { p_++; continue; }
        if (p_ < e_ && *p_ == '}') { p_++; return v; }
        fail("expected ',' or '}'");
      }
    }
    if (c == '[') {
      v.kind = Value::Arr;
      p_++;
      ws();
      if (p_ < e_ && *p_ == ']') { p_++; return v; }
      for (;;) {
        v.arr.push_back(value());
        ws();
        if (p_ < e_ && *p_ == ',') { p_++; continue; }
        if (p_ < e_ && *p_ == ']') { p_++; return v; }
        fail("expected ',' or ']'");
      }
    }
    if (c == '"') { v.kind = Value::Str; v.s = str(); return v; }
    if (lit("true")) { v.kind = Value::Bool; v.b = true; return v; }
    if (lit("false")) { v.kind = Value::Bool; v.b = false; return v; }
    if (lit("null")) return v;
    if (c == '-' || (c >= '0' && c <= '9')) {
      const char* q = p_;
      bool integer = true;
      if (*q == '-') q++;
      while (q < e_ && ((*q >= '0' && *q <= '9') || *q == '.' || *q == 'e' || *q == 'E' || *q == '+' || *q == '-')) {
        if (*q == '.' || *q == 'e' || *q == 'E') integer = false;
        q++;
      }
      std::string t(p_, q);
      p_ = q;
      v.kind = Value::Num;
      v.is_int = integer;
      if (integer) { v.i = strtoll(t.c_str(), nullptr, 10); v.num = (double)v.i; }
      else { v.num = strtod(t.c_str(), nullptr); v.i = (int64_t)v.num; }
      return v;
    }
    fail("unexpected character");
  }
  std::string str() {
    p_++;  // opening quote
    std::string out;
    while (p_ < e_ && *p_ != '"') {
      if (*p_ == '\\') {
        p_++;
        if (p_ >= e_) fail("bad escape");
        char c = *p_++;
        switch (c) {
          case 'n': out += '\n'; break;
          case 't': out += '\t'; break;
          case 'r': out += '\r'; break;
          case 'b': out += '\b'; break;
          case 'f': out += '\f'; break;
          case 'u': {
            if (e_ - p_ < 4) fail("bad \\u escape");
            unsigned cp = (unsigned)strtoul(std::string(p_, p_ + 4).c_str(), nullptr, 16);
            p_ += 4;
            if (cp < 0x80) out += (char)cp;
            else if (cp < 0x800) { out += (char)(0xC0 | (cp >> 6)); out += (char)(0x80 | (cp & 0x3F)); }
            else { out += (char)(0xE0 | (cp >> 12)); out += (char)(0x80 | ((cp >> 6) & 0x3F)); out += (char)(0x80 | (cp & 0x3F)); }
            break;
          }
          default: out += c;
        }
      } else {
        out += *p_++;
      }
    }
    if (p_ >= e_) fail("unterminated string");
    p_++;
    return out;
  }
};

inline Value parse(const std::string& s) { return Parser(s.data(), s.data() + s.size()).parse(); }
inline Value parse(const char* p, size_t n) { return Parser(p, p + n).parse(); }

inline std::string quote(const std::string& s) {
  std::string o = "\"";
  for (char c : s) {
    if (c == '"' || c == '\\') { o += '\\'; o += c; }
    else if (c == '\n') o += "\\n";
    else if ((unsigned char)c < 0x20) { char b[8]; snprintf(b, sizeof b, "\\u%04x", c); o += b; }
    else o += c;
  }
  return o + "\"";
}

}  // namespace json
