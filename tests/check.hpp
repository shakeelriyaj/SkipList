#pragma once

// Minimal zero-dependency test harness.
//   TEST_CASE(name) { ... }   registers a test
//   CHECK(cond)               records a failure and continues
//   CHECK_EQ(a, b)            same, printing both values when they are streamable
//   REQUIRE(cond)             records a failure and aborts the current test
// main() is provided by CHECK_MAIN(); pass a substring to run a subset.

#include <cstdio>
#include <cstring>
#include <exception>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace check {

struct TestCase {
  const char* name;
  void (*fn)();
};

inline std::vector<TestCase>& registry() {
  static std::vector<TestCase> tests;
  return tests;
}

inline int& current_failures() {
  static int failures = 0;
  return failures;
}

struct Registrar {
  Registrar(const char* name, void (*fn)()) { registry().push_back({name, fn}); }
};

struct RequireFailed {};

inline void report(const char* file, int line, const std::string& msg) {
  ++current_failures();
  std::fprintf(stderr, "  %s:%d: FAILED: %s\n", file, line, msg.c_str());
}

template <typename T>
auto stringify(const T& v, int) -> decltype(std::declval<std::ostream&>() << v, std::string()) {
  std::ostringstream os;
  os << +v;  // unary + promotes small integer types to print as numbers
  return os.str();
}
template <typename T>
std::string stringify(const T&, long) {
  return "<unprintable>";
}

inline int run_all(int argc, char** argv) {
  const char* filter = argc > 1 ? argv[1] : nullptr;
  int failed = 0, ran = 0;
  for (const TestCase& t : registry()) {
    if (filter && !std::strstr(t.name, filter)) continue;
    ++ran;
    current_failures() = 0;
    try {
      t.fn();
    } catch (const RequireFailed&) {
    } catch (const std::exception& e) {
      report(t.name, 0, std::string("uncaught exception: ") + e.what());
    } catch (...) {
      report(t.name, 0, "uncaught unknown exception");
    }
    if (current_failures() > 0) {
      ++failed;
      std::fprintf(stderr, "[FAIL] %s\n", t.name);
    } else {
      std::printf("[ ok ] %s\n", t.name);
    }
  }
  std::printf("%d/%d tests passed\n", ran - failed, ran);
  return failed == 0 && ran > 0 ? 0 : 1;
}

}  // namespace check

#define CHECK_CONCAT_(a, b) a##b
#define CHECK_CONCAT(a, b) CHECK_CONCAT_(a, b)

#define TEST_CASE(name)                                                     \
  static void name();                                                       \
  static ::check::Registrar CHECK_CONCAT(name, _registrar)(#name, &name);   \
  static void name()

#define CHECK(cond)                                                         \
  do {                                                                      \
    if (!(cond)) ::check::report(__FILE__, __LINE__, #cond);                \
  } while (0)

#define REQUIRE(cond)                                                       \
  do {                                                                      \
    if (!(cond)) {                                                          \
      ::check::report(__FILE__, __LINE__, #cond);                           \
      throw ::check::RequireFailed{};                                       \
    }                                                                       \
  } while (0)

#define CHECK_EQ(a, b)                                                      \
  do {                                                                      \
    const auto& check_a_ = (a);                                             \
    const auto& check_b_ = (b);                                             \
    if (!(check_a_ == check_b_)) {                                          \
      ::check::report(__FILE__, __LINE__,                                   \
                      std::string(#a " == " #b " (") +                      \
                          ::check::stringify(check_a_, 0) + " vs " +        \
                          ::check::stringify(check_b_, 0) + ")");           \
    }                                                                       \
  } while (0)

#define CHECK_MAIN() \
  int main(int argc, char** argv) { return ::check::run_all(argc, argv); }
