# Security Fixes

This document tracks security vulnerabilities discovered and fixed in the MLXR codebase.

## CWE-78: Command Injection Vulnerability

**Date:** 2025-11-06
**Severity:** Medium
**Status:** ✅ Fixed

### Vulnerability Description

The test daemon (`daemon/test_daemon_main.cpp`) was using the `system()` function to create directories with user-controlled input (HOME environment variable):

```cpp
// VULNERABLE CODE (before fix)
system(("mkdir -p \"" + registry_dir + "\"").c_str());
```

This code was vulnerable to command injection (CWE-78) if the HOME environment variable contained special shell characters such as backticks, semicolons, or other metacharacters.

**Example Attack Vector:**
```bash
export HOME="/tmp/test; rm -rf /important/data #"
./test_daemon
```

This could execute arbitrary commands with the privileges of the daemon process.

### Fix Applied

Replaced the `system()` call with the safe C++17 `std::filesystem::create_directories()` function:

```cpp
// SECURE CODE (after fix)
try {
  std::filesystem::create_directories(registry_dir);
} catch (const std::filesystem::filesystem_error& e) {
  std::cerr << "Failed to create registry directory: " << e.what()
            << std::endl;
  return 1;
}
```

**Benefits:**
- ✅ No shell invocation - direct filesystem API call
- ✅ No command injection vulnerability
- ✅ Better error handling with exceptions
- ✅ Cross-platform compatibility
- ✅ More efficient (no shell process overhead)

### Verification

**Before Fix:**
```bash
$ grep -n "system(" daemon/test_daemon_main.cpp
182:  system(("mkdir -p \"" + registry_dir + "\"").c_str());
```

**After Fix:**
```bash
$ grep -n "system(" daemon/test_daemon_main.cpp
(no matches - vulnerability eliminated)
```

### Related Security Standards

- **CERT C++ Coding Standard:** ENV33-C - Do not call system()
- **OWASP:** [Command Injection](https://owasp.org/www-community/attacks/Command_Injection)
- **CWE-78:** [Improper Neutralization of Special Elements used in an OS Command](https://cwe.mitre.org/data/definitions/78.html)
- **CWE-88:** [Improper Neutralization of Argument Delimiters in a Command](https://cwe.mitre.org/data/definitions/88.html)

## Security Best Practices for MLXR Development

### 1. Never Use Shell-Invoking Functions with User Input

**AVOID:**
- `system()`
- `popen()` / `pclose()`
- `execl()`, `execlp()`, `execvp()` (with shell interpretation)
- String concatenation with shell commands

**USE INSTEAD:**
- `std::filesystem` for file operations (C++17)
- `fork()` + `execve()` for process execution (with argument arrays)
- Direct API calls instead of shell commands

### 2. Input Validation and Sanitization

When external input must be used:
- Validate against a strict whitelist
- Use parameterized interfaces (avoid string concatenation)
- Properly escape/quote all variables

### 3. Principle of Least Privilege

- Run daemon with minimal required permissions
- Use sandboxing and capability restrictions
- Validate all environment variables before use

### 4. Secure Shell Script Development

For bash scripts (like `scripts/install_homebrew_deps.sh`):
- Always quote variables: `"$variable"` not `$variable`
- Use hardcoded values in arrays, not user input
- Enable strict mode: `set -euo pipefail`
- Validate all inputs against expected patterns

### Example - Safe vs Unsafe Code

**❌ UNSAFE:**
```cpp
std::string user_path = getenv("USER_PATH");
system(("mkdir -p " + user_path).c_str());  // VULNERABLE
```

**✅ SAFE:**
```cpp
std::string user_path = getenv("USER_PATH");
// Validate path first
if (!is_valid_path(user_path)) {
  throw std::runtime_error("Invalid path");
}
// Use filesystem API
std::filesystem::create_directories(user_path);
```

### 5. Code Review Checklist

Before merging code, verify:
- [ ] No `system()` calls with user-controlled input
- [ ] No `popen()` with user-controlled commands
- [ ] All filesystem operations use `std::filesystem`
- [ ] Shell scripts properly quote all variables
- [ ] Input validation on all external data
- [ ] Error handling for all security-critical operations

## Audit Status

**Last Security Audit:** 2025-11-06

### Findings:
- ✅ C++ codebase: No remaining `system()` or `popen()` calls
- ✅ Shell scripts: All variables properly quoted
- ✅ No sprintf() with %s format using untrusted input
- ✅ All directory creation uses `std::filesystem`

### Recommendations:
1. Consider adding static analysis tools (e.g., clang-tidy) to CI/CD
2. Add security-focused test cases for input validation
3. Document security requirements in CLAUDE.md
4. Regular security audits (quarterly recommended)

## Reporting Security Issues

If you discover a security vulnerability in MLXR:

1. **DO NOT** open a public GitHub issue
2. Report via GitHub Security Advisories (preferred)
3. Or email: [security contact to be added]
4. Include:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if available)

## References

- [CERT C++ Coding Standard](https://wiki.sei.cmu.edu/confluence/display/cplusplus)
- [OWASP Secure Coding Practices](https://owasp.org/www-project-secure-coding-practices-quick-reference-guide/)
- [CWE Top 25 Most Dangerous Software Weaknesses](https://cwe.mitre.org/top25/)
- [C++ Core Guidelines](https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines)
