// Catch C++ exceptions (FlashInfer throws from host-side dispatch) before they
// cross the extern "C" boundary — a foreign exception reaching Rust aborts the
// process with no message.
#pragma once

#include <exception>

void openinfer_ffi_set_last_error(const char* what);

// Entering a guard clears the previous message so a -1 seen by Rust never
// reads a stale what() from an earlier call on the same thread.
#define OPENINFER_FFI_GUARD_BEGIN \
  openinfer_ffi_set_last_error(""); \
  try {
#define OPENINFER_FFI_GUARD_END(ret_on_throw)              \
  }                                                        \
  catch (const std::exception& e) {                        \
    openinfer_ffi_set_last_error(e.what());                \
    return ret_on_throw;                                   \
  }                                                        \
  catch (...) {                                            \
    openinfer_ffi_set_last_error("unknown C++ exception"); \
    return ret_on_throw;                                   \
  }
