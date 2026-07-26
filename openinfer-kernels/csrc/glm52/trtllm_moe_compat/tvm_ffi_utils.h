#pragma once

#include <sstream>
#include <stdexcept>
#include <utility>

namespace openinfer::trtllm_moe {

class CheckStream {
 public:
  explicit CheckStream(bool failed) : failed_(failed) {}
  CheckStream(const CheckStream&) = delete;
  CheckStream& operator=(const CheckStream&) = delete;
  CheckStream(CheckStream&& other) noexcept
      : failed_(other.failed_), message_(std::move(other.message_)) {
    other.failed_ = false;
  }

  template <typename T>
  CheckStream& operator<<(T&& value) {
    if (failed_) message_ << std::forward<T>(value);
    return *this;
  }

  ~CheckStream() noexcept(false) {
    if (failed_) throw std::runtime_error(message_.str());
  }

 private:
  bool failed_;
  std::ostringstream message_;
};

}  // namespace openinfer::trtllm_moe

#define TVM_FFI_ICHECK(condition) \
  ::openinfer::trtllm_moe::CheckStream(!(condition))
#define TVM_FFI_ICHECK_LE(lhs, rhs) \
  ::openinfer::trtllm_moe::CheckStream(!((lhs) <= (rhs)))
#define TVM_FFI_LOG_AND_THROW(error_type) \
  ::openinfer::trtllm_moe::CheckStream(true)
