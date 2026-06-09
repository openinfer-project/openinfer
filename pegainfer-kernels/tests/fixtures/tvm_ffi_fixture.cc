#include <tvm/ffi/function.h>

namespace ffi = tvm::ffi;

int64_t AddOneScalar(int64_t x) { return x + 1; }

int64_t ApplyCallback(ffi::Function callback, int64_t x) {
  return callback(x).cast<int64_t>();
}

int64_t CallRegisteredHostAddThree(int64_t x) {
  ffi::Function callback =
      ffi::Function::GetGlobalRequired("pegainfer.testing.add_three");
  return callback(x).cast<int64_t>();
}

int64_t CallRegisteredHostFailIfNegative(int64_t x) {
  ffi::Function callback =
      ffi::Function::GetGlobalRequired("pegainfer.testing.fail_if_negative");
  return callback(x).cast<int64_t>();
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(add_one_scalar, AddOneScalar);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(apply_callback, ApplyCallback);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(
    call_registered_host_add_three,
    CallRegisteredHostAddThree);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(
    call_registered_host_fail_if_negative,
    CallRegisteredHostFailIfNegative);
