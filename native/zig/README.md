# Zig native kernel (experimental)

Requires a compatible Zig toolchain; Zig versions can change syntax and ABI tooling.

```bash
zig build-lib -dynamic -O ReleaseFast native/zig/semantic_core.zig -femit-bin=libsemantic_core.so
```

On macOS use `libsemantic_core.dylib`, on Windows `semantic_core.dll`. No compiled binary is committed. Exported functions: `sl_softmax_f64` and `sl_entropy_f64`, returning 0 on success, 1 on invalid shape/pointers and 2 on invalid numeric input.

The C ABI accepts row-major contiguous double arrays. Python and R wrappers should check shapes before passing pointers. This kernel is CPU-only and not an autograd implementation.
