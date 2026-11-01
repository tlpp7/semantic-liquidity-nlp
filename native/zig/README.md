# Zig native kernel (experimental)

Requires a compatible Zig toolchain; Zig versions can change syntax and ABI tooling.

```bash
zig build-lib -dynamic -O ReleaseFast native/zig/semantic_core.zig -femit-bin=libsemantic_core.so
```

On macOS use `libsemantic_core.dylib`, on Windows `semantic_core.dll`. No compiled binary is committed. Exported functions: `sl_softmax_f64` and `sl_entropy_f64`, returning 0 on success, 1 on invalid shape/pointers and 2 on invalid numeric input.

The C ABI accepts row-major contiguous double arrays. Python and R wrappers should check shapes before passing pointers. This kernel is CPU-only and not an autograd implementation.

## R native integration

Compile both kernels and the R-specific `.C` adapter into one shared library. Example on Linux:

```bash
zig build-lib -dynamic -O ReleaseFast native/zig/semantic_core.zig native/zig/r_abi.zig -femit-bin=libsemantic_core.so
Rscript -e 'source("r/slp_native.R"); sl_load(); x <- matrix(c(1,2,3,4,5,6),2,3); print(sl_softmax_zig_safe(x))'
```

The R adapter exports `sl_softmax_r` and `sl_entropy_r` with pointer-only arguments compatible with R's `.C` convention. Use the `*_safe` functions; the native ABI is not a direct `.Call` interface. Build commands and Zig compatibility must be verified against your installed Zig release.

The binary SLP v1 header is defined in `slp_protocol.zig`, with independent Python and R encoders. **Framing is not a network server**; TCP/IPC transport, authentication, integrity checks and timeouts remain future work.
