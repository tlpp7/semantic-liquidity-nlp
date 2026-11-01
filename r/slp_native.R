# R <-> Zig C ABI. The R matrix is column-major; Zig expects row-major.
sl_load <- function(path = "./libsemantic_core.so") {
  dyn.load(normalizePath(path, mustWork = TRUE))
  invisible(TRUE)
}
sl_softmax_zig <- function(x) {
  stopifnot(is.matrix(x), is.double(x), all(is.finite(x)), nrow(x) > 0, ncol(x) >= 2)
  rows <- nrow(x); cols <- ncol(x)
  input <- as.double(t(x))
  result <- .C("sl_softmax_f64", input = input, output = double(length(input)),
               rows = as.integer(rows), cols = as.integer(cols),
               status = integer(1), DUP = FALSE)
  # .C cannot safely call an ABI with size_t dimensions and an integer return.
  # This wrapper is intentionally disabled until the C-int32 adapter is compiled.
  stop("Use sl_softmax_zig_safe() with the dedicated R C ABI adapter.")
}
sl_softmax_zig_safe <- function(x) {
  stopifnot(is.matrix(x), is.double(x), all(is.finite(x)), nrow(x) > 0, ncol(x) >= 2)
  result <- .C("sl_softmax_r", input = as.double(t(x)),
               output = double(length(x)), rows = as.integer(nrow(x)),
               cols = as.integer(ncol(x)), status = integer(1))
  if (result$status != 0L) stop(paste("Zig error", result$status))
  matrix(result$output, nrow=nrow(x), ncol=ncol(x), byrow=TRUE)
}
sl_entropy_zig_safe <- function(p) {
  stopifnot(is.matrix(p), is.double(p), ncol(p) >= 2,
            all(is.finite(p)), all(p >= 0), all(abs(rowSums(p)-1) < 1e-8))
  result <- .C("sl_entropy_r", input = as.double(t(p)),
               output = double(nrow(p)), rows = as.integer(nrow(p)),
               cols = as.integer(ncol(p)), status = integer(1))
  if (result$status != 0L) stop(paste("Zig error", result$status))
  result$output
}
