# SLP v1 binary framing: fixed 32-byte header, row-major float64 payload.
slp_encode <- function(x, op = 1L, request_id = 0) {
  stopifnot(is.matrix(x), is.double(x), all(is.finite(x)),
            nrow(x) > 0, ncol(x) >= 2, op %in% c(1L,2L),
            request_id >= 0, request_id < 2^53)
  payload <- writeBin(as.double(t(x)), raw(), size=8L, endian="little")
  if (length(payload) > 128 * 1024 * 1024) stop("Payload too large")
  u32 <- function(v) writeBin(as.integer(v), raw(), size=4L, endian="little")
  # Request IDs are constrained to <=2^32-1 for exact portable R encoding.
  if (request_id > 2^32-1) stop("R encoder currently supports 32-bit request IDs")
  # Explicit low/high 32-bit words without signed integer overflow.
  u64 <- function(v) {
    low <- v %% 2^32; high <- floor(v / 2^32)
    c(as.raw(floor(low / 256^(0:3)) %% 256),
      as.raw(floor(high / 256^(0:3)) %% 256))
  }
  c(charToRaw("SLP1"), as.raw(c(1L, op, 1L, 0L)),
    u32(nrow(x)), u32(ncol(x)), u64(length(payload)), u64(request_id), payload)
}
