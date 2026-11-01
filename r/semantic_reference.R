# R reference implementation for Zig parity (not a native Zig binding).
sl_softmax <- function(scores) {
  stopifnot(is.matrix(scores), nrow(scores) > 0, ncol(scores) >= 2, all(is.finite(scores)))
  shifted <- scores - apply(scores, 1, max)
  e <- exp(shifted)
  e / rowSums(e)
}
sl_entropy <- function(probabilities) {
  stopifnot(is.matrix(probabilities), ncol(probabilities) >= 2,
            all(is.finite(probabilities)), all(probabilities >= 0),
            all(abs(rowSums(probabilities) - 1) < 1e-8))
  -rowSums(ifelse(probabilities > 0, probabilities * log(pmax(probabilities, .Machine$double.xmin)), 0)) /
    log(ncol(probabilities))
}
scores <- matrix(c(1, 2, 3, 0, -1, 2), nrow=2, byrow=TRUE)
p <- sl_softmax(scores)
stopifnot(all(abs(rowSums(p)-1) < 1e-12))
print(data.frame(token=seq_len(nrow(p)), entropy=sl_entropy(p)))
