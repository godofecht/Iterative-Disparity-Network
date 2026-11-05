# Independent R reference for the C++ 11-10-1 stereo micro-network.
# Usage: Rscript r_script.R left.csv right.csv results/weights.csv results/r_predictions.csv [stride]
args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 4L || length(args) > 5L)
    stop("Usage: Rscript r_script.R left.csv right.csv weights.csv output.csv [stride]")

left <- scan(args[1], quiet = TRUE)
right <- scan(args[2], quiet = TRUE)
w <- read.csv(args[3], check.names = FALSE)$weight
stride <- if (length(args) == 5L) as.integer(args[5]) else 7L
if (length(w) != 120L || length(left) != length(right) ||
    length(left) %% stride != 0L || stride < 5L)
    stop("Invalid input shape or weights")

n <- length(left) / stride
patches <- matrix(0, nrow = n, ncol = 11)
for (p in seq_len(n)) {
    index <- (p - 1L) * stride
    patches[p, 1:5] <- left[index + 1:5]
    patches[p, 6:10] <- right[index + 1:5]
    patches[p, 11] <- 1
}
# Hidden-major ordering: eleven incoming weights for each hidden unit,
# then ten hidden-to-output weights.
input_weights <- matrix(w[1:110], nrow = 11, ncol = 10)
hidden <- tanh(patches %*% input_weights)
y <- as.vector(hidden %*% w[111:120])
write.csv(data.frame(prediction = y), args[4], row.names = FALSE, quote = FALSE)

# The spatial (not temporal) averages wrap around the 1D output lattice.
make_kernel <- function(half_life) {
    distances <- pmin(0:(n - 1L), n - 0:(n - 1L))
    k <- exp(-log(2) * distances / half_life)
    k / sum(k)
}
convolve <- function(values, weights) {
    vapply(seq_len(n), function(i) {
        distances <- (seq_len(n) - i) %% n + 1L
        sum(weights[distances] * values)
    }, numeric(1))
}
short <- convolve(y, make_kernel(32))
long <- convolve(y, make_kernel(3200))
U <- sum((short - y)^2) / 2
V <- sum((long - y)^2) / 2
F <- log((V + 1e-12) / (U + 1e-12))
cat(sprintf("R reference: samples=%d U=%.12g V=%.12g F=%.12g\n", n, U, V, F))
