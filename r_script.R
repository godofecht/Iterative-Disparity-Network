left <- scan("image1.txt", nlines=1)
right <- scan("image2.txt",nlines=1)



calc_z <- function (p) {
  ni = 11
  nj = 10
  nk = 1
  units <- ni + nj + nk
  
  z = rep(NA, units)
  
  
  index = (7*(p-1))+1
  z[1:5] = left[index:(index+4)]
  z[6:10] = right[index:(index+4)]
  z[11] = 1.0
  
  ## input to hidden
  n=1
  for (j in 1:nj) {
    tot = 0.0
    for (i in 1:ni) {
      tot = tot + (w[n] * z[i])
      n = n+1
    }
    z[ni+j] = tanh(tot)
  }
  
  ## hidden to output
  k = 1
  tot = 0
  for (j in 1:nj) {
    tot = tot + (w[n] * z[j+ni])
    n = n+1
  }
  z[ni+nj+1] = tot
  z
}


op <- function(p) {
  ## return the last element of the activations
  all <- calc_z(p)
  return (all[length(all)])
}


######################################################################

z0 <- scan('z.0'); w <- scan('wts.0')

z0 <- scan('z.500'); w <- scan('wts.500')
z0 <- scan('z.990'); w <- scan('wts.990')

allops <- sapply(1:1000, op)
plot(allops, z0)
plot(allops)


