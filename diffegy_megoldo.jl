function RK4M(f::Function, t, Y, step, misc)
  #println("input-check")
  #checkNaN(Y)
  #println("\n")
  k1 = f(t, Y, misc)
  k2 = f(t .+ step ./ 2, Y + step ./ 2 * k1, misc)
  k3 = f(t .+ step ./ 2, Y + step ./ 2 * k2, misc)
  k4 = f(t .+ step, Y + step * k3, misc)
  Y += step * (k1 + 2 * k2 + 2 * k3 + k4) / 6
  #println("output-check")
  #checkNaN(Y)
  #println("\n")
  return Y, t + step
end
