include("typedefs.jl")

function gauss_impulzus(inputs::gaussVars)
  c0 = 3e8
  return inputs.E0 .* exp.(-2 .* log(2) .* inputs.t .^ 2 ./ inputs.sigma_t .^ 2) .*
         exp.(-inputs.x .^ 2 ./ inputs.sigma_x .^ 2) .* exp.(1im .* inputs.omega0 .* inputs.t) .* exp.(-1im .* sin(inputs.gamma) .* inputs.x ./ inputs.lambda0 .* 2 .* pi * neo(inputs.lambda0, 300, inputs.cry))
end
function gauss_impulzus_omega0(inputs::gaussVars)
  return inputs.E0 .* exp.(-2 .* log(2) .* inputs.t .^ 2 ./ inputs.sigma_t .^ 2) .*
                 exp.(-inputs.x .^ 2 ./ inputs.sigma_x .^ 2) .* exp.(-1im .* sin(inputs.gamma) .* inputs.x ./ inputs.lambda0 .* 2 .* pi * neo(inputs.lambda0, 300, inputs.cry))
end
