using CUDA
import Base.+, Base.*, Base./

struct userinputs
  Nx::Int
  Nt::Int
  cry::Int
  sigma_t::Float64
  sigma_x::Float64
  lambda0::Float64
  I0::Float64
  STR::String
  gamma::Float64
  dz::Float64
  z_end::Float64
  x::Vector
  t::Vector
end

struct gaussVars
  E0::Float64
  t::Array{Float64,2}
  sigma_t::Float64
  x::Array{Float64,2}
  sigma_x::Float64
  omega0::Float64
  gamma::Float64
  lambda0::Float64
  cry::Int
end


struct compositeInputGPU
  Akxo::CuArray{ComplexF64,2}
  ATHz_kx_o::CuArray{ComplexF64,2}
  ASH::CuArray{ComplexF64,2}
end

function checkNaN(gpuComposite::compositeInputGPU)
  println("$(sum(isnan.(gpuComposite.Akxo))) NaNs in pump")
  println("$(sum(isnan.(gpuComposite.ATHz_kx_o))) NaNs in THz")
  println("$(sum(isnan.(gpuComposite.ASH))) NaNs in SH")
end

struct compositeInput
  Akxo::Array{ComplexF64,2}
  ATHz_kx_o::Array{ComplexF64,2}
  ASH::Array{ComplexF64,2}
  function compositeInput(inp::compositeInputGPU)
    if sum(isnan.(inp.Akxo)) > 0
      println("NaN values not handled at fetch!")
    end
    new(Array(inp.Akxo), Array(inp.ATHz_kx_o), Array(inp.ASH))
  end
end
function +(a::compositeInput, b::compositeInput)
  return compositeInput(a.Akxo .+ b.Akxo, a.ATHz_kx_o .+ b.ATHz_kx_o, a.ASH .+ b.ASH)
end

function *(a::compositeInput, b::compositeInput)
  return compositeInput(a.Akxo .* b.Akxo, a.ATHz_kx_o .* b.ATHz_kx_o, a.ASH .* b.ASH)
end
function *(a::Float64, b::compositeInput)
  return compositeInput(a .* b.Akxo, a .* b.ATHz_kx_o, a .* b.ASH)
end
function *(a::Int, b::compositeInput)
  return compositeInput(a .* b.Akxo, a .* b.ATHz_kx_o, a .* b.ASH)
end

function +(a::compositeInput, b::Float64)
  return compositeInput(a.Akxo .+ b, a.ATHz_kx_o .+ b, a.ASH .+ b)
end

function /(a::compositeInput, b::Int)
  return compositeInput(a.Akxo ./ b, a.ATHz_kx_o ./ b, a.ASH ./ b)
end

# MethodError: no method matching *(::Float64, ::compositeInputGPU)
function *(a::Float64, b::compositeInputGPU)
  return (compositeInputGPU(a .* b.Akxo, a .* b.ATHz_kx_o, a .* b.ASH))
end

# ERROR: MethodError: no method matching +(::compositeInputGPU, ::compositeInputGPU)
function +(a::compositeInputGPU, b::compositeInputGPU)
  return compositeInputGPU(a.Akxo .+ b.Akxo, a.ATHz_kx_o .+ b.ATHz_kx_o, a.ASH .+ b.ASH)
end

#ERROR: MethodError: no method matching *(::Int64, ::compositeInputGPU)

function *(a::Int, b::compositeInputGPU)
  return compositeInputGPU(a .* b.Akxo, a .* b.ATHz_kx_o, a .* b.ASH)
end

#  MethodError: no method matching /(::compositeInputGPU, ::Int64)
function /(a::compositeInputGPU, b::Int)
  return compositeInputGPU(a.Akxo ./ b, a.ATHz_kx_o ./ b, a.ASH ./ b)
end
# ERROR: MethodError: no method matching +(::compositeInputGPU, ::CuArray{ComplexF64, 2, CUDA.Mem.DeviceBuffer})

function +(a::compositeInputGPU, b::CuArray{ComplexF64,2})
  return compositeInputGPU(a.Akxo .+ b, a.ATHz_kx_o .+ b, a.ASH .+ b)
end

struct fourierOperations
  fft_t_o
  fft_x_kx
  ifft_o_t
  ifft_kx_x
  fast_conv_plan
  fast_conv_fft_plan
end

struct naturalConstants
  e0::Float64
  c0::Float64
  function naturalConstants()
    new(8.854187817e-12, 3e8)
  end
end

struct pumpFieldConstants
  kx_omega::Array{Float64,2}
  kz_omega::Array{Float64,2}
end

struct THzFieldConstants
  alpha::Array{Float64,2}
  kz_omegaTHz::Array{Float64,2}
end

struct SHFieldConstants
  kx_omegaSHG::Array{Float64,2}
  kz_omegaSHG::Array{Float64,2}
end

struct runTimeConstants
  kxMax::Float64
  cx::Array{Float64,2}
  d_eff::Float64
  khi_eff::Float64
  dOmega::Float64
  padding::Array{Float64,2}
  SHG_SHIFT::Int
  ckx::Array{Float64,2}
  comega::Array{Float64,2}
  comegaTHz::Array{Float64,2}
  comegaSHG::Array{Float64,2}
  omegaMax::Float64
  lambda0::Float64
  omega0::Float64
  cry::Int
end

struct miscInputs
  FOPS::fourierOperations
  NC::naturalConstants
  RTC::runTimeConstants
  TFC::THzFieldConstants
  PFC::pumpFieldConstants
  SFC::SHFieldConstants
end

struct runTimeConstantsGPU
  kxMax::Float64
  cx::CuArray{Float64,2}
  d_eff::Float64
  khi_eff::Float64
  dOmega::Float64
  padding::CuArray{Float64,2}
  SHG_SHIFT::Int
  ckx::CuArray{Float64,2}
  comega::CuArray{Float64,2}
  comegaTHz::CuArray{Float64,2}
  comegaSHG::CuArray{Float64,2}
  omegaMax::Float64
  lambda0::Float64
  omega0::Float64
  cry::Int
  function runTimeConstantsGPU(kxMax, cx, d_eff, khi_eff, dOmega, padding, SHG_SHIFT, ckx, comega, comegaTHz, comegaSHG, omegaMax, lambda0, omega0, cry)
    new(kxMax, CuArray(cx), d_eff, khi_eff, dOmega, CuArray(padding), SHG_SHIFT, CuArray(ckx), CuArray(comega), CuArray(comegaTHz), CuArray(comegaSHG), omegaMax, lambda0, omega0, cry)
  end
end

struct THzFieldConstantsGPU
  alpha::CuArray{Float64,2}
  kz_omegaTHz::CuArray{Float64,2}
  function THzFieldConstantsGPU(alpha, kz_omegaTHz)
    new(CuArray(alpha), CuArray(kz_omegaTHz))
  end
end

struct pumpFieldConstantsGPU
  kx_omega::CuArray{Float64,2}
  kz_omega::CuArray{Float64,2}
  function pumpFieldConstantsGPU(kx_omega, kz_omega)
    new(CuArray(kx_omega), CuArray(kz_omega))
  end
end

struct SHFieldConstantsGPU
  kx_omegaSHG::CuArray{Float64,2}
  kz_omegaSHG::CuArray{Float64,2}
  function SHFieldConstantsGPU(kx_omegaSHG, kz_omegaSHG)
    new(CuArray(kx_omegaSHG), CuArray(kz_omegaSHG))
  end
end

struct miscInputsGPU
  FOPS::fourierOperations
  NC::naturalConstants
  RTC::runTimeConstantsGPU
  TFC::THzFieldConstantsGPU
  PFC::pumpFieldConstantsGPU
  SFC::SHFieldConstantsGPU
end
