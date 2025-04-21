"""
Simple forward feeding neural network.  
Input and output layer only.  x --> y.  Activation function is the identity.
With identity activation, da/dz = 1.  

 dC      dC     da     dz 
---- =  ---- . ---- . ----  (Chain Rule)
 dw      da     dz     dw


@Julia: 1.11.5
@Author: james.quinlan
"""

σ(z) = z; # 1 / (1 + exp(-z))
C(a) = (a - y)^2
dcda(a) = 2*a - 1
dadz(z) = 1; #  σ(z) * (1 - σ(z))
dzdw = x

x = 1.5
y = 0.5

w = 0.8
for i = 1:10
  z = w*x
  a = σ(z)
  dcdw = dcda(a) * dadz(z) * dzdw
  w = w - 0.1 * dcdw
  println(w)
end
