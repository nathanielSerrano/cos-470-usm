"""
Simple forward feeding neural network.  
Input and output layer only.  x --> y.  Activation function is the identity.
With identity activation, da/dz = 1.  

 dC      dC     da     dz 
---- =  ---- . ---- . ----  (Chain Rule)
 dw      da     dz     dw

"""
σ(z) = z; # 1 / (1 + exp(-z))
C(a) = (a - y)^2
dcda(a) = 2*a - 1
dadz(z) = 1; #  σ(z) * (1 - σ(z))
dzdw = x

w = 0.8
y = 0.5
x = 1.5

for i = 1:10
  z = w*x
  a = σ(z)
  dcdw = dcda(a) * dadz(z) * dzdw
  w = w - 0.1 * dcdw
  println(w)
end
