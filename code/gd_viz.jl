using Flux, Plots

loss(x) = @. 0.5 * sin(5*x[1]) + 0.1 * (x[1] - 0.5)^4 - 0.25 * x[1]^2 - 0.75 * x[1] + 3
x = [-1.9]  
opt = Descent(0.01)  
state = Flux.setup(opt, x)  
xs = range(-3, 4, length=100)
ys = loss.(xs) # xs .^ 2  
plt = plot(xs, ys, label="Loss Function", linewidth=2, legend=false)
ball = scatter!([x[1]], [loss(x)], color=:red, markersize=6, label="Ball")  
for i in 1:50
    grads = Flux.gradient(loss, x)[1]    
    Flux.update!(state, x, grads)        
    plt = plot(xs, ys, label="Loss Function", linewidth=2, legend=false)
    scatter!([x[1]], [loss(x)], color=:red, markersize=6, label="Ball", markerstrokewidth=0)  
    title!("Iteration $i, x=$(round(x[1], digits=3))")
    display(plt)  
    sleep(0.2)  
end
