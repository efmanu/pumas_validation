using NPZ
using Plots
using Flux
using Flux: Data.DataLoader
using Flux: @epochs

root = "D:\\Pumas\\Projects\\SubspaceInference\\Validation";

include("main.jl")



cd(root);
py_weigths = npzread("abc.npy");

data_ld = npzread("data.npy");
x, y = (data_ld[:, 1], data_ld[:, 2]);

function features(x)
    return hcat(x./2, (x./2).^2)
end

scatter(data_ld[:,1], data_ld[:,2], color=:red, pallette=:seaborn_rocket_gradient);

f = features(x);

f = reshape(f,2,:);
y = reshape(y,1,:);
data =  DataLoader(f,y, batchsize=50, shuffle=true);

dims = [2, 200, 50, 50, 50, 1]
layers = [Dense(dims[i], dims[i+1], Flux.relu) for i in 1:length(dims)-1];
m = Chain(layers...)

θ, re = Flux.destructure(m);
θ = py_weigths;

m = re(θ);

# m = Chain(Dense(2,4),Dense(4,1)) #model

L(x, y) = Flux.Losses.mse(m(x), y) 

ps = Flux.params(m) #model parameters

opt = Momentum(0.01, 0.95)
# opt = ADAM()


epoches = 3000
my_train(epoches, L, ps, data, opt, cyclic_lr = false)

