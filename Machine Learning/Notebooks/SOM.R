# quick SOM example, Alex Ioannidis
# note, you will need to install the two libraries below first

library(rgl)
library(princurve)

# generate points on a helix with random gaussian noise
s = (seq(0,2*pi,length=200))
x1 = cos(s) + 0.1*rnorm(200)
x2 = sin(s) + 0.1*rnorm(200)
x3 = s + 0.1*rnorm(200)
plot3d(x1,x2,x3)

prin_curve = principal.curve(cbind(x1, x2, x3))
# show that point generation worked
lines3d(prin_curve$s, col="blue")

# generate prototypes along 1st principal component axis
n_proto = 30
pc = prcomp(cbind(x1, x2, x3))
proto = t(pc$rotation[,1] %*% t(seq(min(pc$x[,1]), max(pc$x[,1]), length=n_proto)) + pc$center)
plot3d(x1,x2,x3)
points3d(proto, col="red")


x = cbind(x1, x2, x3)[sample(1:length(x1), replace=TRUE),]  # randomize point cloud
h = function(z){1/sqrt(2*pi*r)*exp(-z^2/(2*r^2))}  # Gaussian neighborhood function
r = 1    # neighborhood scaling parameter
l_index = 1:n_proto
proto_dif = 1

while( proto_dif > .00001 ){

a = proto_dif	# learning rate
proto_prev = proto

for (i in 1:length(x1)) {   # loop through observations

xi_mk = -t(t(proto) - x[i,])    # distance vectors from current observation to prototypes
j = which.min(rowSums(xi_mk^2))   # find closest prototype to current observation
proto = proto + a*h(abs(l_index - j)) * xi_mk	# update all prototypes

}
proto_dif = norm(proto-proto_prev, type = "M")   # loop until matrix of protype coords. converges

}

# plot output of SOM against the original points
plot3d(x1,x2,x3)
points3d(proto, col="red")
lines3d(proto, col = "red")