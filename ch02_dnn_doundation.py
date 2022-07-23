import tensorlayerx as tlx
# Single neuro

## matrix multiplication
x=tlx.constant([[1,2,3]])
w=tlx.constant([[-0.5],[0.2],[0.1]])

print("X:\n",x,"\nShape:",x.shape)
print("W:\n",w,"\nShape",w.shape)

## Bias
b=tlx.constant(0.5)

z=tlx.matmul(x,w)+b

print("Z:\n",z,"\nShape",z.shape)
