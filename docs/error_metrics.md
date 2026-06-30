# Error Metrics

## Root Mean Square Error (RMSE)

RMSE is a metric used to understand how accurate our calculated results are to the true value. As the name implies, it is the root of the mean of the squared difference of all values in the actual and exact lists. 

$$\sqrt{\frac{1}{N} \sum_{i=0}^{N - 1} {(x_i - y_i)}^2}$$

## Worked Example

Say our exact values that we are trying to measure are $[2,2,2,2]$, but what we actually get is $[0,4,0,4]$.

Because our measured results are off, we want to find out the error, which we can do using the error metric: 'RMSE'.

$$\sqrt{\frac{1}{N} \sum_{i=0}^{N - 1} {(x_i - y_i)}^2}$$

To go through this formula in steps, we can first find the differences between the exact and actual values.

 - For value 1: $2 - 0 = 2$
 - For value 2: $2 - 4 = -2$
 - For value 3: $2 - 0 = 2$
 - For value 4: $2 - 4 = -2$

Next, we square all values.

 - For value 1: $2^2=4$
 - For value 2: $(-2)^2=4$
 - For value 3: $2^2=4$
 - For value 4: $(-2)^2=4$

Now we have the 'SE' of 'RMSE', the square errors. Next we must find the mean.

The mean is calculated by the sum divided by the value count. The sum is $4+4+4+4=16$ and the value count is $4$, so the mean is $16/4=4$, which is the 'MSE' of 'RMSE'.

The final step is to find the root, which is outermost symbol in the original equation: $\sqrt{}$. The square root of $4$ is $2$, so our root mean square error is **2.0**.
