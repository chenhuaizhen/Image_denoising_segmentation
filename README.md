# Image_denoising_segmentation
Use Gibbs sampling and variational inference to denoise the image and use EM to segment the image

## Requirements
python 3-6  
scipy  
cv2  

## Image_denoising
### Gibbs sampling
#### Algorithm
![image](https://github.com/chenhuaizhen/Image_denoising_segmentation/raw/master/image/gibbs.png)

where the 𝑛𝑏𝑟(𝑖) means all connected nodes (neighbors) of node 𝑥𝑖. Due to use the Ising model, the pairwise potential term 𝛹s𝑖 (𝑥𝑖, 𝑥s)=exp(𝐽𝑥𝑖𝑥s), and the local evidence term 𝛹𝑖(𝑥𝑖)=𝒩(𝑦𝑖|𝑥𝑖, 𝜎^2) where the 𝑦𝑖 is the observed state. 
So the final term of p in the pseudo-code is:  
![image](https://github.com/chenhuaizhen/Image_denoising_segmentation/raw/master/image/gibbs_2.png)

And the 𝑁𝑒𝑥𝑡𝑆𝑖𝑡𝑒(𝑗) function can just return j. But this may cause artifacts as the pixels left and above of 𝑥𝑖 will change before it does, while the pixels right and below will not have changed. Instead, first pass over all of the "even" pixels (i is even) and then make a pass over all of the "odd" pixels will help a lot, which can also simplify the iterations into two matrix operations(instead of n times iterations).  
And when meet the boundary pixels, can try to "wrap" the boundary to deal with this problem, which means that the neighbors wrap around to the other side of the image.  

#### Result
![image](https://github.com/chenhuaizhen/Image_denoising_segmentation/raw/master/image/gibbs_res.png)

### Variational Inference
#### Algorithm
![image](https://github.com/chenhuaizhen/Image_denoising_segmentation/raw/master/image/VI.png)

where the 𝑛𝑏𝑟(𝑖) means all connected nodes (neighbors) of node 𝑥𝑖. 𝜇𝑖 is mean value of node i. And the main update rule is: 
![image](https://github.com/chenhuaizhen/Image_denoising_segmentation/raw/master/image/VI_2.png)

Based on the 𝜇, it will be iterated until convergence.  
The 𝑁𝑒𝑥𝑡𝑆𝑖𝑡𝑒(𝑗) function and boundary problem are the same as the Gibbs sampling algorithm.  

#### Result
![image](https://github.com/chenhuaizhen/Image_denoising_segmentation/raw/master/image/VI_res.png)

## Image_segmentation
### Expectation-Maximization
#### Algorithm
![image](https://github.com/chenhuaizhen/Image_denoising_segmentation/raw/master/image/EM.png)

where   
![image](https://github.com/chenhuaizhen/Image_denoising_segmentation/raw/master/image/EM_2.png)

#### Result
![image](https://github.com/chenhuaizhen/Image_denoising_segmentation/raw/master/image/EM_res.png)
