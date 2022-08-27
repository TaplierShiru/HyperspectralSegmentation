# Hyperspectral segmentation with PyTorch
One of the research-project with the main goal - segmentation of the hyperspectral images.

Segmentation done using PyTorch with PyTorch-Lighting and MLComet for results exploring. 
Beside segmentation, my main goal were to compare pixel strategy segmentation and CNN segmentation.

What is pixel strategy segmentation?

Its then your cut input image (hyperspectral) into block/patches, and classificate each block/patch separate from others. 
Sometimes authors classificate only center pixel (most of the time) and at the end build full mask from each block/patch result. 
Its very time consuming train these kind of the networks, the main problem is  with data pipeline, because this strategy could eat huge amount of memory,
so you need somehow cut input image inplace into block/patches. There is some code with optimizations, that work very well, but its tested on 128gb RAM memory server.
So with lower RAM, it couldn't be done - I think.

CNN Segmentation?

Its classic segmentation, there you to input image - output mask for each pixel, and its done via using full image as input (no cut to block/patches).


