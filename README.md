# Journal

# 2026-01-17

Since the network returned near perfect predictions on a simulated dataset already after one epoch
the key to success might lie in either preprocessing the dataset or increasing the depth of the network.

# 2026-01-18

I tried to apply a median filter on the images but this didn't help. The predictions are still total
gibberish. I decided to try to do some statistics on the simulated and real datasets and compare them.
Maybe the problem is that the real dataset needs normalization of some kind.

I also spent some time on researching the topic of increasing the depth of the network and I stumbled upon
some interesting derivations of the U-Net architecture:

- U-Net++ - looks like it increases the depth of the network slightly and makes it recursive by connecting the skip connections vertically;
- Attention U-Net - somehow makes sure that the U-Net puts the most weight on the most likely place in the image for the feature to appear in;
- Transformer U-Net - adds a transformer to the encoder. There is still a standard U-Net encoder and the transformer is added in parallel to it.

I should look into these architectures if preprocessing proves to be the wrong path forward.

Right now I'm trying to see if decreasing the number of color values in the images would increase
the effectivenes of the net. I'm weighting whether it would be better to quantize to a given small set of
possible color values or just to quantize so that the resulting color space fits to an integer
of a given precision. I should probably just try both options and see which one gets me better results.

I also think that I should try to decrease amount of examples to something really low (like 3) and see
whether the net can fit to that. If it won't fit then we really need to look into the color space
compression.

I deacreased the image count to 9 and nothing meaningful came out of this. I think it's a good idea
to look into how the values of the network predictions look like after passing them through
the sigmoid. This will at least tell us whether the net returns some learned values or just zeroes
everywhere or a direct copy-paste of whatever it sees in the image.
