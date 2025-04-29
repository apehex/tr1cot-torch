# Image Diffusion On Fractal Text

arbitrary division between textual and visual models.

gif with reverse diffusion process

## Encoding Text As RGB

Typically, text is represented in the Unicode format, which is 32 bits long.

As of 2025, there are [only 299,056 codepoints allocated][wiki-unicode] compared to the theorical 4,294,967,296 capacity.

This means that the available codepoints are located on the lower end of the Unicode: most of the bits are null.

In particular, the most significant byte is always zero: any Unicode character can be represented with only 3 bytes.

```python
np.array(list('Hilbert'.encode('utf-32-be'))).reshape((-1, 4))
# array([[  0,   0,   0,  72],
#        [  0,   0,   0, 105],
#        [  0,   0,   0, 108],
#        [  0,   0,   0,  98],
#        [  0,   0,   0, 101],
#        [  0,   0,   0, 114],
#        [  0,   0,   0, 116]])
```

In turn, these 3 bytes can be interpreted as the RGB components of a color:

![][image-rgb-english]

In the image above, each character is represented by a pixel.
It is mostly blue because ASCII characters (western letters) are the smallest codepoints: the red and green channels are null.

For example CJK characters cover a wider range of codepoints / colors:

```python
np.array(list('ヒルベルト曲線'.encode('utf-32-be'))).reshape((-1, 4))
# array([[  0,   0,  48, 210],
#        [  0,   0,  48, 235],
#        [  0,   0,  48, 217],
#        [  0,   0,  48, 235],
#        [  0,   0,  48, 200],
#        [  0,   0, 102, 242],
#        [  0,   0, 125, 218]])
```

![][image-rgb-japanese]

For the rest of this article, text will be displayed both as characters and RGB pixels.

## 2D Text

=> autoregressive sampling, lots of iterations
=> attention patterns exceedingly long

### The Ideal Case: ASCII Art

ASCII art is the closest text can get to rendering graphics:

![][image-2d-asciiart]

Having a single pixel per character loses the shape of the characters, but the overall scene is still visible.

Much like regular text, a model can learn the underlying "meaning" and the relevant associations of characters.

Here you can 

### Regular Pages

Regular text is more tricky because LLMs view it as a 1D array.

Even though we read in a linear pattern, text is still written on 2D supports like this Wikipedia page:

![][image-2d-split]

On the left the text data as the models views it, in RGB, and on the right the labels for clarity.

This layout is very wasteful, with close to half the area covered by padding.

Also the attention on the height axis jumps from one sentence to the next, hardly capturing relevant data.

### Chunking

Splitting text in fixed size chunks removes the empty area:

![][image-2d-chunk]

But the attention patterns are still broken.

### Folding Along A Space Filling Curve

A common solution is to fold the sentence on itself like a thread:

![][image-2d-hilbert]

The black line above is know as the Hilbert curve.
It covers the whole 64 x 64 square with a line of length 4096.

This layout is especially interesting because it preserves locality on both axes.
You can see that squares of different scales capture n-grams, sentences and sections:

![][image-2d-hilbert-zoom]

Instead of the blunt 1D attention, it is now possible to run convolution layers on this visual representation of text.

Similarly to the sliding window attention, the perceptive field grows with the depth of the layers.
Convolutions start by processing words, then sentences, then sections, etc.

## 3D Text

The 2D scheme is practical to illustrate the concepts, but the Hilbert curve can be generalized to any rank:

![][image-3d-hilbert-curve]

Which allows to encode text as a hypercube of shape $(2\^{o})\^{n}$:

![][image-3d-hilbert-rgb]

[image-2d-asciiart]: .assets/2d/asciiart.png
[image-2d-chunk]: .assets/2d/chunk.english.png
[image-2d-split]: .assets/2d/split.english.png

[image-2d-hilbert]: .assets/2d/hilbert.png
[image-2d-hilbert-attention]: .assets/2d/hilbert.attention.png
[image-2d-hilbert-zoom]: .assets/2d/hilbert.zoom.png
[image-3d-hilbert-curve]: .assets/3d/curve.png
[image-3d-hilbert-rgb]: .assets/3d/rgb.png

[image-rgb-english]: .assets/rgb/english.png
[image-rgb-japanese]: .assets/rgb/japanese.png

[wiki-unicode]: https://en.wikipedia.org/wiki/Plane_(Unicode)#Assigned_characters
