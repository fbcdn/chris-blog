---
layout: post
title: "A Case Study on Ethics and Bias in Machine Learning"
permalink: /2020/case-study-on-ethics-and-bias-in-machine-learning/
description: "Most of the issues dealing with machines and bias are insidious - crime, credit, or recidivism prediction, to name a few - where bias is hidden and implicit. This piece is about a foundational (and thankfully lower-stakes) example of bias in machine learning, a discussion of the ethical and social issues that come with it, and things that could be done better in the future."
date: 2020-06-21 00:00:00
tags:
- machine-learning
- bias
- ethics
- gan
- opinion
---

Recently, Denis Malimonov ([@tg_bomze](https://twitter.com/tg_bomze/status/1274098682284163072)) released a project called "Face Depixelizer" which implements [PULSE](https://github.com/adamian98/pulse) supplemented with pretrained data on [Google Colab](https://colab.research.google.com/github/tg-bomze/Face-Depixelizer/blob/master/Face_Depixelizer_Eng.ipynb) for ease-of-use. The project isn't net-new - it's just democratizing access to prior work by the PULSE team and providing some presets. PULSE + Denis' training data can successfully produce "believable" results in many cases, and can generate realistic faces even from 1990s-era game sprites readily.

{:refdef: style="text-align: center;"}
!['De-pixelated' sprite of Wolfenstein 3D's "William Joseph B.J. Blazkowicz on a grey background"](/2020/case-study-on-ethics-and-bias-in-machine-learning/wolfenstein.png)
{: refdef}

However, it incited major criticism centered around two points.

First, there are significant ethical concerns around reconstructing people's faces from pixelated images. For example, identifying people in situations where they wished to remain anonymous by pixelating their faces - protesters, whistleblowers, etc. Thankfully this isn't possible with this tool - hence, "Face Depixelizer" is a very misleading name, and I'll be putting it in quotation marks throughout this article. We'll cover some latent ethical concerns that arise *just from that name*, though.

Second, this particular model tends to output traditionally white faces and features. For example, a pixelated photo of Barack Obama would reliably generate a face that looked like the offspring of Tobey Maguire and Sean Connery - this was first noted by Cédric Sütterlin ([@Chicken3gg](https://twitter.com/Chicken3gg/status/1274314622447820801)), and follow-ups by Ken Chic ([@bitcashio](https://twitter.com/bitcashio/status/1274408339602993155/)) confirmed that it happened consistently under different conditions.

{:refdef: style="text-align: center;"}
!['De-pixelated' photo of Barack Obama which comes out a little too mayonnaise-y.](/2020/case-study-on-ethics-and-bias-in-machine-learning/barack-obama.png)
{: refdef}

Further demonstrations, including a fairly high-resolution photo of Muhammad Ali processed by Zain Amro ([@zainamro](https://twitter.com/zainamro/status/1274361534886469632/)), almost always resulted in an output face with exclusively white-presenting features. Does that make it racist? In this case I'm going to go with "[yes](https://twitter.com/finmckeown/status/1275047175937052672), *accidentally*" - though keep the social outrage low, I believe this is not really Malimonov's fault.

This piece is a comprehensive look at the "Face Depixelizer" debacle through the lens of a security engineer, and is a broad engineering brief on the subject for non-ML engineers. So, let's learn more about how "Face Depixelizer" works so we can understand what is happening, why it's happening, and figure out some things that could be done better - both here and in the machine learning community at large.

## Technical Brief

So, "Face Depixelizer" is effectively a portable and pretrained version of PULSE: Self-Supervised Photo Upsampling via Latent Space Exploration of Generative Models by Duke University researchers Sachit Menon, Alexandru Damian, Shijia Hu, Nikhil Ravi, and Cynthia Rudin ([GitHub](https://github.com/adamian98/pulse), [paper](https://arxiv.org/abs/2003.03808)). PULSE's mission statement is: "Given a low-resolution input image, PULSE searches the outputs of a generative model (here, StyleGAN) for high-resolution images that are perceptually realistic and downscale correctly."

StyleGAN itself was produced by NVIDIA researchers Tero Karras, Samuli Laine, and Timo Aila ([GitHub](https://github.com/NVlabs/stylegan), [paper](https://arxiv.org/abs/1812.04948)). There are some great dives on how StyleGAN works such as Rani Hora's [article](https://towardsdatascience.com/explained-a-style-based-generator-architecture-for-gans-generating-and-tuning-realistic-6cb2be0f431), but the most we need to know about StyleGAN in the context of this article is that it generates net-new images (ex. not just spitting out images it's seen before) from a given seed input, with the ability to manipulate different visual features of the image.

To elucidate, a thousand-foot overview of how PULSE (and therefore Face Depixelizer) works is that, for a given face to "depixelize" (I<sub>LR</sub>), random seed z<sub>init</sub>, and x = 0: 

* The training round starts with a new face I<sub>x</sub> being "hallucinated" by StyleGAN with z<sub>x</sub> seed
* The new face is downsampled (pixelated) and loss is calculated by the difference between the downsampled image and I<sub>LR</sub>
* That loss is applied to z<sub>x</sub> according to the learning rate, creating z<sub>x+1</sub>
* Repeat the above until x == the maximum allowable rounds. The final image, I<sub>final</sub>, is a high-resolution "hallucinated" image which is very similar to I<sub>LR</sub> when downsampled

This is illustrated by the PULSE team's demo, showing the transformation from a totally random initial image to a believable final image. A full example is available [here](https://github.com/adamian98/pulse/blob/445013cc86abd86e89604407f09d7124718b7d9a/readme_resources/transformation.gif) in higher quality.

{:refdef: style="text-align: center;"}
![A clip of PULSE in action.](/2020/case-study-on-ethics-and-bias-in-machine-learning/transform-clipped.webp)
{: refdef}

## Ethical Concerns

One of the key takeaways from the process detailed above is that "Face Depixelizer" and PULSE are not reconstructing anything from the original image - they're exclusively optimizing a random output to be a "perceptually realistic" upscale of the original image.

The wave of concern focused principally on use of "Face Depixelizer" to reassemble photos of protesters on social media - where they'd been intentionally pixelated for privacy and safety - and other socially damaging applications in the wake of Black Lives Matter & ANTIFA protests across the United States. Many even called for Malimonov to remove Face Depixelizer from Google Colab. This was so widespread that PULSE [updated](https://github.com/adamian98/pulse/commit/5a130c91862975234f1a77a982e4cefa58e6e0b9) their README to clarify the function of their model:

> "We have noticed a lot of concern that PULSE will be used to identify individuals whose faces have been blurred out. We want to emphasize that this is impossible - **PULSE makes imaginary faces of people who do not exist, which should not be confused for real people.** It will **not** help identify or reconstruct the original image."

The confusion here mostly stemmed from Malimonov's name for the project. As it's purely an adaptation of PULSE to make it more available to common users, and not extending or augmenting PULSE, the most accurate name would be akin to "pulse-portable" or "pulse-colab." You don't have to be a marketing guru to understand why Malimonov went with the cooler-sounding but technically incorrect "Face Depixelizer" name - it's much higher-interest and therefore higher-impact when shared on social media.

So while I understand why the Malimonov decided to market it this way, I deeply disagree with the "Face Depixelizer" name. Tech needs to be careful with how it names things - especially AI/ML - because it's easy to mislead the public. Consider this a [Vitamin Water](https://www.reuters.com/article/coca-cola-vitaminwater-settlement/coke-to-change-vitaminwater-labels-to-settle-u-s-consumer-lawsuit-idUSL1N1211HX20151001)-esque example - misleading the public about what your [beverage, ML model] does is both possible and profitable, because the people who have the means and reason to check beyond the basic information about it are few. A litmus test I would apply here is: if you introduced "Face Depixelizer" to a jury as-named (assuming it could be admitted as evidence), how many of them would question the powers of technology to reassemble a grainy CCTV image into the very presumed-criminal they see before them? I suspect the answer is "not enough" - and it's not their fault, I expect them to be computer scientists as much as Coca-Cola Co expects customers to be molecular biologists and nutritionists.

Though, will it happen here? I doubt it. This has been very publicly lauded a biased ML model and would be hard to sell anywhere that did even a shred of diligence during acquisitions, but if I'm ever called to testify on the workings of "Face Depixelizer" or PULSE I will be sure to blog about it.

## Bias Concerns

One note to clarify before we talk about bias is that in this context is what exactly bias means. Per Ayodele Odubela's talk "[Combatting Bias in ML](https://www.youtube.com/watch?v=NIxUAlmnqz0)," a bias error is an error from erroneous assumptions in the learning algorithm. High bias can cause an algorithm to miss the relevant relations between features and target outputs (underfitting). For clarity, I'll split the discussion of these issues into "Machine" and "Social" bias subsections.

#### Machine Bias

In a discussion about "Face Depixelizer" and bias in ML, [Yann LeCun](http://yann.lecun.com/) (Chief AI Scientist at Facebook) summarized core [sources](https://threadreaderapp.com/thread/1275162528511860737.html) of bias in machine learning as:

1. The data - how it's collected and formatted.
2. The features - how they are designed.
3. The architecture of the model.
4. The objective function.
5. How it's deployed.

Our focus will rest squarely on #1 and #3 for much of this discussion - while the others are certainly dangerous areas for bias to emerge, the pixel-to-pixel comparison nature of PULSE mostly eliminates bias from #2 and #4 (ex. there are no features or functions, so bias would be minimal in this case), and there is no application to "Face Depixelizer" so #5 is right out.

###### Architecture Evaluation

The first thing to weed out is if this is an issue with "Face Depixelizer" or something further down the dependency chain? "Face Depixelizer" does diverge from PULSE's methodology in one key place: a default seed of `100` is set, as opposed to the random initialization that the PULSE team used for their experiments. By starting from a consistent location, it allows people to generate the same faces, so people in Twitter threads will generate the same image and not wonder why their output image is different from other peoples'. However, if that starting point is white-presenting, this would also create white-biased results - instead of starting the search for a believable face from a "random" (ergo, fair) starting location, perhaps the seed `100` generates a white-presenting face which influences the possible set of outcomes. To quickly debunk this as the *main* issue, I ran a number of tests with random seeds [100 - 100,000,000,000] using a 16x16 pixelated portrait of an Asian woman and a universally white- and female-presenting face was generated every time.

Is it a precise test? No, but it's enough that I strongly doubt this is the root cause, especially since the PULSE team pointed a finger at a very big smoking gun when they [added](https://github.com/adamian98/pulse/commit/fe02cb9dd48ccd1959111143713311ce0d73edcb) a bias section to their paper which discusses sources of bias in depth, leading with:

> "While we initially chose to demonstrate PULSE using StyleGAN ... as the generative model for its impressive image quality, we noticed some bias when evaluated on natural images of faces outside of our test set. In particular, we believe that PULSE may illuminate some biases inherent in StyleGAN."

Further, they identified and considered a number of potential sources of bias that PULSE would introduce over StyleGAN, and ran some tests using a tagged and diverse sample called FairFace by Kimmo Kärkkäinen and Jungseock Joo ([GitHub](https://github.com/dchen236/FairFace), [paper](https://arxiv.org/abs/1908.04913)) to determine PULSE's success rates when generating upsampled images for faces of many races. They found that generally, images were more readily generated for Asian, Indian, and Latino/Hispanic persons, but noted this test:

> "... only reports whether an image was found - which does not reflect the diversity of images found over many runs on many images, an important measure that was difficult to quantify."

##### Data Evaluation

Where would the biases inherent in StyleGAN come from? While StyleGAN might have architectural flaws leading to bias (...it's turtles all the way down...), it appears that the root of bias here is the dataset used to train StyleGAN (which was used by "Face Depixelizer"): [Flickr Faces HQ](https://github.com/NVlabs/ffhq-dataset) (FFHQ). FFHQ is a well-regarded [dataset](https://www.kaggle.com/arnaud58/flickrfaceshq-dataset-ffhq/) of 70,000 human faces at 1024x1024 resolution, crawled from [Flickr](https://en.wikipedia.org/wiki/Flickr) by NVIDIA researchers Tero Karras, Samuli Laine, and Timo Aila, commonly used to train GANs. It is specifically noted by the publishing researchers that FFHQ "[inherits] the biases of [Flickr]", and does not appear to be filtered to remove any race or age bias.

So in effect, FFHQ is fully subject to the usage demographics of Flickr itself. I was unable to find clear demographic information about Flickr from a reputable source to perform my own analysis. Luckily for us, Joni Salminen, Soon-gyo Jung, Shammur Chowdhury, and Bernard J. Jansen from the Qatar Computing Research Institute already wrote a [paper](https://dl.acm.org/doi/pdf/10.1145/3334480.3382791) titled "Analyzing Demographic Bias in Artificially Generated Facial Pictures" which analyzed crowdsourced race and age evaluations for 1,000 randomly generated images using a StyleGAN model trained on FFHQ. Their findings were staggering, and their model generated strongly biased results - 72.6% white-, 13.8% asian-, 10.1% black-, and 3.4% Indian-presenting. While that's not a comprehensive evaluation, it does speak well to FFHQ's biases and possibly Flickr's demographics as a whole. Here's a sample of generated faces as well:

{:refdef: style="text-align: center;"}
![FFHQ simulating a mayonnaise spread.](/2020/case-study-on-ethics-and-bias-in-machine-learning/ffhq-demo.png)
{: refdef}



Many of PULSE's comparisons in their paper come from using [CelebA HQ](https://www.tensorflow.org/datasets/catalog/celeb_a_hq) which is known to be white-biased. 


#### Social Bias



