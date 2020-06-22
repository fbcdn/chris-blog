---
layout: post
title: "Introduction to Bias and Ethics in Machine Learning"
permalink: /2020/introduction-to-bias-and-ethics-in-machine-learning/
description: "Can machines be racist? Sure thing. Most of the issues dealing with machines and bias are insidious - crime, credit, or recidivism prediction, to name a few - where bias is hidden and implicit. This piece is about a foundational (and thankfully lower-stakes) example of bias in machine learning, and a discussion of the ethical and social issues that come with it."
date: 2020-06-21 00:00:00
tags:
- machine-learning
- bias
- ethics
- gan
- opinion
---

So recently, Denis Malimonov ([@tg_bomze](https://twitter.com/tg_bomze/status/1274098682284163072)) released a project called "Face Depixelizer" which implements [PULSE](https://github.com/adamian98/pulse) supplemented with pretrained data on [Google Colab](https://colab.research.google.com/github/tg-bomze/Face-Depixelizer/blob/master/Face_Depixelizer_Eng.ipynb) for ease-of-use. The project isn't net-new - it's just democratizing access to prior work by the PULSE team and providing some presets. PULSE + Denis' training data can successfully produce "believable" results in many cases, and can generate realistic faces even from 1990s-era game sprites readily.

{:refdef: style="text-align: center;"}
!['De-pixelated' sprite of Wolfenstein 3D's "William Joseph B.J. Blazkowicz on a grey background"](/2020/introduction-to-bias-and-ethics-in-machine-learning/wolfenstein.png)
{: refdef}

However, it incited massive, negative discussion centered around three points.

First, there are significant ethical concerns around reconstructing people's faces from pixelated images. For example, identifying people in situations where they wished to remain anonymous by pixelating their faces - protesters, whistleblowers, etc. Thankfully this isn't possible with this tool - hence, "Face Depixelizer" is a very misleading name, and I'll be putting it in quotation marks throughout this article. We'll cover some latent ethical concerns that arise *just from that name*, but not until much later.

Second, this particular model tends to output traditionally white faces and features. A pixelated photo of Barack Obama would reliably generate a face that looked like the offspring of Tobey Maguire and Sean Connery - this was first noted by Cédric Sütterlin ([@Chicken3gg](https://twitter.com/Chicken3gg/status/1274314622447820801)), and follow-ups by Ken Chic ([@bitcashio](https://twitter.com/bitcashio/status/1274408339602993155/)) confirmed that it happened consistently.

{:refdef: style="text-align: center;"}
!['De-pixelated' photo of Barack Obama which comes out a little too mayonnaise-y.](/2020/introduction-to-bias-and-ethics-in-machine-learning/barack-obama.png)
{: refdef}

Further demonstrations, including a fairly high-resolution photo of Muhammad Ali processed by Zain Amro ([@zainamro](https://twitter.com/zainamro/status/1274361534886469632/)), almost always resulted in an output face with exclusively white-presenting features. This lead to the third claim - that the algorithm itself is racist. While the PULSE team is looking into this, I strongly doubt it, since it looks to be the product of biased inputs to StyleGAN.

In order to understand why that's relevant, let's learn more about what's happening under the hood.

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
![A clip of PULSE in action.](/2020/introduction-to-bias-and-ethics-in-machine-learning/transform-clipped.webp)
{: refdef}

Since this is an optimization problem, many successes - and failures - of "Face Depixelizer" and PULSE will likely come from the training data that's available. Since PULSE approaches believable face generation as an optimization problem, we can think of this somewhat like finding the local maximum in a set of data - let's call that the "local maximally relevant face." You can start anywhere on a map of features, skin tones, haircuts, light and dark backdrops, ad infinitum, and need to find to the closest approximate match to - oh, a 16x16 pixel grid. This is an oversimplification, but bear with me for a little longer.

If a biased sample of data is loaded in to StyleGAN, a random starting point is less likely to reach an accurate maximally relevant face - ex. if ninety five out of one hundred faces that are generated by StyleGAN are variants of white faces, and you are searching to 'depixelate' Barack Obama's portrait, the chance that PULSE finds "Daniel Craig in shadow" is higher than "Barack Obama." So with better data, it doesn't mean that a more accurate face will be generated every time, but rather that more diverse faces are readily generated, so instead of a sea of white-presenting faces being generated by "Face Depixelizer" in Twitter threads, there would be more diverse (and more accurate *overall*) generated faces.

Many of PULSE's comparisons in their paper come from using [CelebA HQ](https://www.tensorflow.org/datasets/catalog/celeb_a_hq) which is known to be conventionally-pretty- and white-biased. "Face Depixelizer"'s pretrained model appears to be based on [Flickr Faces HQ](https://github.com/NVlabs/ffhq-dataset) (like StyleGAN), which was compiled by NVIDIA and directly "inherits the biases of [Flickr]" - so no filtering for bias was done.

## Ethical Concerns

One of the key takeaways from the process detailed above is that "Face Depixelizer" and PULSE are not reconstructing anything from the original image - they're exclusively optimizing. A simplistic way of interpreting this is that they're throwing spaghetti at the wall and mashing it around until it's close to a target - not analyzing the original target to make a spaghetti masterpiece. In the wake of the recent outcry, PULSE also [updated](https://github.com/adamian98/pulse/commit/5a130c91862975234f1a77a982e4cefa58e6e0b9) their README to clarify:

> We have noticed a lot of concern that PULSE will be used to identify individuals whose faces have been blurred out. We want to emphasize that this is impossible - **PULSE makes imaginary faces of people who do not exist, which should not be confused for real people.** It will **not** help identify or reconstruct the original image.



Want more? You should look at Ayodele Odubela's talk: "[Combatting Bias in ML](https://www.youtube.com/watch?v=NIxUAlmnqz0)"