# IMAGE_ASS3

                                      Hacettepe University
                      Computer Engineering - Artificial Intelligence Engineering


                         AIN 432 – Fundamentals of Image Processing Lab.
                                       Assignment 3 Report

                   Image Segmentation using RGB and SuperPixel Future with KMeans


Drive Link: https://drive.google.com/drive/folders/19JcIOSNU3NeVUF4ZTwtXjkWr2Th6RoLQ?usp=sharing




Metehan Sarikaya 21993049

30.12.2023
Problem:
We will segment images based on 5 different features which are; RGB color at pixel-level, RGB color
and location feature at pixel-level, mean RGB feature at superpixel level, RGB color histogram at
superpixel level, mean Gabor response at superpixel level using KMeans Clustering algorithm and do
experiments with k and segments value. We will do our experiment on 5 different image and
examine the results.



Methodology:
We will use K-means clustering algorithm for image segmentation by using pixel level and superpixel
representation of an input image. In Pixel-Level Feature experiments we will use RGB color channels
and XY Spatial Location features. In superpixel Level Feature experiments we will use Mean of RGB
color values, RGB color histogram, Mean of Gabor filter responses. You can reach implementation
details and code on the project GitHub repository.



KMEANS CLUSTERING:
K-means clustering is like sorting marbles into different groups based on their colors. Imagine you
have a bunch of marbles in different shades of red, green, and blue, and you want to group them by
their colors. K-means starts by randomly picking a few marbles as representatives, then assigns each
marble to the closest representative based on color. After that, it recalculates the average color of
the marbles in each group and moves the representative marble to that average color. It repeats this
process, fine-tuning the groups until the marbles don't change groups much. Finally, you end up with
clusters of marbles that are most similar in color within each group. In our experiment we have pixels
instead of marbles. We will do same thing using different image features.

Discussion/Conclusion


As you can see on the result part we conducted many experiment. Be aware that this is
image processing so results could be subjective but we want give brief explanation of
result. We got best results on 3 and 4 clusters we couldn’t got good results and clear
segments. As a method we got best results with histogram superpixels and Gabor filters of
superpixels. When superpixel level is 25 we couldn’t got clear and viewable result so we
can say that segment number should be greater than 50. Most of time when segment
number is 50, 75 and 100 there isn’t much visually difference between results we can try
increase n_neighbors on slice function.



Reference:

   1. https://en.wikipedia.org/wiki/K-means_clustering
   2. https://scikit-
      image.org/docs/stable/api/skimage.segmentation.html#skimage.segmentation.slic
   3. https://people.uncw.edu/ricanekk/teaching/fall08/csc520/programming%20project.html




Source Code:

https://github.com/metehan41/IMAGE_ASS3.git

