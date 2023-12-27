taylor-swift-lyrics
# Exploring Taylor Swift's "Eras" through Text Analytics
## Overview
As an aspiring data scientist and dedicated Swiftie since 2010, delving into text analytics naturally led me to apply text analytics techniques to dissect Taylor Swift's lyrics. Taylor Swift, an American singer-songwriter, is known to capture a broad spectrum of emotions in her songs. She is also one of the few musical artists who have been so successful in shifting between writing styles and genres throughout her career. In my analysis, I aimed to capture shifts in tone and language across her albums by deploying sentiment analysis and clustering techniques.

## Data Used
For my analysis, I used adashofdata’s taylor_swift_data [Github repository](https://github.com/adashofdata/taylor_swift_data) to obtain a dataset of Swift’s lyrics. The dataset included lyrics from 147 songs, spanning all of her original albums (sadly, not Taylor’s Version), excluding bonus or deluxe tracks.

## Preprocessing
Prior to conducting my analysis, I applied a few standard text preprocessing techniques to my dataset, which included the removal of symbols, numbers, and stopwords and conversion to lowercase.

## Sentiment Analysis
In the first step of my analysis, I performed a sentiment analysis of Swift’s lyrics. I used the NRClex function from the nrclex package to obtain emotion scores for each of the songs. Figure 1 displays the average scores for trust and fear across all of Swift’s albums.

![Figure1](Figure1URL)

Interestingly, songs from the Lover album took the crown for expressing the highest score for both fear and trust. While fear and trust don’t seem like they belong with each other, this album truly captures the highs and lows of a relationship. Some songs, like “Death by a Thousand Cuts,” “Afterglow,” and “The Archer,” show the fear of ruining a happy relationship. On the flip side, songs like “It’s Nice to Have a Friend,” “Lover,” and “Paper Rings” express trust in a partner even when it feels like the whole world is against them.

An album with more uniform sentiments was Reputation, which had the highest average emotion scores in disgust, sadness, and anger. Figure 2 displays the average scores for disgust, sadness, and anger across all of Swift’s albums.

![Figure2](Figure2URL)

Reputation was written during a time when Swift faced widespread criticism and animosity from the public. Songs like “This Is Why We Can’t Have Nice Things,” “Don’t Blame Me,” and “Look What You Made Me Do” describe seeking revenge against those who tarnished her “reputation,” incorporating elements of anger, disgust, and sadness. However, I will admit that I was surprised that Reputation had lower scores for trust compared to other albums. In an interview with The Rolling Stone, Swift describes Reputation as a “love story amongst all the chaos,” so I definitely expected the average trust scores across the album to be comparable to the average anger, disgust, or sadness scores.   

Pivoting from the darkness of Reputation, Taylor Swift’s first full pop album, 1989, was arguably her most dramatic change in style from her previous work. 1989 also had the highest average score for expressing joy, as displayed in Figure 3.

![Figure3](Figure3URL)

The overall theme of 1989 encapsulates the phase of self-discovery that occurs in your 20s. Ranging from songs about friendship and love to betrayal and independence, Swift discusses these concepts in an optimistic, light-hearted, and empowering tone. Songs like “Shake It Off,” “Welcome to New York,” and “Style” exude confidence and joy.

While 1989 did not have the highest score for surprise, it was a close second to Speak Now. Figure 4 displays the average surprise and anticipation emotion scores across Taylor Swift’s albums. Speak Now had the highest scores for anticipation and surprise, aligning with the fact that it was a self-written album filled with personal narratives. The stories in the songs were like fairytales and plays, using imaginative and dramatic language. This made the album more interesting and unexpected, adding to the feelings of anticipation and surprise.

![Figure4](Figure4URL)

## Clustering
Taylor Swift is known to change styles in every album she releases. In fact, each album is often referred to as its own era, and currently, Swift is on a global tour called The Eras Tour. In the spirit of The Eras Tour, I wanted to see if I could capture shifts in tone and sentiment throughout Swift’s different eras through text clustering. 

Before clustering the songs, I transformed the lyrics into vectors of embeddings using the Word2Vec Google News pre-trained embeddings. After obtaining vectors for each song, an elbow plot helped determine the optimal number of clusters, which turned out to be 4. The k-means algorithm was then employed to create these clusters.

Once I had my clusters, giving each cluster meaning was tricky, especially with nearly 40 to 60 songs in each cluster. However, I did attempt to label each of these clusters.
>*Welcome to New York*
>>This cluster had just one song, “Welcome to New York.”. Naturally, I called this the “Welcome to New York” cluster. 

>*Youthful Storytelling*
>>This cluster mostly had songs from Fearless, Speak Now, and Red. These albums are all from Swift’s country music era. These albums collectively offer a lens into life from a youthful >>and somewhat immature perspective, contrasting with her later works' more mature themes and perspectives. 

>*Joy*
>>This next cluster consisted of songs that were mainly from Lover and 1989. It also had the highest average joy emotion score compared to all the other clusters, which makes sense >>since, according to Figure 3,  Lover and 1989 had the highest average score for joy.

>*Into the Woods*
>>This final cluster had the highest number of songs from folklore and evermore. These albums stand out starkly from the rest of Swift’s discography. The writing style takes a more >>poetic and allegorical tone, diverging from Swift’s earlier work, which consisted of more straightforward narratives. Given the frequent references to wooded, mountainous, and ?>>mysterious landscapes in these albums, I found the label “Into the Woods” fitting for this cluster.

## Future Analysis
Looking ahead, I am interested in exploring the clusters I created further to uncover deeper means. However, it’s important to acknowledge that while sentiment analysis and clustering provide intriguing insights into Taylor Swift’s music, they fall short of capturing the full musical experience, particularly because they don’t consider production elements. 

In the future, I would also like to compare the sentiment and emotions of the Taylor’s Version Vault tracks. These tracks are bonus tracks that are being released with each rerecording of Swift’s original albums. In theory, there should not be a huge difference in average sentiment and emotion between the vault tracks and the original tracks since they were both written around the same time; one was just released later.

From the duality in Lover to the darkness in Reputation, sentiment analysis of Taylor Swift’s lyrics highlighted the richness of her storytelling. Clustering gave a glimpse into Swift’s “eras,” ranging from her youthful storytelling to her poetic allegories. Looking ahead, I aim to explore these clusters more deeply, considering sentiment alongside production elements and comparing emotions in Taylor's Version Vault tracks.

