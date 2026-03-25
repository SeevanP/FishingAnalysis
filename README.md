# **Personal project combining my two favourite things, analyzing data and fishing.**

-This will be an ongoing project and will be updated every few months.

**This project is very important to me as an avid fisherman and lover of nature. I take great pride in the environment and how it is treated. Unlike many projects that rely on downloaded datasets, I wanted to apply knowledge in the real world.**

**My focus is on the prized Spotted Grunter in local South African estuaries, which are fragile ecosystems affected by pollution, overfishing, and habitat loss. I wanted to understand how populations are doing, and over time my curiosity grew—I began exploring how environmental variables influence fish behaviour and catch success.**

**Currently, I’ve collected 247 samples, all caught by my father and me. While this is not yet enough to build a fully predictive model, it forms the foundation for a long-term dataset. My plan is to continue expanding the dataset, including collaborating with local fishing content creators, to target fish sustainably, reduce bycatch, and contribute to conservation.**

**This project reflects both my technical and ethical approach: combining data analysis, predictive modeling, and fieldwork to make fishing practices smarter and more sustainable.**



## **Problems faced:**


1. Small dataset: Although I have spent over a year collecting the data myself I realized my dataset is limited for a reliable prediction model.

**THIS DOES NOT MEAN I WILL GIVE UP!!**

2. So many variables when it comes to fishing!!!

3. While I cannot replicate exact weather, tides and water quality. I did have certain controlled factors such as tackle,bait and selected tidal coefficient days. Each location was consistently fished with 6 months each on every possible weekend.


## **Mindset shift:**

I know they say size doesn't matter and they're right! My small dataset still has valuable information as it can this help keep track of species populations, aiding conservation in 2 important estuaries in Durban, South Africa, that being Durban Harbour and Blue Lagoon.

## **What have I learnt?**
Well fishing is unpredicatable as ever but man is it fun. I can reliably say that tidal coefficients affect these 2 estauries and how feeding is influenced. There will be an update in the coming days showing a graph and fish species amongst them. 


## Here are some plots and little explanations.

<img width="1000" height="500" alt="Figure_1" src="https://github.com/user-attachments/assets/8c2fe9d5-09b5-4c82-b70a-87a30d1182bc" />

> ### Figure.1 (Species Distribution):

Here in the first figure we see what an amazing fisherman I am! Just kidding or am I? What is present though are thornfish. We currently see them dominate these 2 areas especially at Blue Lagoon and once you get through these little bait thieves we can get quality fish like Spotted Grunter. The pink graph represents various species that had few catches to fully define them as their own category such as other grunter species, kingfish(trevally) and kob.











<img width="1000" height="600" alt="Figure_2" src="https://github.com/user-attachments/assets/1e2888b1-924c-4981-b735-afc16d0e76af" />



> ### Figure.2 (Tidal coefficients and feeding):

Here we see how tidal coefficients can affect fish feeding behavior. Goldilocks strikes again—mid-range tidal coefficients seem “just right.” Not too strong currents, not completely slack (zero movement). Fish can comfortably sit in ambush positions without fighting the current, which I was lucky enough to witness firsthand.








<img width="800" height="500" alt="Figure_3" src="https://github.com/user-attachments/assets/1579ac8b-1d5d-47dc-82c3-c9c72b9313f7" />

> ### Figure.3 (Grunters and Blackktails by tidal coefficients):

These are generally highly sought after fish for eating, I would not know as I do not eat seafood. I will have to make a new plot about the difference in grunter catches in Blue Lagoon and Durban Harbour depending on tidal coefficients as the current differs between those 2 systems. What I can tell from this though is Blue Lagoon did have higher grunter catch rates on a low tidal coefficient and in Durban Harbour a high tidal coefficient yielded a higher success rate. This may be due to current being more manageable in Blue Lagoon during low tidal coefficients which is highlighted below in figure 5 in terms of catch rate in these 2 systems.
Blacktails seem to enjoy mid-range tidal coefficients as they generally hug structure and can quickly grab food passing through manageable currents.







<img width="800" height="500" alt="Figure_4" src="https://github.com/user-attachments/assets/621f4983-542b-407c-926a-6c4b2fa84fcc" />

> ### Figure.4 (Bait effectiveness):

Trust me when I say I did try a multitude of baits but we see squid reigns supreme. I guess fish like their calamari to ;-)








<img width="1000" height="600" alt="Figure_6" src="https://github.com/user-attachments/assets/4238cd0a-e6ba-4471-9fa9-31a5234599b2" />

> ### Figure.5 (Area performance by tidal coefficients):

Here we see which area fishes better depending on the tide. Durban Harbour and Blue Lagoon perform differently in certain conditions. Although both are estuaries, the accessible fishing spots vary. Blue Lagoon generally has fishermen along the sides of the river leading to the mouth. Since the mouth is narrow, the current can be strong, making it difficult to keep bait in place. Even when I managed, the fish seemed reluctant to feed. I mean, if I had powerful wind in my face, I’d probably feed my clothes and surroundings more than myself.

Durban Harbour has only one known fishing area: the yacht mole. You can fish off the right side into a massive sandbank. Because the area is wide, the current’s energy spreads out, and feeding fish don’t seem bothered, based on my observation. This does mean that during high tide, casting for specific sandbanks is important—water levels rise, fish aren’t concentrated in one area, and they tend to sit in channels or drop-offs.







<img width="1920" height="967" alt="Figure_7" src="https://github.com/user-attachments/assets/5ef73d23-03a4-4b1e-bf48-c90ead3aedfe" />


> ### Figure.6 (Grunter catch rate between Blue Lagoon and Durban Harbour)

So based on Figure.3, I realized I was seeing something interesting. There is a difference in Spotted Grunter catch rate amongst these 2 systems. The optimal feeding times for each location seem to differ between the same species, depending on the location.




### **Here are some code snippets and their purpose.**



## This is a little snippet of code highlighting how I sorted my data and cleaned it.


<img width="822" height="714" alt="Screenshot (298)" src="https://github.com/user-attachments/assets/14fbce75-3a86-4b9a-a948-bf565395fd7d" />


## A code snippet showing how data will be visualized for others to understand


<img width="791" height="709" alt="Screenshot (299)" src="https://github.com/user-attachments/assets/ccd5e58f-fc6c-486a-856f-46981d5fd16c" />



