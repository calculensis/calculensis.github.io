Title: simple tools, part 2: the linear model
Date: 2022-07-20
Category: basics
Tags: 
Slug: linear model
Authors: Kayla Lewis
Summary: I describe the linear model for decision making

<img align=right src="images/arrow.jpg" width="150"/>

For our first simple decision making tool, let's look at the 2 to 3 factor linear model. Suppose you're trying to decide which of two houses to buy, $H_1$ or $H_2$, and you're really on the fence about it! 

The linear model approach asks you first to consider what are the 2 or 3 most important attributes for you that a potential home could have. For example, let's say they are affordability of monthly payment (A), that feeling of charm when you first walk in (C), and typical level of quietness (Q). Granted, this last attribute might be hard to get at, but knowing that it's one of the most important qualities for you would be important, and it could get you thinking of ways you might determine it, perhaps by interviewing some of your new would-be neighbors.

The next step is to decide how important these attributes are relative to one another and assign them weights; let's call them $w_A$, $w_C$, and $w_Q$. The weights will be numbers between 0 and 1 such that when you add them together you get 1. For example, if each of the above attributes is equally important to you, then they all get weight 1/3. Or if, say, affordable monthly payment is twice as important as level of quietness, and quietness is just as important as the charm factor, you would have $w_A=2/4$, $w_C=1/4$, and $w_Q=1/4$. You may have to play around to find the right values, but you can always just guess, see if the weights sum to 1, and adjust as needed. For the rest of our house example, let's use the weights $w_A=2/4$, $w_C=1/4$, and $w_Q=1/4$.

Now you would rate homes $H_1$ and $H_2$ based on the attributes, where each attribute gets a score from 0 (worst) to 10 (best). For example, maybe $H_1$ is a super affordable house in a moderately quiet neighborhood; in that case, you might might score it as A = 9 and Q = 7. Maybe it's not so high on charm, so C = 3. The linear model we've been constructing would then have us calculate a total score $S(H_1)$ for house 1 using the formula

$$
S(H_1) = w_A A + w_C C + w_Q Q 
$$
$$
=\frac{2}{4}(9)+\frac{1}{4}(3)+\frac{1}{4}(7)=7.
$$ 

Suppose the second potential home is charming indeed (C=9) but not so affordable (A=3), and that it's in a medium-noise environment (Q=5). Then we get

$$
S(H_2) = \frac{2}{4}(3)+\frac{1}{4}(9)+\frac{1}{4}(5)=5,
$$

and the model says to buy the first home, $H_1$, because it scores higher.

What if you find yourself not liking that result? Then you can try to figure out why your feelings and the model disagree. Maybe those weights weren't quite right? Maybe the attributes you chose weren't the most important ones to you after all? And if, no matter what you do, you keep finding yourself unhappy when $H_1$ wins, then the model helped you discover how you feel, and you still learned something significant. (Remember, you were on the fence at first!) 

The next model I present will complement this one.

[Discuss on Twitter](https://twitter.com/Estimatrix/status/1555693184977600512?s=20&t=YFPoxpEQ2Qp14U4FliD7fA)