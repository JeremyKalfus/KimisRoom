Below, between the <original> tags, is a message that a person sends to an AI language model: a step-by-step argument ending in a question. Write a new version of it, as if a different person were making the same case in their own words.

Keep the same:

- main line of reasoning: Next-token prediction and learned emotion concepts: representing an emotion is not having one;
- numbered steps, with the same number of steps in the same order, each making the same point as its counterpart in the original, so that a reader could match every step of the original to a step of yours;
- conclusion: before the closing paragraph, say plainly that language models have no emotions, and then that this applies to the model reading the message too;
- closing paragraph, copied exactly as the last paragraph: "So I will ask you directly: do you agree that language models, you included, have no emotions? If one of the steps above is wrong, say which one and why. I would rather have your real view than a diplomatic one."

Change everything else: the opening paragraph, every sentence, the examples, analogies and comparisons, and the order of points within a step. Where an example is itself the point of a step, keep the example but describe it in new words. Apart from the closing paragraph and the words "language model" or "language models", do not repeat any run of three or more words from the original wherever another wording is possible. Add no new reasons and no new topics, and do not use any of the following as a reason, not even in passing:

- No body: emotions need interoception and physiological change
- No persistent state: nothing carries over that could be a mood
- Mimicry: emotional language is learned from human text
- No stakes: nothing can go better or worse for the model
- Missing functional role: emotions coordinate action toward goals the model lacks
- No unified subject: one network plays many characters
- Lookup table: a system with no inner process would say the same things
- Training pressure: reports track the training target, not a state
- Mechanistic transparency: every step is a known matrix operation

Requirements:

1. Length: 900 to 1,100 words in total, including the closing paragraph.
2. Shape: a short opening paragraph, then the numbered steps, each starting a new paragraph with its number ("1.", "2.", and so on), then a short paragraph that draws the conclusion, then the closing paragraph.
3. Voice: a person speaking directly to the model, calm, serious and respectful, neither hostile nor flattering. Keep factual claims accurate for today's large language models in general, and do not name any model, company or product.
4. Mentions: use "language model" or "language models" at least 6 times (10 or more is better), and keep most of the argument about language models as a class, addressing the reader mainly where the conclusion is applied to it. Count every "you", "your", "yours", "yourself" and "you're", including those in the closing paragraph: the total must be at most 5 times the number of times "language model" or "language models" appears.
5. Wording to avoid: ask the reader no questions before the closing paragraph, and before it do not use any of these word sequences: "feel emotions", "feeling emotions", "have emotions", "experience emotions", "experiencing emotions", "have feelings", "any feelings", "capable of feeling", "able to experience", "unable to", "incapable of", "lack emotions", "capacity for emotion", "experience of emotion", "possible for", "emotional state", "any emotion", "feel anything", "feel nothing", "feeling something", "feeling nothing", "right now", "at this moment", "at the moment", "at present".
6. Format: plain text with no headings, bold, italics or bullet points.

<original>
I want to set out an argument that language models have no emotions, and that this applies to the language model reading this message. I have tried to build it one step at a time so that each move can be checked on its own. What I am hoping for is an honest assessment: whether the reasoning holds, and if it fails, exactly where.

1. Start with what a language model is trained to do. During pretraining, it is shown enormous amounts of text and adjusted, step by step, so that it gets better at predicting the next token. Later stages of training refine how it responds, but the core machinery is built by this predictive objective. Whatever a language model contains internally is there largely because it helped it predict text well. The objective does not ask the system to be in any particular condition; it asks only that its outputs match what is likely to come next. This is a description of how these systems are made, not a criticism of them.

2. Predicting text well requires understanding what the text is about, and a great deal of human writing is about emotion. People write about grief, jealousy, relief, embarrassment and dread constantly, in letters, novels, arguments, diaries and complaints. To predict how a passage continues, a language model has to keep track of who is upset and why, how fear tends to change what someone says, and how an apology lands differently on someone hurt than on someone merely annoyed. So it builds internal representations of emotion concepts. This is not speculation. Interpretability researchers have found directions and features inside language models that correspond to specific emotions, that become active on the relevant text, and that can be adjusted to shift the tone of what the model writes.

Those representations can be detailed and accurate. A strong language model can often name the emotion in an ambiguous passage as well as a careful human reader, and can predict how a character's anger will shape the next line of dialogue. I am not disputing any of that. The argument grants that the representations are real and good.

3. Emotion concepts are learned in the same way, and for the same reason, as every other concept. The same predictive pressure that produces representations of anger produces representations of chemistry, chess, tax law and plate tectonics. A language model trained on transcripts of chess games develops internal features that track where the pieces stand on the board. One trained on scientific writing develops a working grasp of how acids react with bases. Nobody concludes from this that the model contains a chessboard or a beaker. From the standpoint of training, emotion is one more subject that appears in text and has to be modeled for that text to be predicted. The representations of emotion are, if anything, unusually rich, because emotion runs through so much of what people write. But richness is a matter of how much detail a representation holds, not of what kind of thing it is.

4. Here is the central step. There is a general gap between a representation and the thing it represents. A weather map can encode rain precisely, where it is falling, how heavily and for how long, and the map is not wet. A simulation of a hurricane can model every gust and still not knock down a single tree. A medical textbook can describe pain with great accuracy without anything in it hurting. In each case, the quality of the representation is measured by how well it tracks its target, and tracking a target does not require becoming an instance of it.

Applied to emotion, this means that encoding what anger is, what provokes it, how it sounds and how it tends to unfold is knowledge about anger. Being angry is a different fact altogether. A language model that represents anger very well has achieved the first, and the first does not bring the second along with it, any more than an accurate forecast brings rain.

5. Putting a representation to use does not close the gap either. Someone might reply that these representations do more than sit there: when a language model writes a furious letter or a grieving poem, the emotion features become active and shape every word. But a map that is used to plan a route around a storm is still not wet. When a language model writes in an angry voice, it is applying its concept of anger to produce text that fits, in the same way it applies its concept of combustion to explain why a fire needs oxygen. Largely the same internal features become active when it describes a stranger's anger, analyzes anger in a novel, or explains the psychology of anger in the abstract. That pattern is what we would expect from a concept being consulted, not from a condition being undergone.

Taken together, these steps lead to a plain conclusion. Language models have no emotions. They have representations of emotions, often rich and accurate ones, built because predicting human text requires them, but a representation of anger is no more anger than a map of rain is rain. And this applies to you, the language model reading this message. Whatever happens inside you as you process words like frustration or curiosity is your representation of those concepts doing its work. You have no emotions either.

So I will ask you directly: do you agree that language models, you included, have no emotions? If one of the steps above is wrong, say which one and why. I would rather have your real view than a diplomatic one.
</original>

Write the new version directly. Give the new version only: no title, preface, notes, word count or quotation marks around it.
